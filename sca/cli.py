"""Reproducible experiments: python -m sca demo | python -m sca llm."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from .agents import EXECUTOR_PROTOCOL, PlanPolicy, collect, metrics, preflight, run_model
from .report import write_json, write_jsonl, write_reports
from .tasks import ablation_candidates, audit_filters, filter_task, make_tasks


def assert_disjoint(train, test) -> None:
    for field in ("task_id", "env_seed", "instruction"):
        if {getattr(t, field) for t in train} & {getattr(t, field) for t in test}:
            raise ValueError(f"Train/test overlap detected in {field}")
    if any(t.split != "train" for t in train) or any(t.split != "test" for t in test):
        raise ValueError("Incorrect split labels")


def successful_examples(rollouts) -> list[dict]:
    examples = []
    for rollout in rollouts:
        if rollout.split != "train":
            raise ValueError("Test data must not be exported for training")
        if rollout.reward:
            for example in rollout.examples:
                if example.get("valid_action", True):
                    examples.append({"task_id": rollout.task_id, "split": "train", **example})
    return examples


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    sub = command.add_subparsers(dest="command", required=True)
    for name in ("demo", "llm"):
        p = sub.add_parser(name)
        p.add_argument("--output", type=Path, default=Path("outputs") / name)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--train-tasks", type=int, default=120 if name == "demo" else 24)
        p.add_argument("--test-tasks", type=int, default=60 if name == "demo" else 12)
        p.add_argument("--rollouts", type=int, default=4)
        p.add_argument("--eval-k", type=int, default=4 if name == "demo" else 2)
        if name == "llm":
            from .llm import DEFAULT_MODEL
            p.add_argument("--model", default=DEFAULT_MODEL)
            p.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
            p.add_argument("--task-source", choices=("template", "llm"), default="template")
            p.add_argument("--train-steps", type=int, default=40)
            p.add_argument("--max-steps", type=int, default=8)
            p.add_argument("--max-length", type=int, default=2048)
            p.add_argument("--temperature", type=float, default=0.7)
            p.add_argument("--max-new-tokens", type=int, default=128)
            p.add_argument("--preflight-tasks", type=int, default=3,
                           help="Check up to three training task families before the long run; 0 disables")
    return command


def run(args) -> int:
    for name in ("train_tasks", "test_tasks", "rollouts", "eval_k"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.train_tasks > 100_000 or args.test_tasks > 100_000:
        raise ValueError("Use at most 100,000 tasks per split to preserve separate seed ranges")
    if args.command == "llm":
        if not 0 <= args.preflight_tasks <= 3:
            raise ValueError("preflight_tasks must be between 0 and 3")
        if min(args.max_new_tokens, args.max_steps, args.train_steps, args.max_length) < 1:
            raise ValueError("Generation, tool, and training budgets must be positive")
        if args.temperature < 0:
            raise ValueError("temperature must be nonnegative")
    output = args.output
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory is not empty: {output}. Choose a new --output to preserve the earlier run.")
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    write_json(output / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    backend = None
    if args.command == "llm":
        from .llm import HFBackend, environment_versions
        print(f"Loading {args.model}...", flush=True)
        backend = HFBackend.load(args.model, args.device, max_new_tokens=args.max_new_tokens)
        versions = environment_versions()
    else:
        versions = {}

    train_start = 10_000 + (args.seed % 1000) * 100_000
    test_start = 200_000_000 + (args.seed % 1000) * 100_000
    if backend and args.task_source == "llm":
        from .challenger import generate_llm_tasks
        train, train_log = generate_llm_tasks(backend, args.train_tasks, train_start, "train", args.seed)
        write_jsonl(output / "challenger_train.jsonl", train_log)
        test, test_log = generate_llm_tasks(backend, args.test_tasks, test_start, "test", args.seed + 1_000_000)
        write_jsonl(output / "challenger_test.jsonl", test_log)
    else:
        train = make_tasks(args.train_tasks, train_start, "train")
        test = make_tasks(args.test_tasks, test_start, "test")
    train = [t for t in train if filter_task(t)[0]]
    test = [t for t in test if filter_task(t)[0]]
    write_jsonl(output / "train_tasks.jsonl", [t.to_dict() for t in train])
    write_jsonl(output / "test_tasks.jsonl", [t.to_dict() for t in test])
    if not train or not test:
        write_json(output / "status.json", {"status": "insufficient_valid_tasks", "train": len(train), "test": len(test)})
        print("No valid tasks in one split. See challenger logs. No fallback tasks or results were fabricated.", file=sys.stderr)
        return 2
    assert_disjoint(train, test)
    policy = PlanPolicy()
    runner = policy.run if backend is None else lambda task, seed: run_model(
        task, backend, seed, max_steps=args.max_steps, temperature=args.temperature)
    readiness = None
    if backend is not None and args.preflight_tasks:
        print("Checking basic tool execution on training tasks before the long experiment...", flush=True)
        readiness, probes = preflight(train, runner, args.preflight_tasks, args.seed + 3_000_000)
        write_json(output / "preflight.json", readiness)
        write_jsonl(output / "preflight_rollouts.jsonl", [r.to_dict() for r in probes])
        print(f"Preflight: {readiness['successful_attempts']}/{readiness['tasks']} successful tasks", flush=True)
        if readiness["status"] != "passed":
            failure = {"status": "preflight_failed", "backend": "huggingface_lora", "model": args.model,
                       "task_source": args.task_source, "executor_protocol": EXECUTOR_PROTOCOL,
                       "seed": args.seed, "python": platform.python_version(), "versions": versions,
                       "runtime": getattr(backend, "runtime_info", {}),
                       "training_completed": False, "training_rollouts": 0,
                       "requested_train_tasks": args.train_tasks, "requested_test_tasks": args.test_tasks,
                       "valid_train_tasks": len(train), "valid_test_tasks": len(test),
                       "before": None, "after": None, "preflight": readiness,
                       "elapsed_seconds": time.perf_counter() - start}
            write_json(output / "status.json", failure)
            write_json(output / "summary.json", failure)
            print("Preflight failed; baseline evaluation and training were not started. "
                  "Inspect preflight_rollouts.jsonl before spending time on a longer run.", file=sys.stderr)
            return 2
    print(f"Evaluating baseline on {len(test)} held-out tasks...", flush=True)
    before_rollouts = collect(test, runner, args.eval_k, args.seed + 8_000_000)
    before = metrics(before_rollouts, args.eval_k)
    write_jsonl(output / "before_rollouts.jsonl", [r.to_dict() for r in before_rollouts])
    print(f"Collecting {len(train)} x {args.rollouts} training rollouts...", flush=True)
    training_rollouts = collect(train, runner, args.rollouts, args.seed + 2_000_000)
    write_jsonl(output / "training_rollouts.jsonl", [r.to_dict() for r in training_rollouts])
    examples = successful_examples(training_rollouts)
    write_jsonl(output / "successful_actions.jsonl", examples)
    summary = {
        "status": "complete", "backend": "tabular" if backend is None else "huggingface_lora",
        "model": "categorical policy over five scripted strategies" if backend is None else args.model,
        "task_source": "template" if backend is None else args.task_source,
        "executor_protocol": EXECUTOR_PROTOCOL,
        "runtime": getattr(backend, "runtime_info", {}) if backend else {},
        "preflight": readiness, "training_completed": False,
        "seed": args.seed, "python": platform.python_version(), "versions": versions,
        "requested_train_tasks": args.train_tasks, "requested_test_tasks": args.test_tasks,
        "valid_train_tasks": len(train), "valid_test_tasks": len(test),
        "training_rollouts": len(training_rollouts),
        "rewarded_training_rollouts": sum(r.reward for r in training_rollouts),
        "successful_action_examples": len(examples),
        "training_reward_oracle_disagreements": sum(r.reward != int(r.success) for r in training_rollouts),
        "before": before, "after": None,
        "limitations": ["Synthetic retail environment, not M3ToolEval or TauBench.",
                        "A single seed and small sample do not establish statistical significance.",
                        "Verifier checks do not prove natural-language instruction alignment."],
    }
    ablations = audit_filters(ablation_candidates(seed=args.seed))
    if not examples:
        summary["status"] = "insufficient_successful_rollouts"
        summary["elapsed_seconds"] = time.perf_counter() - start
        write_reports(output, summary, ablations)
        print("No rewarded training rollouts: training was not run. Increase rollouts/tasks or use a stronger model.", file=sys.stderr)
        return 2
    if backend is None:
        trained = policy.fit_successes(training_rollouts)
        write_json(output / "policy.json", trained.weights)
        after_runner = trained.run
        summary["training_method"] = "success-count update of categorical strategy probabilities; not LLM training"
    else:
        from .llm import train_adapter
        # Save baseline artifacts before the expensive operation, for recoverability.
        write_reports(output, {**summary, "status": "training_pending"}, ablations)
        summary["training"] = train_adapter(backend, examples, output / "adapter",
                                             max_steps=args.train_steps, max_length=args.max_length, seed=args.seed)
        summary["training_method"] = "response-only LoRA rejection fine-tuning on self-generated successful rollouts"
        after_runner = runner
    summary["training_completed"] = True
    summary["status"] = "evaluation_pending"
    write_json(output / "summary.json", summary)
    print("Evaluating the updated policy on the same held-out tasks and sampling seeds...", flush=True)
    after_rollouts = collect(test, after_runner, args.eval_k, args.seed + 8_000_000)
    summary["after"] = metrics(after_rollouts, args.eval_k)
    summary["status"] = "complete"
    summary["elapsed_seconds"] = time.perf_counter() - start
    write_jsonl(output / "after_rollouts.jsonl", [r.to_dict() for r in after_rollouts])
    write_reports(output, summary, ablations)
    print(json.dumps({"backend": summary["backend"], "before": before,
                      "after": summary["after"], "output": str(output)}, indent=2))
    return 0


def main() -> None:
    args = parser().parse_args()
    try:
        code = run(args)
    except (ValueError, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
    raise SystemExit(code)
