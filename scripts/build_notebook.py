"""Build the Colab notebook using only the standard library."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []


def cell(kind: str, source: str):
    value = {"cell_type": kind, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if value["source"] and not value["source"][-1].endswith("\n"):
        value["source"][-1] += "\n"
    if kind == "code":
        value.update(execution_count=None, outputs=[])
    value["id"] = f"sca-{len(cells):02d}"
    cells.append(value)


cell("markdown", """
# Self-Challenging Language Model Agents: an executable study

Companion code for [Zhou et al. (2025)](https://arxiv.org/abs/2506.01716).
This notebook runs the repository's Python code rather than maintaining a second implementation.

**Two separate experiments:**
1. A CPU demonstration learns probabilities over five scripted strategies. It is **not an LLM**.
2. A GPU experiment samples real LLM tool-use trajectories and trains a LoRA adapter using only rewarded training rollouts.

The GPU path offers a template challenger for an initial smoke run and an interactive LLM
challenger for task authoring. Template-generated tasks must not be described as LLM-generated.
The retail environment and test tasks are our own; no result here reproduces M3ToolEval or TauBench.
All cells are initially unexecuted. Run them to obtain actual outputs; improvement is not guaranteed.
""")
cell("markdown", """
## 1. Get the code

On Colab this clones the public repository. The notebook file must first be pushed to GitHub.
If you opened this notebook locally from the repository, it uses your existing checkout.
""")
cell("code", """
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

REPOSITORY = "https://github.com/Deep-Dhaduk/Self-Challenging-Language-Model-Agents.git"
if not (Path.cwd() / "sca").is_dir():
    if (Path.cwd().parent / "sca").is_dir():
        os.chdir(Path.cwd().parent)
    else:
        checkout = Path.cwd() / "Self-Challenging-Language-Model-Agents"
        if not checkout.exists():
            subprocess.run(["git", "clone", REPOSITORY, str(checkout)], check=True)
        os.chdir(checkout)
ROOT = Path.cwd()
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
print("Repository:", ROOT)
print("Python:", sys.version)
""")
cell("markdown", """
## 2. Inspect a Code-as-Task example

The task includes a request, executable state-check specification, reference tool calls, and
three failure cases. The executor receives only the request, tool descriptions, and its own
observations. Verifiers are interpreted by trusted Python; generated Python is never executed.
""")
cell("code", """
from sca.tasks import template_task, filter_task

example = template_task(123, operation="return")
print(json.dumps(example.to_dict(), indent=2))
print("Full quality filter:", filter_task(example))
""")
cell("markdown", """
## 3. Run the CPU experiment (no packages, API key, or GPU required)

The categorical policy samples five hand-written strategies. A successful-rollout update
changes their probabilities. New customer/order IDs and a different request layout are used
in the held-out split. This is a teaching baseline, not evidence of LLM self-improvement.
""")
cell("code", """
CPU_RUN = ROOT / "outputs" / f"cpu_{RUN_ID}"
subprocess.run([
    sys.executable, "-m", "sca", "demo", "--output", str(CPU_RUN),
    "--seed", "42", "--train-tasks", "120", "--test-tasks", "60",
    "--rollouts", "4", "--eval-k", "4",
], check=True)
""")
cell("code", """
from IPython.display import SVG, display

cpu_summary = json.loads((CPU_RUN / "summary.json").read_text())
display(SVG(filename=str(CPU_RUN / "evaluation.svg")))
display(SVG(filename=str(CPU_RUN / "filter_ablation.svg")))
print(json.dumps(json.loads((CPU_RUN / "filter_ablation.json").read_text()), indent=2))
""")
cell("markdown", """
The ablation intentionally introduces invalid reference solutions, permissive verifiers,
and overstrict verifiers into fixtures. It measures filter behavior on those known defects;
it does not estimate the error rate of LLM-generated tasks. Probe agreement compares a
task verifier against a separately implemented goal oracle on a finite probe set.
""")
cell("markdown", """
## 4. Install the optional LLM dependencies

For the remaining sections select **Runtime → Change runtime type → T4 GPU** (or another
CUDA GPU). This downloads dependencies and model weights. It needs internet access, but no
paid API key. The default 0.5B model is small for demonstration and may have few successes.
The pinned package versions match the repository's tested library API.
""")
cell("code", """
subprocess.run([sys.executable, "-m", "pip", "install", "-e", ".[llm]"], check=True)
""")
cell("code", """
import torch
from sca.llm import environment_versions

print(environment_versions())
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Select a GPU runtime before running the LLM experiment.")
print("GPU:", torch.cuda.get_device_name(0))
""")
cell("markdown", """
## 5. Configure and run real LLM rejection fine-tuning

`TASK_SOURCE = "template"` is the initial smoke configuration: real LLM execution and
training, with tasks authored by the deterministic fixture generator.

To exercise the full interactive **LLM challenger**, change `TASK_SOURCE` to `"llm"`.
For task authoring, consider `Qwen/Qwen2.5-1.5B-Instruct` if your GPU has enough memory.
The challenger explores tools and authors the request, checks, solution, and failure cases.
Invalid proposals are rejected and saved in logs. There is no fallback to reference answers.

The same base LLM authors tasks and executes them. Training uses only its own rewarded
training trajectories, with loss only on assistant responses. This is one offline RFT
iteration using LoRA, not PPO/GRPO or the paper's full training setup. Model improvement is
an experimental question; small runs can stay flat, worsen, or collect zero successes.
""")
cell("code", """
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TASK_SOURCE = "template"  # Change to "llm" for interactive LLM-authored tasks.
TRAIN_TASKS = 24
TEST_TASKS = 12
ROLLOUTS_PER_TRAIN_TASK = 4
EVAL_K = 2
TRAIN_STEPS = 40
LLM_RUN = ROOT / "outputs" / f"llm_{TASK_SOURCE}_{RUN_ID}"

command = [
    sys.executable, "-m", "sca", "llm", "--output", str(LLM_RUN),
    "--model", MODEL, "--task-source", TASK_SOURCE, "--device", "cuda",
    "--train-tasks", str(TRAIN_TASKS), "--test-tasks", str(TEST_TASKS),
    "--rollouts", str(ROLLOUTS_PER_TRAIN_TASK), "--eval-k", str(EVAL_K),
    "--train-steps", str(TRAIN_STEPS), "--seed", "42",
]
completed = subprocess.run(command, check=False)
print("Experiment exit code:", completed.returncode)
if completed.returncode:
    print("Inspect the saved logs/status. Training may not have run. Do not report a missing after score as zero.")
""")
cell("markdown", """
## 6. Inspect measured results and individual trajectories

Pass@1 is the first sampled attempt's success averaged across tasks. Pass@k is the fraction
of tasks with at least one successful attempt among k actual samples. The independent
oracle checks both the requested change and preservation of all other orders.
Verifier/oracle disagreements are logged; finite failure-case tests cannot prove a verifier correct.
Before/after use the same test tasks, seeds, tool budget, and sampling temperature.
Test rollouts never enter the training dataset. Repeat with additional seeds for stronger evidence.
""")
cell("code", """
summary_path = LLM_RUN / "summary.json"
if summary_path.exists():
    llm_summary = json.loads(summary_path.read_text())
    print(json.dumps(llm_summary, indent=2))
    display(SVG(filename=str(LLM_RUN / "evaluation.svg")))
else:
    status_path = LLM_RUN / "status.json"
    print(status_path.read_text() if status_path.exists() else "Run did not reach evaluation; inspect the command output.")

rollout_path = LLM_RUN / "before_rollouts.jsonl"
if rollout_path.exists():
    first_rollout = json.loads(rollout_path.read_text().splitlines()[0])
    for message in first_rollout["messages"]:
        print(message["role"].upper() + ":", message["content"], "\\n")
""")
cell("markdown", """
## 7. Save artifacts and your executed notebook

The archive includes configs, tasks, full trajectories, metrics, charts, and the LoRA adapter
if training succeeded. Download it before the Colab session ends. Also save the executed
notebook to keep the visible outputs. Upload selected measured results to GitHub; model
weights are ignored by default. Medium/slides/video are separate assignment deliverables.
""")
cell("code", """
import shutil

artifact_directory = LLM_RUN if LLM_RUN.exists() else CPU_RUN
archive_path = shutil.make_archive(str(artifact_directory), "zip", root_dir=artifact_directory)
print("Saved:", archive_path)
# On Colab, uncomment the next two lines to download:
# from google.colab import files
# files.download(archive_path)
""")
cell("markdown", """
## References

- Zhou et al., [Self-Challenging Language Model Agents](https://arxiv.org/abs/2506.01716), 2025.
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).
- Hugging Face [chat templates](https://huggingface.co/docs/transformers/chat_templating)
  and [PEFT quick tour](https://huggingface.co/docs/peft/quicktour).

This repository is an independent educational implementation. It does not include code
copied from the authors or claim their benchmark scores as our own.
""")

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                 "language_info": {"name": "python", "version": "3.11"},
                 "colab": {"name": "self_challenging_agents.ipynb", "provenance": []}},
    "nbformat": 4, "nbformat_minor": 5,
}
destination = ROOT / "notebooks" / "self_challenging_agents.ipynb"
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(destination)
