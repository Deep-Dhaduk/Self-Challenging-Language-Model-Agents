import json
import tempfile
import unittest
from pathlib import Path

from sca.agents import Episode, PlanPolicy, collect, metrics, parse_json, run_model
from sca.challenger import generate_llm_tasks
from sca.cli import assert_disjoint, parser, run, successful_examples
from sca.environment import RetailEnv, execute
from sca.schema import Action, Check, Task
from sca.tasks import ablation_candidates, audit_filters, filter_task, make_tasks, template_task, verify


class EnvironmentTests(unittest.TestCase):
    def test_reference_solutions_solve_all_operations(self):
        for operation in ("cancel", "return", "address"):
            task = template_task(123, operation=operation)
            env = execute(task.env_seed, task.solution)
            self.assertTrue(env.goal_met(task.goal))
            self.assertTrue(verify(task, env))
            self.assertFalse(any("error" in t["observation"] for t in env.history))

    def test_mutation_requires_lookup_and_inspection(self):
        env = RetailEnv(123)
        action = Action("cancel_order", {"order_id": "O123-0"})
        self.assertIn("error", env.call(action))
        env.call(Action("lookup_customer", {"email": env.email}))
        self.assertIn("error", env.call(action))
        self.assertEqual(env.orders, env.initial_orders)

    def test_observations_cannot_mutate_environment(self):
        env = RetailEnv(123)
        env.call(Action("lookup_customer", {"email": env.email}))
        observation = env.call(Action("get_order", {"order_id": "O123-0"}))
        observation["status"] = "canceled"
        self.assertEqual(env.orders["O123-0"]["status"], "pending")

    def test_other_order_side_effect_fails_oracle_and_verifier(self):
        task = template_task(123, operation="cancel")
        env = execute(task.env_seed, task.solution)
        env.orders["O123-2"]["address"] = "Wrong address"
        self.assertFalse(env.goal_met(task.goal))
        self.assertFalse(verify(task, env))

    def test_reset_isolation(self):
        task = template_task(123)
        execute(task.env_seed, task.solution)
        self.assertEqual(RetailEnv(123).orders, RetailEnv(123).initial_orders)


class TaskTests(unittest.TestCase):
    def test_serialization_and_failure_cases(self):
        for seed in range(9):
            task = template_task(seed)
            self.assertEqual(Task.from_dict(task.to_dict()), task)
            self.assertTrue(filter_task(task)[0])
            for case in task.failure_cases:
                env = execute(seed, case)
                self.assertFalse(verify(task, env))
                self.assertFalse(env.goal_met(task.goal))

    def test_ablation_detects_fixture_defects(self):
        rows = audit_filters(ablation_candidates(48))
        self.assertEqual([row["accepted"] for row in rows], [48, 24, 12])
        self.assertGreater(rows[0]["false_positive"], 0)
        self.assertGreater(rows[0]["false_negative"], 0)
        self.assertEqual(rows[-1]["false_positive"], 0)
        self.assertEqual(rows[-1]["false_negative"], 0)

    def test_noop_cannot_pass_full_filter(self):
        task = template_task(123)
        task.checks = [Check(task.goal.order_id, "item", RetailEnv(123).orders[task.goal.order_id]["item"])]
        self.assertTrue(filter_task(task, "solution")[0])
        self.assertFalse(filter_task(task, "full")[0])

    def test_schema_rejects_missing_failures_and_executable_fields(self):
        payload = template_task(123).to_dict()
        payload["failure_cases"] = []
        with self.assertRaises(ValueError):
            Task.from_dict(payload)
        payload = template_task(123).to_dict()
        payload["checks"][0]["field"] = "__class__"
        with self.assertRaises(ValueError):
            Task.from_dict(payload)


class AgentTests(unittest.TestCase):
    def test_executors_do_not_receive_goal_solution_or_verifier(self):
        task = template_task(123)
        ep = Episode(task)
        prompt = json.dumps(ep.messages)
        self.assertNotIn(task.goal.order_id, prompt)
        self.assertNotIn('"checks"', prompt)
        class FinishBackend:
            def generate(self, messages, seed, temperature):
                self.messages = messages
                return '{"tool":"finish","args":{}}'
        backend = FinishBackend()
        result = run_model(task, backend, 42)
        self.assertEqual(result.tool_calls, 0)
        self.assertFalse(result.success)

    def test_bad_json_is_recorded_as_error(self):
        ep = Episode(template_task(123))
        self.assertFalse(ep.turn("This is not JSON"))
        self.assertEqual(ep.errors, 1)
        self.assertEqual(ep.env.orders, ep.env.initial_orders)
        with self.assertRaises(ValueError):
            parse_json("[1,2]")

    def test_metrics_use_task_grouping_and_oracle(self):
        policy = PlanPolicy({op: {"complete": 1.0} for op in ("cancel", "return", "address")})
        rows = collect(make_tasks(6, 1000, "test"), policy.run, 4, 42)
        scores = metrics(rows, 4)
        self.assertEqual(scores["pass_at_1"], 1.0)
        self.assertEqual(scores["pass_at_4"], 1.0)
        self.assertEqual(scores["verifier_oracle_disagreements"], 0)
        with self.assertRaises(ValueError):
            metrics(rows[:-1], 4)

    def test_reproducibility(self):
        tasks = make_tasks(6, 1000, "train")
        policy = PlanPolicy()
        first = collect(tasks, policy.run, 4, 42)
        second = collect(tasks, policy.run, 4, 42)
        self.assertEqual([r.to_dict() for r in first], [r.to_dict() for r in second])

    def test_training_excludes_test_data(self):
        train = make_tasks(3, 1000, "train")
        test = make_tasks(3, 1000, "test")
        with self.assertRaises(ValueError):
            assert_disjoint(train, test)
        rows = collect(test, PlanPolicy().run, 1, 42)
        with self.assertRaises(ValueError):
            successful_examples(rows)
        with self.assertRaises(ValueError):
            PlanPolicy().fit_successes(rows)

    def test_llm_challenger_validation_and_no_fallback(self):
        class ProposalBackend:
            max_new_tokens = 50
            def generate(self, messages, seed, temperature):
                return json.dumps({"task": template_task(123).to_dict()})
        backend = ProposalBackend()
        tasks, log = generate_llm_tasks(backend, 1, 123, "train", 42)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0].source, "llm")
        self.assertEqual(backend.max_new_tokens, 50)
        class BadBackend(ProposalBackend):
            def generate(self, messages, seed, temperature):
                return "invalid"
        tasks, log = generate_llm_tasks(BadBackend(), 1, 123, "train", 42, max_attempts=1, max_turns=1)
        self.assertEqual(tasks, [])
        self.assertFalse(log[0]["accepted"])


class CLITests(unittest.TestCase):
    def test_demo_writes_inspectable_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            args = parser().parse_args(["demo", "--output", str(path), "--train-tasks", "12",
                                        "--test-tasks", "6", "--rollouts", "4", "--eval-k", "2"])
            self.assertEqual(run(args), 0)
            summary = json.loads((path / "summary.json").read_text())
            self.assertEqual(summary["status"], "complete")
            self.assertEqual(summary["backend"], "tabular")
            self.assertTrue((path / "evaluation.svg").is_file())
            self.assertTrue((path / "successful_actions.jsonl").is_file())
            with self.assertRaises(ValueError):
                run(args)  # Existing results are preserved.

    def test_notebook_has_valid_unexecuted_cells(self):
        notebook = json.loads(Path("notebooks/self_challenging_agents.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "notebook cell", "exec")
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])


if __name__ == "__main__":
    unittest.main()
