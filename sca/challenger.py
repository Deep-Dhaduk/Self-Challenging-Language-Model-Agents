"""LLM task authoring through the same interactive tool sandbox as the executor."""

import json

from .agents import parse_json
from .environment import RetailEnv, TOOL_HELP
from .schema import Action, Task
from .tasks import filter_task

CHALLENGER = """You design training tasks for a retail tool-use agent.
First explore the environment using the tools. Then propose ONE task for the INITIAL
environment state. Resetting happens automatically when your proposed solution is tested.
Respond each turn with either one tool call JSON or a final {"task": ...} JSON object.
Do not write Python, markdown, or commentary. All fields below are required in task:
instruction: user request with email, item, exact refund method/address when relevant.
goal: {"order_id":"...","operation":"cancel|return|address","value":"..."}.
checks: array of {"order_id":"...","field":"status|refund_method|address|item","equals":"..."}.
solution: array of tool call objects that solves the instruction from a fresh environment.
failure_cases: exactly THREE arrays of tool calls that do NOT solve the instruction.
The checks must accept the solution, reject a no-op, and reject all three failure cases.
Use only IDs and items observed through tools. Make a feasible, unambiguous request.
Address requests need a nonempty value. Return value is original or store_credit.
Cancel value is an empty string. Include lookup_customer, list_orders, and get_order
in the solution before changing an order. Failure cases may include an empty array,
an incomplete solution, and a solution that requests the wrong refund/address/operation.
Your checks should also reject changes to unrelated orders.
""" + TOOL_HELP


def generate_llm_tasks(backend, count: int, start_seed: int, split: str,
                       seed: int, max_attempts: int | None = None,
                       max_turns: int = 8) -> tuple[list[Task], list[dict]]:
    tasks, log = [], []
    original_budget = backend.max_new_tokens
    backend.max_new_tokens = max(1200, original_budget)
    try:
        for candidate in range(max_attempts or count * 2):
            if len(tasks) >= count:
                break
            env_seed = start_seed + candidate
            env = RetailEnv(env_seed)
            messages = [{"role": "system", "content": CHALLENGER},
                        {"role": "user", "content": f"Explore customer email {env.email}; then create a task."}]
            accepted, reason, task = False, "no task proposed within the turn budget", None
            for turn in range(max_turns + 1):
                if turn == max_turns:
                    messages.append({"role": "user", "content": "Now output the final task JSON; no more tool calls."})
                answer = backend.generate(messages, seed + candidate * 1009 + turn, temperature=0.7)
                messages.append({"role": "assistant", "content": answer})
                try:
                    data = parse_json(answer)
                    if "task" in data:
                        payload = data["task"]
                        if not isinstance(payload, dict):
                            raise ValueError("task must be an object")
                        payload.update(task_id=f"{split}-llm-{env_seed}", split=split,
                                       env_seed=env_seed, source="llm", exploration=env.history)
                        task = Task.from_dict(payload)
                        accepted, reason = filter_task(task)
                        break
                    observation = env.call(Action.from_dict(data))
                except (ValueError, KeyError, TypeError) as error:
                    observation = {"error": str(error)}
                    reason = str(error)
                messages.append({"role": "user", "content": "Tool observation: " + json.dumps(observation)})
            if accepted:
                tasks.append(task)
            log.append({"split": split, "env_seed": env_seed, "accepted": accepted,
                        "reason": reason, "messages": messages,
                        "task": task.to_dict() if task else None})
            print(f"Challenger {split}: candidate {candidate + 1}, accepted {len(tasks)}/{count}: {reason}", flush=True)
    finally:
        backend.max_new_tokens = original_budget
    return tasks, log
