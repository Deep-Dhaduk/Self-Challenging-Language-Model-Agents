"""Template challenger, executable verifier, quality filters, and audit ablations."""

import copy
import random

from .environment import RetailEnv, execute
from .schema import Action, Check, Goal, Task


def verify(task: Task, env: RetailEnv) -> bool:
    """Executable verifier with exact state checks; no model-supplied code execution."""
    return bool(task.checks) and all(
        env.orders.get(check.order_id, {}).get(check.field) == check.equals
        for check in task.checks
    )


def mutation(goal: Goal) -> Action:
    if goal.operation == "cancel":
        return Action("cancel_order", {"order_id": goal.order_id})
    if goal.operation == "return":
        return Action("return_order", {"order_id": goal.order_id, "refund_method": goal.value})
    return Action("change_address", {"order_id": goal.order_id, "new_address": goal.value})


def template_task(seed: int, split: str = "train", operation: str | None = None) -> Task:
    """A fixture generator, explicitly not an LLM. Exploration uses actual tool results."""
    env = RetailEnv(seed)
    operation = operation or ("cancel", "return", "address")[seed % 3]
    lookup = Action("lookup_customer", {"email": env.email})
    customer = env.call(lookup)
    listing = Action("list_orders", {"customer_id": customer["customer_id"]})
    orders = env.call(listing)["orders"]
    index = {"cancel": 0, "return": 1, "address": 2}[operation]
    order = env.call(Action("get_order", {"order_id": orders[index]["order_id"]}))
    oid, item = order["order_id"], order["item"]
    value = ("original" if seed % 2 else "store_credit") if operation == "return" else ""
    if operation == "address":
        value = f"{100 + seed % 800} Pine Avenue"
    goal = Goal(oid, operation, value)
    if operation == "cancel":
        request = f"Cancel my {item} order."
        checks = [Check(oid, "status", "canceled")]
        wrong = Action("change_address", {"order_id": oid, "new_address": "999 Wrong Road"})
    elif operation == "return":
        request = f"Return my {item} order and use {value} as the refund method."
        checks = [Check(oid, "status", "returned"), Check(oid, "refund_method", value)]
        wrong = Action("return_order", {"order_id": oid,
                                       "refund_method": "store_credit" if value == "original" else "original"})
    else:
        request = f"Change the shipping address for my {item} order to [{value}]."
        checks = [Check(oid, "address", value)]
        wrong = Action("change_address", {"order_id": oid, "new_address": "999 Wrong Road"})
    # Include unchanged fields and other orders to reject partial/extra side effects.
    solved = execute(seed, [lookup, listing, Action("get_order", {"order_id": oid}), mutation(goal)])
    for order_id, final in solved.orders.items():
        for key in ("status", "refund_method", "address"):
            if not any(c.order_id == order_id and c.field == key for c in checks):
                checks.append(Check(order_id, key, final[key]))
    prefix = [lookup, listing, Action("get_order", {"order_id": oid})]
    instruction = f"My email is {env.email}. {request} Leave my other orders unchanged."
    if split == "test":
        instruction = f"Please help: {request} Contact email: {env.email}. Leave my other orders unchanged."
    return Task(f"{split}-{seed}", split, seed, instruction, goal, checks,
                prefix + [mutation(goal)], [[], prefix, prefix + [wrong]],
                exploration=copy.deepcopy(env.history))


def filter_task(task: Task, mode: str = "full") -> tuple[bool, str]:
    if mode not in ("instruction_only", "solution", "full"):
        raise ValueError(f"Unknown filter mode: {mode}")
    try:
        Task.from_dict(task.to_dict())
    except (ValueError, TypeError, KeyError) as error:
        return False, f"schema: {error}"
    if mode == "instruction_only":
        return True, "schema only; semantic quality unchecked"
    reference = execute(task.env_seed, task.solution)
    if any("error" in turn["observation"] for turn in reference.history):
        return False, "reference tool execution failed"
    if not verify(task, reference):
        return False, "reference solution failed verification"
    if mode == "solution":
        return True, "reference passed"
    if verify(task, RetailEnv(task.env_seed)):
        return False, "no-op incorrectly passes verification"
    for index, actions in enumerate(task.failure_cases):
        if verify(task, execute(task.env_seed, actions)):
            return False, f"failure case {index} incorrectly passes verification"
    return True, "reference passes; no-op and all three failure cases fail"


def make_tasks(count: int, start_seed: int, split: str) -> list[Task]:
    if count < 1:
        raise ValueError("Task count must be positive")
    return [template_task(start_seed + i, split) for i in range(count)]


def ablation_candidates(count: int = 48, seed: int = 42) -> list[Task]:
    """Deliberate fixture defects, not measured LLM generation errors."""
    rng = random.Random(seed)
    candidates = []
    for i in range(count):
        task = template_task(500_000 + i)
        defect = i % 4
        if defect == 1:  # Impossible reference.
            task.solution = task.solution[:-1]
        elif defect == 2:  # Always true: matches the starting state.
            initial = RetailEnv(task.env_seed)
            task.checks = [Check(task.goal.order_id, "item", initial.orders[task.goal.order_id]["item"])]
        elif defect == 3:  # Rejects a valid solution.
            task.checks.append(Check(task.goal.order_id, "status", "impossible-status"))
        task.source = ("fixture_valid", "fixture_bad_solution", "fixture_permissive", "fixture_overstrict")[defect]
        candidates.append(task)
    rng.shuffle(candidates)
    return candidates


def audit_filters(candidates: list[Task]) -> list[dict]:
    """Audit verifiers against the separate goal oracle on a fixed probe set."""
    rows = []
    for mode in ("instruction_only", "solution", "full"):
        kept = [t for t in candidates if filter_task(t, mode)[0]]
        tp = tn = fp = fn = 0
        for task in kept:
            canonical = template_task(task.env_seed, task.split, task.goal.operation)
            probes = [[], task.solution, *task.failure_cases, canonical.solution]
            for actions in probes:
                env = execute(task.env_seed, actions)
                predicted, actual = verify(task, env), env.goal_met(task.goal)
                tp += int(predicted and actual)
                tn += int(not predicted and not actual)
                fp += int(predicted and not actual)
                fn += int(not predicted and actual)
        total = tp + tn + fp + fn
        rows.append({"filter": mode, "candidates": len(candidates), "accepted": len(kept),
                     "acceptance_rate": len(kept) / len(candidates) if candidates else 0.0,
                     "probe_count": total, "true_positive": tp, "true_negative": tn,
                     "false_positive": fp, "false_negative": fn,
                     "probe_agreement": (tp + tn) / total if total else None,
                     "source": "controlled fixture corruption; not an LLM ablation"})
    return rows
