"""Executor rollouts. Executors see instructions and tool observations, never solutions."""

import copy
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Callable

from .environment import RetailEnv, TOOL_HELP
from .schema import Action, Task
from .tasks import verify

SYSTEM = "You are a retail assistant. Complete the user's request with the tools. " + TOOL_HELP
STRATEGIES = ("stop_early", "skip_inspection", "wrong_target", "wrong_value", "complete")


def parse_json(text: str) -> dict:
    """Allow a JSON object or one fenced JSON object; reject surrounding prose."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("Expected one JSON object")
    return value


@dataclass
class Rollout:
    task_id: str
    split: str
    operation: str
    strategy: str
    seed: int
    reward: int
    success: bool
    tool_calls: int
    tool_errors: int
    finish_reason: str
    messages: list[dict]
    examples: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class Episode:
    def __init__(self, task: Task):
        self.task = task
        self.env = RetailEnv(task.env_seed)
        self.messages = [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": task.instruction}]
        self.examples: list[dict] = []
        self.errors = 0

    def turn(self, text: str) -> bool:
        self.examples.append({"messages": copy.deepcopy(self.messages), "response": text})
        self.messages.append({"role": "assistant", "content": text})
        try:
            action = Action.from_dict(parse_json(text))
            if action.tool == "finish":
                return True
            observation = self.env.call(action)
        except (ValueError, TypeError, KeyError, json.JSONDecodeError) as error:
            observation = {"error": f"Invalid action: {error}. Reply with one tool JSON object."}
        self.errors += int("error" in observation)
        self.messages.append({"role": "user", "content": "Tool observation: " + json.dumps(observation)})
        return False

    def action(self, tool: str, **args) -> dict:
        self.turn(json.dumps({"tool": tool, "args": args}))
        return json.loads(self.messages[-1]["content"].removeprefix("Tool observation: "))

    def result(self, seed: int, strategy: str, reason: str = "finished") -> Rollout:
        return Rollout(self.task.task_id, self.task.split, self.task.goal.operation, strategy, seed,
                       int(verify(self.task, self.env)), self.env.goal_met(self.task.goal),
                       len(self.env.history), self.errors, reason,
                       self.messages, self.examples)


def request_fields(instruction: str) -> dict:
    """Small template parser for the CPU baseline; not an LLM or a general language parser."""
    lower = instruction.lower()
    operation = "return" if "return my" in lower else "cancel" if "cancel my" in lower else "address"
    email = re.search(r"[a-zA-Z0-9]+@example\.com", instruction)
    product = re.search(r"my (\w+) order", lower)
    address = re.search(r"\[([^\]]+)\]", instruction)
    if not email or not product:
        raise ValueError("CPU policy supports the template challenger's instructions only")
    return {"operation": operation, "email": email.group(), "item": product.group(1),
            "value": address.group(1) if address else "store_credit" if "store_credit" in lower else "original"}


class PlanPolicy:
    """A categorical policy over five hand-written strategies, for CPU teaching only."""
    def __init__(self, weights: dict | None = None):
        self.weights = weights or {op: {strategy: 1.0 for strategy in STRATEGIES}
                                   for op in ("cancel", "return", "address")}

    def run(self, task: Task, seed: int) -> Rollout:
        request = request_fields(task.instruction)
        weights = self.weights[request["operation"]]
        strategy = random.Random(seed).choices(list(weights), weights=list(weights.values()), k=1)[0]
        ep = Episode(task)
        customer = ep.action("lookup_customer", email=request["email"])
        if strategy != "stop_early":
            orders = ep.action("list_orders", customer_id=customer["customer_id"])["orders"]
            order = next(order for order in orders if order["item"] == request["item"])
            if strategy == "wrong_target":
                order = next(other for other in orders if other["order_id"] != order["order_id"])
            oid = order["order_id"]
            if strategy != "skip_inspection":
                ep.action("get_order", order_id=oid)
            if request["operation"] == "cancel":
                if strategy == "wrong_value":
                    ep.action("change_address", order_id=oid, new_address="999 Wrong Road")
                else:
                    ep.action("cancel_order", order_id=oid)
            elif request["operation"] == "return":
                value = request["value"]
                if strategy == "wrong_value":
                    value = "original" if value == "store_credit" else "store_credit"
                ep.action("return_order", order_id=oid, refund_method=value)
            else:
                value = "999 Wrong Road" if strategy == "wrong_value" else request["value"]
                ep.action("change_address", order_id=oid, new_address=value)
        ep.turn('{"tool":"finish","args":{}}')
        return ep.result(seed, strategy)

    def fit_successes(self, rollouts: list[Rollout]) -> "PlanPolicy":
        weights = copy.deepcopy(self.weights)
        for rollout in rollouts:
            if rollout.split != "train":
                raise ValueError("Refusing to train on test rollouts")
            if rollout.reward:
                weights[rollout.operation][rollout.strategy] += 1
        return PlanPolicy(weights)


def run_model(task: Task, backend, seed: int, max_steps: int = 8,
              temperature: float = 0.7) -> Rollout:
    episode = Episode(task)
    for step in range(max_steps):
        text = backend.generate(episode.messages, seed + step, temperature=temperature)
        if episode.turn(text):
            return episode.result(seed, "llm")
    return episode.result(seed, "llm", "step_budget")


def collect(tasks: list[Task], runner: Callable, repeats: int, seed: int) -> list[Rollout]:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    return [runner(task, seed + i * 1009 + attempt * 7919)
            for i, task in enumerate(tasks) for attempt in range(repeats)]


def metrics(rollouts: list[Rollout], k: int) -> dict:
    if k < 1 or not rollouts:
        raise ValueError("Metrics need rollouts and k >= 1")
    grouped: dict[str, list[Rollout]] = {}
    for rollout in rollouts:
        grouped.setdefault(rollout.task_id, []).append(rollout)
    if any(len(trials) != k for trials in grouped.values()):
        raise ValueError("Every evaluation task must have exactly k attempts")
    calls = sum(r.tool_calls for r in rollouts)
    return {
        "tasks": len(grouped), "attempts_per_task": k,
        "pass_at_1": sum(trials[0].success for trials in grouped.values()) / len(grouped),
        f"pass_at_{k}": sum(any(r.success for r in trials) for trials in grouped.values()) / len(grouped),
        "attempt_success_rate": sum(r.success for r in rollouts) / len(rollouts),
        "mean_tool_calls": calls / len(rollouts),
        "tool_errors": sum(r.tool_errors for r in rollouts),
        "verifier_oracle_disagreements": sum(r.reward != int(r.success) for r in rollouts),
        "by_operation": {op: sum(r.success for r in rollouts if r.operation == op) / n
                         for op, n in Counter(r.operation for r in rollouts).items()},
    }
