"""Serializable tasks. Verifiers are data interpreted by trusted Python code."""

from dataclasses import asdict, dataclass, field
from typing import Any

OPERATIONS = ("cancel", "return", "address")
ORDER_FIELDS = {"status", "refund_method", "address", "item"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass
class Action:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        require(isinstance(data, dict), "Action must be an object")
        require(isinstance(data.get("tool"), str), "Missing tool name")
        require(isinstance(data.get("args", {}), dict), "args must be an object")
        return cls(data["tool"], data.get("args", {}))


@dataclass
class Goal:
    order_id: str
    operation: str
    value: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        require(isinstance(data, dict), "Goal must be an object")
        require(data.get("operation") in OPERATIONS, "Unknown goal operation")
        require(isinstance(data.get("order_id"), str), "Missing order_id")
        require(isinstance(data.get("value", ""), str), "Goal value must be text")
        if data["operation"] == "return":
            require(data.get("value") in ("original", "store_credit"), "Invalid refund method")
        if data["operation"] == "address":
            require(bool(data.get("value", "").strip()), "Address must not be empty")
        return cls(data["order_id"], data["operation"], data.get("value", ""))


@dataclass
class Check:
    order_id: str
    field: str
    equals: str

    @classmethod
    def from_dict(cls, data: dict) -> "Check":
        require(isinstance(data, dict), "Check must be an object")
        require(data.get("field") in ORDER_FIELDS, "Unsupported verifier field")
        require(isinstance(data.get("order_id"), str), "Missing check order_id")
        require(isinstance(data.get("equals"), str), "Check value must be text")
        return cls(data["order_id"], data["field"], data["equals"])


@dataclass
class Task:
    task_id: str
    split: str
    env_seed: int
    instruction: str
    goal: Goal
    checks: list[Check]
    solution: list[Action]
    failure_cases: list[list[Action]]
    source: str = "template"
    exploration: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        require(isinstance(data, dict), "Task must be an object")
        for key in ("task_id", "split", "instruction"):
            require(isinstance(data.get(key), str) and bool(data[key].strip()), f"Missing {key}")
        require(type(data.get("env_seed")) is int, "env_seed must be an integer")
        require(data["split"] in ("train", "test"), "Invalid split")
        for key in ("checks", "solution", "failure_cases"):
            require(isinstance(data.get(key), list), f"{key} must be an array")
        require(1 <= len(data["checks"]) <= 12, "Provide 1-12 verifier checks")
        require(len(data["solution"]) <= 15, "Solution exceeds step budget")
        require(len(data["failure_cases"]) == 3, "Provide exactly three failure cases")
        require(all(isinstance(case, list) and len(case) <= 15 for case in data["failure_cases"]),
                "Invalid failure case")
        return cls(
            task_id=data["task_id"], split=data["split"], env_seed=data["env_seed"],
            instruction=data["instruction"], goal=Goal.from_dict(data["goal"]),
            checks=[Check.from_dict(c) for c in data["checks"]],
            solution=[Action.from_dict(a) for a in data["solution"]],
            failure_cases=[[Action.from_dict(a) for a in case] for case in data["failure_cases"]],
            source=str(data.get("source", "unknown")), exploration=data.get("exploration", []),
        )
