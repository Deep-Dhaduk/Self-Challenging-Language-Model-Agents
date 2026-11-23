"""A deterministic, resettable retail sandbox with hidden order state."""

import copy
import random

from .schema import Action, Goal

TOOL_HELP = """Available tool names and argument signatures:
lookup_customer(email: string) -> customer_id.
list_orders(customer_id: string) -> orders with order_id, item, status.
get_order(order_id: string) -> full order details.
cancel_order(order_id: string) -> cancel a pending order.
return_order(order_id: string, refund_method: string) -> return a delivered order.
  refund_method must be exactly original or store_credit, as requested by the user.
change_address(order_id: string, new_address: string) -> change a pending order's shipping address.
finish() -> end the conversation.
Call format: one JSON object with keys "tool" (tool name) and "args" (argument object).
Use the argument names in these signatures. Use an empty args object for finish.
Rules: look up the customer first. Read get_order for an order BEFORE modifying it.
Use list_orders to discover IDs; never invent IDs or send placeholder values.
A canceled/returned order cannot be changed.
There are no real customers, payments, network calls, or external side effects.
"""


class RetailEnv:
    def __init__(self, seed: int):
        rng = random.Random(seed)
        self.seed = seed
        self.email = f"customer{seed}@example.com"
        self.customer_id = f"C{seed}"
        products = rng.sample(["headphones", "backpack", "keyboard", "lamp", "speaker", "watch"], 3)
        self.orders = {
            f"O{seed}-{i}": {
                "order_id": f"O{seed}-{i}", "item": item,
                "status": "delivered" if i == 1 else "pending",
                "address": f"{rng.randint(1, 99)} Oak Street",
                "refund_method": "", "return_eligible": i == 1,
            }
            for i, item in enumerate(products)
        }
        self.initial_orders = copy.deepcopy(self.orders)
        self.authenticated = False
        self.inspected: set[str] = set()
        self.history: list[dict] = []

    def call(self, action: Action) -> dict:
        try:
            result = self._dispatch(action)
        except (ValueError, TypeError, KeyError) as error:
            result = {"error": str(error)}
        self.history.append({"action": {"tool": action.tool, "args": action.args}, "observation": result})
        return result

    def _dispatch(self, action: Action) -> dict:
        tool, args = action.tool, action.args
        if tool == "lookup_customer":
            if args.get("email") != self.email:
                raise ValueError("Customer not found")
            self.authenticated = True
            return {"customer_id": self.customer_id}
        if not self.authenticated:
            raise ValueError("Look up the customer first")
        if tool == "list_orders":
            if args.get("customer_id") != self.customer_id:
                raise ValueError("Unknown customer_id")
            return {"orders": [{k: order[k] for k in ("order_id", "item", "status")}
                               for order in self.orders.values()]}
        if tool not in ("get_order", "cancel_order", "return_order", "change_address"):
            raise ValueError(f"Unknown tool: {tool}")
        order_id = args.get("order_id")
        if not isinstance(order_id, str) or order_id not in self.orders:
            raise ValueError("Unknown order_id")
        order = self.orders[order_id]
        if tool == "get_order":
            self.inspected.add(order_id)
            return copy.deepcopy(order)
        if order_id not in self.inspected:
            raise ValueError("Read get_order before changing this order")
        if tool == "return_order":
            if order["status"] != "delivered" or not order["return_eligible"]:
                raise ValueError("Only eligible delivered orders can be returned")
            if args.get("refund_method") not in ("original", "store_credit"):
                raise ValueError("Unknown refund_method")
            order.update(status="returned", refund_method=args["refund_method"])
        else:
            if order["status"] != "pending":
                raise ValueError("Only pending orders can be canceled or have their address changed")
            if tool == "cancel_order":
                order["status"] = "canceled"
            else:
                address = args.get("new_address")
                if not isinstance(address, str) or not address.strip():
                    raise ValueError("new_address must be nonempty text")
                order["address"] = address
        return copy.deepcopy(order)

    def goal_met(self, goal: Goal) -> bool:
        """Independent evaluation oracle; also disallow changes to other orders."""
        if goal.order_id not in self.orders:
            return False
        expected = copy.deepcopy(self.initial_orders)
        target = expected[goal.order_id]
        if goal.operation == "cancel":
            target["status"] = "canceled"
        elif goal.operation == "return":
            target.update(status="returned", refund_method=goal.value)
        elif goal.operation == "address":
            target["address"] = goal.value
        else:
            return False
        return self.orders == expected


def execute(seed: int, actions: list[Action]) -> RetailEnv:
    env = RetailEnv(seed)
    for action in actions:
        env.call(action)
    return env
