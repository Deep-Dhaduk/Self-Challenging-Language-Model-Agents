"""Native function-call schemas and a shared inference/training chat representation.

Only the public request and actual prior observations enter this conversion.
No task solutions, verifier checks, goal objects, or invented arguments are used.
"""

import copy
import json

EXECUTOR_PROTOCOL = "retail-native-v3"
EXECUTOR_MARKER = f"[{EXECUTOR_PROTOCOL}]"
EXECUTOR_SYSTEM = EXECUTOR_MARKER + """
You are a retail assistant that acts by calling the provided functions.
Call exactly ONE function per turn and wait for its actual result.
The user already supplied an email. First look up that email, then list that
customer's orders. Select the item requested by the user from the returned list,
and inspect that order before changing it. Never invent customer or order IDs.
Use the requested refund method or new address exactly. Address brackets in the
request are delimiters, not part of the address. Preserve unrelated orders.
After the requested change succeeds, call finish. Correct errors using actual
observations; do not describe imaginary actions or ask for information already given.
"""


def _tool(name, description, properties, required):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": required, "additionalProperties": False},
    }}


EMAIL = {"type": "string", "description": "Exact email from the user's request."}
CUSTOMER_ID = {"type": "string", "description": "customer_id returned by lookup_customer."}
ORDER_ID = {"type": "string", "description": "Exact order_id observed in list_orders or get_order. Never invent it."}
TOOL_SCHEMAS = [
    _tool("lookup_customer", "First step: identify the customer using their email.",
          {"email": EMAIL}, ["email"]),
    _tool("list_orders", "After customer lookup: discover this customer's order IDs, items, and statuses.",
          {"customer_id": CUSTOMER_ID}, ["customer_id"]),
    _tool("get_order", "Read the selected order before making a change.",
          {"order_id": ORDER_ID}, ["order_id"]),
    _tool("cancel_order", "Cancel the inspected pending order, only when cancellation was requested.",
          {"order_id": ORDER_ID}, ["order_id"]),
    _tool("return_order", "Return the inspected eligible delivered order using the requested refund method.",
          {"order_id": ORDER_ID,
           "refund_method": {"type": "string", "enum": ["original", "store_credit"],
                             "description": "Use the exact method requested by the user."}},
          ["order_id", "refund_method"]),
    _tool("change_address", "Update the inspected pending order's shipping address when requested.",
          {"order_id": ORDER_ID,
           "new_address": {"type": "string", "description": "Requested new address without delimiter brackets."}},
          ["order_id", "new_address"]),
    _tool("finish", "End the episode after the requested change succeeded.", {}, []),
]


def uses_native_tools(messages: list[dict]) -> bool:
    return bool(messages and messages[0].get("role") == "system"
                and messages[0].get("content", "").startswith(EXECUTOR_MARKER))


def native_messages(messages: list[dict]) -> list[dict]:
    """Map the audit-friendly internal transcript into native assistant/tool messages."""
    result = []
    for message in messages:
        content, role = message.get("content", ""), message["role"]
        if role == "assistant":
            try:
                value = json.loads(content)
            except (ValueError, TypeError):
                value = None
            if (isinstance(value, dict) and isinstance(value.get("tool"), str)
                    and isinstance(value.get("args"), dict)):
                result.append({"role": "assistant", "content": "", "tool_calls": [
                    {"type": "function", "function": {"name": value["tool"], "arguments": value["args"]}}
                ]})
                continue
        if role == "user" and content.startswith("Tool observation: "):
            observation = content.removeprefix("Tool observation: ")
            # Parsing errors after plain-text responses are feedback to the model,
            # not results of an actual function call.
            if result and result[-1].get("tool_calls"):
                result.append({"role": "tool", "content": observation})
            else:
                result.append({"role": "user", "content": "Protocol feedback: " + observation})
            continue
        result.append(copy.deepcopy(message))
    return result


def apply_chat_template(tokenizer, messages: list[dict], **kwargs):
    """Use the same tool schemas and chat conversion for generation and RFT encoding."""
    if not uses_native_tools(messages):
        # The challenger authors a task JSON object, not an executor function call.
        return tokenizer.apply_chat_template(messages, **kwargs)
    template = tokenizer.get_chat_template()
    if "tools" not in template or "tool_calls" not in template:
        raise ValueError("This executor requires a tokenizer with a native function-calling template")
    return tokenizer.apply_chat_template(native_messages(messages), tools=TOOL_SCHEMAS, **kwargs)
