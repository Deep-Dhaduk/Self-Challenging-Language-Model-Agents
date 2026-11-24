"""Offline checks for the native function-call interface and hardware precision rule."""

import copy
import json
from types import SimpleNamespace
import unittest

from sca.agents import Episode, parse_action
from sca.llm import cuda_dtype
from sca.protocol import TOOL_SCHEMAS, apply_chat_template, native_messages
from sca.tasks import template_task


class ProtocolTests(unittest.TestCase):
    def test_transcript_conversion_uses_observed_arguments_and_tool_roles(self):
        episode = Episode(template_task(123))
        episode.action("lookup_customer", email="customer123@example.com")
        original = copy.deepcopy(episode.messages)
        messages = native_messages(episode.messages)
        self.assertEqual(messages[-2]["role"], "assistant")
        call = messages[-2]["tool_calls"][0]["function"]
        self.assertEqual(call, {"name": "lookup_customer", "arguments": {"email": "customer123@example.com"}})
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertEqual(json.loads(messages[-1]["content"]), {"customer_id": "C123"})
        self.assertEqual(episode.messages, original)

    def test_invalid_model_text_is_feedback_not_a_fabricated_tool_result(self):
        episode = Episode(template_task(123))
        episode.turn("I already returned the order")
        messages = native_messages(episode.messages)
        self.assertNotIn("tool_calls", messages[-2])
        self.assertEqual(messages[-1]["role"], "user")
        self.assertTrue(messages[-1]["content"].startswith("Protocol feedback:"))
        self.assertEqual(episode.env.orders, episode.env.initial_orders)

    def test_native_call_is_not_reported_as_a_format_recovery(self):
        action, recovered = parse_action('<tool_call>\n{"name":"lookup_customer","arguments":{"email":"customer123@example.com"}}\n</tool_call>')
        self.assertEqual(action.tool, "lookup_customer")
        self.assertFalse(recovered)
        with self.assertRaises(ValueError):
            parse_action('<tool_call>{"name":"finish","arguments":{}}</tool_call>\n'
                         '<tool_call>{"name":"finish","arguments":{}}</tool_call>')

    def test_schemas_and_template_receive_no_hidden_task_information(self):
        class SpyTokenizer:
            def get_chat_template(self):
                return "supports tools and tool_calls"
            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.kwargs = messages, kwargs
                return "rendered"
        task = template_task(123)
        tokenizer = SpyTokenizer()
        apply_chat_template(tokenizer, Episode(task).messages, tokenize=False)
        self.assertEqual(tokenizer.kwargs["tools"], TOOL_SCHEMAS)
        rendered_input = json.dumps({"messages": tokenizer.messages, "tools": tokenizer.kwargs["tools"]})
        self.assertNotIn(task.goal.order_id, rendered_input)
        self.assertNotIn('"checks"', rendered_input)
        self.assertNotIn('"solution"', rendered_input)
        self.assertEqual(len(TOOL_SCHEMAS), 7)

    def test_challenger_keeps_its_task_authoring_format(self):
        class SpyTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.kwargs = kwargs
                return "task authoring"
        tokenizer = SpyTokenizer()
        self.assertEqual(apply_chat_template(tokenizer, [{"role": "system", "content": "Author a task JSON"}], tokenize=False), "task authoring")
        self.assertNotIn("tools", tokenizer.kwargs)

    def test_missing_native_template_is_an_explicit_error(self):
        class PlainTokenizer:
            def get_chat_template(self):
                return "plain chat only"
        with self.assertRaises(ValueError):
            apply_chat_template(PlainTokenizer(), Episode(template_task(123)).messages, tokenize=False)

    def test_t4_uses_fp16_even_if_bf16_emulation_is_available(self):
        fake = SimpleNamespace(float16="fp16", bfloat16="bf16", cuda=SimpleNamespace(
            get_device_capability=lambda: (7, 5),
            is_bf16_supported=lambda **kwargs: True))
        self.assertEqual(cuda_dtype(fake), "fp16")

    def test_bf16_requires_native_hardware_support(self):
        queries = []
        def supported(*, including_emulation):
            queries.append(including_emulation)
            return including_emulation
        fake = SimpleNamespace(float16="fp16", bfloat16="bf16", cuda=SimpleNamespace(
            get_device_capability=lambda: (8, 6), is_bf16_supported=supported))
        self.assertEqual(cuda_dtype(fake), "fp16")
        self.assertEqual(queries, [False])


if __name__ == "__main__":
    unittest.main()
