"""Optional real-model smoke tests: tiny random Qwen, local tokenizer, no downloads."""

import importlib.util
import copy
import tempfile
import unittest
from pathlib import Path

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "peft", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "Optional LLM dependencies are not installed")
class LLMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

        torch.set_num_threads(2)
        words = ["[UNK]", "[PAD]", "[EOS]", "<|system|>", "<|user|>", "<|assistant|>",
                 "You", "are", "helpful", "Cancel", "my", "order", "tool", "args", "finish",
                 "{", "}", ":", ",", '"', "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]
        tokenizer = Tokenizer(WordLevel({word: i for i, word in enumerate(words)}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                                unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
        cls.tokenizer.add_special_tokens({"additional_special_tokens": ["<|system|>", "<|user|>", "<|assistant|>"]})
        cls.tokenizer.chat_template = (
            "{% for message in messages %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token }}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
        )
        cls.config = Qwen2Config(vocab_size=len(cls.tokenizer), hidden_size=32,
                                 intermediate_size=64, num_hidden_layers=1,
                                 num_attention_heads=2, num_key_value_heads=1,
                                 max_position_embeddings=4096,
                                 eos_token_id=cls.tokenizer.eos_token_id,
                                 pad_token_id=cls.tokenizer.pad_token_id)
        cls.model_class = Qwen2ForCausalLM
        cls.messages = [{"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Cancel my order"}]

    def backend(self):
        import torch
        from sca.llm import HFBackend
        torch.manual_seed(42)
        return HFBackend(self.model_class(self.config), self.tokenizer, max_new_tokens=4)

    def test_response_mask_and_length_guard(self):
        from sca.llm import encode_example
        example = {"messages": self.messages, "response": '{"tool":"finish","args":{}}'}
        row = encode_example(self.tokenizer, example, 512)
        prefix = self.tokenizer.apply_chat_template(self.messages, tokenize=True, add_generation_prompt=True)
        self.assertEqual(row["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertGreater(sum(value != -100 for value in row["labels"]), 0)
        self.assertIsNone(encode_example(self.tokenizer, example, 1))

    def test_lora_updates_and_saves_reloadable_adapter(self):
        import torch
        from peft import PeftModel
        from sca.llm import train_adapter

        backend = self.backend()
        examples = [{"split": "train", "messages": self.messages,
                     "response": '{"tool":"finish","args":{}}'}]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "adapter"
            result = train_adapter(backend, examples, output, max_steps=2, accumulation=1,
                                   max_length=512, rank=2)
            self.assertEqual(result["optimizer_steps"], 2)
            self.assertTrue((output / "adapter_model.safetensors").is_file())
            changed = any(parameter.detach().abs().sum() > 0 for name, parameter in backend.model.named_parameters()
                          if "lora_B" in name)
            self.assertTrue(changed)
            ids = torch.tensor([[6, 7, 8]])
            with torch.inference_mode():
                expected = backend.model(input_ids=ids).logits
                fresh_base = self.backend().model
                reloaded = PeftModel.from_pretrained(fresh_base, output).eval()
                actual = reloaded(input_ids=ids).logits
            self.assertTrue(torch.allclose(expected, actual, atol=1e-6))
            self.assertIsInstance(backend.generate(self.messages, 42, temperature=0), str)

    def test_training_rejects_test_rows(self):
        from sca.llm import train_adapter
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                train_adapter(self.backend(), [{"split": "test"}], Path(tmp), max_steps=1)

    def test_native_tool_format_uses_the_same_masking_for_real_lora_updates(self):
        from sca.agents import Episode
        from sca.llm import encode_example, train_adapter
        from sca.protocol import apply_chat_template
        from sca.tasks import template_task

        backend = self.backend()
        backend.tokenizer = copy.deepcopy(backend.tokenizer)
        # Independent minimal test fixture, not the downloaded Qwen template.
        backend.tokenizer.chat_template = (
            "{% if tools %}{{ tools | tojson }}{% endif %}"
            "{% for message in messages %}"
            "{% if message.get('tool_calls') %}{{ '<|assistant|>\\n' }}"
            "{% for call in message['tool_calls'] %}{{ '<tool_call>' + (call['function'] | tojson) + '</tool_call>' }}{% endfor %}{{ eos_token }}"
            "{% elif message['role'] == 'tool' %}{{ '<tool_response>' + message['content'] + '</tool_response>' + eos_token }}"
            "{% else %}{{ '<|' + message['role'] + '|>\\n' + message['content'] + eos_token }}{% endif %}{% endfor %}"
            "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% endif %}"
        )
        episode = Episode(template_task(123))
        episode.action("lookup_customer", email="customer123@example.com")
        example = {"split": "train", **episode.examples[0]}
        encoded = encode_example(backend.tokenizer, example, 4096)
        prefix = apply_chat_template(backend.tokenizer, example["messages"], tokenize=True, add_generation_prompt=True)
        self.assertEqual(encoded["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertGreater(sum(x != -100 for x in encoded["labels"]), 0)
        with tempfile.TemporaryDirectory() as tmp:
            result = train_adapter(backend, [example], Path(tmp), max_steps=1,
                                   accumulation=1, max_length=4096, rank=2)
            self.assertEqual(result["used_examples"], 1)
            self.assertIsInstance(backend.generate(example["messages"], 42, temperature=0), str)

    def test_half_precision_base_keeps_trainable_adapter_stable(self):
        import torch
        from sca.llm import train_adapter

        backend = self.backend()
        backend.model.half()
        examples = [{"split": "train", "messages": self.messages,
                     "response": '{"tool":"finish","args":{}}'}]
        with tempfile.TemporaryDirectory() as tmp:
            result = train_adapter(backend, examples, Path(tmp), max_steps=1,
                                   accumulation=1, max_length=512, rank=2)
            self.assertTrue(all(torch.isfinite(p).all() for p in backend.model.parameters()))
            self.assertTrue(all(p.dtype == torch.float32 for p in backend.model.parameters() if p.requires_grad))
            self.assertEqual(result["base_parameter_dtype"], "torch.float16")


if __name__ == "__main__":
    unittest.main()
