"""Optional Hugging Face inference and response-only LoRA rejection fine-tuning."""

import importlib.metadata
import json
import random
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


class HFBackend:
    def __init__(self, model, tokenizer, max_new_tokens: int = 128):
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    @property
    def runtime_info(self) -> dict:
        parameter = next(self.model.parameters())
        return {"device": str(parameter.device), "parameter_dtype": str(parameter.dtype),
                "max_new_tokens": self.max_new_tokens}

    @classmethod
    def load(cls, model_id: str = DEFAULT_MODEL, device: str = "auto",
             max_new_tokens: int = 128) -> "HFBackend":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float32
        if device == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("CUDA is unavailable. Select a GPU runtime before starting this run.")
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa",
        ).to(device)
        return cls(model, tokenizer, max_new_tokens)

    def generate(self, messages: list[dict], seed: int, temperature: float = 0.7) -> str:
        import torch

        torch.manual_seed(seed)
        self.model.eval()
        device = next(self.model.parameters()).device
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        ).to(device)
        # Generic fast tokenizers can emit these, but Qwen's causal LM rejects them.
        inputs.pop("token_type_ids", None)
        options = {"do_sample": temperature > 0, "max_new_tokens": self.max_new_tokens,
                   "pad_token_id": self.tokenizer.pad_token_id,
                   "eos_token_id": self.tokenizer.eos_token_id}
        if temperature > 0:
            options.update(temperature=temperature, top_p=0.95)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **options)
        return self.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def encode_example(tokenizer, example: dict, max_length: int) -> dict | None:
    """Mask prompts/observations. Drop long examples instead of truncating their answers."""
    prefix = tokenizer.apply_chat_template(example["messages"], tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        example["messages"] + [{"role": "assistant", "content": example["response"]}],
        tokenize=True, add_generation_prompt=False,
    )
    if full[:len(prefix)] != prefix:
        raise ValueError("This chat template does not have a stable assistant prefix")
    if len(full) > max_length or len(full) <= len(prefix):
        return None
    return {"input_ids": full, "labels": [-100] * len(prefix) + full[len(prefix):]}


def train_adapter(backend: HFBackend, examples: list[dict], output: Path,
                  max_steps: int = 40, accumulation: int = 4,
                  learning_rate: float = 2e-4, max_length: int = 2048,
                  seed: int = 42, rank: int = 8) -> dict:
    """RFT = supervised updates on successful self-rollout actions, never reference solutions."""
    import torch
    from peft import LoraConfig, get_peft_model

    if max_steps < 1 or accumulation < 1:
        raise ValueError("max_steps and accumulation must be positive")
    if any(example.get("split") != "train" for example in examples):
        raise ValueError("Refusing to train on examples not explicitly marked train")
    torch.manual_seed(seed)
    encoded = [encode_example(backend.tokenizer, ex, max_length) for ex in examples]
    rows = [row for row in encoded if row is not None]
    if not rows:
        raise ValueError("No trainable successful actions. Collect more successful rollouts or raise max_length.")
    model = get_peft_model(backend.model, LoraConfig(
        r=rank, lora_alpha=rank * 2, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM",
    ))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    device = next(model.parameters()).device
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    rng = random.Random(seed)
    order = list(range(len(rows)))
    cursor = len(order)
    losses = []
    model.train()
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for _ in range(accumulation):
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            row = rows[order[cursor]]
            cursor += 1
            ids = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            loss = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss; adapter was not saved")
            (loss / accumulation).backward()
            total_loss += float(loss.detach()) / accumulation
        torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        losses.append({"step": step + 1, "loss": total_loss})
        if (step + 1) % 10 == 0 or step == 0 or step + 1 == max_steps:
            print(f"LoRA update {step + 1}/{max_steps}: loss={total_loss:.4f}", flush=True)
    model.gradient_checkpointing_disable()
    model.config.use_cache = True
    model.eval()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    backend.tokenizer.save_pretrained(output)
    backend.model = model
    result = {"optimizer_steps": max_steps, "gradient_accumulation": accumulation,
              "learning_rate": learning_rate, "rank": rank, "seed": seed,
              "input_examples": len(examples), "used_examples": len(rows),
              "dropped_long_examples": len(examples) - len(rows),
              "response_only_loss": True,
              "device": str(device), "base_parameter_dtype": str(next(model.parameters()).dtype),
              "trainable_parameters": sum(p.numel() for p in parameters),
              "loss": losses}
    (output / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def environment_versions() -> dict:
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed"
    return result
