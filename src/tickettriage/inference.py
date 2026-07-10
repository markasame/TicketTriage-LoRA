"""Model backends for triage inference.

Three backends behind one interface:

- ``TransformersBackend`` — HF transformers, optionally with a PEFT adapter on top
  of the 4-bit base model. Used for training-time eval on the GPU box.
- ``OllamaBackend``      — HTTP calls to a local Ollama server running the exported
  GGUF. Used for lightweight serving and the demo.
- ``EchoBackend``        — deterministic canned outputs for tests and CI.

All backends implement ``generate(user_prompt) -> str`` with the shared system
prompt applied, plus ``triage(ticket) -> TriageResult`` from the base class.
"""

from __future__ import annotations

import time

from .prompts import (
    SYSTEM_PROMPT,
    classify_prompt,
    parse_intent_output,
    parse_priority_output,
    priority_prompt,
    reply_prompt,
)
from .schema import TriageResult


class Backend:
    name = "backend"

    def generate(self, user_prompt: str, max_new_tokens: int = 400) -> str:
        raise NotImplementedError

    def triage(self, ticket: str) -> TriageResult:
        start = time.perf_counter()
        intent_raw = self.generate(classify_prompt(ticket), max_new_tokens=20)
        priority_raw = self.generate(priority_prompt(ticket), max_new_tokens=80)
        reply = self.generate(reply_prompt(ticket), max_new_tokens=400)
        latency = time.perf_counter() - start

        priority, reason = parse_priority_output(priority_raw)
        return TriageResult(
            intent=parse_intent_output(intent_raw),
            priority=priority,
            priority_reason=reason,
            draft_reply=reply.strip(),
            model=self.name,
            latency_s=round(latency, 3),
        )


class EchoBackend(Backend):
    """Deterministic fake backend for tests."""

    name = "echo"

    def generate(self, user_prompt: str, max_new_tokens: int = 400) -> str:
        if "Classify the customer ticket" in user_prompt:
            return "track_order"
        if "Assign a priority" in user_prompt:
            return "priority: medium\nreason: Standard order tracking request."
        return "Hello! I can help you track your order {{Order Number}}."


class OllamaBackend(Backend):
    """Calls a local Ollama server (used for GGUF serving and the demo)."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")
        self.name = f"ollama:{model}"

    def generate(self, user_prompt: str, max_new_tokens: int = 400) -> str:
        import httpx

        resp = httpx.post(
            f"{self.host}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
                "options": {"num_predict": max_new_tokens, "temperature": 0.2},
            },
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


class TransformersBackend(Backend):
    """HF transformers backend; loads base model in 4-bit, optionally with an adapter."""

    def __init__(self, base_model: str, adapter_path: str | None = None, device_map: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model, quantization_config=bnb, device_map=device_map
        )
        self.name = base_model.split("/")[-1]
        if adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.name += "+lora"
        self.model.eval()

    def generate(self, user_prompt: str, max_new_tokens: int = 400) -> str:
        import torch

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)


def make_backend(spec: str) -> Backend:
    """Build a backend from a spec string.

    - ``echo``                       -> EchoBackend
    - ``ollama:<model>``             -> OllamaBackend
    - ``hf:<base_model>``            -> TransformersBackend (base only)
    - ``hf:<base_model>@<adapter>``  -> TransformersBackend with adapter
    """
    if spec == "echo":
        return EchoBackend()
    if spec.startswith("ollama:"):
        return OllamaBackend(spec.split(":", 1)[1])
    if spec.startswith("hf:"):
        rest = spec[3:]
        if "@" in rest:
            base, adapter = rest.split("@", 1)
            return TransformersBackend(base, adapter)
        return TransformersBackend(rest)
    raise ValueError(f"Unknown backend spec: {spec!r}")
