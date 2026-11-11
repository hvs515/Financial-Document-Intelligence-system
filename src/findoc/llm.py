"""LLM backends: local Transformers (with an optional LoRA adapter) or any OpenAI-compatible server."""

from __future__ import annotations

from typing import Protocol

import httpx

from .config import Settings


class LLM(Protocol):
    def generate(self, messages: list[dict], max_new_tokens: int = 400) -> str: ...


class TransformersLLM:
    def __init__(self, model: str, adapter: str = "", load_in_4bit: bool = False) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(adapter or model)
        kwargs: dict = {}
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
            kwargs["dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            if load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=kwargs["dtype"],
                )
        elif load_in_4bit:
            raise RuntimeError("4-bit loading needs a CUDA GPU, and none was found.")
        self._model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        if adapter:
            from peft import PeftModel

            self._model = PeftModel.from_pretrained(self._model, adapter)
        self._model.eval()

    def generate(self, messages: list[dict], max_new_tokens: int = 400) -> str:
        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with self._torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # the same question over the same filing gives the same answer
                pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
            )
        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "not-needed",
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=600,
        )

    def generate(self, messages: list[dict], max_new_tokens: int = 400) -> str:
        response = self._client.post(
            "/chat/completions",
            json={
                "model": self._model,
                "messages": messages,
                "max_tokens": max_new_tokens,
                "temperature": 0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def build_llm(settings: Settings) -> LLM:
    if settings.llm_backend == "transformers":
        return TransformersLLM(settings.llm_model, settings.llm_adapter, settings.llm_load_in_4bit)
    if settings.llm_backend == "openai":
        return OpenAICompatibleLLM(settings.llm_base_url, settings.llm_model, settings.llm_api_key)
    raise ValueError(f"Unknown LLM backend: {settings.llm_backend}")
