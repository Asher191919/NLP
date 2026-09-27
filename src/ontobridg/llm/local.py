from __future__ import annotations

import json
import os
from typing import Any

from ontobridg.llm.base import LLMBackend, LLMResponse


class LocalHTTPLLM(LLMBackend):
    """Optional local OpenAI-compatible server (llama.cpp, vLLM, Ollama, LM Studio)."""

    name = "local_http"

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "local-model",
        timeout: float = 120.0,
    ) -> None:
        self.base_url = (base_url or os.environ.get("ONTOBRIDG_LOCAL_URL") or "http://127.0.0.1:8000/v1").rstrip("/")
        self.model = model
        self.timeout = timeout
        self._impl = None

    def _get_impl(self):
        if self._impl is not None:
            return self._impl
        from ontobridg.llm.openai_compat import OpenAICompatLLM

        self._impl = OpenAICompatLLM(
            base_url=self.base_url,
            api_key=os.environ.get("ONTOBRIDG_LOCAL_KEY") or "local",
            model=self.model,
            temperature=0.0,
            timeout=self.timeout,
        )
        return self._impl

    def complete(self, prompt: str, *, task: str = "", meta: dict[str, Any] | None = None) -> LLMResponse:
        impl = self._get_impl()
        response = impl.complete(prompt, task=task, meta=meta)
        return LLMResponse(text=response.text, raw=response.raw, backend=self.name)


def try_load_local_model(model_name_or_path: str) -> LLMBackend | None:
    """Best-effort local transformers backend. Returns None if unavailable."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception:
        return None

    class _TransformersLLM(LLMBackend):
        name = "transformers"

        def __init__(self, path: str) -> None:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
            self.model = AutoModelForCausalLM.from_pretrained(path)
            self.model.eval()

        def complete(self, prompt: str, *, task: str = "", meta: dict[str, Any] | None = None) -> LLMResponse:
            _ = task, meta
            inputs = self.tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=256, do_sample=False)
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):]
            return LLMResponse(text=text.strip(), raw={}, backend=self.name)

    try:
        return _TransformersLLM(model_name_or_path)
    except Exception:
        return None


def dumps_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
