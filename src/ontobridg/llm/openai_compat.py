from __future__ import annotations

import json
import os
import random
import urllib.error
import urllib.request
from typing import Any

from ontobridg.llm.base import LLMBackend, LLMResponse


class OpenAICompatLLM(LLMBackend):
    """Minimal OpenAI-compatible chat completions client.

    Works with any OpenAI-compatible provider, including dots studio:
      ONTOBRIDG_BASE_URL=https://note3-prev-api.askdiandian.com/v1
      ONTOBRIDG_MODEL=dots3-note-prev
      ONTOBRIDG_API_KEY=ak_...
    """

    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: float = 180.0,
        max_retries: int = 3,
    ) -> None:
        default_base = "https://api.openai.com/v1"
        # Prefer provider-specific env if set
        self.base_url = (
            base_url
            or os.environ.get("ONTOBRIDG_BASE_URL")
            or default_base
        ).rstrip("/")
        self.api_key = (
            api_key
            or os.environ.get("ONTOBRIDG_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or ""
        )
        self.model = (
            model
            if model
            else os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini")
        )
        # If caller left default model but env has dots/other model, prefer env when base_url env is set
        env_model = os.environ.get("ONTOBRIDG_MODEL")
        if env_model and os.environ.get("ONTOBRIDG_BASE_URL") and model in {"", "gpt-4o-mini"}:
            # only override when user didn't explicitly pass a non-default model via factory
            pass
        if not model and env_model:
            self.model = env_model
        if model == "gpt-4o-mini" and env_model and os.environ.get("ONTOBRIDG_BASE_URL"):
            # factory default; allow env override when dots-like base is configured
            if "askdiandian" in (os.environ.get("ONTOBRIDG_BASE_URL") or "") or "dots" in (
                os.environ.get("ONTOBRIDG_BASE_URL") or ""
            ):
                self.model = env_model
        # env timeout override
        try:
            self.timeout = float(os.environ.get("ONTOBRIDG_TIMEOUT") or timeout)
        except ValueError:
            self.timeout = timeout
        self.max_retries = int(os.environ.get("ONTOBRIDG_MAX_RETRIES") or max_retries)
        self.temperature = temperature
        self._client = None

    def _post_json(self, url: str, headers: dict, payload: bytes) -> dict:
        import time

        last_exc: Exception | None = None
        for attempt in range(max(1, self.max_retries)):
            try:
                try:
                    import requests

                    resp = requests.post(
                        url,
                        headers=headers,
                        data=payload,
                        timeout=(15.0, self.timeout),
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    if not isinstance(data, dict):
                        raise ValueError("LLM endpoint returned a non-object JSON payload")
                    return data
                except ImportError:
                    req = urllib.request.Request(
                        url, data=payload, headers=headers, method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        if not isinstance(data, dict):
                            raise ValueError("LLM endpoint returned a non-object JSON payload")
                        return data
            except Exception as exc:
                last_exc = exc
                status = getattr(exc, "code", None)
                response = getattr(exc, "response", None)
                if status is None and response is not None:
                    status = getattr(response, "status_code", None)
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"LLM request rejected with HTTP {status}: {exc}") from exc
                if attempt + 1 >= max(1, self.max_retries):
                    break

                retry_after = None
                response_headers = getattr(response, "headers", None)
                if response_headers:
                    retry_after = response_headers.get("Retry-After")
                if retry_after is None and isinstance(exc, urllib.error.HTTPError):
                    retry_after = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = float(retry_after) if retry_after else 0.0
                except (TypeError, ValueError):
                    delay = 0.0
                if delay <= 0:
                    delay = min(30.0, 1.5 * (2**attempt)) + random.uniform(0.0, 0.5)
                time.sleep(delay)
        raise RuntimeError(f"LLM request failed after retries: {last_exc}")

    def complete(self, prompt: str, *, task: str = "", meta: dict[str, Any] | None = None) -> LLMResponse:
        _ = task, meta
        if not self.api_key:
            raise RuntimeError(
                "Missing API key. Set ONTOBRIDG_API_KEY or OPENAI_API_KEY."
            )
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": int(os.environ.get("ONTOBRIDG_MAX_TOKENS") or 1024),
            "messages": [
                {
                    "role": "system",
                    "content": "You are an ontology learning assistant. Respond with compact JSON only. Keep output under 400 tokens.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        data = self._post_json(url, headers, json.dumps(body).encode("utf-8"))
        try:
            choice = data["choices"][0]
            message = choice["message"]
            text = message.get("content")
            if not text:
                finish_reason = choice.get("finish_reason", "unknown")
                reasoning_chars = len(message.get("reasoning_content") or "")
                raise RuntimeError(
                    "LLM returned no final content "
                    f"(finish_reason={finish_reason}, reasoning_chars={reasoning_chars}). "
                    "Increase ONTOBRIDG_MAX_TOKENS for reasoning models."
                )
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Malformed chat completion response: {data}") from exc
        return LLMResponse(text=text or "", raw=data, backend=self.name)
