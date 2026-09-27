from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class LLMResponse:
    text: str
    raw: dict[str, Any] = field(default_factory=dict)
    backend: str = ""


class LLMBackend(ABC):
    name: str = "base"

    @abstractmethod
    def complete(self, prompt: str, *, task: str = "", meta: dict[str, Any] | None = None) -> LLMResponse:
        raise NotImplementedError

    def complete_json(self, prompt: str, *, task: str = "", meta: dict[str, Any] | None = None) -> dict[str, Any]:
        response = self.complete(prompt, task=task, meta=meta)
        return parse_json_payload(response.text)
def parse_json_payload(text: str) -> dict[str, Any]:
    import json
    import re

    text = text.strip()
    if not text:
        return {}
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {"value": data}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data if isinstance(data, dict) else {"value": data}
        except json.JSONDecodeError:
            return {"raw": text}
    return {"raw": text}
