from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ontobridg.data.schema import OntologyKnowledge
from ontobridg.llm.base import LLMBackend
from ontobridg.llm.mock import MockLLM
from ontobridg.llm.openai_compat import OpenAICompatLLM


def build_llm(
    backend: str = "mock",
    *,
    knowledge: OntologyKnowledge | None = None,
    model: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> LLMBackend:
    backend_l = backend.lower()
    if backend_l in {"mock", "offline"}:
        return MockLLM(knowledge=knowledge, **{k: v for k, v in kwargs.items() if k in {"default_type", "seed"}})
    if backend_l in {"openai", "openai_compat", "api"}:
        # Priority: explicit args > environment > ignored local dots_config.json.
        if not base_url:
            base_url = os.environ.get("ONTOBRIDG_BASE_URL")
        if not api_key:
            api_key = (
                os.environ.get("ONTOBRIDG_API_KEY")
                or os.environ.get("OPENAI_API_KEY")
            )
        if not model or model == "gpt-4o-mini":
            model = os.environ.get("ONTOBRIDG_MODEL") or model
        if not base_url or not api_key or (not model or model == "gpt-4o-mini"):
            # factory.py -> llm/ -> ontobridg/ -> src/dots_config.json
            cfg_path = Path(__file__).resolve().parents[2] / "dots_config.json"
            if cfg_path.exists():
                try:
                    import json as _json

                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    base_url = base_url or cfg.get("base_url") or None
                    api_key = api_key or (cfg.get("api_key") or None)
                    if not model or model == "gpt-4o-mini":
                        model = cfg.get("model") or model
                except Exception:
                    pass
        if not model or model == "gpt-4o-mini":
            model = os.environ.get("ONTOBRIDG_MODEL") or model or "gpt-4o-mini"
        # Last chance: JSON base if still default OpenAI and json has different base
        if (not base_url or "api.openai.com" in str(base_url)) and (
            not model or model == "gpt-4o-mini"
        ):
            cfg_path = Path(__file__).resolve().parents[2] / "dots_config.json"
            if cfg_path.exists():
                try:
                    import json as _json

                    cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
                    if cfg.get("base_url") and not os.environ.get("ONTOBRIDG_BASE_URL"):
                        base_url = cfg["base_url"]
                        if cfg.get("model"):
                            model = cfg["model"]
                        if not api_key and cfg.get("api_key"):
                            api_key = cfg["api_key"]
                except Exception:
                    pass
        return OpenAICompatLLM(base_url=base_url, api_key=api_key, model=model)
    if backend_l in {"local", "local_http"}:
        from ontobridg.llm.local import LocalHTTPLLM

        return LocalHTTPLLM(base_url=base_url, model=model)
    raise ValueError(f"Unknown LLM backend: {backend}")
