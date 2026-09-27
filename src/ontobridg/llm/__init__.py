"""LLM backends for OntoBridg."""
from ontobridg.llm.base import LLMBackend, LLMResponse, parse_json_payload
from ontobridg.llm.mock import MockLLM
from ontobridg.llm.factory import build_llm

try:
    from ontobridg.llm.openai_compat import OpenAICompatLLM
except Exception:  # pragma: no cover
    OpenAICompatLLM = None  # type: ignore[assignment,misc]

try:
    from ontobridg.llm.mock_llm import (
        LLMBackend as SuggestBackend,
        MockLLM as PatternMockLLM,
        get_backend,
        RelationSuggestion,
    )
except Exception:  # pragma: no cover
    PatternMockLLM = None  # type: ignore[assignment,misc]
    SuggestBackend = None  # type: ignore[assignment,misc]
    get_backend = None  # type: ignore[assignment,misc]
    RelationSuggestion = None  # type: ignore[assignment,misc]

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "MockLLM",
    "OpenAICompatLLM",
    "build_llm",
    "parse_json_payload",
    "PatternMockLLM",
    "SuggestBackend",
    "get_backend",
    "RelationSuggestion",
]
