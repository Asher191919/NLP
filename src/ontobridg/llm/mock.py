"""Unified Mock LLM backends.

Policy:
- `MockLLM` (this module) = prompt-path backend implementing LLMBackend from base.py
- `RAGMockLLM` / open-vocab suggestions live in mock_llm.py
- Both accept OntologyKnowledge; no gold leakage unless caller injects knowledge.
"""
from __future__ import annotations

import json
import re
from typing import Any

from ontobridg.data.schema import OntologyKnowledge
from ontobridg.llm.base import LLMBackend as BaseLLMBackend
from ontobridg.llm.base import LLMResponse

# Re-export RAG-path mock symbols for a single import surface
from ontobridg.llm.mock_llm import (  # noqa: F401
    DEFAULT_PATTERNS,
    LLMBackend as RAGLLMBackend,
    MockLLM as RAGMockLLM,
    RelationSuggestion,
)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


class MockLLM(BaseLLMBackend):
    """Deterministic offline backend for prompt-path OntoBridgPipeline."""

    name = "mock"

    def __init__(
        self,
        knowledge: OntologyKnowledge | None = None,
        *,
        default_type: str = "Entity",
        seed: int = 42,
        closed_vocab_only: bool = False,
        noise: float = 0.0,
        **_: Any,
    ) -> None:
        self.knowledge = knowledge or OntologyKnowledge()
        self.default_type = default_type
        self.seed = seed
        self.closed_vocab_only = closed_vocab_only

    def complete(self, prompt: str, *, task: str = "", meta: dict[str, Any] | None = None) -> LLMResponse:
        meta = meta or {}
        resolved_task = task or meta.get("task", "")
        payload = self._predict(prompt, resolved_task, meta)
        return LLMResponse(
            text=json.dumps(payload, ensure_ascii=False),
            raw=payload,
            backend=self.name,
        )

    def suggest_type(self, term: str, types: list[str], context: str | None = None) -> tuple[str, float]:
        term_n = _normalize(term)
        if term_n in self.knowledge.term_to_type:
            return self.knowledge.term_to_type[term_n], 0.95
        for t in types:
            if term_n in _normalize(t) or _normalize(t) in term_n:
                return t, 0.55
        return (types[0] if types else self.default_type), 0.2

    def suggest_taxonomy(self, types: list[str], term_types: dict[str, str]) -> list[tuple[str, str, float]]:
        out: list[tuple[str, str, float]] = []
        for child, parents in self.knowledge.parents.items():
            for parent in parents:
                if child in types or parent in types or not types:
                    out.append((child, parent, 0.85))
        return out

    def suggest_terms(self, corpus: list[str], known: set[str] | None = None) -> list[str]:
        known = known or set(self.knowledge.term_to_type)
        found = list(known)[:50]
        return found

    def _predict(self, prompt: str, task: str, meta: dict[str, Any]) -> dict[str, Any]:
        task_l = str(task).lower()
        if task_l in {"term_typing", "task_a", "a", "typing"}:
            return self._task_a(prompt, meta)
        if task_l in {"taxonomy", "task_b", "b", "is_a", "taxonomy_discovery"}:
            return self._task_b(prompt, meta)
        if task_l in {"relation", "relations", "task_c", "c", "relation_extraction"}:
            return self._task_c(prompt, meta)
        if task_l in {"term_extraction", "task_0", "0", "extract"}:
            return self._task_0(prompt, meta)
        return {"raw": "unknown_task", "task": task}

    def _extract_term(self, prompt: str, meta: dict[str, Any]) -> str:
        if meta.get("term"):
            return str(meta["term"])
        for pat in (
            r'Term:\s*["\']?([^"\'\n]+)',
            r'term\s*=\s*["\']([^"\']+)["\']',
            r'["\']term["\']\s*:\s*["\']([^"\']+)["\']',
        ):
            m = re.search(pat, prompt)
            if m:
                return m.group(1).strip()
        return ""

    def _task_a(self, prompt: str, meta: dict[str, Any]) -> dict[str, Any]:
        terms = meta.get("terms") or meta.get("candidates") or []
        types = meta.get("types") or meta.get("type_inventory") or list(
            set(self.knowledge.term_to_type.values())
        ) or [self.default_type]
        if not terms:
            term = self._extract_term(prompt, meta)
            terms = [term] if term else list(self.knowledge.term_to_type)[:20]
        items = []
        for t in terms:
            ty, conf = self.suggest_type(str(t), list(types), meta.get("context"))
            items.append({"term": t, "type": ty, "score": conf})
        return {"terms": items}

    def _task_b(self, prompt: str, meta: dict[str, Any]) -> dict[str, Any]:
        types = meta.get("types") or []
        pairs = self.suggest_taxonomy(list(types), meta.get("term_types") or {})
        return {
            "taxonomy": [
                {"child": c, "parent": p, "score": s} for c, p, s in pairs
            ]
        }

    def _task_c(self, prompt: str, meta: dict[str, Any]) -> dict[str, Any]:
        corpus = meta.get("corpus") or [prompt]
        if isinstance(corpus, str):
            corpus = [corpus]
        term_types = meta.get("term_types") or self.knowledge.term_to_type
        # delegate to RAG mock relation suggestion for open/closed behavior
        rag = RAGMockLLM(
            knowledge=self.knowledge,
            closed_vocab_only=self.closed_vocab_only,
            seed=self.seed,
        )
        sugg = rag.suggest_relations(
            terms=list(term_types),
            term_types=term_types,
            corpus=list(corpus),
            open_vocabulary=not self.closed_vocab_only,
        )
        return {
            "relations": [
                {
                    "head": s.head,
                    "relation": s.relation,
                    "tail": s.tail,
                    "score": s.score,
                    "source": s.source,
                }
                for s in sugg
            ]
        }

    def _task_0(self, prompt: str, meta: dict[str, Any]) -> dict[str, Any]:
        candidates = meta.get("candidates") or list(self.knowledge.term_to_type)
        text = str(meta.get("text") or prompt).lower()
        terms = []
        for c in candidates:
            if str(c).lower() in text:
                terms.append({"term": c, "type": self.knowledge.term_to_type.get(_normalize(c), self.default_type)})
        return {"terms": terms}


__all__ = [
    "MockLLM",
    "RAGMockLLM",
    "RAGLLMBackend",
    "RelationSuggestion",
    "DEFAULT_PATTERNS",
]
