"""Prompt templates + prompt style enum for OntoBridg LLM backends."""
from __future__ import annotations

from enum import Enum


class PromptStyle(str, Enum):
    INSTRUCTION = "instruction"
    CLOZE = "cloze"
    FEWSHOT = "fewshot"


TASK_A_TERM_TYPING = """You are an ontology learning system.
Given terms from a {domain} corpus and a type inventory, assign each term exactly one type.
Types: {types}
Terms: {terms}
Return JSON list: [{{"term": "...", "type": "..."}}]
"""

TASK_A_TAXONOMY = """Discover is-a (taxonomy) edges among types/terms.
Types: {types}
Typed terms: {term_types}
Return JSON list: [{{"child": "...", "parent": "..."}}]
Constraints: no cycles; parent must be a valid type.
"""

TASK_A_RELATIONS_OPEN = """Extract non-taxonomic relations using an OPEN relation vocabulary.
Do NOT restrict to a closed taxonomy. Consider relations such as:
part-of, causes, located-in, treats, interacts-with, manufactures, has-feature, capital-of...
Corpus: {corpus}
Typed terms: {term_types}
Known domain-range constraints: {domain_ranges}
Return JSON list: [{{"head": "...", "relation": "...", "tail": "...", "score": 0.0}}]
"""

TASK_B_EXTENSION = """Extend the seed ontology with new terms, types, and relations.
Seed terms/types: {seed}
Corpus evidence: {corpus}
Type inventory (extendable): {types}
Predict additional terms->type, is-a edges, and OPEN non-taxonomic relations.
Return JSON: {{"terms": [...], "taxonomy": [...], "relations": [...]}}
"""

CONTAMINATION_PROBE = """Complete the following ontology facts if you know them.
Term: {term}
Likely type: """

TASK0_TERM_EXTRACTION = """Extract domain terms from the text below.
Text: {text}
Candidate glossary (optional): {candidates}
Return JSON: {{"terms": [{{"term": "...", "type": "..."}}]}}
"""


def task0_prompt(text: str, candidates: list[str] | None = None) -> str:
    return TASK0_TERM_EXTRACTION.format(
        text=text,
        candidates=", ".join(candidates[:50]) if candidates else "(none)",
    )


def taska_prompt(terms: list[str], types: list[str], domain: str = "general") -> str:
    return TASK_A_TERM_TYPING.format(
        domain=domain,
        types=", ".join(types[:80]),
        terms=", ".join(terms[:80]),
    )


def taskb_prompt(types: list[str], term_types: dict[str, str] | None = None) -> str:
    tt = term_types or {}
    pairs = "; ".join(f"{k}->{v}" for k, v in list(tt.items())[:60])
    return TASK_A_TAXONOMY.format(types=", ".join(types[:80]), term_types=pairs or "(none)")


def taskc_prompt(
    corpus: list[str] | str,
    term_types: dict[str, str] | None = None,
    domain_ranges: dict | None = None,
) -> str:
    if isinstance(corpus, list):
        corpus_s = "\n".join(corpus[:30])
    else:
        corpus_s = str(corpus)
    return TASK_A_RELATIONS_OPEN.format(
        corpus=corpus_s[:4000],
        term_types=str(term_types or {}),
        domain_ranges=str(domain_ranges or {}),
    )


def taska_extension_prompt(seed: str, corpus: str, types: list[str]) -> str:
    return TASK_B_EXTENSION.format(seed=seed, corpus=corpus[:4000], types=", ".join(types[:80]))


__all__ = [
    "PromptStyle",
    "TASK_A_TERM_TYPING",
    "TASK_A_TAXONOMY",
    "TASK_A_RELATIONS_OPEN",
    "TASK_B_EXTENSION",
    "CONTAMINATION_PROBE",
    "TASK0_TERM_EXTRACTION",
    "task0_prompt",
    "taska_prompt",
    "taskb_prompt",
    "taskc_prompt",
    "taska_extension_prompt",
]
