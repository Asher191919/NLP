"""Mock LLM backend.

Unlike a closed-taxonomy-only extractor (the pro-team failure mode), this mock
generates *open-vocabulary non-taxonomic relation suggestions* from corpus
patterns + a noisy knowledge base, so Task A/B can recover non-zero non-tax
relations under symbolic domain-range constraints.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ontobridg.data.schema import OntologyKnowledge, RelationSample


@dataclass
class RelationSuggestion:
    head: str
    relation: str
    tail: str
    score: float
    source: str  # "llm" | "pattern" | "rag" | "kb"
    evidence: str = ""


class LLMBackend(ABC):
    @abstractmethod
    def suggest_type(self, term: str, types: list[str], context: str | None = None) -> tuple[str, float]:
        ...

    @abstractmethod
    def suggest_taxonomy(self, types: list[str], term_types: dict[str, str]) -> list[tuple[str, str, float]]:
        ...

    @abstractmethod
    def suggest_relations(
        self,
        terms: list[str],
        term_types: dict[str, str],
        corpus: list[str],
        known_relations: set[tuple[str, str, str]] | None = None,
        open_vocabulary: bool = True,
    ) -> list[RelationSuggestion]:
        ...

    @abstractmethod
    def suggest_terms(self, corpus: list[str], known: set[str] | None = None) -> list[str]:
        ...


# Surface patterns used to mine open relation vocabulary from corpus text.
DEFAULT_PATTERNS: list[tuple[str, str]] = [
    (r"(.+?) is used to treat (.+)", "treats"),
    (r"(.+?) treats (.+)", "treats"),
    (r"(.+?) is a therapy for (.+)", "treats"),
    (r"(.+?) causes (.+)", "causes"),
    (r"(.+?) leads to (.+)", "causes"),
    (r"(.+?) results in (.+)", "causes"),
    (r"(.+?) is part of (.+)", "part-of"),
    (r"(.+?) is a component of (.+)", "part-of"),
    (r"(.+?) interacts with (.+)", "interacts-with"),
    (r"(.+?) binds (.+)", "interacts-with"),
    (r"(.+?) is located in (.+)", "located-in"),
    (r"(.+?) lies in (.+)", "located-in"),
    (r"(.+?) is in (.+)", "located-in"),
    (r"(.+?) flows through (.+)", "flows-through"),
    (r"(.+?) runs through (.+)", "flows-through"),
    (r"(.+?) is the capital of (.+)", "capital-of"),
    (r"(.+?) is capital city of (.+)", "capital-of"),
    (r"(.+?) borders (.+)", "borders"),
    (r"(.+?) is adjacent to (.+)", "borders"),
    (r"(.+?) is manufactured by (.+)", "manufactured-by"),
    (r"(.+?) is made by (.+)", "manufactured-by"),
    (r"(.+?) is a (.+?) product", "manufactured-by"),
    (r"(.+?) is made of (.+)", "made-of"),
    (r"(.+?) uses (.+)", "made-of"),
    (r"(.+?) has feature (.+)", "has-feature"),
    (r"(.+?) includes (.+)", "has-feature"),
    (r"(.+?) belongs to (.+?) category", "belongs-to"),
    (r"(.+?) is a (.+)", "belongs-to"),
]


@dataclass
class MockLLM(LLMBackend):
    """Deterministic mock with partial knowledge + pattern-based relation gen.

    Parameters
    ----------
    knowledge:
        Simulated pretraining knowledge. Types for unknown terms fall back to
        embedding-ish heuristic (string shape) + type lexicon overlap.
    noise:
        Probability of injecting a wrong type/relation (simulates LLM error).
    closed_vocab_only:
        If True, mimics the pro-team baseline: only returns is-a / taxonomy
        relations, never non-taxonomic ones. Default False.
    """

    knowledge: OntologyKnowledge = field(default_factory=OntologyKnowledge)
    noise: float = 0.05
    closed_vocab_only: bool = False
    seed: int = 0
    _rng_state: int = 0

    def __post_init__(self) -> None:
        self._rng_state = self.seed

    def _rand(self) -> float:
        # simple LCG for determinism without importing random state coupling
        self._rng_state = (1103515245 * self._rng_state + 12345) % (2**31)
        return self._rng_state / (2**31)

    def suggest_type(self, term: str, types: list[str], context: str | None = None) -> tuple[str, float]:
        if term in self.knowledge.term_to_type:
            t = self.knowledge.term_to_type[term]
            conf = 0.92 - self.noise * self._rand()
            return t, conf
        # lexicon overlap
        best, best_score = "Entity", 0.0
        for ty, lex in self.knowledge.type_lexicon.items():
            for w in lex:
                score = _token_overlap(term, w)
                if score > best_score:
                    best, best_score = ty, score
        if best_score >= 0.3 and best in types:
            return best, 0.55 + 0.3 * best_score
        # shape heuristics
        shape = _shape_guess(term)
        if shape and shape in types:
            return shape, 0.45
        if types:
            return types[0], 0.2
        return "Entity", 0.1

    def suggest_taxonomy(
        self, types: list[str], term_types: dict[str, str]
    ) -> list[tuple[str, str, float]]:
        out: list[tuple[str, str, float]] = []
        for child, parents in self.knowledge.parents.items():
            if child not in types and types:
                continue
            for p in parents:
                out.append((child, p, 0.9 - 0.05 * self._rand()))
        # heuristic: TermType -> plural/parent-ish based on shared prefix
        for t in types:
            if t in self.knowledge.parents:
                continue
            for other in types:
                if t == other:
                    continue
                if other.lower().startswith(t.lower()[:4]) and len(other) > len(t):
                    out.append((t, other, 0.35))
                    break
        # known gold-ish type parents from knowledge type lexicon inheritance
        return out

    _STOP = frozenset(
        {
            "this", "the", "and", "for", "with", "from", "that", "several",
            "evaluation", "about", "within", "when", "using", "are", "was",
            "were", "been", "have", "has", "had", "not", "but", "all", "any",
            "can", "will", "also", "into", "than", "then", "them", "they",
            "their", "there", "here", "over", "such", "only", "other", "more",
            "some", "very", "during", "before", "after", "above", "below",
            "between", "under", "again", "further", "once", "because", "while",
            "document", "ontology", "learning", "domain", "text", "entity",
            "entities", "evidence", "continues", "emerge", "experts", "classify",
            "practitioners", "often", "study", "carefully", "known", "appears",
            "frequently", "regarding", "observations", "reported", "important",
            "role", "plays", "systems", "system", "graph", "similarity", "semantic",
            "firms", "firm", "produces", "builds", "clinicians", "give", "managing",
            "prescribed", "result", "results", "observe", "following", "component",
            "belongs", "structurally", "found", "hosts", "provides", "offered",
            "incorporates", "constituent", "capital", "serves", "admin", "center",
            "runs", "across", "traversed", "adjacent", "hydro", "traverses",
            "molecularly", "binds", "bind", "used", "treat", "therapy", "leads",
            "part", "located", "lies", "flows", "through", "manufactured", "made",
            "product", "feature", "includes", "category", "interacts", "borders",
            "is", "a", "an", "of", "in", "to", "on", "by", "as", "at", "or",
            "it", "its", "be", "do", "does", "if", "so", "no", "yes", "new",
            "one", "two", "three", "we", "you", "your", "our", "us", "me", "my",
        }
    )

    def suggest_terms(self, corpus: list[str], known: set[str] | None = None) -> list[str]:
        known = known or set()
        found: set[str] = set(known)
        lexicon: set[str] = set()
        for words in self.knowledge.type_lexicon.values():
            lexicon.update(words)
        lexicon.update(self.knowledge.term_to_type.keys())
        for line in corpus:
            for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]{1,40}", line):
                tok = m.group(0).lower()
                if tok in self._STOP:
                    continue
                if len(tok) < 3:
                    continue
                # prefer lexicon / known terms; allow novel tokens that look term-like
                if tok in lexicon or tok in known or "_" in tok or tok in self.knowledge.term_to_type:
                    found.add(tok)
                elif tok.isalpha() and len(tok) >= 5 and line.lower().find(tok) >= 0:
                    # keep longer alpha tokens that appear as relation endpoints
                    # only if they sit next to a mined relational cue
                    if any(cue in line.lower() for cue in (
                        "treat", "cause", "part of", "located", "flows", "capital",
                        "borders", "manufactured", "made of", "feature", "belongs",
                        "interacts", "binds", "component", "prescribed", "produces",
                    )):
                        found.add(tok)
        return sorted(found)

    def suggest_relations(
        self,
        terms: list[str],
        term_types: dict[str, str],
        corpus: list[str],
        known_relations: set[tuple[str, str, str]] | None = None,
        open_vocabulary: bool = True,
    ) -> list[RelationSuggestion]:
        """Generate relation suggestions.

        open_vocabulary=True  -> mine open relation inventory from corpus + LLM gen
        open_vocabulary=False -> closed taxonomy vocabulary only (is-a / type-of)
                                 which recovers ZERO non-taxonomic relations
                                 (reproduces pro-team KEY GAP).
        """
        suggestions: list[RelationSuggestion] = []
        known_relations = known_relations or set()
        term_set = set(terms)

        # (0) closed-taxonomy style output: only is-a between terms/types
        for term, ty in term_types.items():
            suggestions.append(
                RelationSuggestion(
                    head=term,
                    relation="is-a",
                    tail=ty,
                    score=0.88,
                    source="llm",
                    evidence="closed-taxonomy typing",
                )
            )
        if self.closed_vocab_only or not open_vocabulary:
            return suggestions

        # (1) KB-suggested relations (partial pretraining memory)
        for h, r, t in known_relations:
            if r in {"is-a", "type-of"}:
                continue
            suggestions.append(
                RelationSuggestion(
                    head=h, relation=r, tail=t, score=0.85, source="kb", evidence="pretraining memory"
                )
            )

        # (2) Open relation vocabulary mining from corpus patterns
        for line in corpus:
            for pattern, rel in DEFAULT_PATTERNS:
                m = re.search(pattern, line, flags=re.IGNORECASE)
                if not m:
                    continue
                h_raw, t_raw = m.group(1).strip(), m.group(2).strip()
                h = _normalize_term(h_raw, term_set)
                t = _normalize_term(t_raw, term_set)
                if not h or not t or h == t:
                    continue
                conf = 0.78 if h in term_set and t in term_set else 0.45
                suggestions.append(
                    RelationSuggestion(
                        head=h,
                        relation=rel,
                        tail=t,
                        score=conf,
                        source="pattern",
                        evidence=line.strip()[:120],
                    )
                )

        # (3) LLM open-vocab generation: invent plausible relations for typed pairs
        #     using domain-range priors from knowledge (symbolic constraints later filter).
        type_of = dict(term_types)
        dr = self.knowledge.domain_ranges or {}
        pairs = list(term_set)
        # limit combinatorial explosion
        max_pairs = min(len(pairs), 40)
        for i, h in enumerate(pairs[:max_pairs]):
            for t in pairs[:max_pairs]:
                if h == t:
                    continue
                th, tt = type_of.get(h, ""), type_of.get(t, "")
                for rel, (dom, ran) in dr.items():
                    if not _type_compatible(th, dom) or not _type_compatible(tt, ran):
                        continue
                    # if KB already has it, skip (already added)
                    if (h, rel, t) in known_relations:
                        continue
                    # probabilistic generation guided by type compatibility
                    p = 0.35 + 0.25 * _type_compat_score(th, dom) * _type_compat_score(tt, ran)
                    if self._rand() < p:
                        suggestions.append(
                            RelationSuggestion(
                                head=h,
                                relation=rel,
                                tail=t,
                                score=0.4 + 0.3 * p,
                                source="llm",
                                evidence=f"typed-pair generation ({th},{tt}) -> {rel}",
                            )
                        )

        # (4) Embedding soft-edge rescue: string-shape co-occurrence neighbors
        for line in corpus:
            toks = [_normalize_term(m.group(0), term_set) for m in re.finditer(r"[A-Za-z][A-Za-z0-9_\-]+", line)]
            toks = [x for x in toks if x and x in term_set]
            for i in range(len(toks)):
                for j in range(i + 1, len(toks)):
                    h, t = toks[i], toks[j]
                    # look up an open rel already mined on this line
                    for s in suggestions:
                        if s.source == "pattern" and {s.head, s.tail} == {h, t}:
                            if s.head == h and s.tail == t:
                                continue  # already have directed
                            # soft reverse suggestion
                            suggestions.append(
                                RelationSuggestion(
                                    head=h,
                                    relation=s.relation,
                                    tail=t,
                                    score=0.35,
                                    source="rag",
                                    evidence=f"soft-edge co-occurrence: {line[:80]}",
                                )
                            )

        # dedup best score
        best: dict[tuple[str, str, str], RelationSuggestion] = {}
        for s in suggestions:
            key = (s.head, s.relation, s.tail)
            if key not in best or s.score > best[key].score:
                best[key] = s
        return list(best.values())


def _normalize_term(raw: str, term_set: set[str]) -> str:
    tok = raw.strip().lower().replace(" ", "_").replace("-", "_")
    tok = re.sub(r"[^a-z0-9_]", "", tok)
    if tok in term_set:
        return tok
    # try singular-ish / underscore variants
    for v in (tok, tok.rstrip("s"), tok.replace("__", "_")):
        if v in term_set:
            return v
    # prefix match
    for t in term_set:
        if t.startswith(tok) or tok.startswith(t):
            return t
    return tok


def _token_overlap(a: str, b: str) -> float:
    sa = set(a.lower().replace("_", " ").split())
    sb = set(b.lower().replace("_", " ").split())
    if not sa or not sb:
        # char overlap fallback
        ca, cb = set(a.lower()), set(b.lower())
        if not ca or not cb:
            return 0.0
        return len(ca & cb) / max(1, len(ca | cb))
    return len(sa & sb) / max(1, len(sa | sb))


def _shape_guess(term: str) -> str:
    t = term.lower()
    if any(k in t for k in ("city", "ville", "burg")):
        return "City"
    if t in {"paris", "tokyo", "cairo", "london", "berlin"}:
        return "City"
    if any(k in t for k in ("disease", "itis", "osis")):
        return "Disease"
    if any(k in t for k in ("drug", "mab", "pril", "statin")):
        return "Drug"
    if any(k in t for k in ("river", "nile", "seine")):
        return "River"
    return ""


def _type_compatible(ty: str, required: str) -> bool:
    if not ty or not required:
        return False
    return ty == required or ty.lower() == required.lower()


def _type_compat_score(ty: str, required: str) -> float:
    if ty == required:
        return 1.0
    if not ty or not required:
        return 0.0
    return _token_overlap(ty, required)


def get_backend(name: str = "mock", knowledge: OntologyKnowledge | None = None, **kwargs: Any) -> LLMBackend:
    if name == "mock":
        return MockLLM(knowledge=knowledge or OntologyKnowledge(), **kwargs)
    raise ValueError(f"Unknown backend {name!r}; only 'mock' is wired in this build")
