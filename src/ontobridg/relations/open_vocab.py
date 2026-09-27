"""Open non-taxonomic relation vocabulary mining.

KEY GAP addressed (pro-team arXiv:2608.27101): closed taxonomy vocabulary
extracted ZERO non-taxonomic relations. This module:

1. Mines an *open* relation inventory from corpus surface patterns
2. Invites LLM relation generation over typed term pairs
3. Applies symbolic domain-range constraints (soft/hard)
4. Keeps associative relations (part-of, causes, located-in, treats, ...)
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from ontobridg.data.schema import OntologyDataset, RelationSample
from ontobridg.llm.mock_llm import LLMBackend, RelationSuggestion, DEFAULT_PATTERNS

# Taxonomic relations are *excluded* from the open non-tax inventory.
TAXONOMIC_RELATIONS = {"is-a", "isa", "type-of", "subclass-of", "subClassOf", "kind-of"}


@dataclass
class OpenRelationInventory:
    """Discovered open relation vocabulary + domain/range constraints."""

    relations: dict[str, Counter] = field(default_factory=dict)  # rel -> {domain: count}
    ranges: dict[str, Counter] = field(default_factory=dict)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def add(self, head_type: str, rel: str, tail_type: str, example: str = "") -> None:
        self.relations.setdefault(rel, Counter())[head_type] += 1
        self.ranges.setdefault(rel, Counter())[tail_type] += 1
        if example:
            self.examples.setdefault(rel, [])
            if example not in self.examples[rel] and len(self.examples[rel]) < 5:
                self.examples[rel].append(example)

    def labels(self, include_taxonomic: bool = False) -> list[str]:
        labs = sorted(self.relations.keys())
        if include_taxonomic:
            labs = sorted(set(labs) | TAXONOMIC_RELATIONS)
        return labs

    def non_taxonomic_labels(self) -> list[str]:
        return [r for r in self.labels() if r not in TAXONOMIC_RELATIONS]

    def domain_range(self, rel: str, top_k: int = 2) -> tuple[set[str], set[str]]:
        dom = {t for t, _ in self.relations.get(rel, Counter()).most_common(top_k)}
        ran = {t for t, _ in self.ranges.get(rel, Counter()).most_common(top_k)}
        return dom, ran

    def to_domain_ranges(self) -> dict[str, tuple[str, str]]:
        out: dict[str, tuple[str, str]] = {}
        for rel in self.non_taxonomic_labels():
            dom, ran = self.domain_range(rel, top_k=1)
            if dom and ran:
                out[rel] = (next(iter(dom)), next(iter(ran)))
        return out

    def merge_gold_constraints(self, gold_dr: dict[str, tuple[str, str]]) -> None:
        """Seed inventory with known symbolic constraints (not gold *instances*)."""
        for rel, (d, r) in gold_dr.items():
            self.relations.setdefault(rel, Counter())[d] += 3
            self.ranges.setdefault(rel, Counter())[r] += 3


@dataclass
class OpenVocabRelationMiner:
    """End-to-end open-vocabulary non-taxonomic relation extractor."""

    backend: LLMBackend
    min_score: float = 0.35
    use_symbolic_constraints: bool = True
    allow_unknown_relations: bool = True
    known_domain_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    # learned constraints filled by fit()
    inventory: OpenRelationInventory = field(default_factory=OpenRelationInventory)

    def mine_inventory_from_corpus(
        self, corpus: list[str], term_types: dict[str, str]
    ) -> OpenRelationInventory:
        inv = self.inventory
        if self.known_domain_ranges:
            inv.merge_gold_constraints(self.known_domain_ranges)
        term_set = set(term_types.keys())
        for line in corpus:
            for pattern, rel in DEFAULT_PATTERNS:
                m = re.search(pattern, line, flags=re.IGNORECASE)
                if not m:
                    continue
                h = _norm(m.group(1), term_set)
                t = _norm(m.group(2), term_set)
                if not h or not t or h == t:
                    continue
                ht = term_types.get(h, "Unknown")
                tt = term_types.get(t, "Unknown")
                inv.add(ht, rel, tt, example=line.strip())
        return inv

    def fit(self, dataset: OntologyDataset, term_types: dict[str, str] | None = None) -> OpenRelationInventory:
        corpus = dataset.metadata.get("corpus", [])
        term_types = term_types or dataset.term_map()
        self.inventory = self.mine_inventory_from_corpus(corpus, term_types)
        # also observe gold *relation labels* (not instances) via domain_ranges
        if dataset.domain_ranges:
            self.inventory.merge_gold_constraints(dataset.domain_ranges)
        return self.inventory

    def extract(
        self,
        terms: list[str],
        term_types: dict[str, str],
        corpus: list[str],
        *,
        filter_taxonomic: bool = True,
    ) -> list[RelationSample]:
        """Extract non-taxonomic relations using open vocab + LLM + constraints."""
        suggestions = self.backend.suggest_relations(
            terms=terms,
            term_types=term_types,
            corpus=corpus,
            open_vocabulary=True,
        )
        kept: list[RelationSuggestion] = []
        for s in suggestions:
            if filter_taxonomic and s.relation in TAXONOMIC_RELATIONS:
                continue
            if s.score < self.min_score:
                continue
            if not self.use_symbolic_constraints:
                kept.append(s)
                continue
            if self._passes_constraints(s, term_types):
                kept.append(s)

        # dedup
        best: dict[tuple[str, str, str], RelationSuggestion] = {}
        for s in kept:
            key = (s.head, s.relation, s.tail)
            if key not in best or s.score > best[key].score:
                best[key] = s
        return [
            RelationSample(head=s.head, relation=s.relation, tail=s.tail)
            for s in sorted(best.values(), key=lambda x: -x.score)
        ]

    def _passes_constraints(self, s: RelationSuggestion, term_types: dict[str, str]) -> bool:
        """Symbolic domain-range + type-consistency filtering."""
        ht = term_types.get(s.head)
        tt = term_types.get(s.tail)
        if ht is None or tt is None:
            # soft fail: unknown endpoints — reject under hard constraints
            return False
        if s.head == s.tail:
            return False
        # constraint sources: known gold-style priors + mined inventory
        constraints: dict[str, tuple[str, str]] = dict(self.known_domain_ranges)
        constraints.update(self.inventory.to_domain_ranges())
        if s.relation in constraints:
            dom, ran = constraints[s.relation]
            # accept if endpoint type matches constraint OR constraint is a root-like type
            if _matches_type(ht, dom) and _matches_type(tt, ran):
                return True
            # soft rescue: high-score pattern/KB evidence with partially compatible types
            if s.score >= 0.75 and (_matches_type(ht, dom) or _matches_type(tt, ran)):
                return True
            return False
        # unknown relation label
        if not self.allow_unknown_relations:
            return False
        # open-vocab novel relation: require both endpoints typed and distinct domains
        # (loose symbolic gate — keeps associative novel labels alive)
        return ht != "" and tt != ""


def _norm(raw: str, term_set: set[str]) -> str:
    tok = raw.strip().lower().replace(" ", "_").replace("-", "_")
    tok = re.sub(r"[^a-z0-9_]", "", tok)
    if tok in term_set:
        return tok
    for v in (tok, tok.rstrip("s")):
        if v in term_set:
            return v
    for t in term_set:
        if t.startswith(tok) or tok.startswith(t):
            return t
    return ""


def _matches_type(ty: str, required: str) -> bool:
    if not ty or not required:
        return False
    if ty == required:
        return True
    # soft hierarchy: any type ending with required, or required is Entity-like root
    if required in {"Entity", "Place", "BiologicalEntity", "CommercialEntity", "MedicalEntity"}:
        return True
    return ty.lower().endswith(required.lower()) or required.lower().endswith(ty.lower())


def closed_vocab_baseline_extract(
    terms: list[str], term_types: dict[str, str]
) -> list[RelationSample]:
    """Reproduce pro-team failure: closed taxonomy vocab => only is-a, zero non-tax."""
    return [
        RelationSample(head=t, relation="is-a", tail=term_types.get(t, "Entity"))
        for t in terms
    ]
