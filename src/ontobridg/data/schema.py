from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TermSample:
    term: str
    type_label: str
    context: str | None = None


@dataclass(frozen=True)
class TaxonomySample:
    child: str
    parent: str

    @property
    def pair(self) -> tuple[str, str]:
        return self.child, self.parent


@dataclass(frozen=True)
class RelationSample:
    head: str
    relation: str
    tail: str

    @property
    def triple(self) -> tuple[str, str, str]:
        return self.head, self.relation, self.tail


@dataclass
class OntologyDataset:
    name: str
    types: list[str]
    terms: list[TermSample]
    taxonomy: list[TaxonomySample]
    relations: list[RelationSample]
    domain_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def term_map(self) -> dict[str, str]:
        return {t.term: t.type_label for t in self.terms}

    def taxonomy_pairs(self) -> set[tuple[str, str]]:
        return {s.pair for s in self.taxonomy}

    def relation_triples(self) -> set[tuple[str, str, str]]:
        return {s.triple for s in self.relations}


@dataclass
class OntologyKnowledge:
    """Prior knowledge a backend may hold (simulates LLM pretraining)."""

    term_to_type: dict[str, str] = field(default_factory=dict)
    parents: dict[str, set[str]] = field(default_factory=dict)
    relations: set[tuple[str, str, str]] = field(default_factory=set)
    domain_ranges: dict[str, tuple[str, str]] = field(default_factory=dict)
    type_lexicon: dict[str, list[str]] = field(default_factory=dict)

    def parents_of(self, type_name: str) -> set[str]:
        return set(self.parents.get(type_name, set()))

    def type_for(self, term: str, default: str = "Entity") -> str:
        return self.term_to_type.get(term, default)

    def to_dataset(self, name: str = "knowledge") -> OntologyDataset:
        terms = [
            TermSample(term=t, type_label=lab)
            for t, lab in sorted(self.term_to_type.items())
        ]
        taxonomy = [
            TaxonomySample(child=c, parent=p)
            for c, parents in sorted(self.parents.items())
            for p in sorted(parents)
        ]
        relations = [
            RelationSample(head=h, relation=r, tail=t)
            for h, r, t in sorted(self.relations)
        ]
        types = sorted({lab for lab in self.term_to_type.values()} | set(self.parents.keys()))
        for parents in self.parents.values():
            types.extend(parents)
        return OntologyDataset(
            name=name,
            types=sorted(set(types)),
            terms=terms,
            taxonomy=taxonomy,
            relations=relations,
            domain_ranges=dict(self.domain_ranges),
        )
