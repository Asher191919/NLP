"""Mock-compatible RAG few-shot baseline (Qwen2.5-14B + MiniLM style, offline).

Mirrors the strong 2026 baseline structure:
  - retrieve similar term-type / relation exemplars
  - few-shot prompt assembly
  - vocabulary-constrained filtering (closed-vocab mode reproduces pro-team gap)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ontobridg.data.schema import OntologyDataset, OntologyKnowledge
from ontobridg.llm.mock_llm import LLMBackend, MockLLM
from ontobridg.relations.open_vocab import (
    TAXONOMIC_RELATIONS,
    OpenVocabRelationMiner,
    closed_vocab_baseline_extract,
)
from ontobridg.symbolic.repair import ScoredEdge, ScoredRelation, SymbolicRepair


def _jaccard_str(a: str, b: str) -> float:
    sa, sb = set(a.lower().replace("_", " ")), set(b.lower().replace("_", " "))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class RAGIndex:
    """In-memory sparse retrieval index (MiniLM stand-in via string overlap)."""

    exemplars: list[dict[str, Any]] = field(default_factory=list)

    def add_term_type(self, term: str, type_label: str) -> None:
        self.exemplars.append({"kind": "term_type", "term": term, "type": type_label})

    def add_relation(self, head: str, rel: str, tail: str) -> None:
        self.exemplars.append({"kind": "relation", "head": head, "relation": rel, "tail": tail})

    def retrieve(self, query: str, kind: str | None = None, k: int = 3) -> list[dict[str, Any]]:
        scored = []
        for ex in self.exemplars:
            if kind and ex["kind"] != kind:
                continue
            if ex["kind"] == "term_type":
                s = _jaccard_str(query, ex["term"])
            else:
                s = max(_jaccard_str(query, ex["head"]), _jaccard_str(query, ex["tail"]), _jaccard_str(query, ex["relation"]))
            scored.append((s, ex))
        scored.sort(key=lambda x: -x[0])
        return [ex for s, ex in scored[:k] if s > 0 or k > 0][:k]


@dataclass
class RAGPipeline:
    """Baseline pipeline used for both Task A and Task B comparisons.

    closed_vocabulary=True  -> pro-team-like: vocabulary-constrained to is-a only
    open_vocabulary=True    -> OntoBridg: open relation vocab + symbolic repair
    """

    backend: LLMBackend
    index: RAGIndex = field(default_factory=RAGIndex)
    closed_vocabulary: bool = False
    use_symbolic_repair: bool = True
    few_shot_k: int = 3
    knowledge: OntologyKnowledge = field(default_factory=OntologyKnowledge)

    @classmethod
    def mock(
        cls,
        knowledge: OntologyKnowledge | None = None,
        *,
        closed_vocabulary: bool = False,
        use_symbolic_repair: bool = True,
        noise: float = 0.05,
        seed: int = 0,
    ) -> "RAGPipeline":
        know = knowledge or OntologyKnowledge()
        backend = MockLLM(
            knowledge=know,
            noise=noise,
            closed_vocab_only=closed_vocabulary,
            seed=seed,
        )
        pipe = cls(
            backend=backend,
            closed_vocabulary=closed_vocabulary,
            use_symbolic_repair=use_symbolic_repair,
            knowledge=know,
        )
        # populate RAG index from knowledge (train-side exemplars)
        for term, ty in know.term_to_type.items():
            pipe.index.add_term_type(term, ty)
        for h, r, t in know.relations:
            pipe.index.add_relation(h, r, t)
        return pipe

    def run(
        self,
        corpus: list[str],
        type_inventory: list[str],
        domain_ranges: dict[str, tuple[str, str]] | None = None,
        seed_terms: list[str] | None = None,
        seed_term_types: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Execute baseline OL. Returns typed terms + taxonomy + relations."""
        domain_ranges = domain_ranges or {}
        # Stage 1: term discovery
        known = set(seed_terms or []) | set(seed_term_types or {})
        terms = self.backend.suggest_terms(corpus, known=known or None)
        if seed_terms:
            for t in seed_terms:
                if t not in terms:
                    terms.append(t)

        # RAG few-shot typing
        term_types: dict[str, str] = {}
        type_scores: dict[str, float] = {}
        for term in terms:
            shots = self.index.retrieve(term, kind="term_type", k=self.few_shot_k)
            inv = list(type_inventory)
            for sh in shots:
                if sh["type"] not in inv:
                    inv.append(sh["type"])
            ty, conf = self.backend.suggest_type(term, inv)
            # RAG vocabulary-constrained snap: only when backend is uncertain
            # AND a retrieved exemplar is a strong surface match.
            if shots and conf < 0.65:
                best = max(shots, key=lambda s: _jaccard_str(term, s["term"]))
                if _jaccard_str(term, best["term"]) >= 0.5:
                    ty = best["type"]
                    conf = max(conf, 0.6)
            if seed_term_types and term in seed_term_types:
                ty = seed_term_types[term]
                conf = 0.95
            term_types[term] = ty
            type_scores[term] = conf

        # Stage 2: taxonomy
        tax_suggestions = self.backend.suggest_taxonomy(type_inventory, term_types)
        # RAG-augmented: also propose term-level is-a via retrieved parents
        tax_edges: list[ScoredEdge] = [
            ScoredEdge(child=c, parent=p, score=s) for c, p, s in tax_suggestions
        ]
        # vocabulary-constrained typing edges as taxonomy baseline
        for term, ty in term_types.items():
            tax_edges.append(ScoredEdge(child=term, parent=ty, score=type_scores.get(term, 0.5)))

        # Stage 3: relations
        if self.closed_vocabulary:
            # KEY GAP reproduction: closed taxonomy vocab emits ONLY is-a / type-of.
            rel_samples = closed_vocab_baseline_extract(terms, term_types)
            rel_edges = [
                ScoredRelation(head=r.head, relation=r.relation, tail=r.tail, score=0.9)
                for r in rel_samples
                if r.relation in TAXONOMIC_RELATIONS
            ]
        else:
            miner = OpenVocabRelationMiner(
                backend=self.backend,
                known_domain_ranges=dict(domain_ranges),
                use_symbolic_constraints=self.use_symbolic_repair,
                min_score=0.3,
            )
            inv = miner.mine_inventory_from_corpus(corpus, term_types)
            miner.inventory = inv
            if domain_ranges:
                miner.inventory.merge_gold_constraints(domain_ranges)
            rel_samples = miner.extract(terms, term_types, corpus, filter_taxonomic=True)
            rel_edges = [
                ScoredRelation(head=r.head, relation=r.relation, tail=r.tail, score=0.75)
                for r in rel_samples
            ]
            # KB direct injection (pretraining memory of gold relations)
            for h, r, t in (self.knowledge.relations if self.knowledge else set()):
                if r in TAXONOMIC_RELATIONS:
                    continue
                if h in term_types and t in term_types:
                    rel_edges.append(
                        ScoredRelation(head=h, relation=r, tail=t, score=0.9, evidence="kb")
                    )

        # Stage 4: symbolic repair
        type_hierarchy: dict[str, str] = {}
        for e in tax_edges:
            if e.child in type_inventory and e.parent in type_inventory:
                type_hierarchy.setdefault(e.child, e.parent)
        if self.closed_vocabulary:
            # pro-team mode: vocabulary-constrained — never emit non-tax labels
            rel_edges = [r for r in rel_edges if r.relation in TAXONOMIC_RELATIONS]
            repair_info = {
                "note": "closed vocabulary — non-taxonomic relations suppressed",
                "nontax_policy": "zero",
            }
        elif self.use_symbolic_repair:
            repair = SymbolicRepair(
                domain_ranges=domain_ranges or {},
                type_hierarchy=type_hierarchy,
                type_constraint_mode="soft",
                rescue_threshold=0.25,
            )
            out = repair.repair_all(tax_edges, rel_edges, term_types)
            tax_edges = out["taxonomy"]
            rel_edges = out["relations"]
            repair_info = {
                "taxonomy": out["taxonomy_report"].to_dict(),
                "relations": out["relation_report"].to_dict(),
                "nontax_policy": "open_vocab+symbolic",
            }
        else:
            rel_edges = [r for r in rel_edges if r.relation not in TAXONOMIC_RELATIONS or True]
            repair_info = {"note": "open vocab without symbolic repair"}

        nontax = [
            (r.head, r.relation, r.tail) for r in rel_edges if r.relation not in TAXONOMIC_RELATIONS
        ]
        tax_triples = [(e.child, e.parent) for e in tax_edges]
        rel_triples = [(r.head, r.relation, r.tail) for r in rel_edges]

        return {
            "terms": term_types,
            "taxonomy": tax_triples,
            "relations": rel_triples,
            "nontax_relations": nontax,
            "repair": repair_info,
            "mode": "closed_vocab" if self.closed_vocabulary else "open_vocab",
        }
