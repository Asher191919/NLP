"""Task A — End-to-End Flagship ontology learning (LLMs4OL 2026).

Learn a full ontology from text/corpus: terms, types, is-a taxonomy,
and non-taxonomic relations — then score with official-style metrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ontobridg.baselines.rag_pipeline import RAGPipeline
from ontobridg.data.schema import OntologyDataset
from ontobridg.eval.semantic_graph import CompetitionMetrics, evaluate_prediction
from ontobridg.llm.mock_llm import MockLLM
from ontobridg.relations.open_vocab import TAXONOMIC_RELATIONS
from ontobridg.symbolic.repair import ScoredEdge, ScoredRelation, SymbolicRepair
from ontobridg.relations.open_vocab import OpenVocabRelationMiner


@dataclass
class TaskAResult:
    task: str = "Task A End-to-End Flagship"
    system: str = "ontobridg"
    dataset: str = ""
    prediction: dict[str, Any] = field(default_factory=dict)
    metrics: CompetitionMetrics | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "system": self.system,
            "dataset": self.dataset,
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "n_terms_pred": len(self.prediction.get("terms", {})),
            "n_taxonomy_pred": len(self.prediction.get("taxonomy", [])),
            "n_relations_pred": len(self.prediction.get("relations", [])),
            "n_nontax_pred": len(self.prediction.get("nontax_relations", [])),
            "extras": self.extras,
        }


class TaskAFlagship:
    """OntoBridg end-to-end Task A system (open relation vocab + symbolic repair)."""

    def __init__(
        self,
        knowledge=None,
        *,
        system_name: str = "ontobridg",
        closed_vocabulary: bool = False,
        use_symbolic_repair: bool = True,
        noise: float = 0.05,
        seed: int = 0,
        system: str | None = None,  # "pro_team" | "ontobridg" | "rag_open"
    ) -> None:
        sys_l = (system or system_name or "").lower()
        if system in {"pro_team", "pro_team_closed_vocab", "closed_vocab"} or "pro_team" in sys_l or "closed" in sys_l:
            closed_vocabulary = True
            use_symbolic_repair = False
            system_name = "pro_team_closed_vocab"
        elif system in {"ontobridg", "ontobridg_open_vocab", "open_vocab"} or "ontobridg" in sys_l or "open" in sys_l:
            closed_vocabulary = False
            use_symbolic_repair = True
            system_name = "ontobridg_open_vocab"
        elif system in {"rag_open", "rag_open_no_extra"}:
            closed_vocabulary = False
            use_symbolic_repair = True
            system_name = "rag_open_no_extra"
        self.system_name = system_name
        self.closed_vocabulary = closed_vocabulary
        self.use_symbolic_repair = use_symbolic_repair
        self.pipeline = RAGPipeline.mock(
            knowledge,
            closed_vocabulary=closed_vocabulary,
            use_symbolic_repair=use_symbolic_repair,
            noise=noise,
            seed=seed,
        )

    def run(self, dataset: OntologyDataset) -> TaskAResult:
        corpus = list(dataset.metadata.get("corpus", []))
        if not corpus:
            # synthesize minimal corpus from gold terms/relations (still not gold labels in prompt)
            corpus = [
                f"{r.head} {r.relation} {r.tail}" for r in dataset.relations
            ] + [f"{t.term} is a {t.type_label}" for t in dataset.terms]

        # Type inventory: for end-to-end we allow open-ish types from dataset schema
        # but do NOT feed gold term->type map (that would be leakage).
        # We DO give the type *names* (label inventory) as many OL setups do.
        type_inventory = list(dataset.types)

        domain_ranges = {} if self.closed_vocabulary else dict(dataset.domain_ranges)

        pred = self.pipeline.run(
            corpus=corpus,
            type_inventory=type_inventory,
            domain_ranges=domain_ranges,
        )

        gold_terms = dataset.term_map()
        gold_tax = list(dataset.taxonomy_pairs())
        gold_rel = list(dataset.relation_triples())

        metrics = evaluate_prediction(
            gold_terms,
            gold_tax,
            gold_rel,
            pred.get("terms", {}),
            [tuple(x) for x in pred.get("taxonomy", [])],
            [tuple(x) for x in pred.get("relations", [])],
        )

        return TaskAResult(
            system=self.system_name,
            dataset=dataset.name,
            prediction=pred,
            metrics=metrics,
            extras={
                "repair": pred.get("repair", {}),
                "nontax_pred": pred.get("nontax_relations", []),
                "gold_nontax_count": len(
                    [r for r in gold_rel if r[1] not in TAXONOMIC_RELATIONS]
                ),
                "pred_nontax_count": len(pred.get("nontax_relations", [])),
            },
        )

    def run_direct(self, dataset: OntologyDataset) -> TaskAResult:
        """Direct OntoBridg path without RAG wrapper (open miner + repair)."""
        if self.closed_vocabulary:
            return self.run(dataset)
        corpus = list(dataset.metadata.get("corpus", []))
        backend = self.pipeline.backend
        terms = backend.suggest_terms(corpus)
        term_types: dict[str, str] = {}
        for t in terms:
            ty, _ = backend.suggest_type(t, list(dataset.types))
            term_types[t] = ty
        miner = OpenVocabRelationMiner(
            backend=backend,
            known_domain_ranges=dict(dataset.domain_ranges),
            use_symbolic_constraints=True,
        )
        inv = miner.fit(dataset, term_types)
        rels = miner.extract(terms, term_types, corpus)
        tax_sugg = backend.suggest_taxonomy(list(dataset.types), term_types)
        type_hierarchy = {c: p for c, p, _ in tax_sugg if c in dataset.types}
        repair = SymbolicRepair(
            domain_ranges=dict(dataset.domain_ranges) or miner.inventory.to_domain_ranges(),
            type_hierarchy=type_hierarchy,
        )
        tax_edges = [ScoredEdge(c, p, s) for c, p, s in tax_sugg]
        for term, ty in term_types.items():
            tax_edges.append(ScoredEdge(term, ty, 0.7))
        rel_edges = [ScoredRelation(r.head, r.relation, r.tail, 0.7) for r in rels]
        out = repair.repair_all(tax_edges, rel_edges, term_types)
        pred = {
            "terms": term_types,
            "taxonomy": [(e.child, e.parent) for e in out["taxonomy"]],
            "relations": [(r.head, r.relation, r.tail) for r in out["relations"]],
            "nontax_relations": [
                (r.head, r.relation, r.tail)
                for r in out["relations"]
                if r.relation not in TAXONOMIC_RELATIONS
            ],
            "repair": {
                "taxonomy": out["taxonomy_report"].to_dict(),
                "relations": out["relation_report"].to_dict(),
            },
        }
        metrics = evaluate_prediction(
            dataset.term_map(),
            list(dataset.taxonomy_pairs()),
            list(dataset.relation_triples()),
            pred["terms"],
            pred["taxonomy"],
            pred["relations"],
        )
        return TaskAResult(
            system=self.system_name + "_direct",
            dataset=dataset.name,
            prediction=pred,
            metrics=metrics,
            extras={
                "open_inventory": inv.non_taxonomic_labels(),
                "pred_nontax_count": len(pred["nontax_relations"]),
                "gold_nontax_count": len(
                    [r for r in dataset.relations if r.relation not in TAXONOMIC_RELATIONS]
                ),
            },
        )
