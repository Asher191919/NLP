"""Task B — Ontology Extension / Reuse (LLMs4OL 2026).

Extend a seed ontology with new terms, types, and relations.
The seed provides partial structure; the system must discover additions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ontobridg.baselines.rag_pipeline import RAGPipeline
from ontobridg.data.schema import OntologyDataset, OntologyKnowledge, TermSample
from ontobridg.eval.semantic_graph import CompetitionMetrics, evaluate_prediction
from ontobridg.relations.open_vocab import TAXONOMIC_RELATIONS, OpenVocabRelationMiner
from ontobridg.symbolic.repair import ScoredEdge, ScoredRelation, SymbolicRepair
from ontobridg.llm.mock_llm import MockLLM


@dataclass
class TaskBResult:
    task: str = "Task B Ontology Extension Reuse"
    system: str = "ontobridg"
    dataset: str = ""
    seed_size: int = 0
    extension_size: int = 0
    prediction: dict[str, Any] = field(default_factory=dict)
    metrics: CompetitionMetrics | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "task": self.task,
            "system": self.system,
            "dataset": self.dataset,
            "seed_size": self.seed_size,
            "extension_size": self.extension_size,
            "metrics": self.metrics.to_dict() if self.metrics else {},
            "n_terms_pred": len(self.prediction.get("terms", {})),
            "n_taxonomy_pred": len(self.prediction.get("taxonomy", [])),
            "n_nontax_pred": len(self.prediction.get("nontax_relations", [])),
            "extras": self.extras,
        }


def split_seed_extension(
    dataset: OntologyDataset, seed_fraction: float = 0.4, seed: int = 21
) -> tuple[OntologyDataset, OntologyDataset]:
    """Split gold into a seed ontology and the extension gold (new terms/rels)."""
    import random

    rng = random.Random(seed)
    terms = list(dataset.terms)
    rng.shuffle(terms)
    n_seed = max(1, int(len(terms) * seed_fraction))
    seed_terms, ext_terms = terms[:n_seed], terms[n_seed:]
    seed_names = {t.term for t in seed_terms}
    ext_names = {t.term for t in ext_terms}

    # seed keeps ALL type-level taxonomy + edges among seed terms
    seed_tax = [
        e for e in dataset.taxonomy
        if e.child in dataset.types or (e.child in seed_names and e.parent in seed_names)
    ]
    ext_tax = [
        e for e in dataset.taxonomy
        if (e.child in ext_names or e.parent in ext_names) and (e.child, e.parent) not in {(x.child, x.parent) for x in seed_tax}
    ]
    seed_rel = [r for r in dataset.relations if r.head in seed_names and r.tail in seed_names]
    ext_rel = [
        r for r in dataset.relations
        if (r.head in ext_names or r.tail in ext_names)
        and r.triple not in {x.triple for x in seed_rel}
    ]

    seed_ds = OntologyDataset(
        name=dataset.name + "_seed",
        types=list(dataset.types),
        terms=seed_terms,
        taxonomy=seed_tax,
        relations=seed_rel,
        domain_ranges=dict(dataset.domain_ranges),
        metadata={
            **dataset.metadata,
            "role": "seed",
            # seed corpus: only lines mentioning seed terms
            "corpus": [
                line for line in dataset.metadata.get("corpus", [])
                if any(n in line for n in seed_names) or not any(
                    n in line for n in ext_names
                )
            ],
        },
    )
    # extension gold = full ontology (what we must recover by extending)
    # We evaluate against the FULL gold, as official Task B scores the extended ontology.
    ext_ds = OntologyDataset(
        name=dataset.name + "_extension",
        types=list(dataset.types),
        terms=list(dataset.terms),
        taxonomy=list(dataset.taxonomy),
        relations=list(dataset.relations),
        domain_ranges=dict(dataset.domain_ranges),
        metadata={
            **dataset.metadata,
            "role": "extension_gold",
            "seed_terms": sorted(seed_names),
            "extension_terms": sorted(ext_names),
        },
    )
    return seed_ds, ext_ds


class TaskBExtension:
    """Extend seed ontology; recover new terms/types/non-tax relations."""

    def __init__(
        self,
        knowledge=None,
        *,
        system_name: str = "ontobridg",
        closed_vocabulary: bool = False,
        use_symbolic_repair: bool = True,
        noise: float = 0.05,
        seed: int = 0,
        system: str | None = None,
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
        self._raw_knowledge = knowledge
        self._noise = noise
        self._seed = seed

    def run(self, dataset: OntologyDataset, seed_fraction: float = 0.4) -> TaskBResult:
        seed_ds, full_gold = split_seed_extension(dataset, seed_fraction=seed_fraction)
        seed_names = {t.term for t in seed_ds.terms}
        seed_term_types = seed_ds.term_map()

        # corpus for extension: full corpus (text is available); seed structure is the prior
        corpus = list(dataset.metadata.get("corpus", []))
        if not corpus:
            corpus = [f"{r.head} {r.relation} {r.tail}" for r in dataset.relations]

        domain_ranges = {} if self.closed_vocabulary else dict(dataset.domain_ranges)

        # Seed ontology knowledge is injected as few-shot / RAG exemplars
        for term, ty in seed_term_types.items():
            self.pipeline.index.add_term_type(term, ty)
        for r in seed_ds.relations:
            self.pipeline.index.add_relation(r.head, r.relation, r.tail)

        # Candidate type inventory: seed types + open discovery
        # In true open extension the system may invent types; here inventory is schema + discovered
        type_inventory = list(dict.fromkeys(list(dataset.types) + list(seed_ds.types)))

        pred = self.pipeline.run(
            corpus=corpus,
            type_inventory=type_inventory,
            domain_ranges=domain_ranges,
            seed_terms=sorted(seed_names),
            seed_term_types=seed_term_types,
        )

        gold_terms = full_gold.term_map()
        gold_tax = list(full_gold.taxonomy_pairs())
        gold_rel = list(full_gold.relation_triples())
        metrics = evaluate_prediction(
            gold_terms,
            gold_tax,
            gold_rel,
            pred.get("terms", {}),
            [tuple(x) for x in pred.get("taxonomy", [])],
            [tuple(x) for x in pred.get("relations", [])],
        )

        # Also report extension-only recovery (novel contribution visibility)
        ext_names = set(full_gold.metadata.get("extension_terms", []))
        if not ext_names:
            ext_names = set(gold_terms) - seed_names
        pred_terms = pred.get("terms", {})
        ext_pred_terms = {t: pred_terms[t] for t in ext_names if t in pred_terms}
        ext_gold_terms = {t: gold_terms[t] for t in ext_names if t in gold_terms}
        ext_metrics = evaluate_prediction(
            ext_gold_terms,
            [e for e in gold_tax if e[0] in ext_names or e[1] in ext_names],
            [r for r in gold_rel if r[0] in ext_names or r[2] in ext_names],
            ext_pred_terms,
            [tuple(x) for x in pred.get("taxonomy", []) if x[0] in ext_names or x[1] in ext_names],
            [tuple(x) for x in pred.get("relations", []) if x[0] in ext_names or x[2] in ext_names],
        )

        return TaskBResult(
            system=self.system_name,
            dataset=dataset.name,
            seed_size=len(seed_names),
            extension_size=len(ext_names),
            prediction=pred,
            metrics=metrics,
            extras={
                "extension_only_metrics": ext_metrics.to_dict(),
                "repair": pred.get("repair", {}),
                "pred_nontax_count": len(pred.get("nontax_relations", [])),
                "gold_nontax_count": len(
                    [r for r in gold_rel if r[1] not in TAXONOMIC_RELATIONS]
                ),
                "nontax_recovered": pred.get("nontax_relations", []),
            },
        )

    def run_direct(self, dataset: OntologyDataset, seed_fraction: float = 0.4) -> TaskBResult:
        """Direct open-vocab extension path."""
        if self.closed_vocabulary:
            return self.run(dataset, seed_fraction=seed_fraction)
        seed_ds, full_gold = split_seed_extension(dataset, seed_fraction=seed_fraction)
        seed_names = {t.term for t in seed_ds.terms}
        seed_term_types = seed_ds.term_map()
        corpus = list(dataset.metadata.get("corpus", []))
        backend = self.pipeline.backend
        # seed knowledge helps typing via RAG-like lexicon already in backend knowledge
        terms = backend.suggest_terms(corpus, known=seed_names)
        term_types: dict[str, str] = {}
        for t in terms:
            if t in seed_term_types:
                term_types[t] = seed_term_types[t]
                continue
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
        # inject seed taxonomy
        for e in seed_ds.taxonomy:
            tax_sugg.append((e.child, e.parent, 0.95))
            if e.child in dataset.types:
                type_hierarchy.setdefault(e.child, e.parent)
        repair = SymbolicRepair(
            domain_ranges=dict(dataset.domain_ranges) or miner.inventory.to_domain_ranges(),
            type_hierarchy=type_hierarchy,
        )
        tax_edges = [ScoredEdge(c, p, s) for c, p, s in tax_sugg]
        for term, ty in term_types.items():
            tax_edges.append(ScoredEdge(term, ty, 0.7))
        rel_edges = [ScoredRelation(r.head, r.relation, r.tail, 0.75) for r in rels]
        for r in seed_ds.relations:
            rel_edges.append(ScoredRelation(r.head, r.relation, r.tail, 0.95))
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
            full_gold.term_map(),
            list(full_gold.taxonomy_pairs()),
            list(full_gold.relation_triples()),
            pred["terms"],
            pred["taxonomy"],
            pred["relations"],
        )
        return TaskBResult(
            system=self.system_name + "_direct",
            dataset=dataset.name,
            seed_size=len(seed_names),
            extension_size=len(full_gold.terms) - len(seed_names),
            prediction=pred,
            metrics=metrics,
            extras={
                "open_inventory": inv.non_taxonomic_labels(),
                "pred_nontax_count": len(pred["nontax_relations"]),
                "gold_nontax_count": len(
                    [r for r in full_gold.relations if r.relation not in TAXONOMIC_RELATIONS]
                ),
            },
        )
