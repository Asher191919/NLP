"""Contamination-aware evaluation protocol for ontology learning.

Novel contribution #3:
- term holdout
- paraphrase template suite
- synthetic novel ontology stress tests
- membership-style contamination scores
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable

from ontobridg.data.schema import OntologyDataset
from ontobridg.eval.semantic_graph import CompetitionMetrics, evaluate_prediction

# Paraphrase templates that re-surface gold terms in novel contexts.
PARAPHRASE_TEMPLATES = [
    "In this domain, {term} plays an important role.",
    "Practitioners often study {term} carefully.",
    "The entity known as {term} appears frequently.",
    "Regarding {term}, several observations have been reported.",
    "New evidence about {term} continues to emerge.",
    "Experts classify {term} within this ontology.",
]

RELATION_PARAPHRASES = {
    "treats": ["{h} is prescribed for {t}", "clinicians give {h} when managing {t}"],
    "causes": ["{t} can result from {h}", "one observes {t} following {h}"],
    "part-of": ["{t} includes {h} as a component", "{h} belongs structurally to {t}"],
    "located-in": ["{h} can be found in {t}", "the {t} hosts {h}"],
    "manufactured-by": ["{t} produces {h}", "the firm {t} builds {h}"],
    "has-feature": ["{h} provides {t}", "{t} is offered on {h}"],
    "made-of": ["{h} incorporates {t}", "{t} is a constituent of {h}"],
    "capital-of": ["{t} has capital {h}", "{h} serves as capital of {t}"],
    "interacts-with": ["{h} binds to {t}", "{h} and {t} interact"],
    "flows-through": ["{h} runs across {t}", "{t} is traversed by {h}"],
}


@dataclass
class ContaminationScore:
    dataset: str
    term_probe_rate: float  # fraction of gold terms the backend "knows"
    relation_probe_rate: float
    level: str  # low | medium | high

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "term_probe_rate": self.term_probe_rate,
            "relation_probe_rate": self.relation_probe_rate,
            "level": self.level,
        }


@dataclass
class StressResult:
    name: str
    metrics: dict
    notes: str = ""


@dataclass
class ContaminationReport:
    dataset: str
    contamination: ContaminationScore
    full_metrics: dict
    holdout_metrics: dict
    paraphrase_metrics: dict
    synthetic_novel_metrics: dict
    drops: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "contamination": self.contamination.to_dict(),
            "full": self.full_metrics,
            "term_holdout": self.holdout_metrics,
            "paraphrase": self.paraphrase_metrics,
            "synthetic_novel": self.synthetic_novel_metrics,
            "drops": self.drops,
        }


def membership_probe(
    known_terms: set[str],
    gold_terms: set[str],
    known_relations: set[tuple[str, str, str]],
    gold_relations: set[tuple[str, str, str]],
    dataset: str = "unknown",
) -> ContaminationScore:
    """Estimate training-data contamination via knowledge-base coverage probes."""
    t_rate = len(known_terms & gold_terms) / max(1, len(gold_terms))
    r_rate = len(known_relations & gold_relations) / max(1, len(gold_relations))
    # non-tax only for relation probe
    gold_nontax = {r for r in gold_relations if r[1] not in {"is-a", "type-of"}}
    known_nontax = known_relations & gold_nontax
    r_rate_nontax = len(known_nontax) / max(1, len(gold_nontax))
    avg = (t_rate + r_rate_nontax) / 2
    level = "high" if avg >= 0.55 else ("medium" if avg >= 0.25 else "low")
    return ContaminationScore(
        dataset=dataset,
        term_probe_rate=t_rate,
        relation_probe_rate=r_rate_nontax,
        level=level,
    )


def build_paraphrase_corpus(
    dataset: OntologyDataset,
    *,
    seed: int = 7,
) -> list[str]:
    """Rewrite gold evidence in novel surface forms (anti-memorization stress)."""
    rng = random.Random(seed)
    corpus: list[str] = []
    for t in dataset.terms:
        tmpl = PARAPHRASE_TEMPLATES[rng.randrange(len(PARAPHRASE_TEMPLATES))]
        corpus.append(tmpl.format(term=t.term))
    for r in dataset.relations:
        templates = RELATION_PARAPHRASES.get(r.relation)
        if not templates:
            templates = [f"{r.head} {r.relation} {r.tail}"]
        tmpl = templates[rng.randrange(len(templates))]
        corpus.append(tmpl.format(h=r.head, t=r.tail))
    rng.shuffle(corpus)
    return corpus


def build_holdout_dataset(
    dataset: OntologyDataset, fraction: float = 0.25, seed: int = 11
) -> tuple[OntologyDataset, OntologyDataset]:
    """Split terms into (train_view, heldout_view) for contamination tests."""
    rng = random.Random(seed)
    terms = list(dataset.terms)
    rng.shuffle(terms)
    n_hold = max(1, int(len(terms) * fraction))
    hold, keep = terms[:n_hold], terms[n_hold:]
    keep_names = {t.term for t in keep}
    hold_names = {t.term for t in hold}

    def filter_ds(name: str, term_list, names: set[str]) -> OntologyDataset:
        return OntologyDataset(
            name=name,
            types=dataset.types,
            terms=term_list,
            taxonomy=[
                e for e in dataset.taxonomy
                if e.child in names or e.parent in names or e.child in dataset.types
            ],
            relations=[
                r for r in dataset.relations if r.head in names and r.tail in names
            ],
            domain_ranges=dict(dataset.domain_ranges),
            metadata={
                **dataset.metadata,
                "corpus": [
                    line for line in dataset.metadata.get("corpus", [])
                    if not any(n in line for n in (names if name.endswith("heldout") else hold_names))
                ]
                if name.endswith("heldout")
                else dataset.metadata.get("corpus", []),
            },
        )

    train_view = filter_ds(dataset.name + "_trainview", keep, keep_names)
    hold_view = filter_ds(dataset.name + "_heldout", hold, hold_names)
    # full gold remains available via original dataset for eval on holdout terms
    return train_view, hold_view


def synthetic_novel_dataset(base: OntologyDataset, *, seed: int = 99) -> OntologyDataset:
    """Create a *novel* ontology (neologism terms) as a contamination stress test.

    Surface forms are rewritten; gold structure preserved. A model relying on
    memorized surface strings should drop sharply on this set.
    """
    rng = random.Random(seed)
    mapping = {}
    for i, t in enumerate(base.terms):
        mapping[t.term] = f"novel_entity_{i:03d}_{rng.choice(['zeta','kappa','sigma','omega'])}"
    rel_map = {
        "treats": "therapeutically_addresses",
        "causes": "induces_state",
        "part-of": "structural_component_of",
        "located-in": "anatomically_sited_in",
        "manufactured-by": "produced_by_firm",
        "made-of": "fabricated_from",
        "has-feature": "equipped_with",
        "belongs-to": "categorized_under",
        "capital-of": "admin_center_of",
        "flows-through": "hydro_traverses",
        "interacts-with": "molecularly_binds",
        "borders": "geographically_adjacent_to",
    }
    terms = [
        type(t)(term=mapping[t.term], type_label=t.type_label, context=t.context)
        for t in base.terms
    ]
    relations = [
        type(r)(
            head=mapping.get(r.head, r.head),
            relation=rel_map.get(r.relation, "assoc_" + r.relation),
            tail=mapping.get(r.tail, r.tail),
        )
        for r in base.relations
    ]
    domain_ranges = {
        rel_map.get(k, "assoc_" + k): v for k, v in base.domain_ranges.items()
    }
    corpus = []
    for line in base.metadata.get("corpus", []):
        new_line = line
        for old, new in mapping.items():
            new_line = re.sub(rf"\b{re.escape(old)}\b", new, new_line)
        for old, new in rel_map.items():
            new_line = new_line.replace(old, new)
        corpus.append(new_line)
    return OntologyDataset(
        name=base.name + "_novel",
        types=base.types,
        terms=terms,
        taxonomy=list(base.taxonomy),
        relations=relations,
        domain_ranges=domain_ranges,
        metadata={"domain": base.metadata.get("domain", "novel"), "corpus": corpus, "novel": True},
    )


def scrub_knowledge_for_holdout(
    knowledge,
    heldout_terms: set[str],
    heldout_relations: set[tuple[str, str, str]] | None = None,
):
    """Copy OntologyKnowledge with holdout terms/relations removed (true blindness)."""
    from ontobridg.data.schema import OntologyKnowledge

    held = {t.lower() for t in heldout_terms}
    held_rel = set(heldout_relations or set())
    out = OntologyKnowledge()
    out.term_to_type = {
        t: ty for t, ty in getattr(knowledge, "term_to_type", {}).items() if t.lower() not in held
    }
    out.parents = {
        c: {p for p in parents if p.lower() not in held}
        for c, parents in getattr(knowledge, "parents", {}).items()
        if c.lower() not in held
    }
    out.parents = {c: ps for c, ps in out.parents.items() if ps}
    out.relations = {
        (h, r, t)
        for h, r, t in getattr(knowledge, "relations", set())
        if h.lower() not in held and t.lower() not in held and (h, r, t) not in held_rel
    }
    out.domain_ranges = dict(getattr(knowledge, "domain_ranges", {}) or {})
    out.type_lexicon = {
        k: [x for x in v if str(x).lower() not in held]
        for k, v in getattr(knowledge, "type_lexicon", {}).items()
    }
    return out


def filter_corpus_for_holdout(corpus: list[str], heldout_terms: set[str]) -> list[str]:
    """Drop corpus lines that mention any held-out term."""
    held = [t.lower() for t in heldout_terms if t]
    if not held:
        return list(corpus)
    out = []
    for line in corpus:
        low = str(line).lower()
        if any(h in low for h in held):
            continue
        out.append(line)
    return out


def _official_gs(m: dict) -> float:
    official = m.get("official_graph_similarity") or {}
    if isinstance(official, dict) and "graph_similarity" in official:
        return float(official["graph_similarity"])
    # older payload shape
    proxy = m.get("semantic_graph_similarity") or {}
    if isinstance(proxy, dict):
        return float(proxy.get("sgs", 0.0))
    return float(m.get("combo_score", 0.0))


def run_contamination_protocol(
    dataset: OntologyDataset,
    predict_fn: Callable[[OntologyDataset, list[str]], dict],
    *,
    known_term_types: dict[str, str] | None = None,
    known_relations: set[tuple[str, str, str]] | None = None,
    holdout_fraction: float = 0.25,
    seed: int = 7,
    knowledge=None,
) -> ContaminationReport:
    """Contamination-aware evaluation with true holdout blindness.

    Holdout arm:
      - evaluates only on held-out terms
      - predicts from corpus lines that do NOT mention held-out surfaces
      - optionally uses knowledge scrubbed of held-out items (`knowledge=`)
    """
    known_term_types = known_term_types or {}
    known_relations = known_relations or set()
    gold_terms = dataset.term_map()
    gold_tax = list(dataset.taxonomy_pairs())
    gold_rel = list(dataset.relation_triples())

    contam = membership_probe(
        set(known_term_types), set(gold_terms), known_relations, set(gold_rel),
        dataset=dataset.name,
    )

    full_pred = predict_fn(dataset, list(dataset.metadata.get("corpus", [])))
    full_m = evaluate_prediction(
        gold_terms, gold_tax, gold_rel,
        full_pred.get("terms", {}),
        [tuple(x) for x in full_pred.get("taxonomy", [])],
        [tuple(x) for x in full_pred.get("relations", [])],
    ).to_dict()

    # holdout with blindness
    _, hold_view = build_holdout_dataset(dataset, fraction=holdout_fraction, seed=seed)
    hold_names = {t.term for t in hold_view.terms}
    blind_corpus = filter_corpus_for_holdout(
        list(dataset.metadata.get("corpus", [])), hold_names
    )
    # Callers may pass a knowledge-aware predictor that inspects dataset.metadata["scrubbed_knowledge"]
    hold_ds = OntologyDataset(
        name=dataset.name + "_holdout_blind",
        types=dataset.types,
        terms=list(hold_view.terms),
        taxonomy=list(dataset.taxonomy),
        relations=list(dataset.relations),
        domain_ranges=dict(dataset.domain_ranges),
        metadata={
            **dataset.metadata,
            "corpus": blind_corpus,
            "heldout_terms": sorted(hold_names),
            "blindness": True,
            **(
                {"scrubbed_knowledge": scrub_knowledge_for_holdout(knowledge, hold_names)}
                if knowledge is not None
                else {}
            ),
        },
    )
    hold_pred = predict_fn(hold_ds, blind_corpus)
    hold_terms = {t.term: t.type_label for t in hold_view.terms}
    pred_terms_hold = {
        t: hold_pred.get("terms", {}).get(t)
        for t in hold_terms
        if hold_pred.get("terms", {}).get(t)
    }
    gold_tax_hold = [e for e in gold_tax if e[0] in hold_names or e[1] in hold_names]
    pred_tax_hold = [
        tuple(x) for x in hold_pred.get("taxonomy", [])
        if x[0] in hold_names or x[1] in hold_names
    ]
    gold_rel_hold = [r for r in gold_rel if r[0] in hold_names or r[2] in hold_names]
    pred_rel_hold = [
        tuple(x) for x in hold_pred.get("relations", [])
        if x[0] in hold_names or x[2] in hold_names
    ]
    hold_m = evaluate_prediction(
        hold_terms, gold_tax_hold, gold_rel_hold,
        pred_terms_hold, pred_tax_hold, pred_rel_hold,
    ).to_dict()

    # paraphrase stress
    para_ds = OntologyDataset(
        name=dataset.name + "_paraphrase",
        types=dataset.types,
        terms=dataset.terms,
        taxonomy=dataset.taxonomy,
        relations=dataset.relations,
        domain_ranges=dataset.domain_ranges,
        metadata={**dataset.metadata, "corpus": build_paraphrase_corpus(dataset, seed=seed)},
    )
    para_pred = predict_fn(para_ds, list(para_ds.metadata["corpus"]))
    para_m = evaluate_prediction(
        gold_terms, gold_tax, gold_rel,
        para_pred.get("terms", {}),
        [tuple(x) for x in para_pred.get("taxonomy", [])],
        [tuple(x) for x in para_pred.get("relations", [])],
    ).to_dict()

    # synthetic novel ontology stress
    novel_ds = synthetic_novel_dataset(dataset, seed=seed + 50)
    novel_pred = predict_fn(novel_ds, list(novel_ds.metadata.get("corpus", [])))
    novel_m = evaluate_prediction(
        novel_ds.term_map(),
        list(novel_ds.taxonomy_pairs()),
        list(novel_ds.relation_triples()),
        novel_pred.get("terms", {}),
        [tuple(x) for x in novel_pred.get("taxonomy", [])],
        [tuple(x) for x in novel_pred.get("relations", [])],
    ).to_dict()

    def _combo(m: dict) -> float:
        official = _official_gs(m)
        tt = (m.get("term_typing") or {}).get("f1", 0.0)
        tax = (m.get("taxonomy") or {}).get("f1", 0.0)
        nrel = (m.get("relations_nontaxonomic") or {}).get("f1", 0.0)
        # pro-team style composite using OFFICIAL graph similarity
        return (official + tt + tax) / 3.0 if official or tt or tax else 0.25 * (
            tt + tax + nrel + official
        )

    def _nontax_f1(m: dict) -> float:
        return float((m.get("relations_nontaxonomic") or {}).get("f1", 0.0))

    drops = {
        "holdout_combo_drop": _combo(full_m) - _combo(hold_m),
        "paraphrase_combo_drop": _combo(full_m) - _combo(para_m),
        "novel_combo_drop": _combo(full_m) - _combo(novel_m),
        "holdout_nontax_f1_drop": _nontax_f1(full_m) - _nontax_f1(hold_m),
        "paraphrase_nontax_f1_drop": _nontax_f1(full_m) - _nontax_f1(para_m),
        "holdout_protocol": "corpus-blind + knowledge-scrub(when provided); positive drop => memorization",
        "metric_note": "drops computed with official Graph Similarity composite",
    }

    return ContaminationReport(
        dataset=dataset.name,
        contamination=contam,
        full_metrics=full_m,
        holdout_metrics=hold_m,
        paraphrase_metrics=para_m,
        synthetic_novel_metrics=novel_m,
        drops=drops,
    )
