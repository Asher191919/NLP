from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ontobridg.data.schema import (
    OntologyDataset,
    OntologyKnowledge,
    RelationSample,
    TaxonomySample,
    TermSample,
)
from ontobridg.eval.metrics import classification_report, taxonomy_metrics

PARAPHRASE_TEMPLATES: list[str] = [
    "the entity known as {term}",
    "{term} (ontology term)",
    "concept: {term}",
    "{term}",
    "{term}! ",
    "  {term}  ",
    "{term}/instance",
]


@dataclass
class ContaminationReport:
    seen_f1: float
    heldout_f1: float
    paraphrase_f1: float
    novel_f1: float | None
    n_seen: int
    n_heldout: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "seen_f1": self.seen_f1,
            "heldout_f1": self.heldout_f1,
            "paraphrase_f1": self.paraphrase_f1,
            "novel_f1": self.novel_f1,
            "n_seen": self.n_seen,
            "n_heldout": self.n_heldout,
            "notes": self.notes,
        }


def holdout_terms(
    dataset: OntologyDataset,
    *,
    holdout_frac: float = 0.3,
    seed: int = 0,
    stratify: bool = True,
) -> tuple[list[TermSample], list[TermSample]]:
    """Split terms into (seen, heldout). Heldout terms should not appear in LLM knowledge."""
    rng = random.Random(seed)
    by_label: dict[str, list[TermSample]] = {}
    for sample in dataset.terms:
        by_label.setdefault(sample.type_label, []).append(sample)

    seen: list[TermSample] = []
    heldout: list[TermSample] = []
    if stratify:
        for _, items in sorted(by_label.items()):
            items = list(items)
            rng.shuffle(items)
            k = max(1, int(round(len(items) * holdout_frac))) if items else 0
            if len(items) == 1:
                seen.extend(items)
                continue
            k = min(k, len(items) - 1)
            heldout.extend(items[:k])
            seen.extend(items[k:])
    else:
        items = list(dataset.terms)
        rng.shuffle(items)
        k = int(round(len(items) * holdout_frac))
        heldout = items[:k]
        seen = items[k:]
    return seen, heldout


def paraphrase_term(term: str, *, template: str | None = None, seed: int | None = None) -> str:
    rng = random.Random(seed)
    tmpl = template or rng.choice(PARAPHRASE_TEMPLATES)
    return tmpl.format(term=term)


def paraphrase_terms(
    terms: Iterable[TermSample],
    *,
    seed: int = 1,
) -> list[TermSample]:
    rng = random.Random(seed)
    out: list[TermSample] = []
    for sample in terms:
        tmpl = rng.choice(PARAPHRASE_TEMPLATES)
        out.append(
            TermSample(
                term=paraphrase_term(sample.term, template=tmpl),
                type_label=sample.type_label,
                context=sample.context,
            )
        )
    return out


def strip_paraphrase(text: str) -> str:
    t = text.strip()
    for prefix in ("the entity known as", "concept:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    t = t.replace("(ontology term)", "").strip()
    t = t.removesuffix("/instance").strip()
    t = t.rstrip("!").strip()
    return t


def build_novel_synthetic_ontology(
    *,
    name: str = "novel_synthetic",
    n_types: int = 8,
    n_terms: int = 40,
    seed: int = 99,
) -> OntologyDataset:
    """Generate a synthetic ontology with invented labels (low contamination risk)."""
    rng = random.Random(seed)
    prefixes = ["Zor", "Vex", "Quil", "Nim", "Brel", "Karn", "Thal", "Mivo", "Pex", "Rul", "Sorn", "Yel"]
    middles = ["an", "eth", "or", "ul", "ish", "ar", "en", "om"]
    suffixes = ["type", "kind", "class", "node", "unit", "set"]

    def coin(n: int = 2) -> str:
        parts = [rng.choice(prefixes) + rng.choice(middles) for _ in range(n)]
        return "".join(parts).capitalize()

    root = "NovelRoot"
    types = [root]
    while len(types) < n_types:
        candidate = coin()
        if candidate not in types:
            types.append(candidate)

    taxonomy: list[TaxonomySample] = []
    for t in types:
        if t == root:
            continue
        parent = rng.choice([x for x in types if x != t] + [root])
        if parent == t:
            parent = root
        taxonomy.append(TaxonomySample(child=t, parent=parent))

    terms: list[TermSample] = []
    used_terms: set[str] = set()
    while len(terms) < n_terms:
        term = coin(rng.randint(1, 2)) + str(rng.randint(1, 99))
        if term in used_terms:
            continue
        used_terms.add(term)
        label = rng.choice([t for t in types if t != root] or types)
        terms.append(TermSample(term=term, type_label=label))

    relations: list[RelationSample] = []
    rel_names = ["linksTo", "bindsWith", "emitsFrom", "storedIn"]
    term_list = [t.term for t in terms]
    for _ in range(max(5, n_terms // 3)):
        h, t = rng.sample(term_list, 2)
        relations.append(RelationSample(head=h, relation=rng.choice(rel_names), tail=t))

    return OntologyDataset(
        name=name,
        types=sorted(set(types)),
        terms=terms,
        taxonomy=taxonomy,
        relations=relations,
        domain_ranges={r: ("NovelRoot", "NovelRoot") for r in rel_names},
        metadata={"source": "novel_synthetic", "seed": seed},
    )


def knowledge_excluding_terms(
    base: OntologyKnowledge,
    excluded_terms: set[str],
) -> OntologyKnowledge:
    return OntologyKnowledge(
        term_to_type={t: lab for t, lab in base.term_to_type.items() if t not in excluded_terms},
        parents={k: set(v) for k, v in base.parents.items()},
        relations=set(base.relations),
        domain_ranges=dict(base.domain_ranges),
        type_lexicon={k: list(v) for k, v in base.type_lexicon.items()},
    )


def evaluate_contamination_slices(
    predict_fn,
    dataset: OntologyDataset,
    *,
    holdout_frac: float = 0.3,
    seed: int = 0,
    novel_dataset: OntologyDataset | None = None,
    novel_predict_fn=None,
    novel_gold_types: Mapping[str, str] | None = None,
) -> ContaminationReport:
    """
    predict_fn(term) -> predicted type string.
    Optional novel_predict_fn/evaluate on invented ontology.
    """
    seen, heldout = holdout_terms(dataset, holdout_frac=holdout_frac, seed=seed)

    def score(samples: list[TermSample], terms_override: list[str] | None = None) -> tuple[float, int]:
        if not samples:
            return 0.0, 0
        y_true = [s.type_label for s in samples]
        if terms_override is not None:
            y_pred = [predict_fn(t) for t in terms_override]
        else:
            y_pred = [predict_fn(s.term) for s in samples]
        report = classification_report(y_true, y_pred)
        micro = report["micro"]["f1"]
        return float(micro), len(samples)

    seen_f1, n_seen = score(seen)
    heldout_f1, n_heldout = score(heldout)
    para_samples = paraphrase_terms(heldout, seed=seed + 5)
    para_terms = [strip_paraphrase(s.term) for s in para_samples]
    # score on paraphrase surface forms by predicting on cleaned and original
    y_true_p = [s.type_label for s in para_samples]
    y_pred_p = [predict_fn(pt) for pt in para_terms]
    para_report = classification_report(y_true_p, y_pred_p)
    paraphrase_f1 = float(para_report["micro"]["f1"])

    novel_f1: float | None = None
    notes: list[str] = []
    if novel_dataset is not None and novel_predict_fn is not None:
        n_true = [t.type_label for t in novel_dataset.terms]
        n_pred = [novel_predict_fn(t.term) for t in novel_dataset.terms]
        n_report = classification_report(n_true, n_pred)
        novel_f1 = float(n_report["micro"]["f1"])
        notes.append(f"novel_terms={len(novel_dataset.terms)}")
    if heldout_f1 + 0.15 < seen_f1:
        notes.append("heldout_much_lower_than_seen_possible_contamination_or_lexical_gap")
    if paraphrase_f1 + 0.1 < heldout_f1:
        notes.append("paraphrase_forms_degrade_typing_model_surface_sensitivity")

    return ContaminationReport(
        seen_f1=seen_f1,
        heldout_f1=heldout_f1,
        paraphrase_f1=paraphrase_f1,
        novel_f1=novel_f1,
        n_seen=n_seen,
        n_heldout=n_heldout,
        notes=notes,
    )


def taxonomy_holdout_edges(
    dataset: OntologyDataset,
    *,
    holdout_frac: float = 0.25,
    seed: int = 0,
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    edges = list(dataset.taxonomy_pairs())
    rng = random.Random(seed)
    rng.shuffle(edges)
    k = int(round(len(edges) * holdout_frac))
    return set(edges[k:]), set(edges[:k])
