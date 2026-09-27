"""Internal metrics + CompetitionMetrics.

`semantic_graph_similarity` here is an INTERNAL PROXY.
External reporting uses ontobridg.eval.official_graph_similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ontobridg.eval.official_graph_similarity import evaluate_structures as _official_gs


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def classification_report(gold: dict[str, str], pred: dict[str, str]) -> dict[str, float]:
    terms = sorted(set(gold) | set(pred))
    tp = fp = fn = 0
    per_type: dict[str, dict[str, int]] = {}
    for t in terms:
        g, p = gold.get(t), pred.get(t)
        if g and p and g == p:
            tp += 1
            per_type.setdefault(g, {"tp": 0, "fp": 0, "fn": 0})["tp"] += 1
        elif p and (not g or g != p):
            fp += 1
            per_type.setdefault(p or "?", {"tp": 0, "fp": 0, "fn": 0})["fp"] += 1
            if g:
                per_type.setdefault(g, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
                fn += 1
        elif g and not p:
            fn += 1
            per_type.setdefault(g, {"tp": 0, "fp": 0, "fn": 0})["fn"] += 1
    micro = _prf(tp, fp, fn)
    macro_f1s = []
    for c in per_type.values():
        macro_f1s.append(_prf(c["tp"], c["fp"], c["fn"])["f1"])
    micro["macro_f1"] = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
    micro["f1"] = micro["f1"]
    return micro


def edge_f1(gold: Iterable[tuple], pred: Iterable[tuple]) -> dict[str, float]:
    gold_set, pred_set = set(gold), set(pred)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return _prf(tp, fp, fn)


def taxonomy_metrics(gold: Iterable[tuple[str, str]], pred: Iterable[tuple[str, str]]) -> dict[str, float]:
    return edge_f1(gold, pred)


def relation_metrics(gold: Iterable[tuple[str, str, str]], pred: Iterable[tuple[str, str, str]]) -> dict[str, float]:
    return edge_f1(gold, pred)


def proxy_semantic_graph_similarity(
    gold_terms: dict[str, str],
    gold_taxonomy: Iterable[tuple[str, str]],
    gold_relations: Iterable[tuple[str, str, str]],
    pred_terms: dict[str, str],
    pred_taxonomy: Iterable[tuple[str, str]],
    pred_relations: Iterable[tuple[str, str, str]],
    *,
    w_terms: float = 0.3,
    w_tax: float = 0.35,
    w_rel: float = 0.35,
) -> dict[str, float]:
    """INTERNAL PROXY only — NOT official Graph Similarity. external_reportable=False."""

    def _jaccard(a: set, b: set) -> float:
        if not a and not b:
            return 1.0
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    g_terms = set(gold_terms.items())
    p_terms = set(pred_terms.items())
    term_j = _jaccard(g_terms, p_terms)
    g_term_names = set(gold_terms)
    p_term_names = set(pred_terms)
    term_coverage = _jaccard(g_term_names, p_term_names)
    term_score = 0.7 * term_j + 0.3 * term_coverage
    tax_score = _jaccard(set(gold_taxonomy), set(pred_taxonomy))
    rel_score = _jaccard(set(gold_relations), set(pred_relations))
    sgs = w_terms * term_score + w_tax * tax_score + w_rel * rel_score
    return {
        "sgs": sgs,
        "proxy_sgs": sgs,
        "term_component": term_score,
        "taxonomy_component": tax_score,
        "relation_component": rel_score,
        "metric_id": "ontobridg_internal_proxy_sgs",
        "external_reportable": False,
        "warning": "NOT official LLMs4OL Graph Similarity",
    }


# historical name
semantic_graph_similarity = proxy_semantic_graph_similarity


@dataclass
class CompetitionMetrics:
    """Scores for internal comparison. Official SGS is stored under `official_sgs`."""

    term_typing: dict[str, float] = field(default_factory=dict)
    taxonomy: dict[str, float] = field(default_factory=dict)
    relations: dict[str, float] = field(default_factory=dict)
    sgs: dict[str, float] = field(default_factory=dict)  # INTERNAL proxy
    nontax_relation: dict[str, float] = field(default_factory=dict)
    official_sgs: dict[str, float] = field(default_factory=dict)  # official exact_match
    external_reportable_sgs_key: str = "official_graph_similarity"

    @property
    def combo_score(self) -> float:
        """Internal diagnostic combo (NOT official leaderboard score)."""
        tt = self.term_typing.get("f1", 0.0)
        tax = self.taxonomy.get("f1", 0.0)
        nrel = self.nontax_relation.get("f1", 0.0)
        sgs = self.official_graph_similarity
        return 0.25 * tt + 0.25 * tax + 0.25 * nrel + 0.25 * sgs

    @property
    def official_graph_similarity(self) -> float:
        return float(self.official_sgs.get("graph_similarity", 0.0))

    @property
    def proxy_sgs_score(self) -> float:
        return float(self.sgs.get("proxy_sgs", self.sgs.get("sgs", 0.0)))

    @property
    def official_like_score(self) -> float:
        """(official Graph Similarity + TermTyping F1 + Taxonomy F1) / 3.

        Matches pro-team 2026 reporting style when those three numbers exist.
        """
        tt = self.term_typing.get("f1", 0.0)
        tax = self.taxonomy.get("f1", 0.0)
        sgs = self.official_graph_similarity
        return (sgs + tt + tax) / 3.0

    @property
    def official_like_score_external(self) -> dict[str, float]:
        return {
            "term_typing_f1": self.term_typing.get("f1", 0.0),
            "taxonomy_f1": self.taxonomy.get("f1", 0.0),
            "graph_similarity": self.official_graph_similarity,
            "composite": self.official_like_score,
        }

    def to_dict(self) -> dict:
        return {
            "term_typing": self.term_typing,
            "taxonomy": self.taxonomy,
            "relations_all": self.relations,
            "relations_nontaxonomic": self.nontax_relation,
            "official_graph_similarity": self.official_sgs,
            "internal_proxy_sgs": {
                **self.sgs,
                "external_reportable": False,
            },
            "official_like_score": self.official_like_score,
            "official_like_score_external": self.official_like_score_external,
            "combo_score_internal": self.combo_score,
            "metric_policy": {
                "external_reportable_key": "official_graph_similarity",
                "forbidden_external_keys": ["internal_proxy_sgs", "sgs", "proxy_sgs"],
            },
        }


def evaluate_prediction(
    gold_terms: dict[str, str],
    gold_taxonomy: Iterable[tuple[str, str]],
    gold_relations: Iterable[tuple[str, str, str]],
    pred_terms: dict[str, str],
    pred_taxonomy: Iterable[tuple[str, str]],
    pred_relations: Iterable[tuple[str, str, str]],
    *,
    taxonomic_labels: set[str] | None = None,
) -> CompetitionMetrics:
    taxonomic_labels = taxonomic_labels or {"is-a", "isa", "type-of", "subclass-of", "kind-of"}
    gold_tax = list(gold_taxonomy)
    pred_tax = list(pred_taxonomy)
    gold_rel = list(gold_relations)
    pred_rel = list(pred_relations)
    gold_nontax = [r for r in gold_rel if r[1] not in taxonomic_labels]
    pred_nontax = [r for r in pred_rel if r[1] not in taxonomic_labels]

    official = _official_gs(
        gold_term_types=gold_terms,
        gold_taxonomy=gold_tax,
        gold_relations=gold_rel,
        pred_term_types=pred_terms,
        pred_taxonomy=pred_tax,
        pred_relations=pred_rel,
    )
    proxy = proxy_semantic_graph_similarity(
        gold_terms, gold_tax, gold_nontax, pred_terms, pred_tax, pred_nontax
    )
    return CompetitionMetrics(
        term_typing=classification_report(gold_terms, pred_terms),
        taxonomy=taxonomy_metrics(gold_tax, pred_tax),
        relations=relation_metrics(gold_rel, pred_rel),
        nontax_relation=relation_metrics(gold_nontax, pred_nontax),
        sgs=proxy,
        official_sgs=official,
    )
