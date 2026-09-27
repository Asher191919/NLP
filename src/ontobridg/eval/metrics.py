"""Internal evaluation helpers + official Graph Similarity integration.

POLICY
------
- External / paper / competition reports MUST use
  `ontobridg.eval.official_graph_similarity` (LLMs4OL 2026 exact_match formula).
- `semantic_graph.semantic_graph_similarity` is an INTERNAL proxy only.
  It is NOT the official metric and MUST NOT be reported as SGS/graph_similarity
  outside ablation/debug tables.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ontobridg.eval.official_graph_similarity import (
    METRIC_ID as OFFICIAL_METRIC_ID,
)
from ontobridg.eval.official_graph_similarity import evaluate_structures as official_graph_similarity

# Backwards-compat names used by older call sites; clearly marked internal.
from ontobridg.eval.semantic_graph import (  # noqa: F401
    CompetitionMetrics,
    classification_report,
    edge_f1,
    evaluate_prediction,
    relation_metrics,
    taxonomy_metrics,
)
from ontobridg.eval.semantic_graph import semantic_graph_similarity as proxy_semantic_graph_similarity_internal


def semantic_graph_similarity(*args, **kwargs):
    """Deprecated alias — returns INTERNAL proxy, not official SGS."""
    out = proxy_semantic_graph_similarity_internal(*args, **kwargs)
    if isinstance(out, dict):
        out = {
            **out,
            "metric_id": "ontobridg_internal_proxy_sgs",
            "external_reportable": False,
            "warning": "NOT official LLMs4OL Graph Similarity; use ontobridg.eval.official_graph_similarity",
        }
    return out


__all__ = [
    "CompetitionMetrics",
    "OFFICIAL_METRIC_ID",
    "classification_report",
    "edge_f1",
    "evaluate_prediction",
    "evaluate_pipeline_result",
    "official_graph_similarity",
    "proxy_semantic_graph_similarity_internal",
    "relation_metrics",
    "semantic_graph_similarity",
    "taxonomy_metrics",
]


def evaluate_pipeline_result(
    gold_term_types,
    pred_term_types,
    gold_taxonomy,
    pred_taxonomy,
    gold_relations=None,
    pred_relations=None,
):
    from ontobridg.eval.semantic_graph import (
        classification_report as _cls,
        relation_metrics as _rel,
        taxonomy_metrics as _tax,
    )

    shared = [t for t in gold_term_types if t in pred_term_types]
    y_true = [gold_term_types[t] for t in shared]
    y_pred = [pred_term_types[t] for t in shared]
    missing = sorted(set(gold_term_types) - set(pred_term_types))
    report = _cls(y_true, y_pred)
    tax_scores = _tax(gold_taxonomy, pred_taxonomy)
    official = official_graph_similarity(
        gold_term_types=gold_term_types,
        gold_taxonomy=gold_taxonomy,
        gold_relations=gold_relations,
        pred_term_types=pred_term_types,
        pred_taxonomy=pred_taxonomy,
        pred_relations=pred_relations,
    )
    rel = None
    if gold_relations is not None:
        rel = _rel(gold_relations, pred_relations or [])
    return {
        "task_a": report,
        "task_a_missing_terms": missing,
        "task_b": tax_scores,
        "task_c": rel,
        "official_graph_similarity": official,
        "graph_similarity": official,  # alias; official exact_match
        "metric_id": OFFICIAL_METRIC_ID,
        "external_reportable": True,
    }
