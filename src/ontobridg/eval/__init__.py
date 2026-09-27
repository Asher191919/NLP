from ontobridg.eval.official_graph_similarity import (
    METRIC_ID as OFFICIAL_METRIC_ID,
)
from ontobridg.eval.official_graph_similarity import (
    evaluate_structures as official_graph_similarity,
)
from ontobridg.eval.official_graph_similarity import exact_match as official_exact_match
from ontobridg.eval.semantic_graph import (
    CompetitionMetrics,
    classification_report as map_classification_report,
    edge_f1,
    evaluate_prediction,
    proxy_semantic_graph_similarity,
    relation_metrics as edge_relation_metrics,
    taxonomy_metrics as edge_taxonomy_metrics,
)
from ontobridg.eval.metrics import (
    evaluate_pipeline_result,
    semantic_graph_similarity,
)

__all__ = [
    "CompetitionMetrics",
    "OFFICIAL_METRIC_ID",
    "classification_report",
    "edge_f1",
    "evaluate_pipeline_result",
    "evaluate_prediction",
    "map_classification_report",
    "official_exact_match",
    "official_graph_similarity",
    "proxy_semantic_graph_similarity",
    "relation_metrics",
    "semantic_graph_similarity",
    "taxonomy_metrics",
]

try:
    from ontobridg.eval.metrics import (
        classification_report,
        relation_metrics,
        taxonomy_metrics,
    )
except Exception:  # pragma: no cover
    classification_report = map_classification_report  # type: ignore
    taxonomy_metrics = edge_taxonomy_metrics  # type: ignore
    relation_metrics = edge_relation_metrics  # type: ignore
