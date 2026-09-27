"""OntoBridg: hybrid neuro-symbolic ontology learning for LLMs4OL tasks."""

__version__ = "0.2.0"

from ontobridg.data.schema import (
    OntologyKnowledge,
    OntologyDataset,
    RelationSample,
    TaxonomySample,
    TermSample,
)

# Pipeline
from ontobridg.pipeline.orchestrator import OntoBridgPipeline, PipelineResult

# Eval — prefer semantic_graph API used by competition sim; keep legacy helpers
from ontobridg.eval.semantic_graph import (
    CompetitionMetrics,
    classification_report as map_classification_report,
    evaluate_prediction,
    semantic_graph_similarity as official_style_sgs,
    taxonomy_metrics as edge_taxonomy_metrics,
)

try:
    from ontobridg.eval.metrics import (
        classification_report,
        evaluate_pipeline_result,
        semantic_graph_similarity,
        taxonomy_metrics,
    )
except Exception:  # pragma: no cover
    classification_report = map_classification_report  # type: ignore
    taxonomy_metrics = edge_taxonomy_metrics  # type: ignore
    semantic_graph_similarity = official_style_sgs  # type: ignore
    evaluate_pipeline_result = None  # type: ignore

# Competition modules (2026 novel contributions)
from ontobridg.tasks.task_a_flagship import TaskAFlagship, TaskAResult
from ontobridg.tasks.task_b_extension import TaskBExtension, TaskBResult
from ontobridg.relations.open_vocab import (
    OpenRelationInventory,
    OpenVocabRelationMiner,
    TAXONOMIC_RELATIONS,
    closed_vocab_baseline_extract,
)
from ontobridg.symbolic.repair import (
    RepairReport,
    ScoredEdge,
    ScoredRelation,
    SymbolicRepair,
)
from ontobridg.baselines.rag_pipeline import RAGIndex, RAGPipeline
from ontobridg.contamination.protocol import (
    ContaminationReport,
    ContaminationScore,
    membership_probe,
    run_contamination_protocol,
    synthetic_novel_dataset,
)

__all__ = [
    "__version__",
    "OntologyKnowledge",
    "OntologyDataset",
    "RelationSample",
    "TaxonomySample",
    "TermSample",
    "OntoBridgPipeline",
    "PipelineResult",
    "CompetitionMetrics",
    "classification_report",
    "taxonomy_metrics",
    "semantic_graph_similarity",
    "evaluate_prediction",
    "evaluate_pipeline_result",
    "official_style_sgs",
    "TaskAFlagship",
    "TaskAResult",
    "TaskBExtension",
    "TaskBResult",
    "OpenRelationInventory",
    "OpenVocabRelationMiner",
    "TAXONOMIC_RELATIONS",
    "closed_vocab_baseline_extract",
    "RepairReport",
    "ScoredEdge",
    "ScoredRelation",
    "SymbolicRepair",
    "RAGIndex",
    "RAGPipeline",
    "ContaminationReport",
    "ContaminationScore",
    "membership_probe",
    "run_contamination_protocol",
    "synthetic_novel_dataset",
]
