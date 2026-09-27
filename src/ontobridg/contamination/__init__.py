from ontobridg.contamination.utils import (
    ContaminationReport,
    build_novel_synthetic_ontology,
    evaluate_contamination_slices,
    holdout_terms,
    paraphrase_term,
    paraphrase_terms,
)

try:
    from ontobridg.contamination.protocol import (
        ContaminationReport as ProtocolContaminationReport,
        ContaminationScore,
        build_holdout_dataset,
        build_paraphrase_corpus,
        filter_corpus_for_holdout,
        membership_probe,
        run_contamination_protocol,
        scrub_knowledge_for_holdout,
        synthetic_novel_dataset,
    )
except Exception:  # pragma: no cover
    ProtocolContaminationReport = None  # type: ignore[assignment,misc]
    ContaminationScore = None  # type: ignore[assignment,misc]
    build_holdout_dataset = None  # type: ignore[assignment,misc]
    build_paraphrase_corpus = None  # type: ignore[assignment,misc]
    filter_corpus_for_holdout = None  # type: ignore[assignment,misc]
    membership_probe = None  # type: ignore[assignment,misc]
    run_contamination_protocol = None  # type: ignore[assignment,misc]
    scrub_knowledge_for_holdout = None  # type: ignore[assignment,misc]
    synthetic_novel_dataset = None  # type: ignore[assignment,misc]

__all__ = [
    "ContaminationReport",
    "ProtocolContaminationReport",
    "ContaminationScore",
    "build_novel_synthetic_ontology",
    "build_holdout_dataset",
    "build_paraphrase_corpus",
    "evaluate_contamination_slices",
    "filter_corpus_for_holdout",
    "holdout_terms",
    "membership_probe",
    "paraphrase_term",
    "paraphrase_terms",
    "run_contamination_protocol",
    "scrub_knowledge_for_holdout",
    "synthetic_novel_dataset",
]
