from ontobridg.symbolic.constraints import (
    SymbolicRepairResult,
    apply_type_range_filter,
    detect_cycles,
    lexical_relation_filter,
    repair_all,
    repair_taxonomy,
    transitive_closure,
)

try:
    from ontobridg.symbolic.repair import (
        RepairReport,
        ScoredEdge,
        ScoredRelation,
        SymbolicRepair,
    )
except Exception:  # pragma: no cover
    RepairReport = None  # type: ignore[assignment,misc]
    ScoredEdge = None  # type: ignore[assignment,misc]
    ScoredRelation = None  # type: ignore[assignment,misc]
    SymbolicRepair = None  # type: ignore[assignment,misc]

__all__ = [
    "SymbolicRepairResult",
    "apply_type_range_filter",
    "detect_cycles",
    "lexical_relation_filter",
    "repair_all",
    "repair_taxonomy",
    "transitive_closure",
    "RepairReport",
    "ScoredEdge",
    "ScoredRelation",
    "SymbolicRepair",
]
