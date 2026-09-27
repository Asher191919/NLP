"""Build a fail-closed grouped-CV audit from gold-labelled v5 candidates.

This command is offline. It pools a prior calibration slice with a validation
slice that has already been inspected, so its out-of-fold metrics are useful
for calibrator diagnostics but cannot serve as confirmatory evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import run_annotated_relation_calibration as grouped
import run_official_blind_eval as blind
import run_relation_v2_offline as offline


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_CALIBRATION_RESULT = RESULTS / (
    "relation_calibration_open_vocab_guarded_v5_axioms_only_"
    "a_indices1419-1554-1796-1824-1935-2676-3130-3516_seed20260924.json"
)
DEFAULT_REUSED_VALIDATION_RESULT = RESULTS / (
    "relation_calibration_open_vocab_guarded_v5_axioms_only_"
    "a_indices678-2171-2643-2780-2920-2958-3064-3676_seed20260925.json"
)
DEFAULT_OUTPUT = RESULTS / (
    "relation_guarded_v5_axioms_only_pooled_grouped_cv_"
    "a16_seed20260925.json"
)
DEFAULT_SYSTEM = "open_vocab_guarded_v5_axioms_only"
SCIENTIFIC_GATE_REASON = (
    "validation_reused_for_calibration_and_no_untouched_confirmatory_set"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_scientific_gate(
    cv_report: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Preserve the numeric result while preventing confirmatory promotion."""
    numeric_gate = dict(cv_report["deployment_support"])
    cv_report["deployment_support"] = {
        **numeric_gate,
        "deployable": False,
        "numeric_gate_passed": bool(numeric_gate["deployable"]),
        "confirmatory_eligible": False,
        "status": SCIENTIFIC_GATE_REASON,
        "reason": SCIENTIFIC_GATE_REASON,
    }
    return cv_report, numeric_gate


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    calibration_path = Path(args.calibration_result).resolve()
    validation_path = Path(args.reused_validation_result).resolve()
    sources = [
        (
            calibration_path,
            json.loads(calibration_path.read_text(encoding="utf-8")),
        ),
        (
            validation_path,
            json.loads(validation_path.read_text(encoding="utf-8")),
        ),
    ]
    merged = offline.merge_source_results(sources, source_system=args.source_system)
    calibration_indices = offline.choose_all_successful_indices(
        merged, source_system=args.source_system
    )
    candidates = offline.collect_candidates(
        merged,
        source_system=args.source_system,
        calibration_indices=calibration_indices,
    )
    for row in candidates:
        row["sample_key"] = f"{row['task']}:{int(row['sample_index'])}"

    cv_report = grouped.grouped_cross_validation_report(
        candidates,
        n_folds=args.folds,
        seed=args.seed,
        exploratory=False,
    )
    cv_report, numeric_gate = apply_scientific_gate(cv_report)
    eligible = offline.deployment_eligible_candidates(candidates)
    source_files = [
        {
            "role": "prior_calibration",
            "path": str(calibration_path),
            "sha256": file_sha256(calibration_path),
        },
        {
            "role": "reused_independent_validation",
            "path": str(validation_path),
            "sha256": file_sha256(validation_path),
        },
    ]
    return {
        "meta": {
            "protocol": "gold_relation_grouped_cv_reused_validation_v1",
            "source_system": args.source_system,
            "source_files": source_files,
            "seed": args.seed,
            "folds": args.folds,
            "api_calls": 0,
            "gold_used_for_candidate_labels_only": True,
            "validation_reused_for_calibration": True,
            "has_untouched_confirmatory_set": False,
        },
        "candidate_summary": {
            "all": offline.summarize_candidates(candidates),
            "deployment_population": offline.summarize_candidates(eligible),
        },
        "numeric_grouped_cv_gate": numeric_gate,
        "grouped_cross_validation": cv_report,
        "scientific_deployable": False,
        "confirmatory_eligible": False,
        "reason": SCIENTIFIC_GATE_REASON,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibration-result", type=Path, default=DEFAULT_CALIBRATION_RESULT
    )
    parser.add_argument(
        "--reused-validation-result",
        type=Path,
        default=DEFAULT_REUSED_VALIDATION_RESULT,
    )
    parser.add_argument("--source-system", default=DEFAULT_SYSTEM)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.folds < 2:
        raise SystemExit("--folds must be at least 2")
    report = build_report(args)
    blind.atomic_write_json(args.output, report)
    cv = report["grouped_cross_validation"]
    raw = cv["pooled_out_of_fold_metrics"]["raw"]
    calibrated = cv["pooled_out_of_fold_metrics"]["calibrated"]
    print(
        f"candidates={cv['coverage']['eligible_candidates']} "
        f"positive_samples={cv['n_positive_sample_groups']} "
        f"numeric_gate={report['numeric_grouped_cv_gate']['deployable']} "
        f"scientific_deployable={report['scientific_deployable']} "
        f"brier={raw['brier']:.4f}->{calibrated['brier']:.4f} "
        f"ece={raw['ece']:.4f}->{calibrated['ece']:.4f}",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
