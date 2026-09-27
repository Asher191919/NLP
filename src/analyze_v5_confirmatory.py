"""Analyze frozen v5 confirmatory results without retuning or API calls."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import analyze_grouped_relation_calibration as uncertainty
import run_official_blind_eval as blind
import run_relation_v2_offline as offline
import run_v5_confirmatory as confirmatory
from ontobridg.relations.calibration import (
    BetaIsotonicCalibrator,
    calibration_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT = RESULTS / (
    "v5_axiom_confirmatory_error_audit_a_n24_seed20260925.json"
)
DEFAULT_REPLICATES = 10_000
DEFAULT_SEED = 20260925


def count_metrics(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def per_sample_exact_counts(result: dict[str, Any]) -> dict[str, dict[str, int]]:
    records = result["tasks"]["task_a"]["systems"][confirmatory.SYSTEM]["samples"]
    return {
        f"a:{int(record['sample_index'])}": {
            field: int(record["metrics"]["nontax"][field])
            for field in ("tp", "fp", "fn")
        }
        for record in records
        if record.get("status") == "ok"
    }


def predicate_metrics(
    predicted: dict[tuple[str, int], set[tuple[str, str, str]]],
    gold: dict[tuple[str, int], set[tuple[str, str, str]]],
) -> dict[str, Any]:
    output = {}
    for predicate in confirmatory.TARGET_PREDICATES:
        pred_filtered = {
            key: {triple for triple in triples if triple[1] == predicate}
            for key, triples in predicted.items()
        }
        gold_filtered = {
            key: {triple for triple in triples if triple[1] == predicate}
            for key, triples in gold.items()
        }
        output[predicate] = offline.relation_set_metrics(
            pred_filtered, gold_filtered
        )
    return output


def stratum_metrics(
    predicted: dict[tuple[str, int], set[tuple[str, str, str]]],
    gold: dict[tuple[str, int], set[tuple[str, str, str]]],
    assignment: dict[int, str],
) -> dict[str, Any]:
    return {
        predicate: offline.relation_set_metrics(
            {
                key: triples
                for key, triples in predicted.items()
                if assignment[int(key[1])] == predicate
            },
            {
                key: triples
                for key, triples in gold.items()
                if assignment[int(key[1])] == predicate
            },
        )
        for predicate in confirmatory.TARGET_PREDICATES
    }


def candidate_error_audit(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = offline.deployment_eligible_candidates(candidates)
    error_types = Counter()
    by_predicate: dict[str, Counter[str]] = {}
    by_confidence: dict[str, Counter[str]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        if row["canonical_correct"]:
            category = "exact_true_positive"
        elif row.get("symmetric_semantic_correct"):
            category = "symmetric_reverse_only"
        elif str(row.get("near_match_type") or "").strip().lower() not in {
            "",
            "none",
        }:
            category = str(row["near_match_type"])
        else:
            category = "no_exact_endpoint_or_predicate_match"
        error_types[category] += 1
        predicate = str(row["canonical_triple"][1])
        by_predicate.setdefault(predicate, Counter())[category] += 1
        confidence = f"{float(row['confidence']):.2f}"
        by_confidence.setdefault(confidence, Counter())[category] += 1
        examples.setdefault(category, [])
        if len(examples[category]) < 5:
            examples[category].append(
                {
                    "sample_index": int(row["sample_index"]),
                    "confidence": float(row["confidence"]),
                    "triple": row["canonical_triple"],
                    "near_gold_triples": row.get("near_gold_triples") or [],
                }
            )
    rejected = [
        row for row in candidates if row["postprocess_status"] == "rejected"
    ]
    return {
        "eligible_candidates": len(eligible),
        "error_type_counts": dict(sorted(error_types.items())),
        "by_predicate": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_predicate.items())
        },
        "by_raw_confidence": {
            key: dict(sorted(value.items()))
            for key, value in sorted(by_confidence.items())
        },
        "rejected_candidate_reasons": dict(
            sorted(
                Counter(
                    reason
                    for row in rejected
                    for reason in row.get("rejection_reasons") or []
                ).items()
            )
        ),
        "examples": examples,
    }


def _interval(values: list[float], estimate: float) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "lower": uncertainty.percentile(values, 0.025),
        "upper": uncertainty.percentile(values, 0.975),
        "bootstrap_valid_replicates": len(values),
    }


def sample_cluster_bootstrap(
    *,
    base_counts: dict[str, dict[str, int]],
    calibrated_counts: dict[str, dict[str, int]],
    candidates: list[dict[str, Any]],
    calibrator: BetaIsotonicCalibrator,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    eligible = offline.deployment_eligible_candidates(candidates)
    candidate_groups: dict[str, list[dict[str, Any]]] = {}
    for row in eligible:
        key = f"{row['task']}:{int(row['sample_index'])}"
        candidate_groups.setdefault(key, []).append(row)
    sample_keys = sorted(base_counts)
    if sample_keys != sorted(calibrated_counts):
        raise ValueError("base and calibrated sample groups differ")
    distributions: dict[str, list[float]] = {
        key: []
        for key in (
            "base_precision",
            "base_recall",
            "base_f1",
            "calibrated_precision",
            "calibrated_recall",
            "calibrated_f1",
            "delta_precision",
            "delta_recall",
            "delta_f1",
            "raw_brier",
            "calibrated_brier",
            "delta_brier",
            "raw_ece",
            "calibrated_ece",
            "delta_ece",
        )
    }
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled_keys = [rng.choice(sample_keys) for _ in sample_keys]
        stage_metrics = {}
        for name, counts in (
            ("base", base_counts),
            ("calibrated", calibrated_counts),
        ):
            totals = {
                field: sum(counts[key][field] for key in sampled_keys)
                for field in ("tp", "fp", "fn")
            }
            metrics = count_metrics(**totals)
            stage_metrics[name] = metrics
            for metric in ("precision", "recall", "f1"):
                distributions[f"{name}_{metric}"].append(float(metrics[metric]))
        for metric in ("precision", "recall", "f1"):
            distributions[f"delta_{metric}"].append(
                float(stage_metrics["calibrated"][metric])
                - float(stage_metrics["base"][metric])
            )

        sampled_candidates = [
            row for key in sampled_keys for row in candidate_groups.get(key, [])
        ]
        labels = [bool(row["canonical_correct"]) for row in sampled_candidates]
        raw_scores = [float(row["confidence"]) for row in sampled_candidates]
        calibrated_scores = [calibrator.calibrate(score) for score in raw_scores]
        raw = calibration_metrics(raw_scores, labels)
        calibrated = calibration_metrics(calibrated_scores, labels)
        for metric in ("brier", "ece"):
            distributions[f"raw_{metric}"].append(float(raw[metric]))
            distributions[f"calibrated_{metric}"].append(
                float(calibrated[metric])
            )
            distributions[f"delta_{metric}"].append(
                float(calibrated[metric]) - float(raw[metric])
            )

    base_totals = {
        field: sum(row[field] for row in base_counts.values())
        for field in ("tp", "fp", "fn")
    }
    calibrated_totals = {
        field: sum(row[field] for row in calibrated_counts.values())
        for field in ("tp", "fp", "fn")
    }
    point = {
        "base": count_metrics(**base_totals),
        "calibrated": count_metrics(**calibrated_totals),
    }
    labels = [bool(row["canonical_correct"]) for row in eligible]
    raw_scores = [float(row["confidence"]) for row in eligible]
    calibrated_scores = [calibrator.calibrate(score) for score in raw_scores]
    raw_point = calibration_metrics(raw_scores, labels)
    calibrated_point = calibration_metrics(calibrated_scores, labels)
    return {
        "protocol": "sample_cluster_bootstrap_confirmatory_v1",
        "resampling_unit": "sample_index",
        "conditional_on": "frozen_confirmatory_samples_predictions_and_gold",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "n_sample_groups": len(sample_keys),
        "exact": {
            stage: {
                metric: _interval(
                    distributions[f"{stage}_{metric}"],
                    float(point[stage][metric]),
                )
                for metric in ("precision", "recall", "f1")
            }
            for stage in ("base", "calibrated")
        }
        | {
            "calibrated_minus_base": {
                metric: _interval(
                    distributions[f"delta_{metric}"],
                    float(point["calibrated"][metric])
                    - float(point["base"][metric]),
                )
                for metric in ("precision", "recall", "f1")
            }
        },
        "candidate_calibration": {
            stage: {
                metric: _interval(
                    distributions[f"{stage}_{metric}"],
                    float(
                        (raw_point if stage == "raw" else calibrated_point)[metric]
                    ),
                )
                for metric in ("brier", "ece")
            }
            for stage in ("raw", "calibrated")
        }
        | {
            "calibrated_minus_raw": {
                metric: _interval(
                    distributions[f"delta_{metric}"],
                    float(calibrated_point[metric]) - float(raw_point[metric]),
                )
                for metric in ("brier", "ece")
            }
        },
    }


def estimation_scale(
    *, precision: float, predictions: int, samples: int, margin: float = 0.10
) -> dict[str, Any]:
    prediction_rate = predictions / samples if samples else 0.0
    z = 1.959963984540054
    required_predictions = math.ceil(
        z * z * precision * (1.0 - precision) / (margin * margin)
    )
    required_samples = (
        math.ceil(required_predictions / prediction_rate)
        if prediction_rate > 0
        else None
    )
    return {
        "method": "normal_approximation_planning_only",
        "target_95_percent_half_width": margin,
        "observed_precision": precision,
        "observed_predictions_per_sample": prediction_rate,
        "required_predictions": required_predictions,
        "estimated_required_samples": required_samples,
        "not_a_hypothesis_test_or_tuning_budget": True,
    }


def build_analysis(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    base = json.loads(Path(args.base_result).read_text(encoding="utf-8"))
    calibrated = json.loads(
        Path(args.calibrated_result).read_text(encoding="utf-8")
    )
    score = json.loads(Path(args.score).read_text(encoding="utf-8"))
    indices = [int(row["sample_index"]) for row in manifest["samples"]]
    assignment = {
        int(row["sample_index"]): str(row["assigned_stratum"])
        for row in manifest["samples"]
    }
    candidates = offline.collect_candidates(
        base,
        source_system=confirmatory.SYSTEM,
        calibration_indices={"a": set(indices)},
        guard_diagnostics_from_result=True,
    )
    calibrator_payload = json.loads(Path(args.calibrator).read_text(encoding="utf-8"))
    frozen_calibrator = BetaIsotonicCalibrator.from_dict(
        calibrator_payload["calibrator"]
    )
    base_predictions = confirmatory._accepted_predictions(base)
    calibrated_predictions = confirmatory._accepted_predictions(calibrated)
    gold = confirmatory._gold_by_sample(indices)
    predicate_exact = {
        "base": predicate_metrics(base_predictions, gold),
        "calibrated": predicate_metrics(calibrated_predictions, gold),
    }
    stratum_exact = {
        "base": stratum_metrics(base_predictions, gold, assignment),
        "calibrated": stratum_metrics(
            calibrated_predictions, gold, assignment
        ),
    }
    bootstrap = sample_cluster_bootstrap(
        base_counts=per_sample_exact_counts(base),
        calibrated_counts=per_sample_exact_counts(calibrated),
        candidates=candidates,
        calibrator=frozen_calibrator,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    base_exact = score["official_exact"]["base_v5"]
    calibrated_exact = score["official_exact"]["frozen_calibrated_v5"]
    scale = {
        "base": estimation_scale(
            precision=float(base_exact["precision"]),
            predictions=int(base_exact["tp"]) + int(base_exact["fp"]),
            samples=len(indices),
        ),
        "calibrated": estimation_scale(
            precision=float(calibrated_exact["precision"]),
            predictions=int(calibrated_exact["tp"])
            + int(calibrated_exact["fp"]),
            samples=len(indices),
        ),
    }
    recommended = max(
        int(row["estimated_required_samples"] or 0) for row in scale.values()
    )
    stratum_sample_counts = Counter(assignment.values())
    scale["per_preregistered_assignment_stratum"] = {
        stage: {
            predicate: estimation_scale(
                precision=float(metrics["precision"]),
                predictions=int(metrics["predictions"]),
                samples=int(stratum_sample_counts[predicate]),
            )
            for predicate, metrics in stratum_exact[stage].items()
        }
        for stage in ("base", "calibrated")
    }
    scale["decision"] = {
        "expand_for_estimation_only": True,
        "launch_now": False,
        "status": "deferred_pending_scope_and_cost_justification",
        "recommended_total_samples": recommended,
        "additional_samples_from_current_n24": max(0, recommended - len(indices)),
        "eligible_endpoint_visible_target_pool": int(
            manifest["pool"]["eligible_endpoint_visible_target_samples"]
        ),
        "remaining_pool_after_current_n24": int(
            manifest["pool"]["eligible_endpoint_visible_target_samples"]
        )
        - len(indices),
        "authorize_threshold_or_rule_retuning": False,
        "interpretation": (
            "Expansion can narrow uncertainty around the frozen negative result; "
            "it cannot reopen model selection on these confirmatory labels. "
            "The pooled estimate hides predicate heterogeneity and does not "
            "remove endpoint-visible gold-stratified selection bias."
        ),
    }
    return {
        "meta": {
            "protocol": "v5_axiom_confirmatory_error_audit_v1",
            "source_score": str(Path(args.score).resolve()),
            "source_score_sha256": confirmatory.file_sha256(Path(args.score)),
            "api_calls": 0,
            "retuning_performed": False,
        },
        "per_predicate_official_exact": {
            "base": predicate_exact["base"],
            "calibrated": predicate_exact["calibrated"],
        },
        "by_preregistered_assignment_stratum": {
            "base": stratum_exact["base"],
            "calibrated": stratum_exact["calibrated"],
        },
        "candidate_error_audit": candidate_error_audit(candidates),
        "sample_cluster_bootstrap": bootstrap,
        "scale_planning": scale,
        "scientific_deployable": False,
        "confirmatory_eligible": False,
        "reason": "confirmatory_gate_failed_and_analysis_does_not_authorize_retuning",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=confirmatory.DEFAULT_MANIFEST)
    parser.add_argument("--base-result", type=Path, default=confirmatory.DEFAULT_BASE_RESULT)
    parser.add_argument(
        "--calibrated-result",
        type=Path,
        default=confirmatory.DEFAULT_CALIBRATED_RESULT,
    )
    parser.add_argument("--score", type=Path, default=confirmatory.DEFAULT_SCORE_OUTPUT)
    parser.add_argument("--calibrator", type=Path, default=confirmatory.DEFAULT_CALIBRATOR)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    analysis = build_analysis(args)
    blind.atomic_write_json(args.output, analysis)
    decision = analysis["scale_planning"]["decision"]
    delta = analysis["sample_cluster_bootstrap"]["exact"][
        "calibrated_minus_base"
    ]["f1"]
    print(
        f"f1_delta={delta['estimate']:.4f} "
        f"ci=[{delta['lower']:.4f},{delta['upper']:.4f}] "
        f"recommended_total_n={decision['recommended_total_samples']}",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
