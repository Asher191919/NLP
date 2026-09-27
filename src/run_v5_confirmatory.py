"""Freeze and execute an untouched, gold-stratified v5 confirmatory slice.

Selection uses official gold locally to require prompt-visible endpoints and to
balance the two axiom predicates. Gold triples and source text never enter the
manifest or model prompt beyond the ordinary legal Task A context field.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind
import run_relation_calibration as calibration
import run_relation_v2_offline as offline
from ontobridg.relations.calibration import (
    BetaIsotonicCalibrator,
    calibration_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PROTOCOL = "v5_axiom_confirmatory_gold_stratified_v1"
SYSTEM = "open_vocab_guarded_v5_axioms_only"
TARGET_PREDICATES = ("equivalent class", "disjoint with")
DEFAULT_COUNT = 24
DEFAULT_SEED = 20260925
DEFAULT_CALIBRATOR = RESULTS / (
    "relation_guarded_v5_axioms_only_pooled_calibrator_a16_seed20260925.json"
)
DEFAULT_MANIFEST = RESULTS / (
    "v5_axiom_confirmatory_manifest_a24_seed20260925.json"
)
DEFAULT_BASE_RESULT = RESULTS / (
    "v5_axiom_confirmatory_base_a_n24_seed20260925.json"
)
DEFAULT_CALIBRATED_RESULT = RESULTS / (
    "v5_axiom_confirmatory_calibrated_a_n24_seed20260925.json"
)
DEFAULT_SCORE_OUTPUT = RESULTS / (
    "v5_axiom_confirmatory_score_a_n24_seed20260925.json"
)
DEFAULT_EXCLUDE_RESULTS = (
    RESULTS / "official_stratified_open_openai_ab_s5_seed20260922.json",
    RESULTS / "relation_calibration_a_n27_seed20260923.json",
    RESULTS / "relation_calibration_ab_n12_seed20260923.json",
    RESULTS
    / "relation_calibration_open_vocab_guarded_v3_relations_"
    "a_indices938-2077-2424_seed20260923.json",
    RESULTS
    / "relation_calibration_open_vocab_guarded_v5_axioms_only_"
    "a_indices1419-1554-1796-1824-1935-2676-3130-3516_seed20260924.json",
    RESULTS
    / "relation_calibration_open_vocab_guarded_v5_axioms_only_"
    "a_indices678-2171-2643-2780-2920-2958-3064-3676_seed20260925.json",
)
DEFAULT_EXCLUDE_ANNOTATIONS = (
    ROOT
    / "experiments"
    / "annotations"
    / "relation_text_model_annotation_v1"
    / "manifest.json",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def stable_rank(seed: int, predicate: str, sample_id: str) -> str:
    value = f"{seed}:{predicate}:{sample_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def select_balanced_samples(
    indexed_samples: list[tuple[int, dict[str, Any]]],
    *,
    excluded_indices: set[int],
    count: int,
    seed: int,
    max_context_chars: int,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[int, dict[str, Any]], dict[str, Any]]:
    if count < 2 or count % len(TARGET_PREDICATES):
        raise ValueError(
            f"count must be a positive multiple of {len(TARGET_PREDICATES)}"
        )
    target_set = set(TARGET_PREDICATES)
    candidates: list[dict[str, Any]] = []
    for index, sample in indexed_samples:
        if index in excluded_indices:
            continue
        features = calibration.sample_relation_features(
            sample, task="a", max_context_chars=max_context_chars
        )
        visible_predicates = target_set & set(features["visible_nontax_predicates"])
        if not visible_predicates:
            continue
        sample_id = str(sample.get("id") or index)
        candidates.append(
            {
                "index": index,
                "sample": sample,
                "sample_id": sample_id,
                "features": features,
                "target_predicates": sorted(visible_predicates),
            }
        )

    per_predicate = count // len(TARGET_PREDICATES)
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    assigned_counts: dict[str, int] = {}
    for predicate in TARGET_PREDICATES:
        ranked = sorted(
            (row for row in candidates if predicate in row["target_predicates"]),
            key=lambda row: (
                len(row["target_predicates"]) != 1,
                stable_rank(seed, predicate, row["sample_id"]),
                row["index"],
            ),
        )
        chosen = [
            row for row in ranked if int(row["index"]) not in selected_indices
        ][:per_predicate]
        if len(chosen) != per_predicate:
            raise ValueError(
                f"only {len(chosen)} unused samples available for {predicate}; "
                f"need {per_predicate}"
            )
        for row in chosen:
            row = dict(row)
            row["assigned_stratum"] = predicate
            selected.append(row)
            selected_indices.add(int(row["index"]))
        assigned_counts[predicate] = len(chosen)

    selected.sort(key=lambda row: int(row["index"]))
    metadata = {
        int(row["index"]): {
            **row["features"],
            "target_predicates": row["target_predicates"],
            "assigned_stratum": row["assigned_stratum"],
            "selection_purpose": "untouched_v5_confirmatory",
        }
        for row in selected
    }
    pool = {
        "eligible_endpoint_visible_target_samples": len(candidates),
        "eligible_by_target_predicate": {
            predicate: sum(
                predicate in row["target_predicates"] for row in candidates
            )
            for predicate in TARGET_PREDICATES
        },
        "eligible_lexically_explicit_target_samples": sum(
            bool(
                target_set
                & set(row["features"]["visible_lexically_explicit_predicates"])
            )
            for row in candidates
        ),
        "selected_assigned_strata": assigned_counts,
        "selected_visible_target_gold": sum(
            metadata[int(row["index"])]["visible_nontax_gold"] for row in selected
        ),
    }
    return (
        [(int(row["index"]), row["sample"]) for row in selected],
        metadata,
        pool,
    )


def exclusion_inputs(args: argparse.Namespace) -> tuple[list[Path], list[Path]]:
    result_paths = [Path(path) for path in DEFAULT_EXCLUDE_RESULTS]
    result_paths.extend(Path(path) for path in args.additional_exclude_result)
    annotation_paths = [Path(path) for path in DEFAULT_EXCLUDE_ANNOTATIONS]
    annotation_paths.extend(Path(path) for path in args.exclude_annotation_manifest)
    return result_paths, annotation_paths


def selected_for_args(
    args: argparse.Namespace,
) -> tuple[
    list[tuple[int, dict[str, Any]]],
    dict[int, dict[str, Any]],
    dict[str, Any],
    set[int],
    list[dict[str, Any]],
]:
    result_paths, annotation_paths = exclusion_inputs(args)
    excluded, ledger = calibration.load_exclusion_ledger(
        result_paths=result_paths,
        annotation_manifest_paths=annotation_paths,
        prefix=args.exclude_prefix,
    )
    selected, metadata, pool = select_balanced_samples(
        blind.load_samples("a", 0, 0),
        excluded_indices=excluded["a"],
        count=args.count,
        seed=args.seed,
        max_context_chars=args.max_context_chars,
    )
    return selected, metadata, pool, excluded["a"], ledger


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    selected, metadata, pool, excluded, ledger = selected_for_args(args)
    calibrator_path = Path(args.calibrator).resolve()
    calibrator_payload = json.loads(calibrator_path.read_text(encoding="utf-8"))
    calibrator = calibrator_payload["calibrator"]
    threshold = float(
        calibrator_payload["calibrated_threshold_selection"][
            "recommended_threshold"
        ]
    )
    rows = []
    generation_hits = 0
    for index, sample in selected:
        prompt = blind.build_prompt(
            sample,
            "a",
            SYSTEM,
            args.max_context_chars,
            args.confidence_threshold,
            args.max_nontax,
            args.implicit_confidence_threshold,
        )
        sample_id = str(sample.get("id") or index)
        cache = blind.cache_path(args.model, "a", SYSTEM, index, sample_id)
        generation_hit = blind.cache_has_reusable_generation(
            cache, prompt_sha256=blind._sha256(prompt)
        )
        generation_hits += int(generation_hit)
        row_metadata = metadata[index]
        rows.append(
            {
                "sample_index": index,
                "sample_id_sha256": blind._sha256(sample_id),
                "context_sha256": blind._sha256(
                    str(sample.get("context") or "")[: args.max_context_chars]
                ),
                "assigned_stratum": row_metadata["assigned_stratum"],
                "target_predicates": row_metadata["target_predicates"],
                "visible_target_gold_count": row_metadata[
                    "visible_nontax_gold"
                ],
                "visible_lexically_explicit_target_count": len(
                    set(TARGET_PREDICATES)
                    & set(row_metadata["visible_lexically_explicit_predicates"])
                ),
                "input_fields": ["context"],
                "protected_gold_field": blind.TASK_CONFIG["a"]["gold_key"],
                "prompt_sha256": blind._sha256(prompt),
                "prompt_chars": len(prompt),
                "generation_cache_hit_at_freeze": generation_hit,
            }
        )
    selected_indices = {int(row["sample_index"]) for row in rows}
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen_not_executed",
            "frozen_before_generation": generation_hits == 0,
            "contains_source_text": False,
            "gold_visible_to_model": False,
            "selection_uses_gold_locally": True,
            "selection_is_gold_stratified_not_unbiased": True,
            "claim_scope": "task_a_endpoint_visible_axiom_slice",
            "task": "a",
            "system": SYSTEM,
            "model": args.model,
            "seed": args.seed,
            "count": args.count,
            "target_predicates": list(TARGET_PREDICATES),
            "selection_profile": "balanced_endpoint_visible_axioms_v1",
            "exclude_prefix": args.exclude_prefix,
            "exclusion_ledger": ledger,
            "selected_overlap_with_exclusions": sorted(selected_indices & excluded),
            "disjoint_from_all_exclusions": not bool(selected_indices & excluded),
            "estimated_generation_cache_hits": generation_hits,
            "estimated_external_calls": len(rows) - generation_hits,
            "calibrator_artifact": str(calibrator_path),
            "calibrator_sha256": file_sha256(calibrator_path),
            "calibrator_status": calibrator["status"],
            "frozen_calibrated_confidence_threshold": threshold,
        },
        "locked_configuration": {
            "primary_system": "base_v5_guard_without_surface_or_reverse_expansion",
            "secondary_system": "same_generation_with_frozen_pooled_calibrator",
            "official_metric": "llms4ol2026_graph_similarity_exact",
            "relation_metric": "official_exact_nontax_micro",
            "surface_canonicalization": False,
            "bidirectional_serialization": False,
            "predicate_allowlist": None,
            "max_context_chars": args.max_context_chars,
            "confidence_threshold": args.confidence_threshold,
            "implicit_confidence_threshold": args.implicit_confidence_threshold,
            "max_implicit_gap_chars": args.max_implicit_gap_chars,
            "max_nontax": args.max_nontax,
            "max_completion_tokens": args.max_completion_tokens,
        },
        "preregistered_gate": {
            "successful_units_required": len(rows),
            "minimum_exact_positive_samples": 5,
            "minimum_exact_precision": 0.20,
            "calibrated_brier_must_not_exceed_raw": True,
            "all_conditions_required": True,
            "threshold_or_rule_retuning_after_freeze": False,
        },
        "pool": pool,
        "samples": rows,
    }


def preserve_or_write_frozen_manifest(
    path: Path, candidate: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    """Write once, then verify immutable protocol fields without overwriting."""
    if not path.exists():
        blind.atomic_write_json(path, candidate)
        return candidate, True
    existing = json.loads(path.read_text(encoding="utf-8"))
    immutable_meta = (
        "protocol",
        "model",
        "seed",
        "count",
        "selection_profile",
        "calibrator_sha256",
        "frozen_calibrated_confidence_threshold",
    )
    for key in immutable_meta:
        if existing.get("meta", {}).get(key) != candidate.get("meta", {}).get(key):
            raise ValueError(f"existing frozen manifest differs on meta.{key}")
    if existing.get("locked_configuration") != candidate.get("locked_configuration"):
        raise ValueError("existing frozen manifest has different locked configuration")
    if existing.get("preregistered_gate") != candidate.get("preregistered_gate"):
        raise ValueError("existing frozen manifest has different preregistered gate")
    row_fields = (
        "sample_index",
        "sample_id_sha256",
        "context_sha256",
        "assigned_stratum",
        "target_predicates",
        "prompt_sha256",
    )
    existing_rows = [
        {key: row.get(key) for key in row_fields}
        for row in existing.get("samples") or []
    ]
    candidate_rows = [
        {key: row.get(key) for key in row_fields}
        for row in candidate.get("samples") or []
    ]
    if existing_rows != candidate_rows:
        raise ValueError("existing frozen manifest has a different sample packet")
    return existing, False


def validate_manifest(
    manifest: dict[str, Any], args: argparse.Namespace
) -> list[tuple[int, dict[str, Any]]]:
    if manifest.get("meta", {}).get("protocol") != PROTOCOL:
        raise ValueError("confirmatory manifest protocol mismatch")
    if manifest.get("locked_configuration", {}).get("surface_canonicalization"):
        raise ValueError("surface canonicalization must remain disabled")
    if manifest.get("locked_configuration", {}).get("bidirectional_serialization"):
        raise ValueError("bidirectional serialization must remain disabled")
    locked = manifest["locked_configuration"]
    expected_runtime = {
        "max_context_chars": args.max_context_chars,
        "confidence_threshold": args.confidence_threshold,
        "implicit_confidence_threshold": args.implicit_confidence_threshold,
        "max_implicit_gap_chars": args.max_implicit_gap_chars,
        "max_nontax": args.max_nontax,
        "max_completion_tokens": args.max_completion_tokens,
    }
    for key, value in expected_runtime.items():
        if locked.get(key) != value:
            raise ValueError(f"runtime {key} does not match frozen manifest")
    if manifest["meta"].get("model") != args.model:
        raise ValueError("runtime model does not match frozen manifest")
    calibrator_path = Path(args.calibrator).resolve()
    if file_sha256(calibrator_path) != manifest["meta"]["calibrator_sha256"]:
        raise ValueError("calibrator artifact hash does not match frozen manifest")
    sample_map = dict(blind.load_samples("a", 0, 0))
    selected = []
    for row in manifest.get("samples") or []:
        index = int(row["sample_index"])
        sample = sample_map[index]
        sample_id = str(sample.get("id") or index)
        context = str(sample.get("context") or "")[: args.max_context_chars]
        if blind._sha256(sample_id) != row["sample_id_sha256"]:
            raise ValueError(f"sample {index} ID hash mismatch")
        if blind._sha256(context) != row["context_sha256"]:
            raise ValueError(f"sample {index} context hash mismatch")
        selected.append((index, sample))
    if len(selected) != int(manifest["meta"]["count"]):
        raise ValueError("confirmatory manifest count mismatch")
    return selected


def _result_payload(path: Path, *, mode: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("meta", {}).get("protocol") != PROTOCOL:
        raise ValueError(f"{path} protocol mismatch")
    if payload.get("meta", {}).get("mode") != mode:
        raise ValueError(f"{path} mode mismatch")
    return payload


def _result_semantic_sha256(result: dict[str, Any]) -> str:
    system_payload = result["tasks"]["task_a"]["systems"][SYSTEM]
    return payload_sha256(
        {
            "protocol": result["meta"]["protocol"],
            "mode": result["meta"]["mode"],
            "manifest_sha256": result["meta"]["manifest_sha256"],
            "selected_indices": result["manifest_snapshot"]["selected_indices"],
            "summary": system_payload["summary"],
            "samples": system_payload["samples"],
        }
    )


def _accepted_predictions(
    result: dict[str, Any],
) -> dict[tuple[str, int], set[tuple[str, str, str]]]:
    records = result["tasks"]["task_a"]["systems"][SYSTEM]["samples"]
    output: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        output[("a", int(record["sample_index"]))] = {
            tuple(candidate["triple"])
            for candidate in (
                record.get("guard_diagnostics", {}).get(
                    "accepted_nontax_candidates", []
                )
            )
        }
    return output


def _gold_by_sample(indices: list[int]) -> dict[tuple[str, int], set[tuple[str, str, str]]]:
    samples = dict(blind.load_samples("a", 0, 0))
    return {
        ("a", index): set(
            blind.categorize(
                blind.normalize_triples(
                    samples[index].get(blind.TASK_CONFIG["a"]["gold_key"]) or []
                )
            )["nontax"]
        )
        for index in indices
    }


def _positive_sample_count(result: dict[str, Any]) -> int:
    records = result["tasks"]["task_a"]["systems"][SYSTEM]["samples"]
    return sum(
        int(record.get("metrics", {}).get("nontax", {}).get("tp", 0)) > 0
        for record in records
        if record.get("status") == "ok"
    )


def score_confirmatory(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    base_path = Path(args.base_result).resolve()
    calibrated_path = Path(args.calibrated_result).resolve()
    calibrator_path = Path(args.calibrator).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = validate_manifest(manifest, args)
    indices = [index for index, _ in selected]
    base = _result_payload(base_path, mode="base")
    calibrated = _result_payload(calibrated_path, mode="calibrated")
    expected_indices = set(indices)
    for name, result in (("base", base), ("calibrated", calibrated)):
        actual = set(result["manifest_snapshot"]["selected_indices"])
        if actual != expected_indices:
            raise ValueError(f"{name} result sample set differs from manifest")
        if result["meta"]["manifest_sha256"] != file_sha256(manifest_path):
            raise ValueError(f"{name} result manifest hash mismatch")

    calibration_indices = {"a": set(indices)}
    candidates = offline.collect_candidates(
        base,
        source_system=SYSTEM,
        calibration_indices=calibration_indices,
        guard_diagnostics_from_result=True,
    )
    eligible = offline.deployment_eligible_candidates(candidates)
    calibrator_payload = json.loads(calibrator_path.read_text(encoding="utf-8"))
    frozen_calibrator = BetaIsotonicCalibrator.from_dict(
        calibrator_payload["calibrator"]
    )
    labels = [bool(row["canonical_correct"]) for row in eligible]
    raw_scores = [float(row["confidence"]) for row in eligible]
    calibrated_scores = [
        frozen_calibrator.calibrate(score) for score in raw_scores
    ]

    base_predictions = _accepted_predictions(base)
    calibrated_predictions = _accepted_predictions(calibrated)
    gold = _gold_by_sample(indices)
    transition = offline.exact_match_transition(
        base_predictions, calibrated_predictions, gold
    )
    base_summary = base["tasks"]["task_a"]["systems"][SYSTEM]["summary"]
    calibrated_summary = calibrated["tasks"]["task_a"]["systems"][SYSTEM][
        "summary"
    ]
    base_exact = dict(base_summary["nontax_f1_micro"])
    calibrated_exact = dict(calibrated_summary["nontax_f1_micro"])
    base_exact["positive_samples"] = _positive_sample_count(base)
    calibrated_exact["positive_samples"] = _positive_sample_count(calibrated)
    raw_calibration = calibration_metrics(raw_scores, labels)
    frozen_calibration = calibration_metrics(calibrated_scores, labels)

    gate = manifest["preregistered_gate"]
    checks = {
        "successful_units": (
            int(calibrated["meta"]["attempt_ledger"]["successful_units"])
            >= int(gate["successful_units_required"])
        ),
        "minimum_exact_positive_samples": (
            int(calibrated_exact["positive_samples"])
            >= int(gate["minimum_exact_positive_samples"])
        ),
        "minimum_exact_precision": (
            float(calibrated_exact["precision"])
            >= float(gate["minimum_exact_precision"])
        ),
        "calibrated_brier_must_not_exceed_raw": (
            float(frozen_calibration["brier"]) <= float(raw_calibration["brier"])
        ),
    }
    passed = all(checks.values())
    failed_checks = [name for name, value in checks.items() if not value]
    return {
        "meta": {
            "protocol": "v5_axiom_confirmatory_scoring_v1",
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "base_result": str(base_path),
            "base_result_semantic_sha256": _result_semantic_sha256(base),
            "calibrated_result": str(calibrated_path),
            "calibrated_result_semantic_sha256": _result_semantic_sha256(
                calibrated
            ),
            "calibrator_artifact": str(calibrator_path),
            "calibrator_sha256": file_sha256(calibrator_path),
            "api_calls_for_scoring": 0,
            "claim_scope": manifest["meta"]["claim_scope"],
        },
        "candidate_population": {
            "all": offline.summarize_candidates(candidates),
            "deployment_eligible": offline.summarize_candidates(eligible),
        },
        "official_exact": {
            "base_v5": base_exact,
            "frozen_calibrated_v5": calibrated_exact,
            "transition_from_base_to_calibrated": transition,
        },
        "candidate_calibration": {
            "raw": raw_calibration,
            "frozen_calibrator": frozen_calibration,
            "delta": {
                "brier": float(frozen_calibration["brier"])
                - float(raw_calibration["brier"]),
                "ece": float(frozen_calibration["ece"])
                - float(raw_calibration["ece"]),
            },
        },
        "preregistered_gate": {
            "requirements": gate,
            "checks": checks,
            "passed": passed,
            "failed_checks": failed_checks,
        },
        "confirmatory_evidence_valid": True,
        "scientific_deployable": passed,
        "confirmatory_eligible": passed,
        "status": "passed" if passed else "failed_closed",
        "reason": "ok" if passed else "+".join(failed_checks),
    }


def execute(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = validate_manifest(manifest, args)
    calibrated = args.mode == "calibrated"
    args.calibration_file = str(Path(args.calibrator).resolve()) if calibrated else None
    args.allowed_nontax_predicates = None
    args.calibrated_confidence_threshold = float(
        manifest["meta"]["frozen_calibrated_confidence_threshold"]
    )
    args.offset = 0
    args.limit = len(selected)
    checkpoint = RESULTS / "checkpoints" / (
        f"v5_confirmatory_{args.mode}_{blind._slug(args.backend)}_"
        f"taska_n{len(selected)}_seed{manifest['meta']['seed']}.json"
    )
    payload = blind.run_system(
        args=args,
        task="a",
        system=SYSTEM,
        indexed_samples=selected,
        checkpoint_path=checkpoint,
    )
    records = payload.get("samples") or []
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": (
                "complete_awaiting_locked_gate_scoring"
                if payload["summary"].get("complete")
                else "incomplete"
            ),
            "manifest": str(manifest_path),
            "manifest_sha256": file_sha256(manifest_path),
            "mode": args.mode,
            "task": "a",
            "system": SYSTEM,
            "model": args.model,
            "calibrator_artifact": args.calibration_file,
            "calibrated_confidence_threshold": args.calibrated_confidence_threshold,
            "api_calls_for_local_calibrated_replay": 0 if calibrated else None,
            "attempt_ledger": {
                "planned_units": len(records),
                "successful_units": sum(row.get("status") == "ok" for row in records),
                "failed_final_units": sum(row.get("status") != "ok" for row in records),
                "low_level_attempts": sum(int(row.get("attempts") or 0) for row in records),
            },
        },
        "manifest_snapshot": {
            "claim_scope": manifest["meta"]["claim_scope"],
            "locked_configuration": manifest["locked_configuration"],
            "preregistered_gate": manifest["preregistered_gate"],
            "selected_indices": [index for index, _ in selected],
        },
        "tasks": {
            "task_a": {
                "meta": {
                    "selected_indices": [index for index, _ in selected],
                    "n_samples": len(selected),
                },
                "systems": {SYSTEM: payload},
            }
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--mode", choices=["base", "calibrated"], default="base")
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--exclude-prefix", type=int, default=40)
    parser.add_argument("--additional-exclude-result", action="append", default=[])
    parser.add_argument("--exclude-annotation-manifest", action="append", default=[])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--calibrator", type=Path, default=DEFAULT_CALIBRATOR)
    parser.add_argument("--base-result", type=Path, default=DEFAULT_BASE_RESULT)
    parser.add_argument(
        "--calibrated-result", type=Path, default=DEFAULT_CALIBRATED_RESULT
    )
    parser.add_argument("--score-output", type=Path, default=DEFAULT_SCORE_OUTPUT)
    parser.add_argument("--backend", choices=["openai", "openai_compat", "api"], default="openai")
    parser.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--implicit-confidence-threshold", type=float, default=0.95)
    parser.add_argument("--max-implicit-gap-chars", type=int, default=600)
    parser.add_argument("--max-nontax", type=int, default=8)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be between 1 and 4")
    os.environ["ONTOBRIDG_MAX_TOKENS"] = str(args.max_completion_tokens)
    if args.audit_only:
        candidate_manifest = build_manifest(args)
        manifest, created = preserve_or_write_frozen_manifest(
            args.manifest, candidate_manifest
        )
        print(
            f"{'Frozen' if created else 'Verified existing frozen'} "
            f"samples={len(manifest['samples'])} "
            f"external_calls={manifest['meta']['estimated_external_calls']} "
            f"eligible_pool={manifest['pool']['eligible_endpoint_visible_target_samples']}",
            flush=True,
        )
        print(
            f"{'Wrote' if created else 'Preserved'} {args.manifest}", flush=True
        )
        return 0

    if args.score_only:
        score = score_confirmatory(args)
        blind.atomic_write_json(args.score_output, score)
        print(
            f"gate={score['status']} "
            f"base_tp={score['official_exact']['base_v5']['tp']} "
            f"calibrated_tp={score['official_exact']['frozen_calibrated_v5']['tp']} "
            f"positive_samples={score['official_exact']['frozen_calibrated_v5']['positive_samples']}",
            flush=True,
        )
        print(f"Wrote {args.score_output}", flush=True)
        return 0

    report = execute(args)
    selected_indices = report["manifest_snapshot"]["selected_indices"]
    manifest_seed = json.loads(Path(args.manifest).read_text(encoding="utf-8"))[
        "meta"
    ]["seed"]
    output = RESULTS / (
        f"v5_axiom_confirmatory_{args.mode}_a_n{len(selected_indices)}_"
        f"seed{manifest_seed}.json"
    )
    blind.atomic_write_json(output, report)
    ledger = report["meta"]["attempt_ledger"]
    print(
        f"mode={args.mode} ok={ledger['successful_units']}/{ledger['planned_units']} "
        f"attempts={ledger['low_level_attempts']}",
        flush=True,
    )
    print(f"Wrote {output}", flush=True)
    return 0 if ledger["failed_final_units"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
