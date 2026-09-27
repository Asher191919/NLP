"""Run guarded-v2 generation for a frozen blind annotation packet.

Raw generation is cached independently from human labels. Once a separately
adjudicated annotation artifact is available, the same command can fit and audit
the calibrator locally without another external request.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import relation_annotation_workflow as workflow
import run_official_blind_eval as blind
import run_relation_v2_offline as offline
from ontobridg.relations.calibration import BetaIsotonicCalibrator, calibration_metrics
from ontobridg.relations.canonicalization import (
    build_entity_inventory,
    canonicalize_triple,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "relation_human_calibration_v1"
SYSTEM = "open_vocab_guarded_v2"
DEFAULT_MANIFEST = (
    ROOT
    / "experiments"
    / "annotations"
    / workflow.PROTOCOL
    / "manifest.json"
)
DEFAULT_GROUPED_CV_FOLDS = 5
DEFAULT_GROUPED_CV_SEED = 20260924


def load_packet_samples(
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, list[tuple[int, dict[str, Any]]]], dict[tuple[str, int], str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The human and dual-model packets share sample geometry and source hashes, so
    # both protocols are accepted here. Provenance mode is decided separately by
    # --annotation-source and enforced again on the frozen artifact.
    accepted_protocols = {workflow.PROTOCOL, workflow.MODEL_PROTOCOL}
    if manifest.get("meta", {}).get("protocol") not in accepted_protocols:
        raise ValueError("annotation manifest protocol mismatch")
    expected_source_hashes = {
        str(row["task"]): str(row["sha256"])
        for row in manifest.get("source_files") or []
    }
    for task, path in workflow.SOURCE_PATHS.items():
        expected = expected_source_hashes.get(task)
        actual = workflow._file_sha256(path)
        if expected != actual:
            raise ValueError(f"task {task} source file hash does not match manifest")

    source_by_task = {
        task: dict(blind.load_samples(task, 0, 0)) for task in ("a", "b")
    }
    selected = {"a": [], "b": []}
    sample_keys: dict[tuple[str, int], str] = {}
    seen_indices = {"a": set(), "b": set()}
    for row in manifest.get("samples") or []:
        task = str(row["task"])
        index = int(row["source_index"])
        if task not in selected:
            raise ValueError(f"unsupported manifest task: {task}")
        if index in seen_indices[task]:
            raise ValueError(f"duplicate task {task} source index {index}")
        sample = source_by_task[task].get(index)
        if sample is None:
            raise ValueError(f"task {task} source index {index} is missing")
        sample_id = str(sample.get("id") or index)
        if blind._sha256(sample_id) != row["source_sample_id_sha256"]:
            raise ValueError(f"task {task} source index {index} ID hash mismatch")
        context = str(sample.get("context") or "")[: int(manifest["meta"]["max_context_chars"])]
        if blind._sha256(context) != row["context_sha256"]:
            raise ValueError(f"task {task} source index {index} context hash mismatch")
        selected[task].append((index, sample))
        sample_keys[(task, index)] = str(row["sample_key"])
        seen_indices[task].add(index)
    for task in selected:
        selected[task].sort(key=lambda item: item[0])
    if sum(len(rows) for rows in selected.values()) != int(manifest["meta"]["n_samples"]):
        raise ValueError("manifest sample count mismatch")
    return manifest, selected, sample_keys


def build_audit_manifest(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    manifest, selected, sample_keys = load_packet_samples(manifest_path)
    tasks: dict[str, Any] = {}
    generation_hits = 0
    exact_hits = 0
    prompt_chars = 0
    for task in ("a", "b"):
        rows = []
        for index, sample in selected[task]:
            prompt = blind.build_prompt(
                sample,
                task,
                SYSTEM,
                args.max_context_chars,
                args.confidence_threshold,
                args.max_nontax,
                args.implicit_confidence_threshold,
            )
            prompt_hash = blind._sha256(prompt)
            postprocess_hash = blind.postprocess_sha256(
                confidence_threshold=args.confidence_threshold,
                implicit_confidence_threshold=args.implicit_confidence_threshold,
                calibrated_confidence_threshold=args.calibrated_confidence_threshold,
                max_nontax=args.max_nontax,
                max_implicit_gap_chars=args.max_implicit_gap_chars,
                calibrator=None,
            )
            sample_id = str(sample.get("id") or index)
            cache = blind.cache_path(args.model, task, SYSTEM, index, sample_id)
            generation_hit = blind.cache_has_reusable_generation(
                cache, prompt_sha256=prompt_hash
            )
            exact_hit = blind.cache_is_replayable(
                cache,
                system=SYSTEM,
                prompt_sha256=prompt_hash,
                postprocess_hash=postprocess_hash,
            )
            generation_hits += int(generation_hit)
            exact_hits += int(exact_hit)
            prompt_chars += len(prompt)
            rows.append(
                {
                    "sample_key": sample_keys[(task, index)],
                    "source_index": index,
                    "source_sample_id_sha256": blind._sha256(sample_id),
                    "input_fields": (
                        ["context"]
                        if task == "a"
                        else [
                            "context",
                            "initial-primitive-ontology-triples",
                            "terms",
                            "types",
                        ]
                    ),
                    "protected_gold_field": blind.TASK_CONFIG[task]["gold_key"],
                    "context_chars_sent": min(
                        len(str(sample.get("context") or "")), args.max_context_chars
                    ),
                    "prompt_sha256": prompt_hash,
                    "generation_cache_hit": generation_hit,
                    "exact_cache_hit": exact_hit,
                }
            )
        tasks[f"task_{task}"] = {"n_samples": len(rows), "samples": rows}
    total = sum(len(rows) for rows in selected.values())
    return {
        "meta": {
            "protocol": (
                "relation_model_weaklabel_calibration_v1"
                if getattr(args, "annotation_source", "human") == "model"
                else PROTOCOL
            ),
            "annotation_source": getattr(args, "annotation_source", "human"),
            "confirmatory_eligible": (
                getattr(args, "annotation_source", "human") != "model"
            ),
            "claim_label": (
                workflow.MODEL_CLAIM_LABEL
                if getattr(args, "annotation_source", "human") == "model"
                else "independently human-annotated calibration"
            ),
            "audit_only": True,
            "packet_id": manifest["meta"]["packet_id"],
            "annotation_manifest_sha256": workflow._file_sha256(manifest_path),
            "system": SYSTEM,
            "model": args.model,
            "external_destination": (
                args.base_url
                or os.environ.get("ONTOBRIDG_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            "contains_source_text": False,
            "selection_uses_official_gold": False,
            "selection_uses_model_output": False,
            "protected_gold_visible_to_model": False,
            "annotation_status": (
                "not_applicable_dual_model_weak_labels"
                if getattr(args, "annotation_source", "human") == "model"
                else "pending_human_double_annotation"
            ),
            "estimated_calls": total,
            "estimated_generation_cache_hits": generation_hits,
            "estimated_exact_cache_replays": exact_hits,
            "estimated_local_reprocesses": generation_hits - exact_hits,
            "estimated_external_calls": total - generation_hits,
            "total_prompt_chars": prompt_chars,
        },
        "tasks": tasks,
    }


def _sanitize_generation_payload(
    payload: dict[str, Any], *, task: str, sample_keys: dict[tuple[str, int], str]
) -> dict[str, Any]:
    records = []
    for source in payload.get("samples") or []:
        record = {key: value for key, value in source.items() if key != "metrics"}
        index = int(record["sample_index"])
        record["sample_key"] = sample_keys[(task, index)]
        records.append(record)
    summary = payload.get("summary") or {}
    kept_summary_keys = {
        "target_n",
        "n_success",
        "n_failed",
        "complete",
        "successful_latency_sec",
        "successful_runner_attempts_total",
        "successful_samples_with_retry",
        "guard",
    }
    sanitized_summary = {
        key: value for key, value in summary.items() if key in kept_summary_keys
    }
    return {
        "meta": {
            **(payload.get("meta") or {}),
            "official_gold_metrics_discarded": True,
            "calibration_labels": "separate_frozen_human_annotation_only",
        },
        "summary": sanitized_summary,
        "samples": records,
    }


def load_frozen_annotations(
    path: Path,
    *,
    packet_id: str,
    expected_keys: set[str],
    annotation_source: str = "human",
) -> dict[str, set[tuple[str, str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    meta = payload.get("meta") or {}
    if annotation_source == "human":
        if meta.get("packet_id") != packet_id or not meta.get("calibration_ready"):
            raise ValueError("frozen annotation artifact is not ready for this packet")
    else:
        # Weak-label path. Accept only a dual-model adjudicated artifact that
        # explicitly declares it is not human-annotated and is exploratory-only.
        if meta.get("packet_id") != packet_id:
            raise ValueError("frozen annotation artifact is not ready for this packet")
        if meta.get("annotation_source") != workflow.MODEL_ANNOTATION_SOURCE_ID:
            raise ValueError(
                "model mode requires annotation_source="
                f"{workflow.MODEL_ANNOTATION_SOURCE_ID!r}"
            )
        if meta.get("human_annotated"):
            raise ValueError(
                "model mode refuses an artifact that declares human annotation"
            )
        if not meta.get("exploratory_calibration_ready"):
            raise ValueError(
                "frozen artifact is not marked exploratory_calibration_ready"
            )
    labels: dict[str, set[tuple[str, str, str]]] = {}
    for row in payload.get("samples") or []:
        key = str(row["sample_key"])
        if not row.get("annotation_complete"):
            raise ValueError(f"sample {key} is not annotation-complete")
        triples: set[tuple[str, str, str]] = set()
        for relation in row.get("relations") or []:
            triple = tuple(str(value) for value in relation.get("triple") or [])
            if len(triple) != 3:
                raise ValueError(f"sample {key} contains a malformed frozen relation")
            subject, predicate, obj = triple
            if predicate in workflow.SYMMETRIC_PREDICATES and obj < subject:
                subject, obj = obj, subject
            triples.add((subject, predicate, obj))
        labels[key] = triples
    if set(labels) != expected_keys:
        raise ValueError("frozen annotation sample keys do not match the packet")
    return labels


def collect_candidates(
    *,
    selected: dict[str, list[tuple[int, dict[str, Any]]]],
    generation: dict[str, Any],
    labels: dict[str, set[tuple[str, str, str]]],
) -> list[dict[str, Any]]:
    source_samples = {
        (task, index): sample
        for task, rows in selected.items()
        for index, sample in rows
    }
    candidates: list[dict[str, Any]] = []
    for task_key, task_payload in generation["tasks"].items():
        task = task_key.removeprefix("task_")
        for record in task_payload.get("samples") or []:
            if record.get("status") != "ok":
                continue
            index = int(record["sample_index"])
            sample_key = str(record["sample_key"])
            sample = source_samples[(task, index)]
            context = str(sample.get("context") or "")
            inventory = build_entity_inventory(sample, task=task)
            cached = json.loads(Path(record["cache"]).read_text(encoding="utf-8"))
            guard = cached.get("guard_diagnostics") or {}
            diagnostics_by_position = {
                int(row["position"]): row
                for name in ("accepted_nontax_candidates", "rejected_nontax_candidates")
                for row in guard.get(name, [])
                if row.get("position") is not None
            }
            accepted_positions = {
                int(row["position"])
                for row in guard.get("accepted_nontax_candidates", [])
                if row.get("position") is not None
            }
            rejected = {
                int(row["position"]): list(row.get("reasons") or [])
                for row in guard.get("rejected_nontax_candidates", [])
                if row.get("position") is not None
            }
            for position, item in enumerate(blind.parse_model_items(cached["raw_text"])):
                if not isinstance(item, dict):
                    continue
                normalized = blind.normalize_triples([item])
                if not normalized:
                    continue
                canonical, trace = canonicalize_triple(
                    normalized[0], context=context, entity_inventory=inventory
                )
                if canonical[1] in workflow.CORE_PREDICATES:
                    continue
                try:
                    confidence = float(item.get("confidence"))
                except (TypeError, ValueError):
                    continue
                match_key = tuple(canonical)
                if (
                    match_key[1] in workflow.SYMMETRIC_PREDICATES
                    and match_key[2] < match_key[0]
                ):
                    match_key = (match_key[2], match_key[1], match_key[0])
                candidate_diagnostics = diagnostics_by_position.get(position, {})
                candidates.append(
                    {
                        "task": task,
                        "sample_index": index,
                        "sample_key": sample_key,
                        "stratum": None,
                        "split": "calibration",
                        "confidence": confidence,
                        "raw_triple": normalized[0],
                        "canonical_triple": canonical,
                        "raw_correct": tuple(normalized[0]) in labels[sample_key],
                        "canonical_correct": match_key in labels[sample_key],
                        "canonicalization_changed": bool(trace["changed"]),
                        "near_match_type": "not_computed_for_human_open_inventory",
                        "near_gold_triples": [],
                        "evidence_mode": blind._norm(item.get("evidence_mode") or ""),
                        "evidence_gap_chars": candidate_diagnostics.get(
                            "evidence_gap_chars"
                        ),
                        "postprocess_status": (
                            "accepted"
                            if position in accepted_positions
                            else "rejected"
                            if position in rejected
                            else "not_recorded"
                        ),
                        "rejection_reasons": rejected.get(position, []),
                    }
                )
    return candidates


def assign_grouped_calibration_folds(
    eligible: list[dict[str, Any]],
    *,
    n_folds: int = DEFAULT_GROUPED_CV_FOLDS,
    seed: int = DEFAULT_GROUPED_CV_SEED,
) -> dict[str, int]:
    """Assign complete sample groups to deterministic, approximately balanced folds."""
    if n_folds < 2:
        raise ValueError("grouped calibration requires at least two folds")

    grouped: dict[str, dict[str, Any]] = {}
    for row in eligible:
        sample_key = str(row.get("sample_key") or "")
        task = str(row.get("task") or "")
        if not sample_key:
            raise ValueError("eligible calibration candidate is missing sample_key")
        if not task:
            raise ValueError(f"sample {sample_key} is missing task")
        group = grouped.setdefault(
            sample_key,
            {"sample_key": sample_key, "task": task, "candidates": 0, "positives": 0},
        )
        if group["task"] != task:
            raise ValueError(f"sample {sample_key} appears under multiple tasks")
        group["candidates"] += 1
        group["positives"] += int(bool(row.get("canonical_correct")))

    if len(grouped) < n_folds:
        raise ValueError(
            f"grouped calibration has {len(grouped)} sample groups for {n_folds} folds"
        )

    def stable_tie(value: str) -> str:
        return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()

    groups = sorted(
        grouped.values(),
        key=lambda group: (
            -int(group["positives"] > 0),
            -int(group["positives"]),
            -int(group["candidates"]),
            stable_tie(str(group["sample_key"])),
            str(group["sample_key"]),
        ),
    )
    folds = [
        {
            "fold": fold,
            "samples": 0,
            "candidates": 0,
            "positive_samples": 0,
            "positives": 0,
            "tasks": Counter(),
        }
        for fold in range(n_folds)
    ]
    assignments: dict[str, int] = {}
    for group in groups:
        task = str(group["task"])
        has_positive = int(group["positives"] > 0)

        def placement_key(fold: dict[str, Any]) -> tuple[Any, ...]:
            if has_positive:
                balance = (
                    fold["positive_samples"],
                    fold["tasks"][task],
                    fold["candidates"],
                    fold["positives"],
                    fold["samples"],
                )
            else:
                balance = (
                    fold["candidates"],
                    fold["tasks"][task],
                    fold["samples"],
                    fold["positive_samples"],
                )
            return (*balance, stable_tie(f"fold:{fold['fold']}"), fold["fold"])

        chosen = min(folds, key=placement_key)
        assignments[str(group["sample_key"])] = int(chosen["fold"])
        chosen["samples"] += 1
        chosen["candidates"] += int(group["candidates"])
        chosen["positive_samples"] += has_positive
        chosen["positives"] += int(group["positives"])
        chosen["tasks"][task] += 1
    return assignments


def grouped_cross_validation_report(
    candidates: list[dict[str, Any]],
    *,
    n_folds: int = DEFAULT_GROUPED_CV_FOLDS,
    seed: int = DEFAULT_GROUPED_CV_SEED,
    exploratory: bool = False,
) -> dict[str, Any]:
    """Fit per-fold calibrators and score every eligible candidate out of fold."""
    eligible = offline.deployment_eligible_candidates(candidates)
    assignments = assign_grouped_calibration_folds(
        eligible, n_folds=n_folds, seed=seed
    )
    all_sample_keys = set(assignments)
    positive_sample_keys = {
        str(row["sample_key"])
        for row in eligible
        if bool(row.get("canonical_correct"))
    }
    pooled_raw: list[float] = []
    pooled_calibrated: list[float] = []
    pooled_labels: list[bool] = []
    out_of_fold_predictions: list[dict[str, Any]] = []
    fold_reports: list[dict[str, Any]] = []

    for fold in range(n_folds):
        validation_keys = {
            sample_key
            for sample_key, assigned_fold in assignments.items()
            if assigned_fold == fold
        }
        train_keys = all_sample_keys - validation_keys
        train = [row for row in eligible if str(row["sample_key"]) in train_keys]
        validation = [
            row for row in eligible if str(row["sample_key"]) in validation_keys
        ]
        calibrator = BetaIsotonicCalibrator().fit(
            [row["confidence"] for row in train],
            [bool(row["canonical_correct"]) for row in train],
        )
        raw_scores = [float(row["confidence"]) for row in validation]
        labels = [bool(row["canonical_correct"]) for row in validation]
        calibrated_scores = [calibrator.calibrate(score) for score in raw_scores]
        pooled_raw.extend(raw_scores)
        pooled_calibrated.extend(calibrated_scores)
        pooled_labels.extend(labels)
        out_of_fold_predictions.extend(
            {
                "sample_key": str(row["sample_key"]),
                "task": str(row["task"]),
                "fold": fold,
                "raw_confidence": raw_score,
                "calibrated_confidence": calibrated_score,
                "weak_label": label,
            }
            for row, raw_score, calibrated_score, label in zip(
                validation, raw_scores, calibrated_scores, labels
            )
        )
        train_tasks = Counter(
            str(row["task"])
            for row in train
            if str(row["sample_key"]) in train_keys
        )
        validation_tasks = Counter(str(row["task"]) for row in validation)
        fold_reports.append(
            {
                "fold": fold,
                "train_sample_keys": sorted(train_keys),
                "validation_sample_keys": sorted(validation_keys),
                "sample_overlap_count": len(train_keys & validation_keys),
                "train_samples": len(train_keys),
                "validation_samples": len(validation_keys),
                "train_candidates": len(train),
                "validation_candidates": len(validation),
                "train_positives": sum(
                    bool(row["canonical_correct"]) for row in train
                ),
                "validation_positives": sum(labels),
                "train_positive_samples": len(train_keys & positive_sample_keys),
                "validation_positive_samples": len(
                    validation_keys & positive_sample_keys
                ),
                "train_candidate_counts_by_task": dict(sorted(train_tasks.items())),
                "validation_candidate_counts_by_task": dict(
                    sorted(validation_tasks.items())
                ),
                "calibrator_status": calibrator.status,
                "calibrator": calibrator.to_dict(),
                "held_out_metrics": {
                    "raw": calibration_metrics(raw_scores, labels),
                    "calibrated": calibration_metrics(calibrated_scores, labels),
                },
            }
        )

    raw_metrics = calibration_metrics(pooled_raw, pooled_labels)
    calibrated_metrics = calibration_metrics(pooled_calibrated, pooled_labels)
    all_fold_calibrators_deployable = all(
        row["calibrator"]["deployable"] for row in fold_reports
    )
    fold_block_counts = [
        len(row["calibrator"]["blocks"]) for row in fold_reports
    ]
    all_single_probability_block = all(count == 1 for count in fold_block_counts)
    sample_support = (
        len(positive_sample_keys) >= offline.MIN_DEPLOYMENT_POSITIVE_SAMPLES
    )
    deployable = all_fold_calibrators_deployable and sample_support and not exploratory
    return {
        "protocol": "sample_grouped_calibration_cv_v1",
        "fit_scope": "other_sample_groups_only",
        "evaluation_scope": "pooled_out_of_fold_candidates",
        "n_folds": n_folds,
        "seed": seed,
        "n_sample_groups": len(all_sample_keys),
        "n_positive_sample_groups": len(positive_sample_keys),
        "fold_assignment": dict(sorted(assignments.items())),
        "folds": fold_reports,
        "out_of_fold_predictions": out_of_fold_predictions,
        "coverage": {
            "eligible_candidates": len(eligible),
            "out_of_fold_predictions": len(pooled_labels),
            "each_candidate_scored_once": len(pooled_labels) == len(eligible),
            "sample_group_overlap_count": sum(
                int(row["sample_overlap_count"]) for row in fold_reports
            ),
        },
        "pooled_out_of_fold_metrics": {
            "raw": raw_metrics,
            "calibrated": calibrated_metrics,
            "delta": {
                "brier": float(calibrated_metrics["brier"])
                - float(raw_metrics["brier"]),
                "ece": float(calibrated_metrics["ece"])
                - float(raw_metrics["ece"]),
            },
        },
        "fold_status_counts": dict(
            sorted(Counter(row["calibrator_status"] for row in fold_reports).items())
        ),
        "calibration_shape": {
            "fold_block_counts": fold_block_counts,
            "all_folds_single_probability_block": all_single_probability_block,
            "learned_confidence_ranking": not all_single_probability_block,
            "interpretation": (
                "base_rate_shrinkage_without_confidence_ranking"
                if all_single_probability_block
                else "at_least_one_fold_retains_score_resolution"
            ),
        },
        "deployment_support": {
            "deployable": deployable,
            "candidate_level_deployable": all_fold_calibrators_deployable,
            "confirmatory_eligible": not exploratory,
            "positive_samples": len(positive_sample_keys),
            "min_positive_samples": offline.MIN_DEPLOYMENT_POSITIVE_SAMPLES,
            "status": (
                "exploratory_weak_labels_confirmatory_gate_closed"
                if exploratory
                else "ok"
                if deployable
                else "insufficient_cross_validated_support"
            ),
        },
    }


def build_calibration_report(
    candidates: list[dict[str, Any]],
    *,
    exploratory: bool = False,
    grouped_cv_folds: int = DEFAULT_GROUPED_CV_FOLDS,
    grouped_cv_seed: int = DEFAULT_GROUPED_CV_SEED,
) -> dict[str, Any]:
    eligible = offline.deployment_eligible_candidates(candidates)
    calibrator = BetaIsotonicCalibrator().fit(
        [row["confidence"] for row in eligible],
        [bool(row["canonical_correct"]) for row in eligible],
    )
    positive_samples = len(
        {row["sample_key"] for row in eligible if row["canonical_correct"]}
    )
    deployable = (
        calibrator.deployable
        and positive_samples >= offline.MIN_DEPLOYMENT_POSITIVE_SAMPLES
    )
    calibrated_scores = [calibrator.calibrate(row["confidence"]) for row in eligible]
    if exploratory:
        # Weak labels can never open the confirmatory gate, regardless of how the
        # numeric calibration turns out. Report the numbers, refuse the promotion.
        deployable = False
    return {
        "candidate_summary": {
            "all": offline.summarize_candidates(candidates),
            "deployment_population": offline.summarize_candidates(eligible),
        },
        "calibrator": calibrator.to_dict(),
        "deployment_support": {
            "deployable": deployable,
            "candidate_level_deployable": calibrator.deployable,
            "confirmatory_eligible": (not exploratory),
            "claim_label": (
                workflow.MODEL_CLAIM_LABEL
                if exploratory
                else "independently human-annotated calibration"
            ),
            "status": (
                "exploratory_weak_labels_confirmatory_gate_closed"
                if exploratory
                else "ok"
                if deployable
                else "insufficient_independent_positive_samples"
                if calibrator.deployable
                else calibrator.status
            ),
            "min_positive_samples": offline.MIN_DEPLOYMENT_POSITIVE_SAMPLES,
            "positive_candidates": sum(
                bool(row["canonical_correct"]) for row in eligible
            ),
            "positive_samples": positive_samples,
        },
        "calibration_metrics": {
            "raw": calibration_metrics(
                [row["confidence"] for row in eligible],
                [bool(row["canonical_correct"]) for row in eligible],
            ),
            "calibrated": calibration_metrics(
                calibrated_scores,
                [bool(row["canonical_correct"]) for row in eligible],
            ),
        },
        "grouped_cross_validation": grouped_cross_validation_report(
            candidates,
            n_folds=grouped_cv_folds,
            seed=grouped_cv_seed,
            exploratory=exploratory,
        ),
        "threshold_selection": offline.calibration_threshold_sweep(
            candidates, calibrator
        ),
        "predicate_policy": offline.calibration_predicate_policy(candidates),
        "exact_positive_candidates": [
            row for row in candidates if row["canonical_correct"]
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--frozen-annotations", type=Path, default=None)
    parser.add_argument(
        "--annotation-source",
        choices=workflow.ANNOTATION_SOURCES,
        default="human",
        help=(
            "human: require a human-adjudicated frozen artifact (confirmatory-"
            "eligible). model: require a dual-model adjudicated weak-label "
            "artifact; results are exploratory only and never confirmatory."
        ),
    )
    parser.add_argument("--backend", choices=["openai", "openai_compat", "api"], default="openai")
    parser.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--prior-failed-unit-attempts", type=int, default=0)
    parser.add_argument("--prior-failed-low-level-attempts", type=int, default=0)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--confidence-threshold", type=float, default=blind.DEFAULT_CONFIDENCE_THRESHOLD)
    parser.add_argument(
        "--implicit-confidence-threshold",
        type=float,
        default=blind.DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument(
        "--calibrated-confidence-threshold",
        type=float,
        default=blind.DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD,
    )
    parser.add_argument("--max-nontax", type=int, default=blind.DEFAULT_MAX_NONTAX)
    parser.add_argument(
        "--grouped-cv-folds", type=int, default=DEFAULT_GROUPED_CV_FOLDS
    )
    parser.add_argument(
        "--grouped-cv-seed", type=int, default=DEFAULT_GROUPED_CV_SEED
    )
    parser.add_argument(
        "--max-implicit-gap-chars",
        type=int,
        default=blind.DEFAULT_MAX_IMPLICIT_GAP_CHARS,
    )
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be between 1 and 4")
    if args.grouped_cv_folds < 2:
        raise SystemExit("--grouped-cv-folds must be at least 2")
    if args.prior_failed_unit_attempts < 0 or args.prior_failed_low_level_attempts < 0:
        raise SystemExit("prior failed attempt counts must be non-negative")
    args.calibration_file = None
    exploratory = args.annotation_source == "model"
    prefix = (
        "model_weaklabel_relation_calibration"
        if exploratory
        else "human_relation_calibration"
    )
    manifest, selected, sample_keys = load_packet_samples(Path(args.manifest))
    packet_id = str(manifest["meta"]["packet_id"])
    tag = f"{packet_id}_n{manifest['meta']['n_samples']}"
    if args.audit_only:
        audit = build_audit_manifest(args)
        output = blind.RESULTS / f"{prefix}_audit_{SYSTEM}_{tag}.json"
        blind.atomic_write_json(output, audit)
        print(
            f"Audit only: samples={audit['meta']['estimated_calls']} "
            f"external_calls={audit['meta']['estimated_external_calls']}"
        )
        print(f"Wrote {output}")
        return 0

    generation: dict[str, Any] = {
        "meta": {
            "protocol": PROTOCOL,
            "packet_id": packet_id,
            "annotation_manifest": str(args.manifest),
            "annotation_manifest_sha256": workflow._file_sha256(Path(args.manifest)),
            "system": SYSTEM,
            "model": args.model,
            "official_gold_used_for_calibration": False,
            "annotation_source": args.annotation_source,
            "confirmatory_eligible": not exploratory,
            "claim_label": (
                workflow.MODEL_CLAIM_LABEL
                if exploratory
                else "independently human-annotated calibration"
            ),
            "human_annotation_status": (
                "frozen" if args.frozen_annotations and not exploratory
                else "not_applicable_dual_model_weak_labels" if exploratory
                else "pending"
            ),
        },
        "tasks": {},
    }
    failed = False
    for task in ("a", "b"):
        args.offset = 0
        args.limit = len(selected[task])
        checkpoint = (
            blind.RESULTS
            / "checkpoints"
            / f"{prefix}_{blind._slug(args.backend)}_task{task}_{SYSTEM}_{tag}.json"
        )
        payload = blind.run_system(
            args=args,
            task=task,
            system=SYSTEM,
            indexed_samples=selected[task],
            checkpoint_path=checkpoint,
        )
        sanitized = _sanitize_generation_payload(
            payload, task=task, sample_keys=sample_keys
        )
        generation["tasks"][f"task_{task}"] = sanitized
        failed = failed or not bool(sanitized["summary"].get("complete"))
        print(
            f"Task {task.upper()}: ok={sanitized['summary'].get('n_success', 0)}/"
            f"{sanitized['summary'].get('target_n', len(selected[task]))} "
            f"raw={sanitized['summary'].get('guard', {}).get('raw_nontax', 0)} "
            f"accepted={sanitized['summary'].get('guard', {}).get('accepted_nontax', 0)}",
            flush=True,
        )
    result: dict[str, Any] = {
        "meta": generation["meta"],
        "generation": generation,
        "calibration": {
            "status": "awaiting_frozen_human_annotations",
            "deployable": False,
        },
    }
    final_records = [
        record
        for task_payload in generation["tasks"].values()
        for record in task_payload.get("samples") or []
    ]
    successful_units = sum(record.get("status") == "ok" for record in final_records)
    failed_units = len(final_records) - successful_units
    result["meta"]["attempt_ledger"] = {
        "planned_units": len(final_records),
        "successful_units": successful_units,
        "failed_final_units": failed_units,
        "prior_failed_unit_attempts": args.prior_failed_unit_attempts,
        "unit_attempts_including_prior_failures": (
            len(final_records) + args.prior_failed_unit_attempts
        ),
        "prior_failed_low_level_attempts": args.prior_failed_low_level_attempts,
        "final_recorded_runner_attempts": sum(
            int(record.get("attempts") or 0) for record in final_records
        ),
        "low_level_attempts_including_prior_failures": (
            sum(int(record.get("attempts") or 0) for record in final_records)
            + args.prior_failed_low_level_attempts
        ),
    }
    token_caps = Counter()
    for record in final_records:
        if record.get("status") != "ok":
            continue
        cache_payload = json.loads(Path(record["cache"]).read_text(encoding="utf-8"))
        token_cap = (cache_payload.get("generation_config") or {}).get(
            "max_completion_tokens"
        )
        token_caps[str(token_cap if token_cap is not None else "unknown")] += 1
    result["meta"]["generation_max_completion_token_counts"] = dict(
        sorted(token_caps.items())
    )
    if args.frozen_annotations:
        expected_keys = set(sample_keys.values())
        labels = load_frozen_annotations(
            args.frozen_annotations,
            packet_id=packet_id,
            expected_keys=expected_keys,
            annotation_source=args.annotation_source,
        )
        candidates = collect_candidates(
            selected=selected, generation=generation, labels=labels
        )
        result["meta"]["frozen_annotations"] = str(args.frozen_annotations)
        result["meta"]["frozen_annotations_sha256"] = workflow._file_sha256(
            args.frozen_annotations
        )
        result["calibration"] = build_calibration_report(
            candidates,
            exploratory=exploratory,
            grouped_cv_folds=args.grouped_cv_folds,
            grouped_cv_seed=args.grouped_cv_seed,
        )
    result["meta"]["protocol"] = (
        "relation_model_weaklabel_calibration_v1"
        if exploratory
        else PROTOCOL
    )
    output = blind.RESULTS / f"{prefix}_{SYSTEM}_{tag}.json"
    blind.atomic_write_json(output, result)
    print(f"Wrote {output}")
    if args.frozen_annotations:
        grouped_cv_output = blind.RESULTS / f"{prefix}_grouped_cv_{tag}.json"
        grouped_cv_artifact = {
            "meta": {
                "protocol": "sample_grouped_calibration_cv_v1",
                "source_artifact": str(output),
                "system": SYSTEM,
                "model": args.model,
                "packet_id": packet_id,
                "annotation_source": args.annotation_source,
                "claim_label": result["meta"]["claim_label"],
                "official_gold_used_for_calibration": False,
                "frozen_annotations": str(args.frozen_annotations),
                "frozen_annotations_sha256": result["meta"][
                    "frozen_annotations_sha256"
                ],
                "additional_external_calls_for_grouped_cv": 0,
                "deployable": result["calibration"]["grouped_cross_validation"][
                    "deployment_support"
                ]["deployable"],
                "confirmatory_eligible": result["calibration"][
                    "grouped_cross_validation"
                ]["deployment_support"]["confirmatory_eligible"],
            },
            "grouped_cross_validation": result["calibration"][
                "grouped_cross_validation"
            ],
        }
        blind.atomic_write_json(grouped_cv_output, grouped_cv_artifact)
        print(f"Wrote {grouped_cv_output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
