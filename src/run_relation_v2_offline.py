"""Offline canonicalization and calibration audit over cached guarded predictions.

This script makes no API calls. Gold is used only to label candidates after a deterministic
sample-level calibration/held-out split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind
from ontobridg.relations.calibration import BetaIsotonicCalibrator, calibration_metrics
from ontobridg.relations.canonicalization import (
    CANONICALIZATION_VERSION,
    build_entity_inventory,
    canonicalize_triple,
)


ROOT = Path(__file__).resolve().parents[1]
MIN_DEPLOYMENT_POSITIVE_SAMPLES = 5
MIN_PROJECTION_POSITIVE_SAMPLES = 5
MIN_PROJECTION_PRECISION = 0.80
STRUCTURAL_PROJECTION_VERSION = "structural_projection_audit_v1"
SYMMETRIC_PREDICATES = frozenset({"disjoint with", "equivalent class"})
SURFACE_MORPHOLOGY_VERSION = "relation_surface_morphology_v1"
_SINGULAR_INVARIANTS = frozenset({"news", "series", "species"})
DEFAULT_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "official_stratified_open_openai_ab_s5_seed20260922.json"
)


def merge_source_results(
    sources: list[tuple[Path, dict[str, Any]]], *, source_system: str
) -> dict[str, Any]:
    """Union calibration results by task and sample index."""
    if len(sources) == 1:
        return sources[0][1]
    records_by_task: dict[str, dict[int, dict[str, Any]]] = {}
    for path, result in sources:
        for task_key, task_payload in result.get("tasks", {}).items():
            try:
                records = task_payload["systems"][source_system]["samples"]
            except KeyError as exc:
                raise ValueError(
                    f"{path} does not contain {task_key}/{source_system} samples"
                ) from exc
            target = records_by_task.setdefault(task_key, {})
            for record in records:
                index = int(record["sample_index"])
                existing = target.get(index)
                if (
                    existing is not None
                    and existing.get("cache") != record.get("cache")
                ):
                    raise ValueError(
                        f"conflicting cache records for {task_key} sample {index}"
                    )
                target[index] = record
    return {
        "meta": {
            "merged_sources": [str(path.resolve()) for path, _ in sources],
        },
        "tasks": {
            task_key: {
                "meta": {
                    "selected_indices": sorted(records),
                    "n_samples": len(records),
                },
                "systems": {
                    source_system: {
                        "samples": [records[index] for index in sorted(records)]
                    }
                },
            }
            for task_key, records in sorted(records_by_task.items())
        },
    }


def _rank(seed: int, task: str, stratum: str, index: int) -> str:
    return hashlib.sha256(f"{seed}:{task}:{stratum}:{index}".encode("utf-8")).hexdigest()


def choose_calibration_indices(
    result: dict[str, Any], *, source_system: str, per_stratum: int, seed: int
) -> dict[str, set[int]]:
    chosen: dict[str, set[int]] = {}
    for task_key, task_payload in result["tasks"].items():
        task = task_key.removeprefix("task_")
        records = task_payload["systems"][source_system]["samples"]
        selected: set[int] = set()
        for stratum in ("low", "medium", "high"):
            candidates = [
                int(record["sample_index"])
                for record in records
                if record.get("stratum") == stratum and record.get("status") == "ok"
            ]
            candidates.sort(key=lambda index: _rank(seed, task, stratum, index))
            if len(candidates) < per_stratum:
                raise ValueError(
                    f"{task_key}/{stratum} has {len(candidates)} records; need {per_stratum}"
                )
            selected.update(candidates[:per_stratum])
        chosen[task] = selected
    return chosen


def choose_all_successful_indices(
    result: dict[str, Any], *, source_system: str
) -> dict[str, set[int]]:
    return {
        task_key.removeprefix("task_"): {
            int(record["sample_index"])
            for record in task_payload["systems"][source_system]["samples"]
            if record.get("status") == "ok"
        }
        for task_key, task_payload in result["tasks"].items()
    }


def classify_near_match(
    triple: list[str], gold: set[tuple[str, str, str]]
) -> tuple[str, list[list[str]]]:
    candidate = tuple(triple)
    if candidate in gold:
        return "exact", [list(candidate)]
    same_endpoints = [
        list(row) for row in gold if row[0] == candidate[0] and row[2] == candidate[2]
    ]
    if same_endpoints:
        return "endpoints_exact_predicate_mismatch", sorted(same_endpoints)
    reversed_endpoints = [
        list(row) for row in gold if row[0] == candidate[2] and row[2] == candidate[0]
    ]
    if reversed_endpoints:
        if candidate[1] in SYMMETRIC_PREDICATES and any(
            row[1] == candidate[1] for row in reversed_endpoints
        ):
            return "symmetric_endpoints_reversed", sorted(reversed_endpoints)
        return "endpoints_reversed", sorted(reversed_endpoints)
    same_predicate_subject = [
        list(row) for row in gold if row[0] == candidate[0] and row[1] == candidate[1]
    ]
    if same_predicate_subject:
        return "predicate_subject_exact", sorted(same_predicate_subject)
    same_predicate_object = [
        list(row) for row in gold if row[1] == candidate[1] and row[2] == candidate[2]
    ]
    if same_predicate_object:
        return "predicate_object_exact", sorted(same_predicate_object)
    return "none", []


def singularize_entity_surface(entity: str) -> str:
    """Conservatively singularize the final English token for offline audit only."""
    words = entity.split()
    if not words:
        return entity
    match = re.fullmatch(r"([a-z0-9]+)([^a-z0-9]*)", words[-1])
    if match is None:
        return entity
    token, suffix = match.groups()
    singular = token
    if (
        len(token) <= 4
        or token in _SINGULAR_INVARIANTS
        or token.endswith(("ss", "us", "is", "ics"))
    ):
        return entity
    if token.endswith("ies"):
        singular = f"{token[:-3]}y"
    elif token.endswith("s") and not token.endswith("es"):
        singular = token[:-1]
    if singular == token:
        return entity
    words[-1] = f"{singular}{suffix}"
    return " ".join(words)


def surface_canonicalize_triple(
    triple: tuple[str, str, str] | list[str],
) -> tuple[str, str, str]:
    subject, predicate, obj = tuple(triple)
    return (
        singularize_entity_surface(subject),
        predicate,
        singularize_entity_surface(obj),
    )


def relation_match_key(
    triple: tuple[str, str, str] | list[str],
    *,
    symmetric_order_invariant: bool,
    surface_form_invariant: bool = False,
) -> tuple[str, str, str]:
    """Return an evaluation key without changing the serialized prediction."""
    subject, predicate, obj = tuple(triple)
    if surface_form_invariant:
        subject = singularize_entity_surface(subject)
        obj = singularize_entity_surface(obj)
    if symmetric_order_invariant and predicate in SYMMETRIC_PREDICATES:
        subject, obj = sorted((subject, obj))
    return subject, predicate, obj


def relation_set_metrics(
    predicted_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
    gold_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
    *,
    symmetric_order_invariant: bool = False,
    surface_form_invariant: bool = False,
) -> dict[str, Any]:
    """Compute sample-keyed micro metrics, optionally ignoring symmetric ordering."""
    predicted = {
        (
            *sample_key,
            *relation_match_key(
                triple,
                symmetric_order_invariant=symmetric_order_invariant,
                surface_form_invariant=surface_form_invariant,
            ),
        )
        for sample_key, triples in predicted_by_sample.items()
        for triple in triples
    }
    gold = {
        (
            *sample_key,
            *relation_match_key(
                triple,
                symmetric_order_invariant=symmetric_order_invariant,
                surface_form_invariant=surface_form_invariant,
            ),
        )
        for sample_key, triples in gold_by_sample.items()
        for triple in triples
    }
    matches = predicted & gold
    tp = len(matches)
    fp = len(predicted - gold)
    fn = len(gold - predicted)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "predictions": len(predicted),
        "gold": len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "positive_samples": len({(row[0], row[1]) for row in matches}),
    }


def canonicalize_prediction_surfaces(
    predicted_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
) -> dict[tuple[str, int], set[tuple[str, str, str]]]:
    return {
        key: {surface_canonicalize_triple(triple) for triple in triples}
        for key, triples in predicted_by_sample.items()
    }


def cap_aware_bidirectional_predictions(
    ranked_predictions_by_sample: dict[
        tuple[str, int], list[tuple[str, str, str]]
    ],
    *,
    max_per_sample: int,
    surface_canonicalization: bool,
) -> tuple[dict[tuple[str, int], set[tuple[str, str, str]]], int]:
    """Preserve base predictions, then spend remaining capacity on reverse edges."""
    output: dict[tuple[str, int], set[tuple[str, str, str]]] = {}
    added_reversals = 0
    for key, ranked in ranked_predictions_by_sample.items():
        ordered: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        transformed = [
            surface_canonicalize_triple(triple)
            if surface_canonicalization
            else triple
            for triple in ranked
        ]
        for triple in transformed:
            if triple not in seen and len(ordered) < max_per_sample:
                ordered.append(triple)
                seen.add(triple)
        for subject, predicate, obj in transformed:
            if len(ordered) >= max_per_sample:
                break
            reverse = (obj, predicate, subject)
            if predicate in SYMMETRIC_PREDICATES and reverse not in seen:
                ordered.append(reverse)
                seen.add(reverse)
                added_reversals += 1
        output[key] = set(ordered)
    return output, added_reversals


def exact_match_transition(
    baseline_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
    variant_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
    gold_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
) -> dict[str, Any]:
    def matches(
        predictions: dict[tuple[str, int], set[tuple[str, str, str]]],
    ) -> set[tuple[str, int, str, str, str]]:
        return {
            (*sample_key, *triple)
            for sample_key, triples in predictions.items()
            for triple in triples & gold_by_sample.get(sample_key, set())
        }

    baseline_matches = matches(baseline_by_sample)
    variant_matches = matches(variant_by_sample)

    def rows(
        values: set[tuple[str, int, str, str, str]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "task": task,
                "sample_index": sample_index,
                "triple": [subject, predicate, obj],
            }
            for task, sample_index, subject, predicate, obj in sorted(values)
        ]

    recovered = variant_matches - baseline_matches
    lost = baseline_matches - variant_matches
    return {
        "recovered_exact": len(recovered),
        "lost_exact": len(lost),
        "recovered": rows(recovered),
        "lost": rows(lost),
    }


def symmetric_serialization_report(
    predicted_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
    gold_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]],
    *,
    scope: str,
    ranked_predictions_by_sample: dict[
        tuple[str, int], list[tuple[str, str, str]]
    ]
    | None = None,
    max_per_sample: int = 8,
) -> dict[str, Any]:
    if ranked_predictions_by_sample is None:
        ranked_predictions_by_sample = {
            key: sorted(triples) for key, triples in predicted_by_sample.items()
        }
    expanded_by_sample = {
        key: set(triples) for key, triples in predicted_by_sample.items()
    }
    added = 0
    for key, triples in predicted_by_sample.items():
        expanded = expanded_by_sample[key]
        for subject, predicate, obj in triples:
            if predicate in SYMMETRIC_PREDICATES:
                before = len(expanded)
                expanded.add((obj, predicate, subject))
                added += len(expanded) - before

    surface_by_sample = canonicalize_prediction_surfaces(predicted_by_sample)
    cap_bidirectional, cap_added = cap_aware_bidirectional_predictions(
        ranked_predictions_by_sample,
        max_per_sample=max_per_sample,
        surface_canonicalization=False,
    )
    surface_cap_bidirectional, surface_cap_added = (
        cap_aware_bidirectional_predictions(
            ranked_predictions_by_sample,
            max_per_sample=max_per_sample,
            surface_canonicalization=True,
        )
    )

    return {
        "scope": scope,
        "symmetric_predicates": sorted(SYMMETRIC_PREDICATES),
        "surface_morphology_version": SURFACE_MORPHOLOGY_VERSION,
        "max_predictions_per_sample": max_per_sample,
        "official_exact_original": relation_set_metrics(
            predicted_by_sample, gold_by_sample
        ),
        "semantic_order_invariant_diagnostic": relation_set_metrics(
            predicted_by_sample,
            gold_by_sample,
            symmetric_order_invariant=True,
        ),
        "semantic_order_and_surface_invariant_diagnostic": relation_set_metrics(
            predicted_by_sample,
            gold_by_sample,
            symmetric_order_invariant=True,
            surface_form_invariant=True,
        ),
        "official_exact_surface_canonicalization_ablation": relation_set_metrics(
            surface_by_sample, gold_by_sample
        ),
        "surface_canonicalization_transition": exact_match_transition(
            predicted_by_sample, surface_by_sample, gold_by_sample
        ),
        "official_exact_bidirectional_serialization_ablation": relation_set_metrics(
            expanded_by_sample, gold_by_sample
        ),
        "bidirectional_predictions_added": added,
        "official_exact_cap_aware_bidirectional_ablation": relation_set_metrics(
            cap_bidirectional, gold_by_sample
        ),
        "cap_aware_bidirectional_transition": exact_match_transition(
            predicted_by_sample, cap_bidirectional, gold_by_sample
        ),
        "cap_aware_bidirectional_predictions_added": cap_added,
        "official_exact_surface_plus_cap_aware_bidirectional_ablation": (
            relation_set_metrics(surface_cap_bidirectional, gold_by_sample)
        ),
        "surface_plus_cap_aware_bidirectional_transition": exact_match_transition(
            predicted_by_sample, surface_cap_bidirectional, gold_by_sample
        ),
        "surface_plus_cap_aware_bidirectional_predictions_added": surface_cap_added,
        "deployable": False,
        "confirmatory_eligible": False,
        "warning": (
            "Symmetric/surface-invariant matching is not the official exact metric. "
            "Surface canonicalization and bidirectional serialization are gold-blind "
            "diagnostic ablations and require adequate independent support before use."
        ),
    }


def audit_symmetric_serialization(
    candidates: list[dict[str, Any]],
    result: dict[str, Any],
    *,
    source_system: str,
) -> dict[str, Any]:
    """Compare semantic unordered matching with gold-blind bidirectional output."""
    predicted_by_sample: dict[
        tuple[str, int], set[tuple[str, str, str]]
    ] = {}
    ranked_predictions_by_sample: dict[
        tuple[str, int], list[tuple[str, str, str]]
    ] = {}
    gold_by_sample: dict[tuple[str, int], set[tuple[str, str, str]]] = {}

    for task_key, task_payload in result["tasks"].items():
        task = task_key.removeprefix("task_")
        samples = dict(blind.load_samples(task, 0, 0))
        for record in task_payload["systems"][source_system]["samples"]:
            if record.get("status") != "ok":
                continue
            index = int(record["sample_index"])
            sample = samples[index]
            gold_by_sample[(task, index)] = set(
                blind.categorize(
                    blind.normalize_triples(
                        sample.get(blind.TASK_CONFIG[task]["gold_key"]) or []
                    )
                )["nontax"]
            )

    for row in candidates:
        if row["postprocess_status"] != "accepted":
            continue
        key = (row["task"], int(row["sample_index"]))
        triple = tuple(row["canonical_triple"])
        predicted_by_sample.setdefault(key, set()).add(triple)
        ranked_predictions_by_sample.setdefault(key, []).append(triple)

    return symmetric_serialization_report(
        predicted_by_sample,
        gold_by_sample,
        scope="accepted_candidates_only",
        ranked_predictions_by_sample=ranked_predictions_by_sample,
        max_per_sample=8,
    )


def collect_candidates(
    result: dict[str, Any],
    *,
    source_system: str,
    calibration_indices: dict[str, set[int]],
    guard_diagnostics_from_result: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for task_key, task_payload in result["tasks"].items():
        task = task_key.removeprefix("task_")
        samples = dict(blind.load_samples(task, 0, 0))
        for record in task_payload["systems"][source_system]["samples"]:
            if record.get("status") != "ok":
                continue
            index = int(record["sample_index"])
            sample = samples[index]
            context = str(sample.get("context") or "")
            inventory = build_entity_inventory(sample, task=task)
            gold = blind.categorize(
                blind.normalize_triples(sample.get(blind.TASK_CONFIG[task]["gold_key"]) or [])
            )["nontax"]
            cached = json.loads(Path(record["cache"]).read_text(encoding="utf-8"))
            guard = (
                record.get("guard_diagnostics")
                if guard_diagnostics_from_result
                else cached.get("guard_diagnostics")
            ) or {}
            diagnostics_by_position = {
                int(row["position"]): row
                for key in (
                    "accepted_nontax_candidates",
                    "rejected_nontax_candidates",
                )
                for row in guard.get(key, [])
                if row.get("position") is not None
            }
            accepted_positions = {
                int(row["position"])
                for row in guard.get("accepted_nontax_candidates", [])
                if row.get("position") is not None
            }
            rejected_by_position = {
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
                raw_triple = normalized[0]
                try:
                    confidence = float(item.get("confidence"))
                except (TypeError, ValueError):
                    continue
                canonical, trace = canonicalize_triple(
                    raw_triple,
                    context=context,
                    entity_inventory=inventory,
                )
                if (
                    canonical[1] in blind.INSTANCE_RELATIONS
                    or canonical[1] in blind.TAXONOMY_RELATIONS
                ):
                    continue
                near_match_type, near_gold_triples = classify_near_match(
                    canonical, gold
                )
                symmetric_semantic_correct = relation_match_key(
                    canonical, symmetric_order_invariant=True
                ) in {
                    relation_match_key(row, symmetric_order_invariant=True)
                    for row in gold
                }
                evidence_mode = blind._norm(item.get("evidence_mode") or "")
                if not evidence_mode:
                    evidence = item.get("evidence")
                    evidence_mode = (
                        "cross_sentence"
                        if isinstance(evidence, list) and len(evidence) > 1
                        else "explicit"
                    )
                candidates.append(
                    {
                        "task": task,
                        "sample_index": index,
                        "stratum": record.get("stratum"),
                        "split": (
                            "calibration"
                            if index in calibration_indices[task]
                            else "heldout"
                        ),
                        "confidence": confidence,
                        "position": position,
                        "raw_triple": raw_triple,
                        "canonical_triple": canonical,
                        "raw_correct": tuple(raw_triple) in gold,
                        "canonical_correct": tuple(canonical) in gold,
                        "symmetric_semantic_correct": symmetric_semantic_correct,
                        "canonicalization_changed": bool(trace["changed"]),
                        "near_match_type": near_match_type,
                        "near_gold_triples": near_gold_triples,
                        "evidence_mode": evidence_mode,
                        "evidence_gap_chars": diagnostics_by_position.get(
                            position, {}
                        ).get("evidence_gap_chars"),
                        "postprocess_status": (
                            "accepted"
                            if position in accepted_positions
                            else "rejected"
                            if position in rejected_by_position
                            else "not_recorded"
                        ),
                        "rejection_reasons": rejected_by_position.get(position, []),
                    }
                )
    return candidates


def structural_projections_for_triple(
    triple: list[str], *, evidence: list[str] | None = None
) -> list[tuple[str, list[str]]]:
    """Return gold-blind relation projections from a generated taxonomy edge."""
    subject, predicate, obj = triple
    if predicate != "is-a":
        return []

    projections = [
        ("taxonomy_to_part_of_unrestricted", [subject, "part_of", obj]),
        ("taxonomy_inverse_has_part_unrestricted", [obj, "has part", subject]),
    ]
    subject_norm = blind._norm(subject)
    object_norm = blind._norm(obj)
    if "(whole)" in subject_norm and object_norm.endswith(" substance"):
        projections.append(
            ("whole_substance_derivation", [obj, "derives from", subject])
        )

    subject_tokens = subject_norm.split()
    object_tokens = object_norm.split()
    if (
        len(subject_tokens) == len(object_tokens)
        and len(subject_tokens) >= 2
        and subject_tokens[:-1] == object_tokens[:-1]
        and subject_tokens[-1] in {"area", "diameter", "height", "length", "width"}
        and object_tokens[-1] == "size"
    ):
        projections.append(
            ("matched_measurement_size_part", [subject, "part_of", obj])
        )

    evidence_norm = " ".join(blind._norm(value) for value in (evidence or []))
    if "nested position" in evidence_norm:
        projections.append(
            ("nested_taxonomy_inverse_has_part", [obj, "has part", subject])
        )
    return projections


def collect_structural_projection_candidates(
    result: dict[str, Any],
    *,
    source_system: str,
    calibration_indices: dict[str, set[int]],
) -> list[dict[str, Any]]:
    """Project generated taxonomy edges without consulting gold during generation."""
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, tuple[str, str, str]]] = set()
    for task_key, task_payload in result["tasks"].items():
        task = task_key.removeprefix("task_")
        samples = dict(blind.load_samples(task, 0, 0))
        for record in task_payload["systems"][source_system]["samples"]:
            if record.get("status") != "ok":
                continue
            index = int(record["sample_index"])
            sample = samples[index]
            context = str(sample.get("context") or "")
            inventory = build_entity_inventory(sample, task=task)
            gold = blind.categorize(
                blind.normalize_triples(
                    sample.get(blind.TASK_CONFIG[task]["gold_key"]) or []
                )
            )["nontax"]
            cached = json.loads(Path(record["cache"]).read_text(encoding="utf-8"))
            for position, item in enumerate(
                blind.parse_model_items(cached["raw_text"])
            ):
                if not isinstance(item, dict):
                    continue
                normalized = blind.normalize_triples([item])
                if not normalized:
                    continue
                canonical, _ = canonicalize_triple(
                    normalized[0],
                    context=context,
                    entity_inventory=inventory,
                )
                evidence_raw = item.get("evidence")
                evidence = (
                    [str(value) for value in evidence_raw]
                    if isinstance(evidence_raw, list)
                    else [str(evidence_raw)]
                    if evidence_raw
                    else []
                )
                try:
                    confidence = float(item.get("confidence"))
                except (TypeError, ValueError):
                    confidence = None
                for rule, projected in structural_projections_for_triple(
                    canonical, evidence=evidence
                ):
                    key = (task, index, rule, tuple(projected))
                    if key in seen:
                        continue
                    seen.add(key)
                    candidates.append(
                        {
                            "task": task,
                            "sample_index": index,
                            "split": (
                                "calibration"
                                if index in calibration_indices[task]
                                else "heldout"
                            ),
                            "rule": rule,
                            "source_position": position,
                            "source_triple": canonical,
                            "projected_triple": projected,
                            "source_confidence": confidence,
                            "correct": tuple(projected) in gold,
                        }
                    )
    return candidates


def _projection_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    true_positives = [row for row in rows if row["correct"]]
    positive_samples = {
        (row["task"], row["sample_index"]) for row in true_positives
    }
    return {
        "n": len(rows),
        "tp": len(true_positives),
        "fp": len(rows) - len(true_positives),
        "positive_samples": len(positive_samples),
        "precision": len(true_positives) / len(rows) if rows else 0.0,
    }


def audit_structural_projections(
    candidates: list[dict[str, Any]],
    *,
    sample_keys: set[tuple[str, int]],
    base_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Audit rule families with leave-one-sample-out selection."""
    calibration = [row for row in candidates if row["split"] == "calibration"]
    rules = sorted({row["rule"] for row in calibration})
    by_rule: dict[str, Any] = {}
    replicated_rules: list[str] = []

    for rule in rules:
        rule_rows = [row for row in calibration if row["rule"] == rule]
        pooled = _projection_metrics(rule_rows)
        loo_predictions: list[dict[str, Any]] = []
        selected_folds = 0
        for heldout_key in sorted(sample_keys):
            training = [
                row
                for row in rule_rows
                if (row["task"], row["sample_index"]) != heldout_key
            ]
            training_metrics = _projection_metrics(training)
            eligible = (
                training_metrics["positive_samples"] >= 1
                and training_metrics["precision"] >= MIN_PROJECTION_PRECISION
            )
            if not eligible:
                continue
            selected_folds += 1
            loo_predictions.extend(
                row
                for row in rule_rows
                if (row["task"], row["sample_index"]) == heldout_key
            )
        loo = _projection_metrics(loo_predictions)
        replicated = (
            pooled["positive_samples"] >= 2
            and loo["positive_samples"] >= 2
            and loo["precision"] >= MIN_PROJECTION_PRECISION
        )
        if replicated:
            replicated_rules.append(rule)
        by_rule[rule] = {
            "pooled": pooled,
            "leave_one_sample_out": {
                **loo,
                "selected_folds": selected_folds,
                "selection_min_training_positive_samples": 1,
                "selection_min_training_precision": MIN_PROJECTION_PRECISION,
            },
            "cross_sample_replicated": replicated,
        }

    selected = [row for row in calibration if row["rule"] in replicated_rules]
    selected_by_key: dict[tuple[str, int, tuple[str, str, str]], dict[str, Any]] = {}
    for row in selected:
        key = (row["task"], row["sample_index"], tuple(row["projected_triple"]))
        selected_by_key.setdefault(key, row)
    selected_unique = list(selected_by_key.values())
    selected_metrics = _projection_metrics(selected_unique)
    projection_deployable = (
        selected_metrics["positive_samples"] >= MIN_PROJECTION_POSITIVE_SAMPLES
        and selected_metrics["precision"] >= MIN_PROJECTION_PRECISION
    )

    base_positive_keys = {
        (row["task"], row["sample_index"])
        for row in base_candidates
        if row["canonical_correct"]
    }
    projected_positive_keys = {
        (row["task"], row["sample_index"])
        for row in selected_unique
        if row["correct"]
    }
    return {
        "version": STRUCTURAL_PROJECTION_VERSION,
        "uses_gold_for_candidate_labels_only": True,
        "api_calls": 0,
        "by_rule": by_rule,
        "replicated_rules": replicated_rules,
        "exact_positive_candidates": [
            row for row in calibration if row["correct"]
        ],
        "selected_projection_candidates": selected_unique,
        "selected_projection_summary": selected_metrics,
        "deployment_support": {
            "deployable": projection_deployable,
            "status": (
                "ok"
                if projection_deployable
                else "insufficient_projection_positive_samples"
            ),
            "min_positive_samples": MIN_PROJECTION_POSITIVE_SAMPLES,
            "min_precision": MIN_PROJECTION_PRECISION,
            "positive_samples": selected_metrics["positive_samples"],
            "precision": selected_metrics["precision"],
        },
        "combined_with_base_diagnostic": {
            "positive_samples": len(base_positive_keys | projected_positive_keys),
            "base_positive_samples": len(base_positive_keys),
            "new_projection_positive_samples": len(
                projected_positive_keys - base_positive_keys
            ),
            "passes_sample_count_only": len(
                base_positive_keys | projected_positive_keys
            )
            >= MIN_DEPLOYMENT_POSITIVE_SAMPLES,
            "deployable": False,
            "warning": (
                "The combined sample count is diagnostic only until the projection "
                "channel independently reaches its support gate."
            ),
        },
    }


def summarize_candidates(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(candidates),
        "raw_true_positives": sum(bool(row["raw_correct"]) for row in candidates),
        "canonical_true_positives": sum(
            bool(row["canonical_correct"]) for row in candidates
        ),
        "symmetric_semantic_true_positives": sum(
            bool(row.get("symmetric_semantic_correct", row["canonical_correct"]))
            for row in candidates
        ),
        "recovered_by_symmetric_order_invariance": sum(
            not row["canonical_correct"]
            and row.get("symmetric_semantic_correct", row["canonical_correct"])
            for row in candidates
        ),
        "canonicalization_changed": sum(
            bool(row["canonicalization_changed"]) for row in candidates
        ),
        "recovered_by_canonicalization": sum(
            not row["raw_correct"] and row["canonical_correct"] for row in candidates
        ),
        "lost_by_canonicalization": sum(
            row["raw_correct"] and not row["canonical_correct"] for row in candidates
        ),
    }


def calibration_threshold_sweep(
    candidates: list[dict[str, Any]], calibrator: BetaIsotonicCalibrator
) -> dict[str, Any]:
    eligible = deployment_eligible_candidates(candidates)
    scored = [
        (calibrator.calibrate(row["confidence"]), bool(row["canonical_correct"]))
        for row in eligible
    ]
    probabilities = sorted({score for score, _ in scored})
    thresholds = [0.0]
    thresholds.extend(
        (left + right) / 2.0
        for left, right in zip(probabilities, probabilities[1:])
    )
    if probabilities:
        thresholds.append(probabilities[-1])
    thresholds = sorted(set(thresholds))
    positives = sum(label for _, label in scored)
    rows = []
    for threshold in thresholds:
        tp = sum(score >= threshold and label for score, label in scored)
        fp = sum(score >= threshold and not label for score, label in scored)
        fn = positives - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "selected": tp + fp,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    best = max(
        rows,
        key=lambda row: (row["f1"], row["precision"], row["threshold"]),
        default=None,
    )
    return {
        "scope": "post_evidence_pre_quantity_candidates_from_calibration_samples_only",
        "n_candidates": len(scored),
        "n_positives": positives,
        "recommended_threshold": best["threshold"] if best else None,
        "recommended_metrics": best,
        "rows": rows,
    }


def summarize_candidate_groups(
    candidates: list[dict[str, Any]], *, field: str
) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        if field == "predicate":
            value = row["canonical_triple"][1]
        else:
            value = row.get(field)
        key = str(value)
        groups.setdefault(key, []).append(row)
    return {
        key: {
            "n": len(rows),
            "positives": sum(bool(row["canonical_correct"]) for row in rows),
            "accepted": sum(row["postprocess_status"] == "accepted" for row in rows),
            "accepted_positives": sum(
                row["postprocess_status"] == "accepted" and row["canonical_correct"]
                for row in rows
            ),
        }
        for key, rows in sorted(groups.items())
    }


def deployment_eligible_candidates(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Candidates that reach calibration before the per-sample quantity cap."""
    return [
        row
        for row in candidates
        if row.get("postprocess_status") == "accepted"
        or row.get("rejection_reasons") == ["quantity_limit"]
    ]


def calibration_predicate_policy(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = deployment_eligible_candidates(candidates)
    positive_predicates = sorted(
        {
            row["canonical_triple"][1]
            for row in eligible
            if row["canonical_correct"]
        }
    )
    selected = [
        row
        for row in eligible
        if row["canonical_triple"][1] in positive_predicates
    ]
    tp = sum(bool(row["canonical_correct"]) for row in selected)
    fp = len(selected) - tp
    total_positives = sum(bool(row["canonical_correct"]) for row in eligible)
    fn = total_positives - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "scope": "post_evidence_pre_quantity_candidates_from_calibration_samples_only",
        "allowed_predicates": positive_predicates,
        "selected": len(selected),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "warning": (
            "This policy is selected on sparse calibration labels and must be reported "
            "as a calibration-supported ablation, not a universal predicate inventory."
        ),
    }


def summarize_near_matches(candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in candidates:
        key = str(row["near_match_type"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    result_paths = [Path(args.result), *(Path(value) for value in args.additional_result)]
    sources = [
        (path, json.loads(path.read_text(encoding="utf-8")))
        for path in result_paths
    ]
    result = merge_source_results(sources, source_system=args.source_system)
    calibration_indices = (
        choose_all_successful_indices(result, source_system=args.source_system)
        if args.all_samples_calibration
        else choose_calibration_indices(
            result,
            source_system=args.source_system,
            per_stratum=args.calibration_per_stratum,
            seed=args.seed,
        )
    )
    candidates = collect_candidates(
        result,
        source_system=args.source_system,
        calibration_indices=calibration_indices,
    )
    projection_candidates = collect_structural_projection_candidates(
        result,
        source_system=args.source_system,
        calibration_indices=calibration_indices,
    )
    calibration = [row for row in candidates if row["split"] == "calibration"]
    heldout = [row for row in candidates if row["split"] == "heldout"]
    calibration_deployment = deployment_eligible_candidates(calibration)
    heldout_deployment = deployment_eligible_candidates(heldout)
    calibrator = BetaIsotonicCalibrator().fit(
        [row["confidence"] for row in calibration_deployment],
        [row["canonical_correct"] for row in calibration_deployment],
    )
    heldout_raw_scores = [row["confidence"] for row in heldout_deployment]
    heldout_labels = [row["canonical_correct"] for row in heldout_deployment]
    heldout_calibrated_scores = [
        calibrator.calibrate(score) for score in heldout_raw_scores
    ]
    calibration_calibrated_scores = [
        calibrator.calibrate(row["confidence"]) for row in calibration_deployment
    ]
    implicit = [row for row in candidates if row["evidence_mode"] == "cross_sentence"]
    exact_positive_candidates = [
        {
            key: row[key]
            for key in (
                "task",
                "sample_index",
                "stratum",
                "split",
                "confidence",
                "raw_triple",
                "canonical_triple",
                "raw_correct",
                "canonical_correct",
                "symmetric_semantic_correct",
                "canonicalization_changed",
                "evidence_mode",
                "evidence_gap_chars",
                "postprocess_status",
                "rejection_reasons",
            )
        }
        for row in candidates
        if row["canonical_correct"]
    ]
    actionable_near_matches = [
        {
            key: row[key]
            for key in (
                "task",
                "sample_index",
                "split",
                "confidence",
                "raw_triple",
                "canonical_triple",
                "near_match_type",
                "near_gold_triples",
                "evidence_mode",
                "evidence_gap_chars",
                "postprocess_status",
                "rejection_reasons",
            )
        }
        for row in candidates
        if row["near_match_type"]
        in {
            "endpoints_exact_predicate_mismatch",
            "endpoints_reversed",
            "symmetric_endpoints_reversed",
            "predicate_subject_exact",
            "predicate_object_exact",
        }
    ]
    all_selected_indices = {
        task_key.removeprefix("task_"): {
            int(record["sample_index"])
            for record in task_payload["systems"][args.source_system]["samples"]
            if record.get("status") == "ok"
        }
        for task_key, task_payload in result["tasks"].items()
    }
    positive_sample_count = len(
        {
            (row["task"], row["sample_index"])
            for row in calibration_deployment
            if row["canonical_correct"]
        }
    )
    calibration_sample_keys = {
        (task, index)
        for task, indices in calibration_indices.items()
        for index in indices
    }
    projection_audit = audit_structural_projections(
        projection_candidates,
        sample_keys=calibration_sample_keys,
        base_candidates=calibration_deployment,
    )
    symmetric_serialization_audit = audit_symmetric_serialization(
        candidates, result, source_system=args.source_system
    )
    predicate_policy = calibration_predicate_policy(calibration)
    allowed_predicates = set(predicate_policy["allowed_predicates"])
    deployment_allowed = (
        calibrator.deployable
        and positive_sample_count >= MIN_DEPLOYMENT_POSITIVE_SAMPLES
    )
    return {
        "meta": {
            "protocol": "relation_guarded_v2_offline_audit_v1",
            "source_result": str(result_paths[0].resolve()),
            "source_results": [str(path.resolve()) for path in result_paths],
            "source_system": args.source_system,
            "seed": args.seed,
            "calibration_per_stratum": args.calibration_per_stratum,
            "all_samples_calibration": args.all_samples_calibration,
            "canonicalization_version": CANONICALIZATION_VERSION,
            "source_evaluation_selection_uses_gold_locally": True,
            "source_attempt_ledgers": [
                payload.get("meta", {}).get("attempt_ledger")
                for _, payload in sources
            ],
            "calibration_split_uses_candidate_labels": False,
            "gold_used_for_candidate_labels_only": True,
            "calibrator_fit_population": "post_evidence_pre_quantity_candidates",
            "api_calls": 0,
        },
        "split": {
            "calibration_indices": {
                task: sorted(indices) for task, indices in calibration_indices.items()
            },
            "heldout_indices": {
                task: sorted(all_selected_indices[task] - calibration_indices[task])
                for task in calibration_indices
            },
        },
        "candidate_summary": {
            "all": summarize_candidates(candidates),
            "calibration": summarize_candidates(calibration),
            "heldout": summarize_candidates(heldout),
            "calibration_deployment_population": summarize_candidates(
                calibration_deployment
            ),
            "heldout_deployment_population": summarize_candidates(
                heldout_deployment
            ),
        },
        "calibrator": calibrator.to_dict(),
        "deployment_support": {
            "deployable": deployment_allowed,
            "candidate_level_deployable": calibrator.deployable,
            "status": (
                "ok"
                if deployment_allowed
                else "insufficient_independent_positive_samples"
                if calibrator.deployable
                else calibrator.status
            ),
            "min_positive_samples": MIN_DEPLOYMENT_POSITIVE_SAMPLES,
            "positive_candidates": sum(
                bool(row["canonical_correct"]) for row in calibration_deployment
            ),
            "positive_samples": positive_sample_count,
            "support_level": (
                "strong"
                if positive_sample_count >= 10
                else "moderate"
                if positive_sample_count >= 5
                else "minimal"
                if positive_sample_count >= 2
                else "insufficient"
            ),
            "warning": (
                "Candidate-level deployment checks pass, but positive evidence comes "
                "from fewer than five independent samples. Treat this as a precision "
                "operating point, not a stable probability model."
                if calibrator.deployable and positive_sample_count < 5
                else None
            ),
        },
        "calibration_metrics": {
            "calibration_raw": calibration_metrics(
                [row["confidence"] for row in calibration_deployment],
                [row["canonical_correct"] for row in calibration_deployment],
            ),
            "calibration_calibrated": calibration_metrics(
                calibration_calibrated_scores,
                [row["canonical_correct"] for row in calibration_deployment],
            ),
            "heldout_raw": calibration_metrics(heldout_raw_scores, heldout_labels),
            "heldout_calibrated_diagnostic_only": calibration_metrics(
                heldout_calibrated_scores, heldout_labels
            ),
        },
        "calibrated_threshold_selection": calibration_threshold_sweep(
            calibration, calibrator
        ),
        "calibration_predicate_policy": predicate_policy,
        "predicate_constrained_threshold_selection": calibration_threshold_sweep(
            [
                row
                for row in calibration
                if row["canonical_triple"][1] in allowed_predicates
            ],
            calibrator,
        ),
        "candidate_groups": {
            "by_task": summarize_candidate_groups(calibration, field="task"),
            "by_evidence_mode": summarize_candidate_groups(
                calibration, field="evidence_mode"
            ),
            "by_predicate": summarize_candidate_groups(
                calibration, field="predicate"
            ),
            "by_raw_confidence": summarize_candidate_groups(
                calibration, field="confidence"
            ),
        },
        "implicit_evidence": {
            "candidates_observed": len(implicit),
            "accepted_candidates": sum(
                row["postprocess_status"] == "accepted" for row in implicit
            ),
            "canonical_true_positives": sum(
                bool(row["canonical_correct"]) for row in implicit
            ),
            "accepted_canonical_true_positives": sum(
                row["postprocess_status"] == "accepted" and row["canonical_correct"]
                for row in implicit
            ),
            "requires_new_guarded_v2_generation": (
                args.source_system != "open_vocab_guarded_v2" and not implicit
            ),
        },
        "structural_projection_audit": projection_audit,
        "symmetric_serialization_audit": symmetric_serialization_audit,
        "exact_positive_candidates": exact_positive_candidates,
        "near_match_summary": summarize_near_matches(candidates),
        "actionable_near_matches": actionable_near_matches,
        "interpretation": {
            "calibrated_metrics_are_deployable": deployment_allowed,
            "warning": (
                "Do not apply this calibrator: the calibration split has too few positive "
                "candidates. The calibrated held-out metrics are diagnostic only."
                if not calibrator.deployable
                else (
                    "Do not apply this calibrator: candidate-level class checks pass, but "
                    f"positive evidence comes from fewer than {MIN_DEPLOYMENT_POSITIVE_SAMPLES} "
                    "independent samples."
                    if not deployment_allowed
                    else "Calibrator passed candidate- and sample-level support checks."
                )
            ),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", default=str(DEFAULT_RESULT))
    parser.add_argument(
        "--additional-result",
        action="append",
        default=[],
        help="Additional result to union by task/sample before calibration.",
    )
    parser.add_argument("--source-system", default="open_vocab_guarded")
    parser.add_argument("--calibration-per-stratum", type=int, default=2)
    parser.add_argument(
        "--all-samples-calibration",
        action="store_true",
        help="Use every successful sample in a dedicated calibration result for fitting.",
    )
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument(
        "--output",
        default=str(
            ROOT
            / "experiments"
            / "results"
            / "relation_guarded_v2_offline_audit_seed20260922.json"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.calibration_per_stratum < 1:
        raise SystemExit("--calibration-per-stratum must be positive")
    report = build_report(args)
    output = Path(args.output)
    blind.atomic_write_json(output, report)
    summary = report["candidate_summary"]
    calibrator = report["calibrator"]
    deployment = report["deployment_support"]
    print(
        f"candidates={summary['all']['n']} calibration={summary['calibration']['n']} "
        f"heldout={summary['heldout']['n']} canonical_tp={summary['all']['canonical_true_positives']} "
        f"calibrator={calibrator['status']} "
        f"candidate_deployable={calibrator['deployable']} "
        f"sample_deployable={deployment['deployable']}",
        flush=True,
    )
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
