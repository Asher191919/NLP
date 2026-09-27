"""Post-hoc error audit for the completed official relation transfer run.

This script is strictly offline. Diagnostic surface matching is deliberately
kept separate from official exact scoring and cannot authorize retuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import run_official_blind_eval as blind
import run_official_relation_transfer as transfer
import run_relation_v2_offline as offline
from ontobridg.relations.canonicalization import canonicalize_predicate


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_RESULT = RESULTS / "official_relation_transfer_openai_a_n30_seed20260927.json"
DEFAULT_OUTPUT = RESULTS / (
    "official_relation_transfer_error_audit_a_n30_seed20260927.json"
)
PROTOCOL = "official_relation_transfer_error_audit_v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def diagnostic_entity_key(value: str) -> str:
    """Return a conservative, non-official key for endpoint-boundary diagnosis."""
    text = blind._norm(value)
    text = re.sub(r"\s*\([^()]*\)\s*$", "", text)
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return offline.singularize_entity_surface(text)


def _surface_triple_key(
    triple: Iterable[str], *, reverse: bool = False
) -> tuple[str, str, str]:
    subject, predicate, obj = tuple(triple)
    left, right = (
        (diagnostic_entity_key(obj), diagnostic_entity_key(subject))
        if reverse
        else (diagnostic_entity_key(subject), diagnostic_entity_key(obj))
    )
    return left, blind._norm(predicate), right


def classify_prediction(
    triple: tuple[str, str, str], gold: set[tuple[str, str, str]]
) -> tuple[str, list[list[str]]]:
    """Assign one mutually exclusive local error category to a prediction."""
    exact_type, exact_rows = offline.classify_near_match(list(triple), gold)
    if exact_type in {
        "exact",
        "endpoints_exact_predicate_mismatch",
        "symmetric_endpoints_reversed",
        "endpoints_reversed",
    }:
        return exact_type, exact_rows

    direct_key = _surface_triple_key(triple)
    reverse_key = _surface_triple_key(triple, reverse=True)
    direct_surface = [
        list(row)
        for row in gold
        if _surface_triple_key(row)[0::2] == direct_key[0::2]
    ]
    if direct_surface:
        same_predicate = [row for row in direct_surface if row[1] == triple[1]]
        if same_predicate:
            return "surface_endpoint_variant_same_predicate", sorted(same_predicate)
        return "surface_endpoints_predicate_mismatch", sorted(direct_surface)

    reverse_surface = [
        list(row)
        for row in gold
        if _surface_triple_key(row)[0::2] == reverse_key[0::2]
    ]
    if reverse_surface:
        same_predicate = [row for row in reverse_surface if row[1] == triple[1]]
        if same_predicate and triple[1] in offline.SYMMETRIC_PREDICATES:
            return "surface_symmetric_reverse_variant", sorted(same_predicate)
        if same_predicate:
            return "surface_endpoints_reversed", sorted(same_predicate)
        return "surface_reversed_endpoints_predicate_mismatch", sorted(reverse_surface)

    if exact_type != "none":
        return exact_type, exact_rows

    exact_endpoints = {endpoint for row in gold for endpoint in (row[0], row[2])}
    exact_overlap = sum(endpoint in exact_endpoints for endpoint in (triple[0], triple[2]))
    if exact_overlap == 2:
        return "both_endpoints_exact_not_paired", []
    if exact_overlap == 1:
        return "one_endpoint_exact", []

    surface_endpoints = {diagnostic_entity_key(endpoint) for endpoint in exact_endpoints}
    surface_overlap = sum(
        diagnostic_entity_key(endpoint) in surface_endpoints
        for endpoint in (triple[0], triple[2])
    )
    if surface_overlap == 2:
        return "both_endpoints_surface_not_paired", []
    if surface_overlap == 1:
        return "one_endpoint_surface", []
    if triple[1] in {row[1] for row in gold}:
        return "predicate_only", []
    return "no_local_gold_overlap", []


def classify_gold_coverage(
    triple: tuple[str, str, str], predictions: set[tuple[str, str, str]]
) -> str:
    """Classify the closest evidence available for one missed gold relation."""
    if triple in predictions:
        return "exact"
    reverse = (triple[2], triple[1], triple[0])
    if triple[1] in offline.SYMMETRIC_PREDICATES and reverse in predictions:
        return "symmetric_reverse_only"
    if any(row[0] == triple[0] and row[2] == triple[2] for row in predictions):
        return "exact_endpoints_wrong_predicate"
    if any(row[0] == triple[2] and row[2] == triple[0] for row in predictions):
        return "reversed_endpoints_wrong_predicate"

    direct = _surface_triple_key(triple)
    reverse_surface = _surface_triple_key(triple, reverse=True)
    if any(_surface_triple_key(row) == direct for row in predictions):
        return "surface_endpoint_variant_same_predicate"
    if (
        triple[1] in offline.SYMMETRIC_PREDICATES
        and any(_surface_triple_key(row) == reverse_surface for row in predictions)
    ):
        return "surface_symmetric_reverse_variant"
    if any(
        _surface_triple_key(row)[0::2] == direct[0::2] for row in predictions
    ):
        return "surface_endpoints_wrong_predicate"

    same_predicate = {row for row in predictions if row[1] == triple[1]}
    if any(
        row[0] in {triple[0], triple[2]} or row[2] in {triple[0], triple[2]}
        for row in same_predicate
    ):
        return "same_predicate_one_exact_endpoint"
    if same_predicate:
        return "predicate_generated_wrong_endpoints"

    predicted_endpoints = {
        endpoint for row in predictions for endpoint in (row[0], row[2])
    }
    endpoint_count = sum(endpoint in predicted_endpoints for endpoint in (triple[0], triple[2]))
    if endpoint_count == 2:
        return "both_endpoints_generated_unpaired"
    if endpoint_count == 1:
        return "one_endpoint_generated"
    surface_endpoints = {diagnostic_entity_key(row) for row in predicted_endpoints}
    surface_count = sum(
        diagnostic_entity_key(endpoint) in surface_endpoints
        for endpoint in (triple[0], triple[2])
    )
    if surface_count == 2:
        return "both_surface_endpoints_generated_unpaired"
    if surface_count == 1:
        return "one_surface_endpoint_generated"
    return "no_endpoint_or_predicate_generated"


def _examples_add(
    examples: dict[str, list[dict[str, Any]]],
    category: str,
    row: dict[str, Any],
    *,
    limit: int = 5,
) -> None:
    examples.setdefault(category, [])
    if len(examples[category]) < limit:
        examples[category].append(row)


def _counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda row: (-row[1], row[0])))


def _load_cache(sample: dict[str, Any], *, system: str, model: str) -> dict[str, Any]:
    path = blind.cache_path(
        model,
        "a",
        system,
        int(sample["source_index"]),
        str(sample["id"]),
    )
    if not path.exists():
        raise ValueError(f"missing completed cache: {path}")
    cached = json.loads(path.read_text(encoding="utf-8"))
    if cached.get("status") != "ok":
        raise ValueError(f"non-success cache in completed audit: {path}")
    return cached


def audit_system(
    samples: list[dict[str, Any]], *, system: str, model: str
) -> dict[str, Any]:
    global_gold_predicates = {
        triple[1]
        for sample in samples
        for triple in blind.categorize(sample["gold_triples"])["nontax"]
        if triple[0] != triple[2]
    }
    prediction_classes: Counter[str] = Counter()
    gold_coverage: Counter[str] = Counter()
    prediction_predicates: Counter[str] = Counter()
    gold_predicates: Counter[str] = Counter()
    predicate_scope: Counter[str] = Counter()
    by_stratum: dict[str, Counter[str]] = {}
    prediction_examples: dict[str, list[dict[str, Any]]] = {}
    gold_examples: dict[str, list[dict[str, Any]]] = {}
    samples_with_gold_predicate_output = 0
    predicted_total = 0
    gold_total = 0
    raw_guard_classes: Counter[str] = Counter()
    rejected_guard_classes: Counter[str] = Counter()
    raw_guard_predicate_scope: Counter[str] = Counter()
    rejected_reasons: Counter[str] = Counter()
    raw_guard_total = 0
    rejected_guard_total = 0

    for sample in samples:
        gold = {
            row
            for row in blind.categorize(sample["gold_triples"])["nontax"]
            if row[0] != row[2]
        }
        cached = _load_cache(sample, system=system, model=model)
        predicted = {
            row
            for row in blind.categorize(cached.get("triples") or [])["nontax"]
            if row[0] != row[2]
        }
        predicted_total += len(predicted)
        gold_total += len(gold)
        sample_gold_predicates = {row[1] for row in gold}
        if any(row[1] in sample_gold_predicates for row in predicted):
            samples_with_gold_predicate_output += 1

        for triple in sorted(gold):
            gold_predicates[triple[1]] += 1
            category = classify_gold_coverage(triple, predicted)
            gold_coverage[category] += 1
            _examples_add(
                gold_examples,
                category,
                {
                    "source_index": int(sample["source_index"]),
                    "gold_triple": list(triple),
                },
            )

        stratum_counts = by_stratum.setdefault(sample["assigned_stratum"], Counter())
        for triple in sorted(predicted):
            prediction_predicates[triple[1]] += 1
            category, near_gold = classify_prediction(triple, gold)
            prediction_classes[category] += 1
            stratum_counts[category] += 1
            canonical_predicate, _, alias = canonicalize_predicate(triple[1])
            if triple[1] in global_gold_predicates:
                scope = "exact_global_gold_predicate"
            elif canonical_predicate in blind.INSTANCE_RELATIONS:
                scope = "typing_alias_misserialized_as_nontax"
            elif canonical_predicate in blind.TAXONOMY_RELATIONS:
                scope = "taxonomy_alias_misserialized_as_nontax"
            else:
                scope = "outside_global_gold_predicate_inventory"
            predicate_scope[scope] += 1
            _examples_add(
                prediction_examples,
                category,
                {
                    "source_index": int(sample["source_index"]),
                    "prediction": list(triple),
                    "near_gold_triples": near_gold,
                    "predicate_alias": alias,
                },
            )

        if system == "open_vocab_guarded_v2":
            guard = cached.get("guard_diagnostics") or {}
            accepted = list(guard.get("accepted_nontax_candidates") or [])
            rejected = list(guard.get("rejected_nontax_candidates") or [])
            accepted_set = {tuple(row["triple"]) for row in accepted}
            if accepted_set != predicted:
                raise ValueError(
                    f"guarded accepted candidates differ from scored predictions: "
                    f"source_index={sample['source_index']}"
                )
            raw_guard_total += len(accepted) + len(rejected)
            rejected_guard_total += len(rejected)
            for status, rows in (("accepted", accepted), ("rejected", rejected)):
                for row in rows:
                    triple = tuple(blind.normalize_triples([row["triple"]])[0])
                    category, _ = classify_prediction(triple, gold)
                    raw_guard_classes[category] += 1
                    canonical_predicate, _, _ = canonicalize_predicate(triple[1])
                    if triple[1] in global_gold_predicates:
                        raw_scope = "exact_global_gold_predicate"
                    elif canonical_predicate in blind.INSTANCE_RELATIONS:
                        raw_scope = "typing_alias"
                    elif canonical_predicate in blind.TAXONOMY_RELATIONS:
                        raw_scope = "taxonomy_alias"
                    else:
                        raw_scope = "outside_global_gold_predicate_inventory"
                    raw_guard_predicate_scope[raw_scope] += 1
                    if status == "rejected":
                        rejected_guard_classes[category] += 1
                        rejected_reasons.update(row.get("reasons") or [])

    output = {
        "predicted_nonself_nontax": predicted_total,
        "gold_nonself_nontax": gold_total,
        "prediction_error_categories": _counter_dict(prediction_classes),
        "missed_gold_coverage_categories": _counter_dict(gold_coverage),
        "prediction_predicate_counts": _counter_dict(prediction_predicates),
        "gold_predicate_counts": _counter_dict(gold_predicates),
        "prediction_predicate_scope": _counter_dict(predicate_scope),
        "samples_with_any_sample_gold_predicate_output": (
            samples_with_gold_predicate_output
        ),
        "by_assigned_stratum_prediction_errors": {
            key: _counter_dict(value) for key, value in sorted(by_stratum.items())
        },
        "prediction_error_examples": prediction_examples,
        "missed_gold_examples": gold_examples,
    }
    if system == "open_vocab_guarded_v2":
        output["guard_candidate_audit"] = {
            "raw_nontax_candidates": raw_guard_total,
            "accepted_nontax_candidates": predicted_total,
            "rejected_nontax_candidates": rejected_guard_total,
            "raw_candidate_error_categories": _counter_dict(raw_guard_classes),
            "raw_candidate_predicate_scope": _counter_dict(
                raw_guard_predicate_scope
            ),
            "rejected_candidate_error_categories": _counter_dict(
                rejected_guard_classes
            ),
            "rejected_candidate_reasons": _counter_dict(rejected_reasons),
            "raw_exact_true_positives": int(raw_guard_classes.get("exact", 0)),
            "raw_candidates_using_exact_global_gold_predicate": int(
                raw_guard_predicate_scope.get("exact_global_gold_predicate", 0)
            ),
            "rejected_exact_true_positives": int(
                rejected_guard_classes.get("exact", 0)
            ),
        }
    return output


def build_audit(
    *, result_path: Path = DEFAULT_RESULT, package_dir: Path, eval_manifest: Path
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = str(result["meta"]["model"])
    package_manifest, dataset, evaluation = transfer.load_frozen(
        package_dir, eval_manifest, model
    )
    ledger = result["meta"]["attempt_ledger"]
    if (
        result["meta"].get("status") != "complete"
        or int(ledger["successful_units"]) != int(ledger["planned_units"])
        or int(ledger["failed_final_units"]) != 0
    ):
        raise ValueError("official transfer result is not complete")
    if result["meta"].get("package_id") != package_manifest["meta"]["package_id"]:
        raise ValueError("result/package identity mismatch")
    if result.get("confirmatory_eligible") is not False:
        raise ValueError("completed result unexpectedly opened confirmatory gate")
    if result.get("scientific_deployable") is not False:
        raise ValueError("completed result unexpectedly opened deployment gate")

    samples = list(dataset["samples"])
    systems = {
        system: audit_system(samples, system=system, model=model)
        for system in transfer.holdout.SYSTEMS
    }
    guarded = systems["open_vocab_guarded_v2"]
    guard_audit = guarded["guard_candidate_audit"]
    guarded_gold_scope = guarded["prediction_predicate_scope"].get(
        "exact_global_gold_predicate", 0
    )
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "complete",
            "post_hoc_diagnostic": True,
            "official_metric": False,
            "contains_source_text": False,
            "contains_protected_gold": True,
            "result_path": str(result_path.resolve()),
            "result_sha256": file_sha256(result_path),
            "package_id": package_manifest["meta"]["package_id"],
            "evaluation_manifest_sha256": file_sha256(eval_manifest),
            "model": model,
            "recorded_unit_attempts": int(ledger["recorded_unit_attempts"]),
        },
        "diagnostic_policy": {
            "official_exact_scores_unchanged": True,
            "surface_matching_is_non_official": True,
            "surface_matching_operations": [
                "strip_trailing_parenthetical_qualifier",
                "normalize_hyphen_and_underscore",
                "remove_ascii_punctuation",
                offline.SURFACE_MORPHOLOGY_VERSION,
            ],
            "may_tune_prompt_or_thresholds": False,
            "may_fit_or_apply_calibrator": False,
            "may_authorize_external_calls": False,
            "may_authorize_deployment": False,
        },
        "systems": systems,
        "conclusions": {
            "guarded_raw_generator_contained_exact_gold": bool(
                guard_audit["raw_exact_true_positives"]
            ),
            "guard_rejected_exact_gold": bool(
                guard_audit["rejected_exact_true_positives"]
            ),
            "guarded_accepted_predictions_using_exact_gold_predicate": int(
                guarded_gold_scope
            ),
            "guarded_raw_candidates_using_exact_gold_predicate": int(
                guard_audit["raw_candidates_using_exact_global_gold_predicate"]
            ),
            "dominant_guarded_failure": (
                "predicate_inventory_mismatch"
                if guarded_gold_scope == 0
                else "mixed_predicate_and_endpoint_mismatch"
            ),
            "nonce_gain_transferred_to_official_text": False,
            "retuning_authorized": False,
            "scientific_deployable": False,
            "confirmatory_eligible": False,
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, default=DEFAULT_RESULT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=transfer.holdout.DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=transfer.holdout.DEFAULT_EVAL_MANIFEST,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_audit(
        result_path=Path(args.result),
        package_dir=Path(args.package_dir),
        eval_manifest=Path(args.eval_manifest),
    )
    blind.atomic_write_json(Path(args.output), payload)
    guarded = payload["systems"]["open_vocab_guarded_v2"]
    guard = guarded["guard_candidate_audit"]
    print(
        "Completed official relation transfer error audit "
        f"guarded_predictions={guarded['predicted_nonself_nontax']} "
        f"raw_candidates={guard['raw_nontax_candidates']} "
        f"raw_exact_tp={guard['raw_exact_true_positives']} "
        f"gold_predicate_predictions="
        f"{payload['conclusions']['guarded_accepted_predictions_using_exact_gold_predicate']}",
        flush=True,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
