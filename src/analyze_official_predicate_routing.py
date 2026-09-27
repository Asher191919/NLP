"""Offline post-hoc audit for the completed predicate-routing evaluation.

The audit explains official exact-score failures without changing any score or
authorizing same-slice tuning. Evidence-bound predictions absent from the
official target gold remain exact false positives; this script does not infer
their semantic correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import analyze_official_relation_transfer as transfer_audit
import run_official_blind_eval as blind
import run_official_predicate_routing as routing


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_RESULT = RESULTS / (
    "official_predicate_routing_openai_a_n27_seed20260927.json"
)
DEFAULT_OUTPUT = RESULTS / (
    "official_predicate_routing_error_audit_a_n27_seed20260927.json"
)
PROTOCOL = "official_predicate_routing_error_audit_v1"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def counter_dict(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda row: (-row[1], row[0])))


def normalized_triple(value: Iterable[str]) -> tuple[str, str, str]:
    rows = blind.normalize_triples([list(value)])
    if len(rows) != 1:
        raise ValueError(f"candidate did not normalize to one triple: {value!r}")
    return tuple(rows[0])


def canonicalization_binding(candidate: dict[str, Any]) -> str:
    """Describe endpoint binding without treating it as correctness evidence."""
    canonicalization = candidate.get("canonicalization") or {}
    methods = (
        canonicalization.get("subject_method"),
        canonicalization.get("object_method"),
    )
    exact = sum(method == "context_exact" for method in methods)
    if exact == 2:
        return "both_endpoints_context_exact"
    if exact == 1:
        return "one_endpoint_context_exact"
    return "no_endpoint_context_exact"


def official_exact_category(
    triple: tuple[str, str, str], gold: set[tuple[str, str, str]]
) -> tuple[str, list[list[str]]]:
    """Reuse the frozen diagnostic taxonomy while preserving exact scoring."""
    return transfer_audit.classify_prediction(triple, gold)


def _add_example(
    examples: dict[str, list[dict[str, Any]]],
    category: str,
    row: dict[str, Any],
    *,
    limit: int = 5,
) -> None:
    examples.setdefault(category, [])
    if len(examples[category]) < limit:
        examples[category].append(row)


def load_cache(sample: dict[str, Any], *, system: str, model: str) -> dict[str, Any]:
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
    target_predicates = set(routing.holdout.TARGET_PREDICATES)
    accepted_predicates: Counter[str] = Counter()
    raw_predicates: Counter[str] = Counter()
    rejected_predicates: Counter[str] = Counter()
    exact_categories: Counter[str] = Counter()
    missed_gold_categories: Counter[str] = Counter()
    fp_evidence_modes: Counter[str] = Counter()
    fp_endpoint_binding: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    by_stratum_predicates: dict[str, Counter[str]] = {}
    by_stratum_exact: dict[str, Counter[str]] = {}
    examples: dict[str, list[dict[str, Any]]] = {}

    accepted_nontax_total = 0
    accepted_target_total = 0
    rejected_target_total = 0
    raw_target_total = 0
    samples_at_accepted_limit = 0
    samples_with_quantity_rejection = 0

    for sample in samples:
        source_index = int(sample["source_index"])
        stratum = str(sample["assigned_stratum"])
        gold = {tuple(row) for row in sample["target_gold"]}
        cached = load_cache(sample, system=system, model=model)
        predicted_nontax = {
            row
            for row in blind.categorize(cached.get("triples") or [])["nontax"]
            if row[0] != row[2]
        }
        guard = cached.get("guard_diagnostics") or {}
        accepted = list(guard.get("accepted_nontax_candidates") or [])
        rejected = list(guard.get("rejected_nontax_candidates") or [])
        accepted_set = {normalized_triple(row["triple"]) for row in accepted}
        if accepted_set != predicted_nontax:
            raise ValueError(
                "accepted guard candidates differ from scored predictions: "
                f"source_index={source_index} system={system}"
            )

        accepted_nontax_total += len(accepted)
        if len(accepted) >= routing.holdout.MAX_NONTAX:
            samples_at_accepted_limit += 1
        if any("quantity_limit" in (row.get("reasons") or []) for row in rejected):
            samples_with_quantity_rejection += 1

        stratum_predicates = by_stratum_predicates.setdefault(stratum, Counter())
        stratum_exact = by_stratum_exact.setdefault(stratum, Counter())
        predicted_target: set[tuple[str, str, str]] = set()
        for candidate in accepted:
            triple = normalized_triple(candidate["triple"])
            if triple[1] not in target_predicates:
                continue
            predicted_target.add(triple)
            accepted_target_total += 1
            accepted_predicates[triple[1]] += 1
            stratum_predicates[triple[1]] += 1
            category, near_gold = official_exact_category(triple, gold)
            exact_categories[category] += 1
            stratum_exact[category] += 1
            if category != "exact":
                fp_evidence_modes[str(candidate.get("evidence_mode") or "missing")] += 1
                fp_endpoint_binding[canonicalization_binding(candidate)] += 1
                _add_example(
                    examples,
                    category,
                    {
                        "source_index": source_index,
                        "assigned_stratum": stratum,
                        "prediction": list(triple),
                        "near_gold_triples": near_gold,
                        "evidence_mode": candidate.get("evidence_mode"),
                        "endpoint_binding": canonicalization_binding(candidate),
                    },
                )

        for triple in sorted(gold):
            missed_gold_categories[
                transfer_audit.classify_gold_coverage(triple, predicted_target)
            ] += 1

        for status, candidates in (("accepted", accepted), ("rejected", rejected)):
            for candidate in candidates:
                triple = normalized_triple(candidate["triple"])
                if triple[1] not in target_predicates:
                    continue
                raw_target_total += 1
                raw_predicates[triple[1]] += 1
                if status == "rejected":
                    rejected_target_total += 1
                    rejected_predicates[triple[1]] += 1
                    rejection_reasons.update(candidate.get("reasons") or [])

    exact_tp = int(exact_categories.get("exact", 0))
    exact_fp = accepted_target_total - exact_tp
    return {
        "samples": len(samples),
        "accepted_nonself_nontax_candidates": accepted_nontax_total,
        "raw_target_candidates": raw_target_total,
        "accepted_target_predictions": accepted_target_total,
        "rejected_target_candidates": rejected_target_total,
        "official_exact_tp": exact_tp,
        "official_exact_fp": exact_fp,
        "accepted_target_predicate_counts": counter_dict(accepted_predicates),
        "raw_target_predicate_counts": counter_dict(raw_predicates),
        "rejected_target_predicate_counts": counter_dict(rejected_predicates),
        "official_exact_prediction_categories": counter_dict(exact_categories),
        "missed_gold_coverage_categories": counter_dict(missed_gold_categories),
        "off_gold_evidence_modes": counter_dict(fp_evidence_modes),
        "off_gold_endpoint_binding": counter_dict(fp_endpoint_binding),
        "rejected_target_reasons": counter_dict(rejection_reasons),
        "samples_at_accepted_quantity_limit": samples_at_accepted_limit,
        "samples_with_quantity_limit_rejection": samples_with_quantity_rejection,
        "by_assigned_stratum_target_predicates": {
            key: counter_dict(value)
            for key, value in sorted(by_stratum_predicates.items())
        },
        "by_assigned_stratum_exact_categories": {
            key: counter_dict(value)
            for key, value in sorted(by_stratum_exact.items())
        },
        "off_gold_examples": examples,
    }


def build_audit(
    *,
    result_path: Path = DEFAULT_RESULT,
    package_dir: Path = routing.holdout.DEFAULT_OUTPUT_DIR,
    eval_manifest: Path = routing.holdout.DEFAULT_EVAL_MANIFEST,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    model = str(result["meta"]["model"])
    package, dataset, evaluation = routing.load_frozen(
        package_dir, eval_manifest, model
    )
    ledger = result["meta"]["attempt_ledger"]
    if (
        result["meta"].get("status") != "complete"
        or int(ledger["successful_units"]) != int(ledger["planned_units"])
        or int(ledger["failed_final_units"]) != 0
    ):
        raise ValueError("predicate-routing result is not complete")
    if result["meta"].get("package_id") != package["meta"]["package_id"]:
        raise ValueError("result/package identity mismatch")
    if result.get("confirmatory_eligible") is not False:
        raise ValueError("completed result unexpectedly opened confirmatory gate")
    if result.get("scientific_deployable") is not False:
        raise ValueError("completed result unexpectedly opened deployment gate")

    systems = {
        system: audit_system(list(dataset["samples"]), system=system, model=model)
        for system in routing.holdout.SYSTEMS
    }
    general = systems[routing.holdout.SYSTEMS[0]]
    routed = systems[routing.holdout.SYSTEMS[1]]
    routed_type = int(routed["accepted_target_predicate_counts"].get("type", 0))
    routed_outputs = int(routed["accepted_target_predictions"])
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
            "package_id": package["meta"]["package_id"],
            "evaluation_manifest_sha256": file_sha256(eval_manifest),
            "model": model,
            "recorded_unit_attempts": int(ledger["recorded_unit_attempts"]),
        },
        "diagnostic_policy": {
            "official_exact_scores_unchanged": True,
            "off_gold_does_not_imply_semantically_false": True,
            "evidence_binding_does_not_imply_official_correctness": True,
            "may_tune_prompt_or_thresholds": False,
            "may_fit_or_apply_calibrator": False,
            "may_authorize_external_calls": False,
            "may_authorize_deployment": False,
        },
        "systems": systems,
        "conclusions": {
            "general_target_predictions": int(
                general["accepted_target_predictions"]
            ),
            "routed_target_predictions": routed_outputs,
            "routed_type_predictions": routed_type,
            "routed_type_prediction_fraction": (
                routed_type / routed_outputs if routed_outputs else 0.0
            ),
            "routed_official_exact_tp": int(routed["official_exact_tp"]),
            "routed_official_exact_fp": int(routed["official_exact_fp"]),
            "dominant_observed_behavior": "broad_type_overgeneration",
            "official_gold_completeness_not_established": True,
            "semantic_correctness_of_off_gold_predictions_not_adjudicated": True,
            "mechanism_gate_reopened": False,
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
        "--package-dir", type=Path, default=routing.holdout.DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--eval-manifest", type=Path, default=routing.holdout.DEFAULT_EVAL_MANIFEST
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
    routed = payload["systems"][routing.holdout.SYSTEMS[1]]
    print(
        "Completed predicate-routing error audit "
        f"target_predictions={routed['accepted_target_predictions']} "
        f"type_predictions={routed['accepted_target_predicate_counts'].get('type', 0)} "
        f"exact_tp={routed['official_exact_tp']} "
        f"exact_fp={routed['official_exact_fp']}",
        flush=True,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
