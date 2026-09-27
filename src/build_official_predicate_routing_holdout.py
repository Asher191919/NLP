"""Freeze an independent official-text test of predicate-family routing.

The hypothesis comes from a protected-gold post-hoc audit, so every sample in
this packet is disjoint from that audit and all earlier project selections.
Only the legal Task A ``id``/``context`` projection may leave the machine.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_official_relation_transfer_holdout as prior
import run_official_blind_eval as blind


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "holdouts" / "official_predicate_routing_v1"
DEFAULT_EVAL_MANIFEST = RESULTS / (
    "official_predicate_routing_manifest_a_n27_seed20260927.json"
)
PRIOR_TRANSFER_DATASET = prior.DEFAULT_OUTPUT_DIR / "dataset.json"
PROTOCOL = "official_predicate_routing_holdout_v1"
EVAL_PROTOCOL = "official_predicate_routing_eval_v1"
DEFAULT_SEED = 20260927
DEFAULT_COUNTS_PER_STRATUM = {
    "equivalent_class": 8,
    "disjoint_with": 8,
    "type": 3,
    "domain_specific": 8,
}
DEFAULT_MAX_CONTEXT_CHARS = 6000
DEFAULT_MODEL = "dots3-note-prev"
SYSTEMS = (
    "open_vocab_guarded_v3_relations",
    "open_vocab_guarded_v6_predicate_routed",
)
STRATA = (
    "equivalent_class",
    "disjoint_with",
    "type",
    "domain_specific",
)
TARGET_PREDICATES = (
    "equivalent class",
    "disjoint with",
    "type",
    "database_cross_reference",
    "has role",
    "same as",
    "is_conjugate_acid_of",
    "is_conjugate_base_of",
)
DOMAIN_PREDICATES = frozenset(TARGET_PREDICATES[3:])
CONFIDENCE_THRESHOLD = 0.85
IMPLICIT_CONFIDENCE_THRESHOLD = 0.95
CALIBRATED_CONFIDENCE_THRESHOLD = 0.50
MAX_NONTAX = 8
MAX_IMPLICIT_GAP_CHARS = 600
MAX_COMPLETION_TOKENS = 16384


def stable_rank(seed: int, stratum: str, sample_id: str) -> str:
    return prior.text_sha256(f"{PROTOCOL}:{seed}:{stratum}:{sample_id}")


def build_prompt(sample: dict[str, Any], system: str, max_context_chars: int) -> str:
    return blind.build_prompt(
        {"id": sample["id"], "context": sample["context"]},
        "a",
        system,
        max_context_chars,
        CONFIDENCE_THRESHOLD,
        MAX_NONTAX,
        IMPLICIT_CONFIDENCE_THRESHOLD,
    )


def postprocess_sha256() -> str:
    return blind.postprocess_sha256(
        confidence_threshold=CONFIDENCE_THRESHOLD,
        implicit_confidence_threshold=IMPLICIT_CONFIDENCE_THRESHOLD,
        calibrated_confidence_threshold=CALIBRATED_CONFIDENCE_THRESHOLD,
        max_nontax=MAX_NONTAX,
        max_implicit_gap_chars=MAX_IMPLICIT_GAP_CHARS,
        calibrator=None,
        allowed_nontax_predicates=None,
    )


def exclusion_ledger() -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    excluded, ledger = prior.exclusion_ledger()
    payload = json.loads(PRIOR_TRANSFER_DATASET.read_text(encoding="utf-8"))
    indices = {int(row["source_index"]) for row in payload["samples"]}
    before = len(excluded["a"])
    excluded["a"].update(indices)
    ledger.append(
        {
            "kind": "completed_official_relation_transfer_package",
            "path": str(PRIOR_TRANSFER_DATASET.resolve()),
            "sha256": prior.file_sha256(PRIOR_TRANSFER_DATASET),
            "protocol": payload["meta"]["protocol"],
            "task": "a",
            "indices": len(indices),
            "new_indices": len(excluded["a"]) - before,
        }
    )
    return excluded, ledger


def _eligible_strata(predicates: set[str]) -> set[str]:
    strata = set()
    if "equivalent class" in predicates:
        strata.add("equivalent_class")
    if "disjoint with" in predicates:
        strata.add("disjoint_with")
    if "type" in predicates:
        strata.add("type")
    if predicates & DOMAIN_PREDICATES:
        strata.add("domain_specific")
    return strata


def candidate_rows(
    *, excluded: dict[str, set[int]], max_context_chars: int, model: str
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded_gold, excluded_contexts = prior.excluded_evidence(excluded)
    target_set = set(TARGET_PREDICATES)
    candidates: list[dict[str, Any]] = []
    rejected: Counter[str] = Counter()
    for index, sample in blind.load_samples("a", 0, 0):
        if index in excluded["a"]:
            rejected["used_sample_index"] += 1
            continue
        context = str(sample.get("context") or "")
        gold = prior.nonself_nontax(sample, "a")
        visible = prior.visible_relations(
            sample, task="a", max_context_chars=max_context_chars
        )
        target_gold = {row for row in gold if row[1] in target_set}
        visible_target = {row for row in visible if row[1] in target_set}
        if not 1 <= len(visible_target) <= MAX_NONTAX:
            rejected["visible_target_count_outside_1_to_8"] += 1
            continue
        if gold & excluded_gold:
            rejected["exact_nontax_gold_overlap_with_prior_evidence"] += 1
            continue
        if blind._compact(context) in excluded_contexts:
            rejected["exact_compact_context_overlap_with_prior_evidence"] += 1
            continue
        predicates = {row[1] for row in visible_target}
        strata = _eligible_strata(predicates)
        if not strata:
            rejected["no_target_stratum"] += 1
            continue
        sample_id = str(sample.get("id") or index)
        projection = {"id": sample_id, "context": context}
        cache_hit = False
        for system in SYSTEMS:
            prompt = blind.build_prompt(
                projection,
                "a",
                system,
                max_context_chars,
                CONFIDENCE_THRESHOLD,
                MAX_NONTAX,
                IMPLICIT_CONFIDENCE_THRESHOLD,
            )
            cache = blind.cache_path(model, "a", system, index, sample_id)
            cache_hit = cache_hit or blind.cache_has_reusable_generation(
                cache, prompt_sha256=blind._sha256(prompt)
            )
        if cache_hit:
            rejected["prefrozen_generation_cache"] += 1
            continue
        candidates.append(
            {
                "source_index": index,
                "sample": sample,
                "sample_id": sample_id,
                "context": context,
                "gold": gold,
                "target_gold": target_gold,
                "visible_target": visible_target,
                "visible_target_predicates": predicates,
                "eligible_strata": strata,
            }
        )
    return candidates, dict(sorted(rejected.items()))


def select_rows(
    candidates: list[dict[str, Any]],
    *,
    seed: int,
    counts_per_stratum: dict[str, int],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    selected_target_gold: set[tuple[str, str, str]] = set()
    for stratum in ("domain_specific", "type", "equivalent_class", "disjoint_with"):
        target_count = int(counts_per_stratum[stratum])
        pool = [row for row in candidates if stratum in row["eligible_strata"]]
        pool.sort(
            key=lambda row: (
                row["eligible_strata"] != {stratum},
                -sum(
                    triple[1]
                    in (
                        DOMAIN_PREDICATES
                        if stratum == "domain_specific"
                        else {
                            "equivalent_class": {"equivalent class"},
                            "disjoint_with": {"disjoint with"},
                            "type": {"type"},
                        }[stratum]
                    )
                    for triple in row["visible_target"]
                ),
                stable_rank(seed, stratum, row["sample_id"]),
                row["source_index"],
            )
        )
        chosen = []
        for row in pool:
            if row["source_index"] in selected_indices:
                continue
            if row["target_gold"] & selected_target_gold:
                continue
            chosen.append(row)
            selected_indices.add(row["source_index"])
            selected_target_gold.update(row["target_gold"])
            if len(chosen) == target_count:
                break
        if len(chosen) != target_count:
            raise ValueError(
                f"only {len(chosen)} disjoint candidates available for {stratum}; "
                f"need {target_count}"
            )
        for row in chosen:
            copied = dict(row)
            copied["assigned_stratum"] = stratum
            selected.append(copied)
    selected.sort(key=lambda row: int(row["source_index"]))
    return selected


def package_sample(row: dict[str, Any], package_index: int) -> dict[str, Any]:
    sample = row["sample"]
    all_gold = blind.normalize_triples(
        sample.get(blind.TASK_CONFIG["a"]["gold_key"]) or []
    )
    return {
        "package_index": package_index,
        "source_task": "a",
        "source_index": int(row["source_index"]),
        "id": row["sample_id"],
        "context": row["context"],
        "assigned_stratum": row["assigned_stratum"],
        "gold_triples": all_gold,
        "target_gold": [list(triple) for triple in sorted(row["target_gold"])],
        "visible_target_gold": [
            list(triple) for triple in sorted(row["visible_target"])
        ],
    }


def independence_audit(
    samples: list[dict[str, Any]], excluded: dict[str, set[int]]
) -> dict[str, Any]:
    excluded_gold, excluded_contexts = prior.excluded_evidence(excluded)
    selected_gold = {
        tuple(triple)
        for sample in samples
        for triple in blind.categorize(sample["gold_triples"])["nontax"]
        if triple[0] != triple[2]
    }
    visible_target = [
        tuple(triple)
        for sample in samples
        for triple in sample["visible_target_gold"]
    ]
    return {
        "selected_indices": [sample["source_index"] for sample in samples],
        "selected_index_overlap_with_prior_task_a": sorted(
            {sample["source_index"] for sample in samples} & excluded["a"]
        ),
        "selected_context_overlap_with_prior_evidence": sum(
            blind._compact(sample["context"]) in excluded_contexts for sample in samples
        ),
        "selected_exact_nontax_gold_overlap_with_prior_evidence": len(
            selected_gold & excluded_gold
        ),
        "duplicate_selected_sample_ids": len(samples)
        - len({sample["id"] for sample in samples}),
        "duplicate_visible_target_gold_across_selected_samples": len(visible_target)
        - len(set(visible_target)),
    }


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    visible = [
        tuple(triple)
        for sample in samples
        for triple in sample["visible_target_gold"]
    ]
    target = [
        tuple(triple) for sample in samples for triple in sample["target_gold"]
    ]
    return {
        "samples": len(samples),
        "assigned_strata": dict(
            sorted(Counter(row["assigned_stratum"] for row in samples).items())
        ),
        "visible_target_gold": len(visible),
        "all_target_gold": len(target),
        "visible_target_count_min": min(
            len(row["visible_target_gold"]) for row in samples
        ),
        "visible_target_count_max": max(
            len(row["visible_target_gold"]) for row in samples
        ),
        "visible_target_predicate_counts": dict(
            sorted(Counter(row[1] for row in visible).items())
        ),
    }


def build_package(
    seed: int = DEFAULT_SEED,
    counts_per_stratum: dict[str, int] | None = None,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    model: str = DEFAULT_MODEL,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    counts_per_stratum = dict(counts_per_stratum or DEFAULT_COUNTS_PER_STRATUM)
    if set(counts_per_stratum) != set(STRATA):
        raise ValueError("counts_per_stratum must specify every frozen stratum")
    excluded, ledger = exclusion_ledger()
    candidates, rejections = candidate_rows(
        excluded=excluded, max_context_chars=max_context_chars, model=model
    )
    selected = select_rows(
        candidates, seed=seed, counts_per_stratum=counts_per_stratum
    )
    samples = [package_sample(row, i) for i, row in enumerate(selected)]
    overlap = independence_audit(samples, excluded)
    for key, value in overlap.items():
        if key == "selected_indices":
            continue
        if value:
            raise ValueError(f"holdout independence audit failed: {key}={value}")

    dataset = {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen",
            "seed": seed,
            "source": "LLMs4OL 2026 Task A official training data",
            "hypothesis_origin": "prior_protected_gold_error_audit",
            "selection_uses_gold_locally": True,
            "gold_visible_to_model": False,
            "selection_is_gold_stratified_not_unbiased": True,
        },
        "samples": samples,
    }
    dataset_payload = prior.json_bytes(dataset)
    input_rows = [{"id": row["id"], "context": row["context"]} for row in samples]
    inputs_payload = prior.jsonl_bytes(input_rows)
    source_path = Path(blind.TASK_CONFIG["a"]["path"])
    package_id = prior.sha256_bytes(dataset_payload)[:20]
    balance = balance_audit(samples)
    manifest = {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen",
            "package_id": package_id,
            "seed": seed,
            "samples": len(samples),
            "contains_source_text": False,
            "contains_protected_gold": False,
            "gold_visible_to_model": False,
            "posthoc_hypothesis_independent_evaluation": True,
            "claim_scope": "gold_stratified_official_train_predicate_routing_diagnostic",
            "naturalistic_benchmark": False,
            "verified_temporal_holdout": False,
        },
        "source": {
            "task": "a",
            "path": str(source_path.resolve()),
            "sha256": prior.file_sha256(source_path),
            "legal_model_input_fields": ["id", "context"],
            "protected_gold_field": blind.TASK_CONFIG["a"]["gold_key"],
        },
        "selection": {
            "counts_per_stratum": counts_per_stratum,
            "capacity_note": (
                "Only three fully independent type-stratum samples remain after "
                "all exclusions; the frozen 8/8/3/8 allocation preserves them "
                "without reusing inspected evidence."
            ),
            "strata": list(STRATA),
            "frozen_target_predicates": list(TARGET_PREDICATES),
            "visible_target_per_sample": {"min": 1, "max": MAX_NONTAX},
            "max_context_chars": max_context_chars,
            "candidate_count": len(candidates),
            "rejection_counts": rejections,
            "exclusion_ledger": ledger,
        },
        "balance_audit": balance,
        "independence_audit": overlap,
        "files": {
            "dataset.json": {"sha256": prior.sha256_bytes(dataset_payload)},
            "model_inputs.jsonl": {"sha256": prior.sha256_bytes(inputs_payload)},
        },
        "samples": [
            {
                "package_index": row["package_index"],
                "source_index": row["source_index"],
                "sample_id_sha256": prior.text_sha256(row["id"]),
                "context_sha256": prior.text_sha256(row["context"]),
                "assigned_stratum": row["assigned_stratum"],
                "target_gold_count": len(row["target_gold"]),
                "visible_target_gold_count": len(row["visible_target_gold"]),
            }
            for row in samples
        ],
    }
    manifest_payload = prior.json_bytes(manifest)
    postprocess_hash = postprocess_sha256()
    eval_rows = []
    generation_hits = 0
    for row in samples:
        prompt_hashes = {}
        for system in SYSTEMS:
            prompt_hash = prior.text_sha256(
                build_prompt(row, system, max_context_chars)
            )
            prompt_hashes[system] = prompt_hash
            cache = blind.cache_path(
                model, "a", system, row["source_index"], row["id"]
            )
            generation_hits += blind.cache_has_reusable_generation(
                cache, prompt_sha256=prompt_hash
            )
        eval_rows.append(
            {
                "package_index": row["package_index"],
                "source_index": row["source_index"],
                "sample_id_sha256": prior.text_sha256(row["id"]),
                "context_sha256": prior.text_sha256(row["context"]),
                "assigned_stratum": row["assigned_stratum"],
                "target_gold_count": len(row["target_gold"]),
                "visible_target_gold_count": len(row["visible_target_gold"]),
                "prompt_sha256": prompt_hashes,
            }
        )
    if generation_hits:
        raise ValueError(f"new holdout has {generation_hits} prefrozen cache hits")
    planned_units = len(samples) * len(SYSTEMS)
    evaluation = {
        "meta": {
            "protocol": EVAL_PROTOCOL,
            "status": "frozen_not_executed",
            "frozen_before_generation": True,
            "model": model,
            "systems": list(SYSTEMS),
            "planned_units": planned_units,
            "generation_cache_hits": 0,
            "estimated_external_calls": planned_units,
            "contains_source_text": False,
            "contains_protected_gold": False,
            "gold_visible_to_model": False,
            "confirmatory_eligible": False,
            "scientific_deployable": False,
        },
        "source_package": {
            "package_id": package_id,
            "manifest_sha256": prior.sha256_bytes(manifest_payload),
            "dataset_sha256": prior.sha256_bytes(dataset_payload),
            "model_inputs_sha256": prior.sha256_bytes(inputs_payload),
        },
        "locked_protocol": {
            "task_prompt": "llms4ol2026_task_a_context_only_relation_only",
            "hypothesis": "frozen_predicate_routing_improves_exact_target_relation_recovery",
            "primary_metric": "macro_exact_target_f1_across_assigned_strata",
            "primary_contrast": "routed_v6_minus_general_relation_v3",
            "secondary_metrics": [
                "exact_target_predicate_micro_f1",
                "exact_all_nontax_micro_f1",
                "official_graph_similarity_exact",
            ],
            "diagnostics": [
                "exact_target_f1_by_assigned_stratum",
                "exact_target_f1_by_gold_predicate",
                "target_predicate_adherence",
                "visible_vs_nonvisible_target_recall",
                "stratified_paired_sample_cluster_bootstrap_10000",
            ],
            "target_predicates": list(TARGET_PREDICATES),
            "mechanism_support_gate": {
                "routed_target_f1_strictly_greater_than_general": True,
                "minimum_routed_exact_tp_samples": 5,
                "retuning_if_gate_fails": False,
            },
            "bootstrap_replicates": 10000,
            "bootstrap_seed": seed,
            "max_context_chars": max_context_chars,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "implicit_confidence_threshold": IMPLICIT_CONFIDENCE_THRESHOLD,
            "calibrated_confidence_threshold": CALIBRATED_CONFIDENCE_THRESHOLD,
            "max_nontax": MAX_NONTAX,
            "max_implicit_gap_chars": MAX_IMPLICIT_GAP_CHARS,
            "postprocess_sha256": postprocess_hash,
            "calibrator": None,
            "allowed_nontax_predicates": None,
            "allow_prompt_tuning": False,
            "allow_threshold_tuning": False,
            "allow_calibration": False,
            "allow_same_slice_repair": False,
            "allow_deployment_change": False,
            "interpretation": "independent_posthoc_hypothesis_test_on_gold_stratified_official_train",
        },
        "samples": eval_rows,
    }
    return dataset_payload, inputs_payload, manifest_payload, evaluation


def write_or_verify(path: Path, payload: bytes, *, label: str) -> bool:
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError(f"existing frozen {label} differs: {path}")
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset, inputs, manifest, evaluation = build_package(
        seed=args.seed,
        max_context_chars=args.max_context_chars,
        model=args.model,
    )
    outputs = {
        Path(args.output_dir) / "dataset.json": (dataset, "dataset"),
        Path(args.output_dir) / "model_inputs.jsonl": (inputs, "model inputs"),
        Path(args.output_dir) / "manifest.json": (manifest, "package manifest"),
        Path(args.eval_manifest): (
            prior.json_bytes(evaluation),
            "evaluation manifest",
        ),
    }
    if args.audit_only:
        missing = [str(path) for path in outputs if not path.exists()]
        if missing:
            raise SystemExit(f"frozen files are missing: {missing}")
        for path, (payload, label) in outputs.items():
            write_or_verify(path, payload, label=label)
        action = "Verified"
    else:
        created = [
            path
            for path, (payload, label) in outputs.items()
            if write_or_verify(path, payload, label=label)
        ]
        action = "Frozen" if created else "Verified"
    package = json.loads(manifest)
    balance = package["balance_audit"]
    print(
        f"{action} official predicate-routing holdout samples={balance['samples']} "
        f"visible_target_gold={balance['visible_target_gold']} "
        f"package_id={package['meta']['package_id']} external_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
