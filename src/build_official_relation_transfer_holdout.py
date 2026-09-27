"""Freeze a disjoint, relation-rich official-text transfer holdout.

The packet is selected from unused LLMs4OL 2026 Task A training samples. Gold is
used locally for stratification, endpoint-visibility checks, and overlap audits;
only the legal ``id``/``context`` projection is eligible for model input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind
import run_relation_calibration as calibration


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT_DIR = ROOT / "experiments" / "holdouts" / "official_relation_transfer_v1"
DEFAULT_EVAL_MANIFEST = RESULTS / (
    "official_relation_transfer_manifest_a_n30_seed20260927.json"
)
PROTOCOL = "official_relation_transfer_holdout_v1"
EVAL_PROTOCOL = "official_relation_transfer_eval_v1"
DEFAULT_SEED = 20260927
DEFAULT_COUNT_PER_STRATUM = 10
DEFAULT_MAX_CONTEXT_CHARS = 6000
DEFAULT_MODEL = "dots3-note-prev"
SYSTEMS = ("open_vocab", "open_vocab_guarded_v2")
STRATA = ("equivalent_class", "disjoint_with", "other_open_predicate")
AXIOM_PREDICATES = {"equivalent class", "disjoint with"}
CONFIDENCE_THRESHOLD = 0.85
IMPLICIT_CONFIDENCE_THRESHOLD = 0.95
CALIBRATED_CONFIDENCE_THRESHOLD = 0.50
MAX_NONTAX = 8
MAX_IMPLICIT_GAP_CHARS = 600

DEFAULT_EXCLUDE_RESULTS = (
    RESULTS / "official_stratified_open_openai_ab_s5_seed20260922.json",
    RESULTS / "relation_calibration_a_n27_seed20260923.json",
    RESULTS / "relation_calibration_ab_n12_seed20260923.json",
    RESULTS
    / "relation_calibration_open_vocab_guarded_v5_axioms_only_"
    "a_indices1419-1554-1796-1824-1935-2676-3130-3516_seed20260924.json",
    RESULTS
    / "relation_calibration_open_vocab_guarded_v5_axioms_only_"
    "a_indices678-2171-2643-2780-2920-2958-3064-3676_seed20260925.json",
)
DEFAULT_EXCLUDE_ANNOTATIONS = (
    ROOT / "experiments" / "annotations" / "relation_text_annotation_v1" / "manifest.json",
    ROOT
    / "experiments"
    / "annotations"
    / "relation_text_model_annotation_v1"
    / "manifest.json",
)
DEFAULT_EXCLUDE_MANIFESTS = (
    RESULTS / "v5_axiom_confirmatory_manifest_a24_seed20260925.json",
)
DEFAULT_EXCLUDE_SELECTION_AUDITS = (
    RESULTS / "relation_calibration_audit_a_n24_seed20260923.json",
    RESULTS / "relation_calibration_audit_a_n27_seed20260923.json",
    RESULTS / "relation_calibration_audit_ab_n12_seed20260923.json",
    RESULTS / "relation_calibration_audit_ab_n12_seed20260924.json",
    RESULTS / "relation_calibration_audit_b_n24_seed20260923.json",
    RESULTS
    / "relation_calibration_audit_open_vocab_guarded_v3_relations_"
    "a_indices938-2077-2424_seed20260923.json",
    RESULTS
    / "relation_calibration_audit_open_vocab_guarded_v5_axioms_only_"
    "a_indices1419-1554-1796-1824-1935-2676-3130-3516_seed20260924.json",
    RESULTS
    / "relation_calibration_audit_open_vocab_guarded_v5_axioms_only_"
    "a_indices678-2171-2643-2780-2920-2958-3064-3676_seed20260925.json",
    RESULTS
    / "relation_calibration_audit_open_vocab_guarded_v5_axioms_only_"
    "a_n12_seed20260925.json",
    RESULTS
    / "relation_calibration_audit_open_vocab_guarded_v5_axioms_only_"
    "ab_n12_seed20260924.json",
)


def json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "\n".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            for row in rows
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def stable_rank(seed: int, stratum: str, sample_id: str) -> str:
    return text_sha256(f"{PROTOCOL}:{seed}:{stratum}:{sample_id}")


def nonself_nontax(sample: dict[str, Any], task: str) -> set[tuple[str, str, str]]:
    gold = blind.normalize_triples(sample.get(blind.TASK_CONFIG[task]["gold_key"]) or [])
    return {
        tuple(triple)
        for triple in blind.categorize(gold)["nontax"]
        if triple[0] != triple[2]
    }


def visible_relations(
    sample: dict[str, Any], *, task: str, max_context_chars: int
) -> set[tuple[str, str, str]]:
    context = blind._compact(str(sample.get("context") or "")[:max_context_chars])
    return {
        triple
        for triple in nonself_nontax(sample, task)
        if blind._compact(triple[0]) in context and blind._compact(triple[2]) in context
    }


def exclusion_ledger(
    *, prefix: int = 40
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    excluded, ledger = calibration.load_exclusion_ledger(
        result_paths=list(DEFAULT_EXCLUDE_RESULTS),
        annotation_manifest_paths=list(DEFAULT_EXCLUDE_ANNOTATIONS),
        prefix=prefix,
    )
    for path in DEFAULT_EXCLUDE_MANIFESTS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        task = str(payload.get("meta", {}).get("task") or "")
        indices = {
            int(row["sample_index"])
            for row in payload.get("samples") or []
            if row.get("sample_index") is not None
        }
        before = len(excluded.get(task, set()))
        if task in excluded:
            excluded[task].update(indices)
        ledger.append(
            {
                "kind": "frozen_evaluation_manifest",
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "protocol": payload.get("meta", {}).get("protocol"),
                "task": task,
                "indices": len(indices),
                "new_indices": len(excluded.get(task, set())) - before,
            }
        )
    for path in DEFAULT_EXCLUDE_SELECTION_AUDITS:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = {"a": set(), "b": set()}
        for task in source:
            task_payload = payload.get("tasks", {}).get(f"task_{task}", {})
            source[task].update(
                int(row["sample_index"])
                for row in task_payload.get("samples") or []
                if row.get("sample_index") is not None
            )
        before = {task: len(excluded[task]) for task in source}
        for task in source:
            excluded[task].update(source[task])
        ledger.append(
            {
                "kind": "audit_only_selection",
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "protocol": payload.get("meta", {}).get("protocol"),
                "task_a_indices": len(source["a"]),
                "task_b_indices": len(source["b"]),
                "new_task_a_indices": len(excluded["a"]) - before["a"],
                "new_task_b_indices": len(excluded["b"]) - before["b"],
            }
        )
    return excluded, ledger


def excluded_evidence(
    excluded: dict[str, set[int]],
) -> tuple[set[tuple[str, str, str]], set[str]]:
    triples: set[tuple[str, str, str]] = set()
    contexts: set[str] = set()
    for task in ("a", "b"):
        samples = dict(blind.load_samples(task, 0, 0))
        for index in excluded[task]:
            sample = samples.get(index)
            if sample is None:
                continue
            triples.update(nonself_nontax(sample, task))
            contexts.add(blind._compact(sample.get("context") or ""))
    return triples, contexts


def candidate_rows(
    *, excluded: dict[str, set[int]], max_context_chars: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    excluded_gold, excluded_contexts = excluded_evidence(excluded)
    candidates: list[dict[str, Any]] = []
    rejection_counts: Counter[str] = Counter()
    for index, sample in blind.load_samples("a", 0, 0):
        if index in excluded["a"]:
            rejection_counts["used_sample_index"] += 1
            continue
        context = str(sample.get("context") or "")
        context_window = context[:max_context_chars]
        gold = nonself_nontax(sample, "a")
        visible = visible_relations(
            sample, task="a", max_context_chars=max_context_chars
        )
        if not 2 <= len(visible) <= MAX_NONTAX:
            rejection_counts["visible_relation_count_outside_2_to_8"] += 1
            continue
        if gold & excluded_gold:
            rejection_counts["exact_nontax_gold_overlap_with_prior_evidence"] += 1
            continue
        if blind._compact(context) in excluded_contexts:
            rejection_counts["exact_compact_context_overlap_with_prior_evidence"] += 1
            continue
        predicates = {triple[1] for triple in visible}
        strata = set()
        if "equivalent class" in predicates:
            strata.add("equivalent_class")
        if "disjoint with" in predicates:
            strata.add("disjoint_with")
        if predicates - AXIOM_PREDICATES:
            strata.add("other_open_predicate")
        if not strata:
            rejection_counts["no_target_stratum"] += 1
            continue
        candidates.append(
            {
                "source_index": index,
                "sample": sample,
                "sample_id": str(sample.get("id") or index),
                "context": context,
                "context_window": context_window,
                "gold": gold,
                "visible": visible,
                "visible_predicates": predicates,
                "eligible_strata": strata,
            }
        )
    return candidates, dict(sorted(rejection_counts.items()))


def select_rows(
    candidates: list[dict[str, Any]], *, seed: int, count_per_stratum: int
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    selected_indices: set[int] = set()
    selected_gold: set[tuple[str, str, str]] = set()
    # The heterogeneous stratum is scarce, so reserve it before the two large
    # axiom strata. Within it, prefer samples that add more non-axiom predicates.
    selection_order = (
        "other_open_predicate",
        "equivalent_class",
        "disjoint_with",
    )
    for stratum in selection_order:
        pool = [row for row in candidates if stratum in row["eligible_strata"]]
        pool.sort(
            key=lambda row: (
                0
                if row["eligible_strata"] == {stratum}
                else 1,
                -len(row["visible_predicates"] - AXIOM_PREDICATES)
                if stratum == "other_open_predicate"
                else 0,
                stable_rank(seed, stratum, row["sample_id"]),
                row["source_index"],
            )
        )
        chosen = []
        for row in pool:
            if row["source_index"] in selected_indices:
                continue
            if row["gold"] & selected_gold:
                continue
            chosen.append(row)
            selected_indices.add(row["source_index"])
            selected_gold.update(row["gold"])
            if len(chosen) == count_per_stratum:
                break
        if len(chosen) != count_per_stratum:
            raise ValueError(
                f"only {len(chosen)} disjoint candidates available for {stratum}; "
                f"need {count_per_stratum}"
            )
        for row in chosen:
            row = dict(row)
            row["assigned_stratum"] = stratum
            selected.append(row)
    selected.sort(key=lambda row: row["source_index"])
    return selected


def package_sample(row: dict[str, Any], package_index: int) -> dict[str, Any]:
    sample = row["sample"]
    all_gold = blind.normalize_triples(
        sample.get(blind.TASK_CONFIG["a"]["gold_key"]) or []
    )
    return {
        "package_index": package_index,
        "source_task": "a",
        "source_index": row["source_index"],
        "id": row["sample_id"],
        "context": row["context"],
        "assigned_stratum": row["assigned_stratum"],
        "gold_triples": all_gold,
        "visible_nontax_gold": [list(triple) for triple in sorted(row["visible"])],
    }


def audit_selected(
    samples: list[dict[str, Any]], excluded: dict[str, set[int]]
) -> dict[str, Any]:
    excluded_gold, excluded_contexts = excluded_evidence(excluded)
    selected_gold = [
        tuple(triple)
        for sample in samples
        for triple in blind.categorize(sample["gold_triples"])["nontax"]
        if triple[0] != triple[2]
    ]
    visible_gold = [
        tuple(triple) for sample in samples for triple in sample["visible_nontax_gold"]
    ]
    nonce_path = ROOT / "experiments" / "holdouts" / "nonce_relation_v1" / "dataset.json"
    nonce_gold: set[tuple[str, str, str]] = set()
    if nonce_path.exists():
        nonce = json.loads(nonce_path.read_text(encoding="utf-8"))
        nonce_gold = {
            tuple(triple)
            for sample in nonce.get("samples") or []
            for triple in sample.get("gold_relations") or []
        }
    return {
        "selected_indices": [sample["source_index"] for sample in samples],
        "selected_index_overlap_with_prior_task_a": sorted(
            {sample["source_index"] for sample in samples} & excluded["a"]
        ),
        "selected_context_overlap_with_prior_evidence": sum(
            blind._compact(sample["context"]) in excluded_contexts for sample in samples
        ),
        "selected_exact_nontax_gold_overlap_with_prior_evidence": len(
            set(selected_gold) & excluded_gold
        ),
        "selected_exact_nontax_gold_overlap_with_nonce_holdout": len(
            set(selected_gold) & nonce_gold
        ),
        "duplicate_selected_sample_ids": len(samples)
        - len({sample["id"] for sample in samples}),
        "duplicate_visible_gold_across_selected_samples": len(visible_gold)
        - len(set(visible_gold)),
    }


def balance_audit(samples: list[dict[str, Any]]) -> dict[str, Any]:
    strata = Counter(sample["assigned_stratum"] for sample in samples)
    visible = [
        tuple(triple) for sample in samples for triple in sample["visible_nontax_gold"]
    ]
    all_nontax = [
        tuple(triple)
        for sample in samples
        for triple in blind.categorize(sample["gold_triples"])["nontax"]
        if triple[0] != triple[2]
    ]
    density = Counter(
        "medium_2_to_4" if len(sample["visible_nontax_gold"]) <= 4 else "high_5_to_8"
        for sample in samples
    )
    return {
        "samples": len(samples),
        "assigned_strata": dict(sorted(strata.items())),
        "visible_nontax_gold": len(visible),
        "all_nonself_nontax_gold": len(all_nontax),
        "visible_relation_count_min": min(len(row["visible_nontax_gold"]) for row in samples),
        "visible_relation_count_max": max(len(row["visible_nontax_gold"]) for row in samples),
        "density_counts": dict(sorted(density.items())),
        "visible_predicate_counts": dict(sorted(Counter(t[1] for t in visible).items())),
    }


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


def build_package(
    seed: int = DEFAULT_SEED,
    count_per_stratum: int = DEFAULT_COUNT_PER_STRATUM,
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    model: str = DEFAULT_MODEL,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    excluded, ledger = exclusion_ledger()
    candidates, rejections = candidate_rows(
        excluded=excluded, max_context_chars=max_context_chars
    )
    selected = select_rows(
        candidates, seed=seed, count_per_stratum=count_per_stratum
    )
    samples = [package_sample(row, i) for i, row in enumerate(selected)]
    overlap = audit_selected(samples, excluded)
    if any(
        overlap[key]
        for key in (
            "selected_index_overlap_with_prior_task_a",
            "selected_context_overlap_with_prior_evidence",
            "selected_exact_nontax_gold_overlap_with_prior_evidence",
            "selected_exact_nontax_gold_overlap_with_nonce_holdout",
            "duplicate_selected_sample_ids",
            "duplicate_visible_gold_across_selected_samples",
        )
    ):
        raise ValueError(f"holdout independence audit failed: {overlap}")
    dataset = {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen",
            "seed": seed,
            "source": "LLMs4OL 2026 Task A official training data",
            "selection_uses_gold_locally": True,
            "gold_visible_to_model": False,
            "selection_is_gold_stratified_not_unbiased": True,
        },
        "samples": samples,
    }
    dataset_payload = json_bytes(dataset)
    input_rows = [{"id": row["id"], "context": row["context"]} for row in samples]
    inputs_payload = jsonl_bytes(input_rows)
    source_path = Path(blind.TASK_CONFIG["a"]["path"])
    manifest = {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen",
            "package_id": sha256_bytes(dataset_payload)[:20],
            "seed": seed,
            "samples": len(samples),
            "contains_source_text": False,
            "contains_protected_gold": False,
            "gold_visible_to_model": False,
            "claim_scope": "disjoint_gold_stratified_official_train_transfer_diagnostic",
            "naturalistic_benchmark": False,
            "verified_temporal_holdout": False,
        },
        "source": {
            "task": "a",
            "path": str(source_path.resolve()),
            "sha256": file_sha256(source_path),
            "legal_model_input_fields": ["id", "context"],
            "protected_gold_field": blind.TASK_CONFIG["a"]["gold_key"],
        },
        "selection": {
            "count_per_stratum": count_per_stratum,
            "strata": list(STRATA),
            "visible_nonself_nontax_per_sample": {"min": 2, "max": MAX_NONTAX},
            "max_context_chars": max_context_chars,
            "candidate_count": len(candidates),
            "rejection_counts": rejections,
            "exclusion_ledger": ledger,
        },
        "balance_audit": balance_audit(samples),
        "independence_audit": overlap,
        "files": {
            "dataset.json": {"sha256": sha256_bytes(dataset_payload)},
            "model_inputs.jsonl": {"sha256": sha256_bytes(inputs_payload)},
        },
        "samples": [
            {
                "package_index": row["package_index"],
                "source_index": row["source_index"],
                "sample_id_sha256": text_sha256(row["id"]),
                "context_sha256": text_sha256(row["context"]),
                "assigned_stratum": row["assigned_stratum"],
                "visible_nontax_gold_count": len(row["visible_nontax_gold"]),
            }
            for row in samples
        ],
    }
    manifest_payload = json_bytes(manifest)
    eval_manifest = {
        "meta": {
            "protocol": EVAL_PROTOCOL,
            "status": "frozen_not_executed",
            "frozen_before_generation": True,
            "model": model,
            "systems": list(SYSTEMS),
            "planned_units": len(samples) * len(SYSTEMS),
            "generation_cache_hits": 0,
            "estimated_external_calls": len(samples) * len(SYSTEMS),
            "contains_source_text": False,
            "contains_protected_gold": False,
            "gold_visible_to_model": False,
            "confirmatory_eligible": False,
            "scientific_deployable": False,
        },
        "source_package": {
            "package_id": manifest["meta"]["package_id"],
            "manifest_sha256": sha256_bytes(manifest_payload),
            "dataset_sha256": sha256_bytes(dataset_payload),
            "model_inputs_sha256": sha256_bytes(inputs_payload),
        },
        "locked_protocol": {
            "task_prompt": "llms4ol2026_task_a_context_only",
            "primary_metric": "exact_nontax_micro_f1",
            "secondary_metric": "official_graph_similarity_exact",
            "stratum_metric": "exact_precision_recall_f1_by_assigned_stratum",
            "predicate_metric": "exact_precision_recall_f1_by_gold_predicate",
            "density_metric": "exact_precision_recall_f1_medium_2_to_4_vs_high_5_to_8",
            "visibility_metric": "exact_recall_visible_vs_nonvisible_gold",
            "max_context_chars": max_context_chars,
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "implicit_confidence_threshold": IMPLICIT_CONFIDENCE_THRESHOLD,
            "calibrated_confidence_threshold": CALIBRATED_CONFIDENCE_THRESHOLD,
            "max_nontax": MAX_NONTAX,
            "max_implicit_gap_chars": MAX_IMPLICIT_GAP_CHARS,
            "postprocess_sha256": postprocess_sha256(),
            "calibrator": None,
            "allow_prompt_tuning": False,
            "allow_threshold_tuning": False,
            "allow_calibration": False,
            "allow_deployment_change": False,
            "interpretation": "official_train_text_transfer_diagnostic_not_naturalistic_or_temporal",
        },
        "samples": [
            {
                "package_index": row["package_index"],
                "source_index": row["source_index"],
                "sample_id_sha256": text_sha256(row["id"]),
                "context_sha256": text_sha256(row["context"]),
                "assigned_stratum": row["assigned_stratum"],
                "visible_nontax_gold_count": len(row["visible_nontax_gold"]),
                "prompt_sha256": {
                    system: text_sha256(build_prompt(row, system, max_context_chars))
                    for system in SYSTEMS
                },
            }
            for row in samples
        ],
    }
    return dataset_payload, inputs_payload, manifest_payload, eval_manifest


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
    parser.add_argument("--count-per-stratum", type=int, default=DEFAULT_COUNT_PER_STRATUM)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-manifest", type=Path, default=DEFAULT_EVAL_MANIFEST)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset, inputs, manifest, eval_manifest = build_package(
        seed=args.seed,
        count_per_stratum=args.count_per_stratum,
        max_context_chars=args.max_context_chars,
        model=args.model,
    )
    outputs = {
        Path(args.output_dir) / "dataset.json": (dataset, "dataset"),
        Path(args.output_dir) / "model_inputs.jsonl": (inputs, "model inputs"),
        Path(args.output_dir) / "manifest.json": (manifest, "package manifest"),
        Path(args.eval_manifest): (json_bytes(eval_manifest), "evaluation manifest"),
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
    package_manifest = json.loads(manifest)
    audit = package_manifest["balance_audit"]
    print(
        f"{action} official relation transfer holdout "
        f"samples={audit['samples']} visible_gold={audit['visible_nontax_gold']} "
        f"package_id={package_manifest['meta']['package_id']} external_calls=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
