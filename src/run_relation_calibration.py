"""Build and run an independent, relation-enriched guarded-v2 calibration split.

Gold triples are used locally only to select samples whose non-taxonomic relation
endpoints are visible in the actual prompt window. Gold is never serialized into a
model prompt. The default exclusions keep the calibration split disjoint from the
blind n=30 prefix and the density-stratified evaluation split.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind
from ontobridg.relations.canonicalization import canonicalize_predicate


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "official_stratified_open_openai_ab_s5_seed20260922.json"
)
CALIBRATION_PROTOCOL = "relation_guarded_v2_independent_calibration_v1"
DEFAULT_SYSTEM = "open_vocab_guarded_v2"
SUPPORTED_CALIBRATION_SYSTEMS = (
    "open_vocab_guarded_v2",
    "open_vocab_guarded_v3_relations",
    "open_vocab_guarded_v5_axioms_only",
)
SUPPORTED_EXACT_PREDICATES = {
    "part_of",
    "has part",
    "develops_from",
    "derives from",
    "located in",
    "positively regulates",
    "regulates",
}
RELATION_CUES = {
    "equivalent class": (
        "equivalent",
        "same as",
        "synonym",
        "also known as",
        "known as",
        "interchangeable",
        "interchangeably",
        "referred to as",
        "also called",
        "called",
        "abbreviated",
        "denoted",
    ),
    "disjoint with": (
        "disjoint",
        "mutually exclusive",
        "distinct from",
        "incompatible",
    ),
    "same as": ("same as", "also known as", "synonym"),
    "database_cross_reference": (
        "database cross reference",
        "database_cross_reference",
        "xref",
    ),
    "term replaced by": ("term replaced by", "replaced by"),
}


def _rank(seed: int, task: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task}:{sample_id}".encode("utf-8")).hexdigest()


def parse_sample_indices(value: str) -> list[int]:
    if not value.strip():
        return []
    indices = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if any(index < 0 for index in indices):
        raise ValueError("sample indices must be non-negative")
    return indices


def load_excluded_indices(path: Path | None, *, prefix: int) -> dict[str, set[int]]:
    excluded, _ = load_exclusion_ledger(
        result_paths=[path] if path is not None else [],
        annotation_manifest_paths=[],
        prefix=prefix,
    )
    return excluded


def _result_indices(payload: dict[str, Any]) -> dict[str, set[int]]:
    indices_by_task = {"a": set(), "b": set()}
    for task in ("a", "b"):
        task_payload = payload.get("tasks", {}).get(f"task_{task}", {})
        indices = task_payload.get("meta", {}).get("selected_indices")
        if indices is None:
            systems = task_payload.get("systems", {})
            first_system = next(iter(systems.values()), {})
            indices = [
                row["sample_index"]
                for row in first_system.get("samples", [])
                if row.get("sample_index") is not None
            ]
        indices_by_task[task].update(int(index) for index in indices or [])
    return indices_by_task


def load_exclusion_ledger(
    *,
    result_paths: list[Path],
    annotation_manifest_paths: list[Path],
    prefix: int,
) -> tuple[dict[str, set[int]], list[dict[str, Any]]]:
    excluded = {"a": set(range(max(0, prefix))), "b": set(range(max(0, prefix)))}
    ledger: list[dict[str, Any]] = [
        {
            "kind": "prefix",
            "description": f"indices [0, {max(0, prefix)})",
            "task_a_indices": max(0, prefix),
            "task_b_indices": max(0, prefix),
        }
    ]
    for path in result_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = _result_indices(payload)
        before = {task: len(excluded[task]) for task in ("a", "b")}
        for task in ("a", "b"):
            excluded[task].update(source[task])
        ledger.append(
            {
                "kind": "result",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "protocol": payload.get("meta", {}).get("protocol"),
                "task_a_indices": len(source["a"]),
                "task_b_indices": len(source["b"]),
                "new_task_a_indices": len(excluded["a"]) - before["a"],
                "new_task_b_indices": len(excluded["b"]) - before["b"],
            }
        )
    for path in annotation_manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        source = {"a": set(), "b": set()}
        for row in payload.get("samples") or []:
            task = str(row.get("task") or "")
            if task in source and row.get("source_index") is not None:
                source[task].add(int(row["source_index"]))
        before = {task: len(excluded[task]) for task in ("a", "b")}
        for task in ("a", "b"):
            excluded[task].update(source[task])
        ledger.append(
            {
                "kind": "annotation_manifest",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "protocol": payload.get("meta", {}).get("protocol"),
                "task_a_indices": len(source["a"]),
                "task_b_indices": len(source["b"]),
                "new_task_a_indices": len(excluded["a"]) - before["a"],
                "new_task_b_indices": len(excluded["b"]) - before["b"],
            }
        )
    return excluded, ledger


def sample_relation_features(
    sample: dict[str, Any],
    *,
    task: str,
    max_context_chars: int,
) -> dict[str, Any]:
    context = str(sample.get("context") or "")[:max_context_chars]
    compact_context = blind._compact(context)
    gold = blind.normalize_triples(
        sample.get(blind.TASK_CONFIG[task]["gold_key"]) or []
    )
    nontax = {
        triple
        for triple in blind.categorize(gold)["nontax"]
        if triple[0] != triple[2]
    }
    visible = []
    visible_supported = []
    visible_guard_compatible = []
    visible_lexically_explicit = []
    segments = [
        segment
        for segment in re.split(r"(?<=[.!?;])\s+|[\r\n]+", context)
        if segment.strip()
    ]
    for triple in sorted(nontax):
        subject, predicate, obj = triple
        endpoints_visible = bool(
            blind._compact(subject)
            and blind._compact(obj)
            and blind._compact(subject) in compact_context
            and blind._compact(obj) in compact_context
        )
        if not endpoints_visible:
            continue
        visible.append(triple)
        cues = RELATION_CUES.get(predicate, (predicate.replace("_", " "),))
        for segment in segments:
            normalized_segment = blind._norm(segment)
            compact_segment = blind._compact(segment)
            if (
                blind._compact(subject) in compact_segment
                and blind._compact(obj) in compact_segment
                and any(blind._norm(cue) in normalized_segment for cue in cues)
            ):
                visible_lexically_explicit.append(triple)
                break
        canonical_predicate, _, _ = canonicalize_predicate(predicate)
        if (
            predicate == canonical_predicate
            and predicate not in blind.GENERIC_RELATIONS
        ):
            visible_guard_compatible.append(triple)
        if (
            predicate == canonical_predicate
            and predicate in SUPPORTED_EXACT_PREDICATES
        ):
            visible_supported.append(triple)
    return {
        "official_nontax_gold_nonself": len(nontax),
        "visible_nontax_gold": len(visible),
        "visible_supported_gold": len(visible_supported),
        "visible_guard_compatible_exact_gold": len(visible_guard_compatible),
        "visible_lexically_explicit_gold": len(visible_lexically_explicit),
        "visible_nontax_predicates": sorted({triple[1] for triple in visible}),
        "visible_lexically_explicit_predicates": sorted(
            {triple[1] for triple in visible_lexically_explicit}
        ),
        "visible_supported_predicates": sorted(
            {triple[1] for triple in visible_supported}
        ),
        "visible_guard_compatible_predicates": sorted(
            {triple[1] for triple in visible_guard_compatible}
        ),
    }


def select_calibration_samples(
    indexed_samples: list[tuple[int, dict[str, Any]]],
    *,
    task: str,
    count: int,
    seed: int,
    excluded_indices: set[int],
    max_context_chars: int,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[int, dict[str, Any]], dict[str, Any]]:
    candidates: list[
        tuple[int, int, int, int, int, int, str, dict[str, Any], dict[str, Any]]
    ] = []
    predicate_counts: dict[str, int] = {}
    visible_predicate_counts: dict[str, int] = {}
    for index, sample in indexed_samples:
        if index in excluded_indices:
            continue
        features = sample_relation_features(
            sample,
            task=task,
            max_context_chars=max_context_chars,
        )
        if not features["visible_nontax_gold"]:
            continue
        for predicate in features["visible_supported_predicates"]:
            predicate_counts[predicate] = predicate_counts.get(predicate, 0) + 1
        for predicate in features["visible_nontax_predicates"]:
            visible_predicate_counts[predicate] = (
                visible_predicate_counts.get(predicate, 0) + 1
            )
        sample_id = str(sample.get("id") or index)
        candidates.append(
            (
                -int(features["visible_supported_gold"]),
                -int(features["visible_lexically_explicit_gold"]),
                -int(features["visible_guard_compatible_exact_gold"]),
                -int(features["visible_nontax_gold"]),
                -int(features["official_nontax_gold_nonself"]),
                index,
                _rank(seed, task, sample_id),
                sample,
                features,
            )
        )
    candidates.sort(
        key=lambda row: (row[0], row[1], row[2], row[3], row[4], row[6], row[5])
    )
    if len(candidates) < count:
        raise ValueError(
            f"task {task} has {len(candidates)} independent endpoint-visible samples; "
            f"need {count}"
        )

    chosen = candidates[:count]
    selected = sorted(((row[5], row[7]) for row in chosen), key=lambda row: row[0])
    metadata = {
        row[5]: {
            **row[8],
            "selection_purpose": "independent_relation_calibration",
        }
        for row in chosen
    }
    pool = {
        "eligible_samples": len(candidates),
        "eligible_with_supported_predicate": sum(row[0] < 0 for row in candidates),
        "eligible_with_lexically_explicit_relation": sum(
            row[1] < 0 for row in candidates
        ),
        "eligible_with_guard_compatible_exact_predicate": sum(
            row[2] < 0 for row in candidates
        ),
        "supported_predicate_sample_counts": predicate_counts,
        "visible_predicate_sample_counts": dict(
            sorted(
                visible_predicate_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
        "selected_visible_gold": sum(row[8]["visible_nontax_gold"] for row in chosen),
        "selected_visible_supported_gold": sum(
            row[8]["visible_supported_gold"] for row in chosen
        ),
        "selected_visible_lexically_explicit_gold": sum(
            row[8]["visible_lexically_explicit_gold"] for row in chosen
        ),
        "selected_visible_guard_compatible_exact_gold": sum(
            row[8]["visible_guard_compatible_exact_gold"] for row in chosen
        ),
    }
    return selected, metadata, pool


def selected_for_tasks(
    args: argparse.Namespace,
    tasks: list[str],
) -> tuple[
    dict[str, list[tuple[int, dict[str, Any]]]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[str, set[int]],
    list[dict[str, Any]],
]:
    result_paths = (
        [Path(args.exclude_eval_result)] if args.exclude_eval_result else []
    ) + [Path(value) for value in args.additional_exclude_result]
    annotation_manifest_paths = [
        Path(value) for value in args.exclude_annotation_manifest
    ]
    excluded, exclusion_ledger = load_exclusion_ledger(
        result_paths=result_paths,
        annotation_manifest_paths=annotation_manifest_paths,
        prefix=args.exclude_prefix,
    )
    selected_by_task: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    metadata_by_task: dict[str, dict[int, dict[str, Any]]] = {}
    pool_by_task: dict[str, dict[str, Any]] = {}
    explicit_indices = parse_sample_indices(args.sample_indices)
    if explicit_indices and len(tasks) != 1:
        raise ValueError("--sample-indices requires exactly one task")
    for task in tasks:
        if explicit_indices:
            sample_map = dict(blind.load_samples(task, 0, 0))
            missing = sorted(set(explicit_indices) - set(sample_map))
            overlap = sorted(set(explicit_indices) & excluded[task])
            if missing:
                raise ValueError(f"task {task} sample indices do not exist: {missing}")
            if overlap:
                raise ValueError(f"task {task} explicit indices are excluded: {overlap}")
            selected = [(index, sample_map[index]) for index in explicit_indices]
            metadata = {
                index: {
                    **sample_relation_features(
                        sample_map[index],
                        task=task,
                        max_context_chars=args.max_context_chars,
                    ),
                    "selection_purpose": "fixed_relation_generation_pilot",
                }
                for index in explicit_indices
            }
            pool = {
                "selection": "explicit_sample_indices",
                "selected_visible_gold": sum(
                    row["visible_nontax_gold"] for row in metadata.values()
                ),
                "selected_visible_supported_gold": sum(
                    row["visible_supported_gold"] for row in metadata.values()
                ),
            }
        else:
            selected, metadata, pool = select_calibration_samples(
                blind.load_samples(task, 0, 0),
                task=task,
                count=args.samples_per_task,
                seed=args.seed,
                excluded_indices=excluded[task],
                max_context_chars=args.max_context_chars,
            )
        selected_by_task[task] = selected
        metadata_by_task[task] = metadata
        pool_by_task[task] = pool
    return (
        selected_by_task,
        metadata_by_task,
        pool_by_task,
        excluded,
        exclusion_ledger,
    )


def build_audit_manifest(
    args: argparse.Namespace,
    tasks: list[str],
) -> dict[str, Any]:
    explicit_indices = parse_sample_indices(args.sample_indices)
    (
        selected_by_task,
        metadata_by_task,
        pool_by_task,
        excluded,
        exclusion_ledger,
    ) = selected_for_tasks(args, tasks)
    manifest: dict[str, Any] = {
        "meta": {
            "protocol": CALIBRATION_PROTOCOL,
            "audit_only": True,
            "selection_uses_gold_locally": True,
            "selection_is_calibration_only_not_unbiased_evaluation": True,
            "gold_visible_to_model": False,
            "contains_source_text": False,
            "external_destination": (
                args.base_url
                or os.environ.get("ONTOBRIDG_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            "model": args.model,
            "system": args.system,
            "samples_per_task": args.samples_per_task,
            "sample_indices": explicit_indices,
            "seed": args.seed,
            "prior_failed_connection_attempts": args.prior_failed_connection_attempts,
            "prior_failed_unit_attempts": args.prior_failed_unit_attempts,
            "prior_failed_low_level_attempts": args.prior_failed_low_level_attempts,
            "exclude_prefix": args.exclude_prefix,
            "exclude_eval_result": args.exclude_eval_result,
            "additional_exclude_results": args.additional_exclude_result,
            "exclude_annotation_manifests": args.exclude_annotation_manifest,
            "exclusion_ledger": exclusion_ledger,
            "confidence_threshold": args.confidence_threshold,
            "implicit_confidence_threshold": args.implicit_confidence_threshold,
            "max_nontax": args.max_nontax,
            "max_implicit_gap_chars": args.max_implicit_gap_chars,
        },
        "tasks": {},
    }
    generation_hits = 0
    exact_hits = 0
    prompt_chars = 0
    disjoint_from_excluded = True
    for task in tasks:
        rows = []
        selected_indices = {index for index, _ in selected_by_task[task]}
        overlap = sorted(selected_indices & excluded[task])
        disjoint_from_excluded = disjoint_from_excluded and not overlap
        for index, sample in selected_by_task[task]:
            prompt = blind.build_prompt(
                sample,
                task,
                args.system,
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
            path = blind.cache_path(args.model, task, args.system, index, sample_id)
            generation_hit = blind.cache_has_reusable_generation(
                path, prompt_sha256=prompt_hash
            )
            exact_hit = blind.cache_is_replayable(
                path,
                system=args.system,
                prompt_sha256=prompt_hash,
                postprocess_hash=postprocess_hash,
            )
            generation_hits += int(generation_hit)
            exact_hits += int(exact_hit)
            prompt_chars += len(prompt)
            rows.append(
                {
                    "sample_index": index,
                    "sample_id_sha256": blind._sha256(sample_id),
                    **metadata_by_task[task][index],
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
                        len(str(sample.get("context") or "")),
                        args.max_context_chars,
                    ),
                    "prompt": {
                        "sha256": prompt_hash,
                        "chars": len(prompt),
                        "generation_cache_hit": generation_hit,
                        "exact_cache_hit": exact_hit,
                    },
                }
            )
        manifest["tasks"][f"task_{task}"] = {
            "n_samples": len(rows),
            "excluded_indices_count": len(excluded[task]),
            "selected_overlap_with_excluded": overlap,
            "pool": pool_by_task[task],
            "samples": rows,
        }
    total = sum(len(rows) for rows in selected_by_task.values())
    manifest["meta"].update(
        {
            "estimated_calls": total,
            "estimated_generation_cache_hits": generation_hits,
            "estimated_exact_cache_replays": exact_hits,
            "estimated_local_reprocesses": generation_hits - exact_hits,
            "estimated_external_calls": total - generation_hits,
            "total_prompt_chars": prompt_chars,
            "disjoint_from_blind_prefix_and_density_eval": disjoint_from_excluded,
            "disjoint_from_all_exclusions": disjoint_from_excluded,
        }
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["a", "b", "ab"], default="ab")
    parser.add_argument("--samples-per-task", type=int, default=12)
    parser.add_argument("--sample-indices", default="")
    parser.add_argument(
        "--system",
        choices=SUPPORTED_CALIBRATION_SYSTEMS,
        default=DEFAULT_SYSTEM,
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--prior-failed-connection-attempts", type=int, default=0)
    parser.add_argument("--prior-failed-unit-attempts", type=int, default=0)
    parser.add_argument("--prior-failed-low-level-attempts", type=int, default=0)
    parser.add_argument("--exclude-prefix", type=int, default=30)
    parser.add_argument("--exclude-eval-result", default=str(DEFAULT_EVAL_RESULT))
    parser.add_argument("--additional-exclude-result", action="append", default=[])
    parser.add_argument("--exclude-annotation-manifest", action="append", default=[])
    parser.add_argument("--backend", choices=["openai", "openai_compat", "api"], default="openai")
    parser.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
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
    args.calibration_file = None
    if args.samples_per_task < 1:
        raise SystemExit("--samples-per-task must be positive")
    if args.exclude_prefix < 0:
        raise SystemExit("--exclude-prefix must be non-negative")
    if any(
        value < 0
        for value in (
            args.prior_failed_connection_attempts,
            args.prior_failed_unit_attempts,
            args.prior_failed_low_level_attempts,
        )
    ):
        raise SystemExit("prior failed attempt counts must be non-negative")
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be between 1 and 4")
    tasks = ["a", "b"] if args.task == "ab" else [args.task]
    explicit_indices = parse_sample_indices(args.sample_indices)
    sample_tag = (
        "indices" + "-".join(str(index) for index in explicit_indices)
        if explicit_indices
        else f"n{args.samples_per_task}"
    )
    system_tag = "" if args.system == DEFAULT_SYSTEM else f"_{blind._slug(args.system)}"
    tag = f"{args.task}_{sample_tag}_seed{args.seed}"

    if args.audit_only:
        manifest = build_audit_manifest(args, tasks)
        path = blind.RESULTS / f"relation_calibration_audit{system_tag}_{tag}.json"
        blind.atomic_write_json(path, manifest)
        print(
            f"Audit only: samples={manifest['meta']['estimated_calls']} "
            f"external_calls={manifest['meta']['estimated_external_calls']} "
            f"prompt_chars={manifest['meta']['total_prompt_chars']}",
            flush=True,
        )
        print(f"Wrote {path}", flush=True)
        return 0

    (
        selected_by_task,
        metadata_by_task,
        pool_by_task,
        excluded,
        exclusion_ledger,
    ) = selected_for_tasks(args, tasks)
    result: dict[str, Any] = {
        "meta": {
            "protocol": CALIBRATION_PROTOCOL,
            "model": args.model,
            "system": args.system,
            "tasks": tasks,
            "samples_per_task": args.samples_per_task,
            "sample_indices": explicit_indices,
            "seed": args.seed,
            "prior_failed_connection_attempts": args.prior_failed_connection_attempts,
            "prior_failed_unit_attempts": args.prior_failed_unit_attempts,
            "prior_failed_low_level_attempts": args.prior_failed_low_level_attempts,
            "selection_uses_gold_locally": True,
            "selection_is_calibration_only_not_unbiased_evaluation": True,
            "gold_visible_to_model": False,
            "exclude_prefix": args.exclude_prefix,
            "exclude_eval_result": args.exclude_eval_result,
            "additional_exclude_results": args.additional_exclude_result,
            "exclude_annotation_manifests": args.exclude_annotation_manifest,
            "exclusion_ledger": exclusion_ledger,
            "confidence_threshold": args.confidence_threshold,
            "implicit_confidence_threshold": args.implicit_confidence_threshold,
            "calibrated_confidence_threshold": args.calibrated_confidence_threshold,
            "max_nontax": args.max_nontax,
            "max_implicit_gap_chars": args.max_implicit_gap_chars,
        },
        "tasks": {},
    }
    any_failed = False
    for task in tasks:
        selected = selected_by_task[task]
        args.offset = 0
        args.limit = len(selected)
        checkpoint = (
            blind.RESULTS
            / "checkpoints"
            / f"relation_calibration_{blind._slug(args.backend)}_task{task}_{args.system}_{sample_tag}_seed{args.seed}.json"
        )
        payload = blind.run_system(
            args=args,
            task=task,
            system=args.system,
            indexed_samples=selected,
            checkpoint_path=checkpoint,
        )
        for record in payload["samples"]:
            record.update(metadata_by_task[task][int(record["sample_index"])])
        payload["meta"].update(
            {
                "selection": "independent_relation_calibration",
                "selected_indices": [index for index, _ in selected],
                "excluded_indices_count": len(excluded[task]),
                "pool": pool_by_task[task],
            }
        )
        blind.atomic_write_json(checkpoint, payload)
        result["tasks"][f"task_{task}"] = {
            "meta": {
                "task": task,
                "selected_indices": [index for index, _ in selected],
                "n_samples": len(selected),
            },
            "systems": {args.system: payload},
        }
        summary = payload["summary"]
        any_failed = any_failed or not summary["complete"]
        print(
            f"Task {task.upper()}: ok={summary['n_success']}/{summary['target_n']} "
            f"candidates={summary.get('guard', {}).get('raw_nontax', 0)} "
            f"accepted={summary.get('guard', {}).get('accepted_nontax', 0)}",
            flush=True,
        )

    summaries = [
        result["tasks"][f"task_{task}"]["systems"][args.system]["summary"]
        for task in tasks
    ]
    successful_units = sum(int(summary["n_success"]) for summary in summaries)
    failed_final_units = sum(int(summary["n_failed"]) for summary in summaries)
    successful_runner_attempts = sum(
        int(summary["successful_runner_attempts_total"]) for summary in summaries
    )
    current_low_level_attempts = sum(
        int(record.get("attempts") or 0)
        for task in tasks
        for record in result["tasks"][f"task_{task}"]["systems"][args.system][
            "samples"
        ]
    )
    prior_failed_units = (
        args.prior_failed_unit_attempts + args.prior_failed_connection_attempts
    )
    prior_failed_low_level = (
        args.prior_failed_low_level_attempts
        + args.prior_failed_connection_attempts
    )
    planned_units = sum(len(rows) for rows in selected_by_task.values())
    result["meta"]["attempt_ledger"] = {
        "planned_units": planned_units,
        "successful_units": successful_units,
        "failed_final_units": failed_final_units,
        "prior_failed_connection_attempts": args.prior_failed_connection_attempts,
        "prior_failed_unit_attempts": args.prior_failed_unit_attempts,
        "unit_attempts_including_prior_failures": planned_units + prior_failed_units,
        "current_low_level_attempts": current_low_level_attempts,
        "prior_failed_low_level_attempts": args.prior_failed_low_level_attempts,
        "successful_runner_attempts_including_retries": successful_runner_attempts,
        "low_level_attempts_including_prior_failures": (
            current_low_level_attempts + prior_failed_low_level
        ),
    }

    output = blind.RESULTS / f"relation_calibration{system_tag}_{tag}.json"
    blind.atomic_write_json(output, result)
    print(f"Wrote {output}", flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
