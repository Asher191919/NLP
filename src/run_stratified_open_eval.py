"""Density-stratified blind evaluation for legacy and guarded open-vocabulary prompts.

Selection uses local gold triples only to assign evaluation strata. Gold triples never enter
model prompts. The same selected samples are evaluated by both open-vocabulary systems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind


STRATA = ("low", "medium", "high")
STRATUM_DEFINITIONS = {
    "low": "exactly 1 non-self non-taxonomic gold triple",
    "medium": "2-4 non-self non-taxonomic gold triples",
    "high": "at least 5 non-self non-taxonomic gold triples",
}
DEFAULT_OPEN_SYSTEMS = ("open_vocab", "open_vocab_guarded")
SUPPORTED_OPEN_SYSTEMS = (*DEFAULT_OPEN_SYSTEMS, "open_vocab_guarded_v2")
HYBRID_SYSTEM = "open_vocab_guarded_hybrid"
V2_HYBRID_SYSTEM = "open_vocab_guarded_v2_hybrid"


def density_stratum(nontax_gold: int) -> str | None:
    if nontax_gold == 1:
        return "low"
    if 2 <= nontax_gold <= 4:
        return "medium"
    if nontax_gold >= 5:
        return "high"
    return None


def _selection_rank(seed: int, task: str, sample_id: str) -> str:
    return hashlib.sha256(f"{seed}:{task}:{sample_id}".encode("utf-8")).hexdigest()


def select_density_strata(
    indexed_samples: list[tuple[int, dict[str, Any]]],
    *,
    task: str,
    samples_per_stratum: int,
    seed: int,
) -> tuple[list[tuple[int, dict[str, Any]]], dict[int, dict[str, Any]]]:
    gold_key = blind.TASK_CONFIG[task]["gold_key"]
    buckets: dict[str, list[tuple[str, int, dict[str, Any], int, int]]] = {
        name: [] for name in STRATA
    }
    for index, sample in indexed_samples:
        gold = blind.normalize_triples(sample.get(gold_key) or [])
        official_nontax = blind.categorize(gold)["nontax"]
        nontax_gold = len(official_nontax)
        nonself_nontax_gold = len(
            {triple for triple in official_nontax if triple[0] != triple[2]}
        )
        stratum = density_stratum(nonself_nontax_gold)
        if stratum is None:
            continue
        sample_id = str(sample.get("id") or index)
        buckets[stratum].append(
            (
                _selection_rank(seed, task, sample_id),
                index,
                sample,
                nontax_gold,
                nonself_nontax_gold,
            )
        )

    selected: list[tuple[int, dict[str, Any]]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for stratum in STRATA:
        candidates = sorted(buckets[stratum], key=lambda row: row[0])
        if len(candidates) < samples_per_stratum:
            raise ValueError(
                f"task {task} stratum {stratum} has {len(candidates)} candidates; "
                f"need {samples_per_stratum}"
            )
        for _, index, sample, nontax_gold, nonself_nontax_gold in candidates[:samples_per_stratum]:
            selected.append((index, sample))
            metadata[index] = {
                "stratum": stratum,
                "official_nontax_gold": nontax_gold,
                "selection_nontax_gold_nonself": nonself_nontax_gold,
            }
    selected.sort(key=lambda row: row[0])
    return selected, metadata


def _metric_delta(guarded: dict[str, Any], legacy: dict[str, Any]) -> dict[str, Any]:
    return {
        "typing_f1": (
            guarded["term_typing_f1_micro"]["f1"]
            - legacy["term_typing_f1_micro"]["f1"]
        ),
        "taxonomy_f1": (
            guarded["taxonomy_f1_micro"]["f1"]
            - legacy["taxonomy_f1_micro"]["f1"]
        ),
        "nontax_f1": guarded["nontax_f1_micro"]["f1"] - legacy["nontax_f1_micro"]["f1"],
        "nontax_nonself_f1": (
            guarded["nontax_nonself_f1_micro"]["f1"]
            - legacy["nontax_nonself_f1_micro"]["f1"]
        ),
        "official_graph_similarity": (
            guarded["official_graph_similarity_avg"]
            - legacy["official_graph_similarity_avg"]
        ),
        "composite": (
            guarded["composite_typing_tax_gs"]
            - legacy["composite_typing_tax_gs"]
        ),
        "nontax_pred_total": guarded["nontax_pred_total"] - legacy["nontax_pred_total"],
    }


def merge_core_with_guarded_relations(
    legacy_triples: list[list[str]],
    guarded_triples: list[list[str]],
) -> list[list[str]]:
    legacy = blind.normalize_triples(legacy_triples)
    guarded = blind.normalize_triples(guarded_triples)
    core = [
        triple
        for triple in legacy
        if triple[1] in blind.INSTANCE_RELATIONS or triple[1] in blind.TAXONOMY_RELATIONS
    ]
    relations = [
        triple
        for triple in guarded
        if triple[1] not in blind.INSTANCE_RELATIONS
        and triple[1] not in blind.TAXONOMY_RELATIONS
    ]
    return blind.normalize_triples(core + relations)


def add_stratified_summaries(
    payload: dict[str, Any],
    selection: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    for record in payload["samples"]:
        record.update(selection[int(record["sample_index"])])
    payload["strata"] = {}
    for stratum in STRATA:
        records = [r for r in payload["samples"] if r["stratum"] == stratum]
        payload["strata"][stratum] = blind.aggregate_records(records, len(records))
    return payload


def build_hybrid_payload(
    *,
    task: str,
    selected: list[tuple[int, dict[str, Any]]],
    selection: dict[int, dict[str, Any]],
    legacy_payload: dict[str, Any],
    guarded_payload: dict[str, Any],
    hybrid_system: str = HYBRID_SYSTEM,
    relation_source: str = "open_vocab_guarded",
) -> dict[str, Any]:
    legacy_records = {int(r["sample_index"]): r for r in legacy_payload["samples"]}
    guarded_records = {int(r["sample_index"]): r for r in guarded_payload["samples"]}
    records: list[dict[str, Any]] = []
    gold_key = blind.TASK_CONFIG[task]["gold_key"]
    for index, sample in selected:
        legacy_record = legacy_records[index]
        guarded_record = guarded_records[index]
        record: dict[str, Any] = {
            "sample_index": index,
            "sample_id": str(sample.get("id") or index),
            "status": "error",
            "attempts": 0,
            "latency_sec": None,
            "source_caches": {
                "core": legacy_record["cache"],
                "relations": guarded_record["cache"],
            },
            **selection[index],
        }
        if legacy_record["status"] == "ok" and guarded_record["status"] == "ok":
            legacy_cache = json.loads(Path(legacy_record["cache"]).read_text(encoding="utf-8"))
            guarded_cache = json.loads(Path(guarded_record["cache"]).read_text(encoding="utf-8"))
            triples = merge_core_with_guarded_relations(
                legacy_cache["triples"], guarded_cache["triples"]
            )
            gold = blind.normalize_triples(sample.get(gold_key) or [])
            record["status"] = "ok"
            record["metrics"] = blind.evaluate_sample(gold, triples)
            record["pred_counts"] = {
                key: len(value) for key, value in blind.categorize(triples).items()
            }
            if guarded_cache.get("guard_diagnostics") is not None:
                record["guard_diagnostics"] = guarded_cache["guard_diagnostics"]
        else:
            record["errors"] = ["one or both source systems failed"]
        records.append(record)

    payload = {
        "meta": {
            "challenge": "LLMs4OL 2026",
            "protocol": "llms4ol2026_stratified_open_v1",
            "task": task,
            "task_name": blind.TASK_CONFIG[task]["name"],
            "system": hybrid_system,
            "derived_without_additional_api_calls": True,
            "core_source": "open_vocab",
            "relation_source": relation_source,
            "selected_indices": [index for index, _ in selected],
            "gold_visible_to_model": False,
        },
        "summary": blind.aggregate_records(records, len(records)),
        "samples": records,
    }
    return add_stratified_summaries(payload, selection)


def build_audit_manifest(
    args: argparse.Namespace,
    tasks: list[str],
    systems: list[str],
) -> dict[str, Any]:
    calibrator = None
    if args.calibration_file:
        calibration_payload = json.loads(Path(args.calibration_file).read_text(encoding="utf-8"))
        calibrator = blind.calibrator_from_artifact_payload(calibration_payload)
    allowed_predicates = blind.normalize_allowed_nontax_predicates(
        args.allowed_nontax_predicates
    )
    manifest: dict[str, Any] = {
        "meta": {
            "protocol": "llms4ol2026_stratified_open_v1",
            "audit_only": True,
            "selection_uses_gold_locally": True,
            "gold_visible_to_model": False,
            "contains_source_text": False,
            "external_destination": (
                args.base_url
                or os.environ.get("ONTOBRIDG_BASE_URL")
                or "https://api.openai.com/v1"
            ),
            "model": args.model,
            "systems": systems,
            "samples_per_stratum": args.samples_per_stratum,
            "seed": args.seed,
            "confidence_threshold": args.confidence_threshold,
            "implicit_confidence_threshold": args.implicit_confidence_threshold,
            "calibrated_confidence_threshold": args.calibrated_confidence_threshold,
            "max_nontax": args.max_nontax,
            "max_implicit_gap_chars": args.max_implicit_gap_chars,
            "calibration_file": args.calibration_file,
            "allowed_nontax_predicates": (
                sorted(allowed_predicates)
                if allowed_predicates is not None
                else None
            ),
            "strata": STRATUM_DEFINITIONS,
        },
        "tasks": {},
    }
    total_calls = 0
    total_cache_replays = 0
    total_exact_cache_replays = 0
    total_local_reprocesses = 0
    total_prompt_chars = 0
    for task in tasks:
        all_samples = blind.load_samples(task, 0, 0)
        selected, selection = select_density_strata(
            all_samples,
            task=task,
            samples_per_stratum=args.samples_per_stratum,
            seed=args.seed,
        )
        rows = []
        for index, sample in selected:
            prompts = {
                system: blind.build_prompt(
                    sample,
                    task,
                    system,
                    args.max_context_chars,
                    args.confidence_threshold,
                    args.max_nontax,
                    args.implicit_confidence_threshold,
                )
                for system in systems
            }
            postprocess_hash = blind.postprocess_sha256(
                confidence_threshold=args.confidence_threshold,
                implicit_confidence_threshold=args.implicit_confidence_threshold,
                calibrated_confidence_threshold=args.calibrated_confidence_threshold,
                max_nontax=args.max_nontax,
                max_implicit_gap_chars=args.max_implicit_gap_chars,
                calibrator=calibrator,
                allowed_nontax_predicates=allowed_predicates,
            )
            prompt_metadata: dict[str, dict[str, Any]] = {}
            sample_id = str(sample.get("id") or index)
            for system, prompt in prompts.items():
                prompt_hash = blind._sha256(prompt)
                path = blind.cache_path(args.model, task, system, index, sample_id)
                generation_hit = blind.cache_has_reusable_generation(
                    path,
                    prompt_sha256=prompt_hash,
                )
                exact_cache_hit = blind.cache_is_replayable(
                    path,
                    system=system,
                    prompt_sha256=prompt_hash,
                    postprocess_hash=postprocess_hash,
                )
                prompt_metadata[system] = {
                    "sha256": prompt_hash,
                    "chars": len(prompt),
                    "cache_replay_expected": generation_hit,
                    "local_reprocess_expected": generation_hit and not exact_cache_hit,
                }
                total_cache_replays += int(generation_hit)
                total_exact_cache_replays += int(exact_cache_hit)
                total_local_reprocesses += int(generation_hit and not exact_cache_hit)
            rows.append(
                {
                    "sample_index": index,
                    "sample_id_sha256": blind._sha256(str(sample.get("id") or index)),
                    **selection[index],
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
                    "prompts": prompt_metadata,
                }
            )
            total_calls += len(systems)
            total_prompt_chars += sum(len(prompt) for prompt in prompts.values())
        manifest["tasks"][f"task_{task}"] = {
            "task_name": blind.TASK_CONFIG[task]["name"],
            "n_samples": len(selected),
            "estimated_calls": len(selected) * len(systems),
            "samples": rows,
        }
    manifest["meta"]["estimated_calls"] = total_calls
    manifest["meta"]["estimated_generation_cache_hits"] = total_cache_replays
    manifest["meta"]["estimated_exact_cache_replays"] = total_exact_cache_replays
    manifest["meta"]["estimated_local_reprocesses"] = total_local_reprocesses
    manifest["meta"]["estimated_external_calls"] = total_calls - total_cache_replays
    manifest["meta"]["total_prompt_chars"] = total_prompt_chars
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["a", "b", "ab"], default="ab")
    parser.add_argument("--samples-per-stratum", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--systems", default=",".join(DEFAULT_OPEN_SYSTEMS))
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
    parser.add_argument("--calibration-file", default=None)
    parser.add_argument(
        "--allowed-nontax-predicates",
        default="",
        help="Optional comma-separated canonical predicate allowlist for precision mode.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    allowed_predicates = blind.normalize_allowed_nontax_predicates(
        args.allowed_nontax_predicates
    )
    if args.samples_per_stratum < 1:
        raise SystemExit("--samples-per-stratum must be positive")
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("--workers must be between 1 and 4")
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.implicit_confidence_threshold <= 1.0:
        raise SystemExit("--implicit-confidence-threshold must be between 0 and 1")
    if not 0.0 <= args.calibrated_confidence_threshold <= 1.0:
        raise SystemExit("--calibrated-confidence-threshold must be between 0 and 1")
    if args.max_nontax < 0:
        raise SystemExit("--max-nontax must be non-negative")
    if args.max_implicit_gap_chars < 0:
        raise SystemExit("--max-implicit-gap-chars must be non-negative")
    systems = [value.strip() for value in args.systems.split(",") if value.strip()]
    unknown = sorted(set(systems) - set(SUPPORTED_OPEN_SYSTEMS))
    if unknown:
        raise SystemExit(f"unsupported systems: {', '.join(unknown)}")
    tasks = ["a", "b"] if args.task == "ab" else [args.task]

    tag = f"s{args.samples_per_stratum}_seed{args.seed}"
    if args.audit_only:
        manifest = build_audit_manifest(args, tasks, systems)
        system_tag = ""
        if tuple(systems) != DEFAULT_OPEN_SYSTEMS:
            system_tag = "_" + "_".join(blind._slug(system) for system in systems)
        path = (
            blind.RESULTS
            / f"official_stratified_open_audit_{args.task}{system_tag}_{tag}.json"
        )
        blind.atomic_write_json(path, manifest)
        print(
            f"Audit only: calls={manifest['meta']['estimated_calls']} "
            f"prompt_chars={manifest['meta']['total_prompt_chars']} destination="
            f"{manifest['meta']['external_destination']}",
            flush=True,
        )
        print(f"Wrote {path}", flush=True)
        return 0

    result: dict[str, Any] = {
        "meta": {
            "challenge": "LLMs4OL 2026",
            "protocol": "llms4ol2026_stratified_open_v1",
            "model": args.model,
            "tasks": tasks,
            "systems": systems,
            "samples_per_stratum": args.samples_per_stratum,
            "seed": args.seed,
            "selection_uses_gold_locally": True,
            "gold_visible_to_model": False,
            "confidence_threshold": args.confidence_threshold,
            "implicit_confidence_threshold": args.implicit_confidence_threshold,
            "calibrated_confidence_threshold": args.calibrated_confidence_threshold,
            "max_nontax": args.max_nontax,
            "max_implicit_gap_chars": args.max_implicit_gap_chars,
            "calibration_file": args.calibration_file,
            "allowed_nontax_predicates": (
                sorted(allowed_predicates)
                if allowed_predicates is not None
                else None
            ),
            "strata": STRATUM_DEFINITIONS,
        },
        "tasks": {},
    }
    any_failed = False
    for task in tasks:
        all_samples = blind.load_samples(task, 0, 0)
        selected, selection = select_density_strata(
            all_samples,
            task=task,
            samples_per_stratum=args.samples_per_stratum,
            seed=args.seed,
        )
        args.offset = 0
        args.limit = len(selected)
        task_payload: dict[str, Any] = {
            "meta": {
                "task": task,
                "task_name": blind.TASK_CONFIG[task]["name"],
                "selected_indices": [index for index, _ in selected],
                "n_samples": len(selected),
            },
            "systems": {},
        }
        print(
            f"Task {task.upper()} {blind.TASK_CONFIG[task]['name']}: "
            f"stratified_n={len(selected)} systems={','.join(systems)} workers={args.workers}",
            flush=True,
        )
        for system in systems:
            checkpoint = (
                blind.RESULTS
                / "checkpoints"
                / f"stratified_open_{blind._slug(args.backend)}_task{task}_{system}_{tag}.json"
            )
            payload = blind.run_system(
                args=args,
                task=task,
                system=system,
                indexed_samples=selected,
                checkpoint_path=checkpoint,
            )
            payload = add_stratified_summaries(payload, selection)
            payload["meta"]["selection"] = "non_tax_gold_density_stratified_local_only"
            payload["meta"]["selected_indices"] = [index for index, _ in selected]
            blind.atomic_write_json(checkpoint, payload)
            task_payload["systems"][system] = payload
            summary = payload["summary"]
            any_failed = any_failed or not summary["complete"]
            print(
                f"  {system}: ok={summary['n_success']}/{summary['target_n']} "
                f"NonTaxF1={summary['nontax_f1_micro']['f1']:.4f} "
                f"NonSelfF1={summary['nontax_nonself_f1_micro']['f1']:.4f} "
                f"pred/gold={summary['nontax_pred_total']}/{summary['nontax_gold_total']} "
                f"GS={summary['official_graph_similarity_avg']:.4f}",
                flush=True,
            )

        if set(DEFAULT_OPEN_SYSTEMS).issubset(task_payload["systems"]):
            legacy = task_payload["systems"]["open_vocab"]
            guarded = task_payload["systems"]["open_vocab_guarded"]
            hybrid = build_hybrid_payload(
                task=task,
                selected=selected,
                selection=selection,
                legacy_payload=legacy,
                guarded_payload=guarded,
            )
            task_payload["systems"][HYBRID_SYSTEM] = hybrid
            task_payload["guarded_minus_legacy"] = {
                "overall": _metric_delta(guarded["summary"], legacy["summary"]),
                "strata": {
                    stratum: _metric_delta(
                        guarded["strata"][stratum], legacy["strata"][stratum]
                    )
                    for stratum in STRATA
                },
            }
            task_payload["guarded_hybrid_minus_legacy"] = {
                "overall": _metric_delta(hybrid["summary"], legacy["summary"]),
                "strata": {
                    stratum: _metric_delta(
                        hybrid["strata"][stratum], legacy["strata"][stratum]
                    )
                    for stratum in STRATA
                },
            }
        if {"open_vocab", "open_vocab_guarded_v2"}.issubset(task_payload["systems"]):
            legacy = task_payload["systems"]["open_vocab"]
            guarded_v2 = task_payload["systems"]["open_vocab_guarded_v2"]
            hybrid_v2 = build_hybrid_payload(
                task=task,
                selected=selected,
                selection=selection,
                legacy_payload=legacy,
                guarded_payload=guarded_v2,
                hybrid_system=V2_HYBRID_SYSTEM,
                relation_source="open_vocab_guarded_v2",
            )
            task_payload["systems"][V2_HYBRID_SYSTEM] = hybrid_v2
            task_payload["guarded_v2_minus_legacy"] = {
                "overall": _metric_delta(guarded_v2["summary"], legacy["summary"]),
                "strata": {
                    stratum: _metric_delta(
                        guarded_v2["strata"][stratum], legacy["strata"][stratum]
                    )
                    for stratum in STRATA
                },
            }
            task_payload["guarded_v2_hybrid_minus_legacy"] = {
                "overall": _metric_delta(hybrid_v2["summary"], legacy["summary"]),
                "strata": {
                    stratum: _metric_delta(
                        hybrid_v2["strata"][stratum], legacy["strata"][stratum]
                    )
                    for stratum in STRATA
                },
            }
        result["tasks"][f"task_{task}"] = task_payload

    output = Path(args.output) if args.output else (
        blind.RESULTS
        / f"official_stratified_open_{blind._slug(args.backend)}_{args.task}_{tag}.json"
    )
    blind.atomic_write_json(output, result)
    print(f"Wrote {output}", flush=True)
    return 1 if any_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
