"""Run a frozen endpoint-alias surface stress on official-train samples."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind
import run_official_contamination_probe as probe
from ontobridg.llm.factory import build_llm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PROTOCOL = "llms4ol2026_endpoint_alias_surface_stress_v1"
SYSTEMS = ("closed_vocab", "open_vocab")
DEFAULT_SEED = 20260926
DEFAULT_MANIFEST = RESULTS / (
    "official_contamination_surface_manifest_ab_n15_seed20260926.json"
)
DEFAULT_OUTPUT = RESULTS / (
    "official_contamination_surface_openai_ab_n15_seed20260926.json"
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def alias_for(term: str, *, task: str, sample_index: int, seed: int) -> str:
    digest = _sha256_text(f"{seed}|{task}|{sample_index}|{term}")[:10]
    return f"zeta_{digest}"


def _replace(text: str, mapping: dict[str, str]) -> str:
    output = str(text)
    for original in sorted(mapping, key=lambda value: (-len(value), value)):
        output = re.sub(re.escape(original), mapping[original], output, flags=re.I)
    return output


def _replace_nested(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        return _replace(value, mapping)
    if isinstance(value, list):
        return [_replace_nested(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _replace_nested(item, mapping) for key, item in value.items()}
    return value


def transform_sample(
    sample: dict[str, Any], *, task: str, sample_index: int, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    gold_key = blind.TASK_CONFIG[task]["gold_key"]
    gold = blind.normalize_triples(sample.get(gold_key) or [])
    endpoints = sorted({value for triple in gold for value in (triple[0], triple[2])})
    mapping = {
        endpoint: alias_for(
            endpoint, task=task, sample_index=sample_index, seed=seed
        )
        for endpoint in endpoints
    }
    legal_input = str(sample.get("context") or "")
    if task == "b":
        legal_input += json.dumps(
            {
                "terms": sample.get("terms") or [],
                "types": sample.get("types") or [],
                "initial": sample.get("initial-primitive-ontology-triples") or [],
            },
            ensure_ascii=False,
        )
    visible = [endpoint for endpoint in endpoints if endpoint.lower() in legal_input.lower()]

    transformed = {"id": sample.get("id")}
    for field in ("context", "terms", "types", "initial-primitive-ontology-triples", gold_key):
        if field in sample:
            transformed[field] = _replace_nested(sample[field], mapping)
    transformed_gold = blind.normalize_triples(transformed.get(gold_key) or [])
    metadata = {
        "endpoint_count": len(endpoints),
        "input_visible_endpoint_count": len(visible),
        "input_visible_endpoint_fraction": len(visible) / len(endpoints) if endpoints else 0.0,
        "gold_triple_count": len(gold),
        "fully_input_visible_gold_triple_count": sum(
            triple[0] in visible and triple[2] in visible for triple in gold
        ),
        "transformed_payload_sha256": _sha256_text(
            json.dumps(transformed, ensure_ascii=False, sort_keys=True)
        ),
        "transformed_gold_sha256": _sha256_text(
            json.dumps(transformed_gold, ensure_ascii=False, sort_keys=True)
        ),
    }
    return transformed, metadata


def cache_path(model: str, task: str, system: str, index: int, sample_id: str) -> Path:
    return (
        RESULTS
        / "contamination_surface_cache"
        / blind._slug(model)
        / f"task_{task}"
        / system
        / f"{index:05d}_{_sha256_text(sample_id)[:10]}.json"
    )


def cache_is_replayable(path: Path, prompt_sha256: str) -> bool:
    if not path.exists():
        return False
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        cached.get("protocol") == PROTOCOL
        and cached.get("status") == "ok"
        and cached.get("prompt_sha256") == prompt_sha256
    )


def _source_samples() -> dict[str, dict[int, dict[str, Any]]]:
    return {task: dict(blind.load_samples(task, 0, 0)) for task in ("a", "b")}


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_probe = json.loads(Path(args.probe_manifest).read_text(encoding="utf-8"))
    sources = _source_samples()
    rows = []
    cache_hits = 0
    total_units = 0
    for task in ("a", "b"):
        for index in source_probe["selected_indices"][task]:
            sample = sources[task][int(index)]
            transformed, metadata = transform_sample(
                sample,
                task=task,
                sample_index=int(index),
                seed=args.seed,
            )
            prompt_hashes = {}
            for system in SYSTEMS:
                prompt_hash = _sha256_text(
                    blind.build_prompt(
                        transformed,
                        task,
                        system,
                        args.max_context_chars,
                    )
                )
                prompt_hashes[system] = prompt_hash
                total_units += 1
                path = cache_path(
                    args.model,
                    task,
                    system,
                    int(index),
                    str(sample.get("id") or index),
                )
                cache_hits += cache_is_replayable(path, prompt_hash)
            rows.append(
                {
                    "task": task,
                    "sample_index": int(index),
                    "sample_id": str(sample.get("id") or index),
                    **metadata,
                    "prompt_sha256": prompt_hashes,
                }
            )
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen_not_executed",
            "frozen_before_generation": True,
            "model": args.model,
            "seed": args.seed,
            "systems": list(SYSTEMS),
            "contains_source_text": False,
            "contains_protected_gold": False,
            "gold_visible_to_model": False,
            "transformation": "deterministic_endpoint_aliasing_in_all_legal_inputs",
            "planned_units": total_units,
            "cache_hits": cache_hits,
            "estimated_external_calls": total_units - cache_hits,
        },
        "locked_protocol": {
            "sample_source": "frozen_closed_book_probe_sample_indices",
            "max_context_chars": args.max_context_chars,
            "primary_contrast": "original_minus_endpoint_alias",
            "exact_match_remains_primary": True,
            "allow_generation_tuning": False,
            "allow_calibration_tuning": False,
            "allow_deployment_change": False,
            "interpretation": "surface_dependence_stress_not_verified_contamination",
        },
        "sources": {
            "probe_manifest": {
                "path": str(Path(args.probe_manifest).resolve()),
                "sha256": file_sha256(Path(args.probe_manifest)),
            },
            "task_data": {
                task: {
                    "path": str(blind.TASK_CONFIG[task]["path"].resolve()),
                    "sha256": file_sha256(blind.TASK_CONFIG[task]["path"]),
                }
                for task in ("a", "b")
            },
            "blind_results": {
                task: {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
                for task, path in probe.DEFAULT_BLIND_RESULTS.items()
            },
        },
        "samples": rows,
    }


def preserve_or_write_manifest(
    path: Path, candidate: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        existing_comparable = copy.deepcopy(existing)
        candidate_comparable = copy.deepcopy(candidate)
        for payload in (existing_comparable, candidate_comparable):
            payload["meta"].pop("cache_hits", None)
            payload["meta"].pop("estimated_external_calls", None)
        if existing_comparable != candidate_comparable:
            raise ValueError("existing frozen surface-stress manifest differs")
        return existing, False
    blind.atomic_write_json(path, candidate)
    return candidate, True


def validate_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    meta = manifest["meta"]
    if meta["protocol"] != PROTOCOL or not meta["frozen_before_generation"]:
        raise ValueError("surface-stress manifest is not frozen")
    if meta["model"] != args.model:
        raise ValueError("runtime model differs from frozen surface-stress model")
    if meta["contains_source_text"] or meta["contains_protected_gold"]:
        raise ValueError("surface manifest must contain neither text nor gold")
    for group in ("task_data", "blind_results"):
        for source in manifest["sources"][group].values():
            if file_sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError(f"frozen {group} source hash mismatch")
    probe_source = manifest["sources"]["probe_manifest"]
    if file_sha256(Path(probe_source["path"])) != probe_source["sha256"]:
        raise ValueError("frozen probe manifest hash mismatch")
    if any(
        manifest["locked_protocol"][key]
        for key in (
            "allow_generation_tuning",
            "allow_calibration_tuning",
            "allow_deployment_change",
        )
    ):
        raise ValueError("surface stress cannot authorize tuning or deployment")

    sources = _source_samples()
    for row in manifest["samples"]:
        sample = sources[row["task"]][int(row["sample_index"])]
        transformed, metadata = transform_sample(
            sample,
            task=row["task"],
            sample_index=int(row["sample_index"]),
            seed=int(meta["seed"]),
        )
        for key, value in metadata.items():
            if row[key] != value:
                raise ValueError(f"surface transformation drifted in {key}")
        for system in SYSTEMS:
            prompt = blind.build_prompt(
                transformed,
                row["task"],
                system,
                int(manifest["locked_protocol"]["max_context_chars"]),
            )
            if _sha256_text(prompt) != row["prompt_sha256"][system]:
                raise ValueError("surface-stress prompt hash mismatch")


def reconstruct(
    manifest: dict[str, Any], row: dict[str, Any]
) -> dict[str, Any]:
    sample = dict(blind.load_samples(row["task"], 0, 0))[int(row["sample_index"])]
    transformed, _ = transform_sample(
        sample,
        task=row["task"],
        sample_index=int(row["sample_index"]),
        seed=int(manifest["meta"]["seed"]),
    )
    return transformed


def run_one(
    *,
    llm: Any,
    manifest: dict[str, Any],
    row: dict[str, Any],
    system: str,
    model: str,
    parse_retries: int,
) -> dict[str, Any]:
    path = cache_path(
        model,
        row["task"],
        system,
        int(row["sample_index"]),
        row["sample_id"],
    )
    prompt_hash = row["prompt_sha256"][system]
    if cache_is_replayable(path, prompt_hash):
        return json.loads(path.read_text(encoding="utf-8"))
    previous_attempts = 0
    previous_errors = []
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            previous_attempts = int(previous.get("attempts") or 0)
            previous_errors = list(previous.get("errors") or [])
        except (OSError, json.JSONDecodeError):
            pass
    transformed = reconstruct(manifest, row)
    prompt = blind.build_prompt(
        transformed,
        row["task"],
        system,
        int(manifest["locked_protocol"]["max_context_chars"]),
    )
    started = time.perf_counter()
    errors = []
    raw_text = ""
    for attempt in range(1, max(1, parse_retries) + 1):
        try:
            suffix = (
                ""
                if attempt == 1
                else "\n\nYour previous response was invalid. Return the required JSON only."
            )
            response = llm.complete(
                prompt + suffix,
                task=f"contamination_surface_{row['task']}_{system}",
                meta={
                    "sample_id": row["sample_id"],
                    "sample_index": row["sample_index"],
                },
            )
            raw_text = response.text
            triples = blind.parse_model_payload(raw_text)
            payload = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "ok",
                "model": model,
                "task": row["task"],
                "system": system,
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "prompt_sha256": prompt_hash,
                "triples": triples,
                "raw_text": raw_text,
                "attempts": previous_attempts + attempt,
                "attempts_this_run": attempt,
                "previous_errors": previous_errors,
                "latency_sec": round(time.perf_counter() - started, 3),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            blind.atomic_write_json(path, payload)
            return payload
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    payload = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "error",
        "model": model,
        "task": row["task"],
        "system": system,
        "sample_index": row["sample_index"],
        "sample_id": row["sample_id"],
        "prompt_sha256": prompt_hash,
        "raw_text": raw_text,
        "errors": previous_errors + errors,
        "attempts": previous_attempts + max(1, parse_retries),
        "attempts_this_run": max(1, parse_retries),
        "latency_sec": round(time.perf_counter() - started, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    blind.atomic_write_json(path, payload)
    return payload


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [record for record in records if record.get("status") == "ok"]
    aggregate = blind.aggregate_records(successful, len(records))
    return {
        "n": len(records),
        "n_success": len(successful),
        "typing_f1": aggregate["term_typing_f1_micro"]["f1"],
        "taxonomy_f1": aggregate["taxonomy_f1_micro"]["f1"],
        "nontax_f1": aggregate["nontax_f1_micro"]["f1"],
        "official_graph_similarity_avg": aggregate["official_graph_similarity_avg"],
        "composite_typing_tax_gs": aggregate["composite_typing_tax_gs"],
    }


def original_summary(
    manifest: dict[str, Any], task: str, system: str
) -> dict[str, Any]:
    source = json.loads(
        Path(manifest["sources"]["blind_results"][task]["path"]).read_text(
            encoding="utf-8"
        )
    )
    indices = {
        int(row["sample_index"])
        for row in manifest["samples"]
        if row["task"] == task
    }
    records = [
        row
        for row in source["systems"][system]["samples"]
        if int(row["sample_index"]) in indices
    ]
    return probe._summarize_blind_records(records)


def execute(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    llm = build_llm(args.backend, model=args.model, base_url=args.base_url)
    results_by_key = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                run_one,
                llm=llm,
                manifest=manifest,
                row=row,
                system=system,
                model=args.model,
                parse_retries=args.parse_retries,
            ): (row, system)
            for row in manifest["samples"]
            for system in SYSTEMS
        }
        for future in as_completed(futures):
            row, system = futures[future]
            results_by_key[(row["task"], int(row["sample_index"]), system)] = future.result()

    records = [results_by_key[key] for key in sorted(results_by_key)]
    sample_rows = {(row["task"], int(row["sample_index"])): row for row in manifest["samples"]}
    for record in records:
        if record.get("status") != "ok":
            continue
        source_row = sample_rows[(record["task"], int(record["sample_index"]))]
        transformed = reconstruct(manifest, source_row)
        gold = blind.normalize_triples(
            transformed.get(blind.TASK_CONFIG[record["task"]]["gold_key"]) or []
        )
        record["metrics"] = blind.evaluate_sample(gold, record["triples"])

    summaries = {}
    for task in ("a", "b"):
        summaries[task] = {}
        for system in SYSTEMS:
            alias_records = [
                row for row in records if row["task"] == task and row["system"] == system
            ]
            alias_summary = _summary(alias_records)
            original = original_summary(manifest, task, system)
            fields = (
                "typing_f1",
                "taxonomy_f1",
                "nontax_f1",
                "official_graph_similarity_avg",
                "composite_typing_tax_gs",
            )
            summaries[task][system] = {
                "original": original,
                "endpoint_alias": alias_summary,
                "original_minus_alias": {
                    field: original[field] - alias_summary[field] for field in fields
                },
            }
    successful = [row for row in records if row.get("status") == "ok"]
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "complete" if len(successful) == len(records) else "incomplete",
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": file_sha256(Path(args.manifest)),
            "model": args.model,
            "attempt_ledger": {
                "planned_units": len(records),
                "successful_units": len(successful),
                "failed_final_units": len(records) - len(successful),
                "recorded_unit_attempts": sum(int(row.get("attempts") or 0) for row in records),
                "transport_retries_inside_unit_attempts_observable": False,
            },
        },
        "input_visibility": {
            "endpoints": sum(int(row["endpoint_count"]) for row in manifest["samples"]),
            "visible_endpoints": sum(
                int(row["input_visible_endpoint_count"]) for row in manifest["samples"]
            ),
            "gold_triples": sum(int(row["gold_triple_count"]) for row in manifest["samples"]),
            "fully_visible_gold_triples": sum(
                int(row["fully_input_visible_gold_triple_count"])
                for row in manifest["samples"]
            ),
        },
        "paired_summaries": summaries,
        "samples": [
            {
                "task": row["task"],
                "system": row["system"],
                "sample_index": row["sample_index"],
                "status": row["status"],
                "attempts": row.get("attempts", 0),
                "metrics": row.get("metrics"),
                "errors": row.get("errors"),
            }
            for row in records
        ],
        "interpretation": {
            "contamination_confirmed": False,
            "valid_claim": "paired_endpoint_alias_surface_dependence",
            "invalid_claim": "verified_pretraining_membership",
            "authorize_tuning_or_deployment_change": False,
        },
        "confirmatory_eligible": False,
        "scientific_deployable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--probe-manifest", type=Path, default=probe.DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--backend", choices=["openai", "openai_compat", "api"], default="openai")
    parser.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-completion-tokens", type=int, default=16384)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    candidate = build_manifest(args)
    manifest, created = preserve_or_write_manifest(args.manifest, candidate)
    validate_manifest(manifest, args)
    hits = sum(
        cache_is_replayable(
            cache_path(
                args.model,
                row["task"],
                system,
                int(row["sample_index"]),
                row["sample_id"],
            ),
            row["prompt_sha256"][system],
        )
        for row in manifest["samples"]
        for system in SYSTEMS
    )
    if args.audit_only:
        endpoints = sum(int(row["endpoint_count"]) for row in manifest["samples"])
        visible = sum(
            int(row["input_visible_endpoint_count"]) for row in manifest["samples"]
        )
        print(
            f"{'Frozen' if created else 'Verified'} surface stress "
            f"units={manifest['meta']['planned_units']} cache_hits={hits} "
            f"external_calls={manifest['meta']['planned_units'] - hits} "
            f"visible_endpoints={visible}/{endpoints}",
            flush=True,
        )
        print(f"{'Wrote' if created else 'Preserved'} {args.manifest}", flush=True)
        return 0
    os.environ["ONTOBRIDG_MAX_TOKENS"] = str(args.max_completion_tokens)
    report = execute(args, manifest)
    blind.atomic_write_json(args.output, report)
    ledger = report["meta"]["attempt_ledger"]
    print(
        f"ok={ledger['successful_units']}/{ledger['planned_units']} "
        f"attempts={ledger['recorded_unit_attempts']}",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0 if ledger["failed_final_units"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
