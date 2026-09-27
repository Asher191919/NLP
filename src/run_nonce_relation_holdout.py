"""Freeze, execute, and score the nonce-entity relation holdout."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import build_nonce_relation_holdout as holdout
import run_official_blind_eval as blind
from ontobridg.llm.factory import build_llm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PROTOCOL = "nonce_relation_holdout_eval_v1"
SYSTEMS = ("open_vocab", "open_vocab_guarded_v2")
DEFAULT_MANIFEST = RESULTS / "nonce_relation_holdout_manifest_n30_seed20260926.json"
DEFAULT_OUTPUT = RESULTS / "nonce_relation_holdout_openai_n30_seed20260926.json"
DEFAULT_CONFIDENCE_THRESHOLD = 0.85
DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD = 0.95
DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD = 0.50
DEFAULT_MAX_NONTAX = 8
DEFAULT_MAX_IMPLICIT_GAP_CHARS = 600


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def package_paths(package_dir: Path) -> dict[str, Path]:
    return {
        "manifest": package_dir / "manifest.json",
        "dataset": package_dir / "dataset.json",
        "model_inputs": package_dir / "model_inputs.jsonl",
    }


def load_and_validate_package(package_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    paths = package_paths(package_dir)
    package_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
    if package_manifest["meta"]["protocol"] != holdout.PROTOCOL:
        raise ValueError("unexpected holdout package protocol")
    if package_manifest["meta"]["status"] != "frozen":
        raise ValueError("holdout package is not frozen")
    for key, filename in (("dataset", "dataset.json"), ("model_inputs", "model_inputs.jsonl")):
        if file_sha256(paths[key]) != package_manifest["files"][filename]["sha256"]:
            raise ValueError(f"frozen {filename} hash mismatch")
    if package_manifest["meta"]["package_id"] != file_sha256(paths["dataset"])[:20]:
        raise ValueError("holdout package id does not match dataset hash")
    for source in package_manifest["official_overlap_audit"]["official_2026_sources"].values():
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("official source hash changed after holdout freeze")
    if dataset["meta"]["protocol"] != holdout.PROTOCOL:
        raise ValueError("dataset and package manifest protocols differ")
    balance = holdout.audit_samples(dataset["samples"])
    if balance != package_manifest["balance_audit"]:
        raise ValueError("holdout balance audit drifted")
    overlap = holdout.official_overlap_audit(dataset["samples"])
    if overlap != package_manifest["official_overlap_audit"]:
        raise ValueError("holdout official-overlap audit drifted")
    input_rows = [
        json.loads(line)
        for line in paths["model_inputs"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_inputs = [
        {"id": row["id"], "context": row["context"]} for row in dataset["samples"]
    ]
    if input_rows != expected_inputs:
        raise ValueError("model inputs do not match the frozen id/context projection")
    if any(set(row) != {"id", "context"} for row in input_rows):
        raise ValueError("model inputs may contain only id and context")
    by_id = {row["id"]: row for row in dataset["samples"]}
    for row in package_manifest["samples"]:
        sample = by_id[row["sample_id"]]
        if _sha256_text(sample["context"]) != row["context_sha256"]:
            raise ValueError("holdout context hash mismatch")
    return package_manifest, dataset


def cache_path(model: str, system: str, sample_index: int, sample_id: str) -> Path:
    return (
        RESULTS
        / "nonce_relation_holdout_cache"
        / blind._slug(model)
        / system
        / f"{sample_index:05d}_{_sha256_text(sample_id)[:10]}.json"
    )


def postprocess_hash() -> str:
    return blind.postprocess_sha256(
        confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
        implicit_confidence_threshold=DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
        calibrated_confidence_threshold=DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD,
        max_nontax=DEFAULT_MAX_NONTAX,
        max_implicit_gap_chars=DEFAULT_MAX_IMPLICIT_GAP_CHARS,
        calibrator=None,
        allowed_nontax_predicates=None,
    )


def cache_has_generation(path: Path, prompt_sha256: str) -> bool:
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
        and isinstance(cached.get("raw_text"), str)
    )


def previous_attempt_state(
    path: Path, prompt_sha256: str
) -> tuple[int, list[str]]:
    if not path.exists():
        return 0, []
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0, []
    if (
        cached.get("protocol") != PROTOCOL
        or cached.get("prompt_sha256") != prompt_sha256
    ):
        return 0, []
    errors = list(cached.get("errors") or cached.get("previous_errors") or [])
    return int(cached.get("attempts") or 0), errors


def cache_is_fully_replayable(
    path: Path, *, system: str, prompt_sha256: str
) -> bool:
    if not cache_has_generation(path, prompt_sha256):
        return False
    if system != "open_vocab_guarded_v2":
        return True
    cached = json.loads(path.read_text(encoding="utf-8"))
    return cached.get("postprocess_sha256") == postprocess_hash()


def model_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {"id": sample["id"], "context": sample["context"]}


def build_prompt(sample: dict[str, Any], system: str, max_context_chars: int) -> str:
    return blind.build_prompt(
        model_sample(sample),
        "a",
        system,
        max_context_chars,
        DEFAULT_CONFIDENCE_THRESHOLD,
        DEFAULT_MAX_NONTAX,
        DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    package_manifest, dataset = load_and_validate_package(Path(args.package_dir))
    rows = []
    generation_hits = 0
    postprocess_hits = 0
    for sample in dataset["samples"]:
        prompt_hashes = {}
        for system in SYSTEMS:
            prompt_hash = _sha256_text(
                build_prompt(sample, system, args.max_context_chars)
            )
            prompt_hashes[system] = prompt_hash
            path = cache_path(
                args.model, system, int(sample["sample_index"]), sample["id"]
            )
            generation_hits += cache_has_generation(path, prompt_hash)
            postprocess_hits += cache_is_fully_replayable(
                path, system=system, prompt_sha256=prompt_hash
            )
        rows.append(
            {
                "sample_index": int(sample["sample_index"]),
                "sample_id": sample["id"],
                "surface_form": sample["surface_form"],
                "context_sha256": _sha256_text(sample["context"]),
                "prompt_sha256": prompt_hashes,
            }
        )
    planned = len(rows) * len(SYSTEMS)
    package_paths_resolved = package_paths(Path(args.package_dir))
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen_not_executed",
            "frozen_before_generation": True,
            "model": args.model,
            "systems": list(SYSTEMS),
            "planned_units": planned,
            "generation_cache_hits": generation_hits,
            "postprocess_cache_hits": postprocess_hits,
            "estimated_external_calls": planned - generation_hits,
            "contains_source_text": False,
            "contains_protected_gold": False,
            "gold_visible_to_model": False,
            "confirmatory_eligible": False,
            "scientific_deployable": False,
        },
        "source_package": {
            "package_id": package_manifest["meta"]["package_id"],
            "manifest": {
                "path": str(package_paths_resolved["manifest"].resolve()),
                "sha256": file_sha256(package_paths_resolved["manifest"]),
            },
            "dataset": {
                "path": str(package_paths_resolved["dataset"].resolve()),
                "sha256": file_sha256(package_paths_resolved["dataset"]),
            },
            "model_inputs": {
                "path": str(package_paths_resolved["model_inputs"].resolve()),
                "sha256": file_sha256(package_paths_resolved["model_inputs"]),
            },
        },
        "locked_protocol": {
            "task_prompt": "llms4ol2026_task_a_context_only",
            "primary_metric": "exact_nontax_micro_f1",
            "secondary_metric": "official_graph_similarity_exact",
            "surface_strata": ["canonical", "paraphrased"],
            "hard_negative_metric": "control_false_positive_rate",
            "predicate_metric": "exact_precision_recall_f1_by_target_predicate",
            "evidence_metric": "exact_gold_recall_by_evidence_mode",
            "max_context_chars": args.max_context_chars,
            "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
            "implicit_confidence_threshold": DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
            "calibrated_confidence_threshold": DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD,
            "max_nontax": DEFAULT_MAX_NONTAX,
            "max_implicit_gap_chars": DEFAULT_MAX_IMPLICIT_GAP_CHARS,
            "postprocess_sha256": postprocess_hash(),
            "calibrator": None,
            "allow_prompt_tuning": False,
            "allow_threshold_tuning": False,
            "allow_calibration": False,
            "allow_deployment_change": False,
            "interpretation": "synthetic_low_exposure_relation_holdout",
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
        dynamic = (
            "generation_cache_hits",
            "postprocess_cache_hits",
            "estimated_external_calls",
        )
        for payload in (existing_comparable, candidate_comparable):
            for key in dynamic:
                payload["meta"].pop(key, None)
        if existing_comparable != candidate_comparable:
            raise ValueError("existing frozen nonce-holdout manifest differs")
        return existing, False
    blind.atomic_write_json(path, candidate)
    return candidate, True


def validate_manifest(
    manifest: dict[str, Any], args: argparse.Namespace
) -> dict[str, Any]:
    if manifest["meta"]["protocol"] != PROTOCOL:
        raise ValueError("unexpected nonce-holdout evaluation protocol")
    if not manifest["meta"]["frozen_before_generation"]:
        raise ValueError("evaluation manifest was not frozen before generation")
    if manifest["meta"]["model"] != args.model:
        raise ValueError("runtime model differs from frozen manifest")
    if manifest["meta"]["contains_source_text"] or manifest["meta"]["contains_protected_gold"]:
        raise ValueError("evaluation manifest must contain neither text nor gold")
    if manifest["locked_protocol"]["calibrator"] is not None:
        raise ValueError("calibrator use is prohibited for this holdout")
    for key in ("allow_prompt_tuning", "allow_threshold_tuning", "allow_calibration", "allow_deployment_change"):
        if manifest["locked_protocol"][key]:
            raise ValueError(f"{key} must remain false")
    for source in manifest["source_package"].values():
        if isinstance(source, dict) and "path" in source:
            if file_sha256(Path(source["path"])) != source["sha256"]:
                raise ValueError("frozen source-package hash mismatch")
    _, dataset = load_and_validate_package(Path(args.package_dir))
    by_id = {row["id"]: row for row in dataset["samples"]}
    for row in manifest["samples"]:
        sample = by_id[row["sample_id"]]
        if _sha256_text(sample["context"]) != row["context_sha256"]:
            raise ValueError("context changed after evaluation freeze")
        for system in SYSTEMS:
            prompt = build_prompt(
                sample,
                system,
                int(manifest["locked_protocol"]["max_context_chars"]),
            )
            if _sha256_text(prompt) != row["prompt_sha256"][system]:
                raise ValueError("prompt changed after evaluation freeze")
    return dataset


def parse_response(
    raw_text: str, *, sample: dict[str, Any], system: str, max_context_chars: int
) -> tuple[list[list[str]], dict[str, Any] | None]:
    if system == "open_vocab_guarded_v2":
        return blind.parse_guarded_v2_model_payload(
            raw_text,
            context=sample["context"][:max_context_chars],
            sample=model_sample(sample),
            task="a",
            confidence_threshold=DEFAULT_CONFIDENCE_THRESHOLD,
            implicit_confidence_threshold=DEFAULT_IMPLICIT_CONFIDENCE_THRESHOLD,
            calibrated_confidence_threshold=DEFAULT_CALIBRATED_CONFIDENCE_THRESHOLD,
            max_nontax=DEFAULT_MAX_NONTAX,
            max_implicit_gap_chars=DEFAULT_MAX_IMPLICIT_GAP_CHARS,
            calibrator=None,
            allowed_nontax_predicates=None,
        )
    return blind.parse_model_payload(raw_text), None


def run_one(
    *,
    llm: Any,
    manifest: dict[str, Any],
    sample: dict[str, Any],
    system: str,
    model: str,
    parse_retries: int,
) -> dict[str, Any]:
    index = int(sample["sample_index"])
    sample_id = sample["id"]
    manifest_row = manifest["samples"][index]
    prompt_hash = manifest_row["prompt_sha256"][system]
    path = cache_path(model, system, index, sample_id)
    max_context_chars = int(manifest["locked_protocol"]["max_context_chars"])
    if cache_has_generation(path, prompt_hash):
        cached = json.loads(path.read_text(encoding="utf-8"))
        triples, diagnostics = parse_response(
            cached["raw_text"],
            sample=sample,
            system=system,
            max_context_chars=max_context_chars,
        )
        cached["triples"] = triples
        cached["postprocess_sha256"] = postprocess_hash()
        if diagnostics is not None:
            cached["guard_diagnostics"] = diagnostics
        blind.atomic_write_json(path, cached)
        return cached

    previous_attempts, previous_errors = previous_attempt_state(path, prompt_hash)

    prompt = build_prompt(sample, system, max_context_chars)
    started = time.perf_counter()
    errors = []
    raw_text = ""
    for attempt in range(1, max(1, parse_retries) + 1):
        try:
            suffix = "" if attempt == 1 else "\n\nReturn the required JSON object only."
            response = llm.complete(
                prompt + suffix,
                task=f"nonce_relation_holdout_{system}",
                meta={"sample_id": sample_id, "sample_index": index},
            )
            raw_text = response.text
            triples, diagnostics = parse_response(
                raw_text,
                sample=sample,
                system=system,
                max_context_chars=max_context_chars,
            )
            record = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "ok",
                "model": model,
                "system": system,
                "sample_index": index,
                "sample_id": sample_id,
                "prompt_sha256": prompt_hash,
                "postprocess_sha256": postprocess_hash(),
                "triples": triples,
                "raw_text": raw_text,
                "attempts": previous_attempts + attempt,
                "attempts_this_run": attempt,
                "previous_attempts": previous_attempts,
                "previous_errors": previous_errors,
                "attempt_errors_this_run": list(errors),
                "errors": previous_errors + errors,
                "latency_sec": round(time.perf_counter() - started, 3),
                "generated_at": datetime.now(timezone.utc).isoformat(),
            }
            if diagnostics is not None:
                record["guard_diagnostics"] = diagnostics
            blind.atomic_write_json(path, record)
            return record
        except Exception as exc:
            errors.append(f"attempt {attempt}: {type(exc).__name__}: {exc}")
    record = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "status": "error",
        "model": model,
        "system": system,
        "sample_index": index,
        "sample_id": sample_id,
        "prompt_sha256": prompt_hash,
        "raw_text": raw_text,
        "errors": previous_errors + errors,
        "attempts": previous_attempts + max(1, parse_retries),
        "attempts_this_run": max(1, parse_retries),
        "previous_attempts": previous_attempts,
        "previous_errors": previous_errors,
        "attempt_errors_this_run": errors,
        "latency_sec": round(time.perf_counter() - started, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    blind.atomic_write_json(path, record)
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in records if row.get("status") == "ok"]
    aggregate = blind.aggregate_records(successful, len(records))
    control_total = sum(len(row["hard_negative_controls"]) for row in successful)
    control_fp = 0
    for row in successful:
        predicted = {
            tuple(triple) for triple in blind.normalize_triples(row.get("triples") or [])
        }
        control_fp += sum(
            tuple(control) in predicted for control in row["hard_negative_controls"]
        )
    by_predicate = {}
    for predicate in holdout.PREDICATES:
        gold = set()
        predicted = set()
        for row in successful:
            gold.update(
                tuple(annotation["triple"])
                for annotation in row["_gold_relation_annotations"]
                if annotation["triple"][1] == predicate
            )
            predicted.update(
                tuple(triple)
                for triple in blind.normalize_triples(row.get("triples") or [])
                if triple[1] == predicate
            )
        by_predicate[predicate] = blind.prf(gold, predicted)
    by_evidence_mode = {}
    for evidence_mode in ("explicit", "cross_sentence"):
        support = 0
        true_positives = 0
        for row in successful:
            predicted = {
                tuple(triple)
                for triple in blind.normalize_triples(row.get("triples") or [])
            }
            targets = [
                tuple(annotation["triple"])
                for annotation in row["_gold_relation_annotations"]
                if annotation["evidence_mode"] == evidence_mode
            ]
            support += len(targets)
            true_positives += sum(target in predicted for target in targets)
        by_evidence_mode[evidence_mode] = {
            "support": support,
            "true_positives": true_positives,
            "recall": true_positives / support if support else 0.0,
        }
    return {
        "n": len(records),
        "n_success": len(successful),
        "typing_f1": aggregate["term_typing_f1_micro"]["f1"],
        "taxonomy_f1": aggregate["taxonomy_f1_micro"]["f1"],
        "nontax_precision": aggregate["nontax_f1_micro"]["precision"],
        "nontax_recall": aggregate["nontax_f1_micro"]["recall"],
        "nontax_f1": aggregate["nontax_f1_micro"]["f1"],
        "official_graph_similarity_avg": aggregate["official_graph_similarity_avg"],
        "hard_negative_controls": control_total,
        "hard_negative_false_positives": control_fp,
        "hard_negative_false_positive_rate": (
            control_fp / control_total if control_total else 0.0
        ),
        "by_predicate": by_predicate,
        "gold_recall_by_evidence_mode": by_evidence_mode,
    }


def execute(
    args: argparse.Namespace, manifest: dict[str, Any], dataset: dict[str, Any]
) -> dict[str, Any]:
    llm = build_llm(args.backend, model=args.model, base_url=args.base_url)
    futures = {}
    records_by_key = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for sample in dataset["samples"]:
            for system in SYSTEMS:
                future = pool.submit(
                    run_one,
                    llm=llm,
                    manifest=manifest,
                    sample=sample,
                    system=system,
                    model=args.model,
                    parse_retries=args.parse_retries,
                )
                futures[future] = (system, int(sample["sample_index"]))
        for future in as_completed(futures):
            records_by_key[futures[future]] = future.result()

    sample_by_index = {
        int(sample["sample_index"]): sample for sample in dataset["samples"]
    }
    records = [records_by_key[key] for key in sorted(records_by_key)]
    for record in records:
        sample = sample_by_index[int(record["sample_index"])]
        record["surface_form"] = sample["surface_form"]
        record["hard_negative_controls"] = sample["hard_negative_controls"]
        record["_gold_relation_annotations"] = sample["gold_relation_annotations"]
        if record.get("status") == "ok":
            record["metrics"] = blind.evaluate_sample(
                sample["gold_relations"], record["triples"]
            )

    summaries = {}
    for system in SYSTEMS:
        system_records = [row for row in records if row["system"] == system]
        summaries[system] = {
            "all": summarize(system_records),
            "by_surface_form": {
                surface_form: summarize(
                    [
                        row
                        for row in system_records
                        if row["surface_form"] == surface_form
                    ]
                )
                for surface_form in ("canonical", "paraphrased")
            },
        }
    successful = [row for row in records if row.get("status") == "ok"]
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "complete" if len(successful) == len(records) else "incomplete",
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": file_sha256(Path(args.manifest)),
            "package_id": manifest["source_package"]["package_id"],
            "model": args.model,
            "attempt_ledger": {
                "planned_units": len(records),
                "successful_units": len(successful),
                "failed_final_units": len(records) - len(successful),
                "recorded_unit_attempts": sum(int(row.get("attempts") or 0) for row in records),
            },
        },
        "summaries": summaries,
        "samples": [
            {
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "surface_form": row["surface_form"],
                "system": row["system"],
                "status": row["status"],
                "attempts": row.get("attempts", 0),
                "metrics": row.get("metrics"),
                "hard_negative_controls": len(row["hard_negative_controls"]),
                "errors": row.get("errors"),
            }
            for row in records
        ],
        "interpretation": {
            "synthetic_low_exposure_holdout": True,
            "naturalistic_benchmark": False,
            "verified_temporal_holdout": False,
            "may_tune_generation_or_thresholds": False,
            "may_fit_or_apply_calibrator": False,
            "may_authorize_deployment": False,
        },
        "confirmatory_eligible": False,
        "scientific_deployable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--package-dir", type=Path, default=holdout.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=["openai", "openai_compat", "api"], default="openai")
    parser.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--max-context-chars", type=int, default=6000)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    candidate = build_manifest(args)
    manifest, created = preserve_or_write_manifest(Path(args.manifest), candidate)
    dataset = validate_manifest(manifest, args)
    generation_hits = sum(
        cache_has_generation(
            cache_path(args.model, system, row["sample_index"], row["sample_id"]),
            row["prompt_sha256"][system],
        )
        for row in manifest["samples"]
        for system in SYSTEMS
    )
    if args.audit_only:
        print(
            f"{'Frozen' if created else 'Verified'} nonce holdout "
            f"units={manifest['meta']['planned_units']} generation_cache_hits={generation_hits} "
            f"external_calls={manifest['meta']['planned_units'] - generation_hits}",
            flush=True,
        )
        print(f"{'Wrote' if created else 'Preserved'} {args.manifest}", flush=True)
        return 0
    os.environ["ONTOBRIDG_MAX_TOKENS"] = str(args.max_completion_tokens)
    report = execute(args, manifest, dataset)
    blind.atomic_write_json(Path(args.output), report)
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
