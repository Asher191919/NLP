"""Audit or execute the frozen official-text relation transfer holdout.

The command is audit-only by default. External calls require the explicit
``--execute`` switch after separate authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import build_official_relation_transfer_holdout as holdout
import run_official_blind_eval as blind
from ontobridg.llm.factory import build_llm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT = RESULTS / "official_relation_transfer_openai_a_n30_seed20260927.json"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen(
    package_dir: Path, eval_manifest_path: Path, model: str
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    paths = {
        "dataset": package_dir / "dataset.json",
        "inputs": package_dir / "model_inputs.jsonl",
        "manifest": package_dir / "manifest.json",
    }
    for path in (*paths.values(), eval_manifest_path):
        if not path.exists():
            raise ValueError(f"missing frozen artifact: {path}")
    package_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
    evaluation = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if package_manifest["meta"]["protocol"] != holdout.PROTOCOL:
        raise ValueError("unexpected package protocol")
    if evaluation["meta"]["protocol"] != holdout.EVAL_PROTOCOL:
        raise ValueError("unexpected evaluation protocol")
    if evaluation["meta"]["model"] != model:
        raise ValueError("runtime model differs from frozen evaluation manifest")
    if evaluation["meta"]["systems"] != list(holdout.SYSTEMS):
        raise ValueError("runtime systems differ from frozen evaluation manifest")
    for key, filename in (("dataset", "dataset.json"), ("inputs", "model_inputs.jsonl")):
        if file_sha256(paths[key]) != package_manifest["files"][filename]["sha256"]:
            raise ValueError(f"frozen {filename} hash mismatch")
    if package_manifest["meta"]["package_id"] != file_sha256(paths["dataset"])[:20]:
        raise ValueError("package id does not match dataset hash")
    if file_sha256(Path(package_manifest["source"]["path"])) != package_manifest["source"]["sha256"]:
        raise ValueError("official source changed after freeze")
    expected_dataset, expected_inputs, expected_manifest, expected_eval = holdout.build_package(
        model=model
    )
    if paths["dataset"].read_bytes() != expected_dataset:
        raise ValueError("frozen dataset differs from deterministic rebuild")
    if paths["inputs"].read_bytes() != expected_inputs:
        raise ValueError("frozen model inputs differ from deterministic rebuild")
    if paths["manifest"].read_bytes() != expected_manifest:
        raise ValueError("frozen package manifest differs from deterministic rebuild")
    if eval_manifest_path.read_bytes() != holdout.json_bytes(expected_eval):
        raise ValueError("frozen evaluation manifest differs from deterministic rebuild")
    input_rows = [
        json.loads(line)
        for line in paths["inputs"].read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_projection = [
        {"id": row["id"], "context": row["context"]}
        for row in dataset["samples"]
    ]
    if input_rows != expected_projection:
        raise ValueError("model input projection drifted")
    if any(set(row) != {"id", "context"} for row in input_rows):
        raise ValueError("model inputs may contain only id and context")
    by_package_index = {
        int(row["package_index"]): row for row in dataset["samples"]
    }
    for row in evaluation["samples"]:
        sample = by_package_index[int(row["package_index"])]
        if holdout.text_sha256(sample["context"]) != row["context_sha256"]:
            raise ValueError("frozen context hash mismatch")
        for system in holdout.SYSTEMS:
            prompt = holdout.build_prompt(
                sample,
                system,
                int(evaluation["locked_protocol"]["max_context_chars"]),
            )
            if holdout.text_sha256(prompt) != row["prompt_sha256"][system]:
                raise ValueError("frozen prompt hash mismatch")
    return package_manifest, dataset, evaluation


def cache_status(
    dataset: dict[str, Any], evaluation: dict[str, Any], model: str
) -> tuple[int, int]:
    hits = 0
    replayable = 0
    manifest_rows = {
        int(row["package_index"]): row for row in evaluation["samples"]
    }
    postprocess_hash = evaluation["locked_protocol"]["postprocess_sha256"]
    for sample in dataset["samples"]:
        row = manifest_rows[int(sample["package_index"])]
        for system in holdout.SYSTEMS:
            path = blind.cache_path(
                model,
                "a",
                system,
                int(sample["source_index"]),
                sample["id"],
            )
            prompt_hash = row["prompt_sha256"][system]
            hits += blind.cache_has_reusable_generation(
                path, prompt_sha256=prompt_hash
            )
            replayable += blind.cache_is_replayable(
                path,
                system=system,
                prompt_sha256=prompt_hash,
                postprocess_hash=postprocess_hash,
            )
    return hits, replayable


def run_one(
    *,
    llm: Any,
    sample: dict[str, Any],
    evaluation: dict[str, Any],
    system: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    package_index = int(sample["package_index"])
    source_index = int(sample["source_index"])
    frozen = evaluation["samples"][package_index]
    prediction = blind.predict_one(
        llm=llm,
        sample={"id": sample["id"], "context": sample["context"]},
        index=source_index,
        task="a",
        system=system,
        model=args.model,
        max_context_chars=int(evaluation["locked_protocol"]["max_context_chars"]),
        confidence_threshold=holdout.CONFIDENCE_THRESHOLD,
        implicit_confidence_threshold=holdout.IMPLICIT_CONFIDENCE_THRESHOLD,
        calibrated_confidence_threshold=holdout.CALIBRATED_CONFIDENCE_THRESHOLD,
        max_nontax=holdout.MAX_NONTAX,
        max_implicit_gap_chars=holdout.MAX_IMPLICIT_GAP_CHARS,
        calibrator=None,
        parse_retries=args.parse_retries,
        refresh_cache=False,
        allowed_nontax_predicates=None,
    )
    if prediction.get("prompt_sha256") != frozen["prompt_sha256"][system]:
        raise ValueError("prediction prompt identity differs from frozen manifest")
    record = {
        "package_index": package_index,
        "source_index": source_index,
        "sample_id": sample["id"],
        "assigned_stratum": sample["assigned_stratum"],
        "density_stratum": (
            "medium_2_to_4"
            if len(sample["visible_nontax_gold"]) <= 4
            else "high_5_to_8"
        ),
        "system": system,
        "status": prediction["status"],
        "attempts": int(prediction.get("attempts") or 0),
        "latency_sec": prediction.get("latency_sec"),
    }
    if prediction["status"] == "ok":
        predicted = blind.normalize_triples(prediction.get("triples") or [])
        record["metrics"] = blind.evaluate_sample(sample["gold_triples"], predicted)
        record["_gold"] = sample["gold_triples"]
        record["_visible_gold"] = sample["visible_nontax_gold"]
        record["_predicted"] = predicted
        if prediction.get("guard_diagnostics") is not None:
            record["guard_diagnostics"] = prediction["guard_diagnostics"]
    else:
        record["errors"] = prediction.get("errors") or []
    return record


def predicate_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    predicates = sorted(
        {
            tuple(triple)[1]
            for record in records
            if record.get("status") == "ok"
            for triple in blind.categorize(record["_gold"])["nontax"]
        }
    )
    output = {}
    for predicate in predicates:
        tp = fp = fn = 0
        for record in records:
            if record.get("status") != "ok":
                continue
            gold = {
                triple
                for triple in blind.categorize(record["_gold"])["nontax"]
                if triple[1] == predicate
            }
            pred = {
                triple
                for triple in blind.categorize(record["_predicted"])["nontax"]
                if triple[1] == predicate
            }
            scores = blind.prf(gold, pred)
            tp += int(scores["tp"])
            fp += int(scores["fp"])
            fn += int(scores["fn"])
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        output[predicate] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return output


def visibility_recall(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {
        "endpoint_visible": {"support": 0, "true_positives": 0},
        "not_endpoint_visible": {"support": 0, "true_positives": 0},
    }
    for record in records:
        if record.get("status") != "ok":
            continue
        gold = {
            triple
            for triple in blind.categorize(record["_gold"])["nontax"]
            if triple[0] != triple[2]
        }
        visible = {tuple(triple) for triple in record["_visible_gold"]}
        predicted = set(blind.categorize(record["_predicted"])["nontax"])
        for key, targets in (
            ("endpoint_visible", visible),
            ("not_endpoint_visible", gold - visible),
        ):
            counts[key]["support"] += len(targets)
            counts[key]["true_positives"] += len(targets & predicted)
    for row in counts.values():
        row["recall"] = (
            row["true_positives"] / row["support"] if row["support"] else 0.0
        )
    return counts


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = blind.aggregate_records(records, len(records))
    aggregate["by_gold_predicate"] = predicate_metrics(records)
    aggregate["gold_recall_by_endpoint_visibility"] = visibility_recall(records)
    return aggregate


def report(
    records: list[dict[str, Any]],
    evaluation: dict[str, Any],
    package_manifest: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summaries = {}
    for system in holdout.SYSTEMS:
        system_rows = [row for row in records if row["system"] == system]
        summaries[system] = {
            "all": summarize(system_rows),
            "by_assigned_stratum": {
                stratum: summarize(
                    [row for row in system_rows if row["assigned_stratum"] == stratum]
                )
                for stratum in holdout.STRATA
            },
            "by_density": {
                density: summarize(
                    [row for row in system_rows if row["density_stratum"] == density]
                )
                for density in ("medium_2_to_4", "high_5_to_8")
            },
        }
    successful = [row for row in records if row.get("status") == "ok"]
    return {
        "meta": {
            "protocol": holdout.EVAL_PROTOCOL,
            "status": "complete" if len(successful) == len(records) else "incomplete",
            "manifest": str(Path(args.eval_manifest).resolve()),
            "manifest_sha256": file_sha256(Path(args.eval_manifest)),
            "package_id": package_manifest["meta"]["package_id"],
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
                key: value
                for key, value in row.items()
                if not key.startswith("_") and key != "guard_diagnostics"
            }
            for row in records
        ],
        "interpretation": {
            "official_training_text": True,
            "gold_stratified_not_unbiased": True,
            "naturalistic_benchmark": False,
            "verified_temporal_holdout": False,
            "may_tune_generation_or_thresholds": False,
            "may_fit_or_apply_calibrator": False,
            "may_authorize_deployment": False,
        },
        "confirmatory_eligible": False,
        "scientific_deployable": False,
    }


def execute(
    args: argparse.Namespace,
    package_manifest: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    llm = build_llm(args.backend, model=args.model, base_url=args.base_url)
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_one,
                llm=llm,
                sample=sample,
                evaluation=evaluation,
                system=system,
                args=args,
            )
            for system in holdout.SYSTEMS
            for sample in dataset["samples"]
        ]
        for future in as_completed(futures):
            records.append(future.result())
    records.sort(key=lambda row: (holdout.SYSTEMS.index(row["system"]), row["package_index"]))
    return report(records, evaluation, package_manifest, args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--package-dir", type=Path, default=holdout.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--eval-manifest", type=Path, default=holdout.DEFAULT_EVAL_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--backend", choices=["openai", "openai_compat", "api"], default="openai")
    parser.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", holdout.DEFAULT_MODEL))
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--max-completion-tokens", type=int, default=8192)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    package_manifest, dataset, evaluation = load_frozen(
        Path(args.package_dir), Path(args.eval_manifest), args.model
    )
    hits, replayable = cache_status(dataset, evaluation, args.model)
    planned = int(evaluation["meta"]["planned_units"])
    print(
        f"Verified official relation transfer units={planned} "
        f"generation_cache_hits={hits} postprocess_cache_hits={replayable} "
        f"external_calls={planned - hits}",
        flush=True,
    )
    if not args.execute:
        print("Audit-only default: add --execute only after separate API authorization.")
        return 0
    os.environ["ONTOBRIDG_MAX_TOKENS"] = str(args.max_completion_tokens)
    payload = execute(args, package_manifest, dataset, evaluation)
    blind.atomic_write_json(Path(args.output), payload)
    ledger = payload["meta"]["attempt_ledger"]
    print(
        f"ok={ledger['successful_units']}/{ledger['planned_units']} "
        f"attempts={ledger['recorded_unit_attempts']}",
        flush=True,
    )
    print(f"Wrote {args.output}")
    return 0 if ledger["failed_final_units"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
