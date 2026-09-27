"""Audit or execute the frozen official predicate-routing holdout.

The runner is audit-only by default. External calls require ``--execute`` after
separate authorization for the frozen ``id``/``context`` payloads.
"""
from __future__ import annotations

import argparse
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import build_official_predicate_routing_holdout as holdout
import run_official_blind_eval as blind
from ontobridg.llm.factory import build_llm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT = RESULTS / (
    "official_predicate_routing_openai_a_n27_seed20260927.json"
)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


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
    package = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    dataset = json.loads(paths["dataset"].read_text(encoding="utf-8"))
    evaluation = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
    if package["meta"]["protocol"] != holdout.PROTOCOL:
        raise ValueError("unexpected package protocol")
    if evaluation["meta"]["protocol"] != holdout.EVAL_PROTOCOL:
        raise ValueError("unexpected evaluation protocol")
    if evaluation["meta"]["model"] != model:
        raise ValueError("runtime model differs from frozen manifest")
    if evaluation["meta"]["systems"] != list(holdout.SYSTEMS):
        raise ValueError("runtime systems differ from frozen manifest")
    for key, filename in (("dataset", "dataset.json"), ("inputs", "model_inputs.jsonl")):
        if holdout.prior.file_sha256(paths[key]) != package["files"][filename]["sha256"]:
            raise ValueError(f"frozen {filename} hash mismatch")
    if package["meta"]["package_id"] != holdout.prior.file_sha256(paths["dataset"])[:20]:
        raise ValueError("package id does not match dataset hash")
    if holdout.prior.file_sha256(Path(package["source"]["path"])) != package["source"]["sha256"]:
        raise ValueError("official source changed after freeze")

    frozen_selected_indices = {
        int(row["source_index"])
        for row in dataset.get("samples") or []
    }
    expected_dataset, expected_inputs, expected_manifest, expected_evaluation = (
        holdout.build_package(
            model=model,
            frozen_selected_indices=frozen_selected_indices,
        )
    )
    if paths["dataset"].read_bytes() != expected_dataset:
        raise ValueError("frozen dataset differs from deterministic rebuild")
    if paths["inputs"].read_bytes() != expected_inputs:
        raise ValueError("frozen model inputs differ from deterministic rebuild")
    if paths["manifest"].read_bytes() != expected_manifest:
        raise ValueError("frozen package manifest differs from deterministic rebuild")
    if eval_manifest_path.read_bytes() != holdout.prior.json_bytes(expected_evaluation):
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
        raise ValueError("model inputs differ from frozen id/context projection")
    if any(set(row) != {"id", "context"} for row in input_rows):
        raise ValueError("external model inputs may contain only id and context")

    manifest_rows = {
        int(row["package_index"]): row for row in evaluation["samples"]
    }
    for sample in dataset["samples"]:
        frozen = manifest_rows[int(sample["package_index"])]
        if holdout.prior.text_sha256(sample["context"]) != frozen["context_sha256"]:
            raise ValueError("frozen context hash mismatch")
        for system in holdout.SYSTEMS:
            prompt = holdout.build_prompt(
                sample,
                system,
                int(evaluation["locked_protocol"]["max_context_chars"]),
            )
            if holdout.prior.text_sha256(prompt) != frozen["prompt_sha256"][system]:
                raise ValueError("frozen prompt hash mismatch")
    return package, dataset, evaluation


def cache_status(
    dataset: dict[str, Any], evaluation: dict[str, Any], model: str
) -> tuple[int, int]:
    hits = replayable = 0
    frozen_rows = {
        int(row["package_index"]): row for row in evaluation["samples"]
    }
    postprocess_hash = evaluation["locked_protocol"]["postprocess_sha256"]
    for sample in dataset["samples"]:
        frozen = frozen_rows[int(sample["package_index"])]
        for system in holdout.SYSTEMS:
            path = blind.cache_path(
                model,
                "a",
                system,
                int(sample["source_index"]),
                sample["id"],
            )
            prompt_hash = frozen["prompt_sha256"][system]
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
    frozen = evaluation["samples"][package_index]
    prediction = blind.predict_one(
        llm=llm,
        sample={"id": sample["id"], "context": sample["context"]},
        index=int(sample["source_index"]),
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
        raise ValueError("prediction prompt differs from frozen manifest")
    record = {
        "package_index": package_index,
        "source_index": int(sample["source_index"]),
        "sample_id": sample["id"],
        "assigned_stratum": sample["assigned_stratum"],
        "system": system,
        "status": prediction["status"],
        "attempts": int(prediction.get("attempts") or 0),
        "latency_sec": prediction.get("latency_sec"),
    }
    if prediction["status"] != "ok":
        record["errors"] = prediction.get("errors") or []
        return record

    predicted = blind.normalize_triples(prediction.get("triples") or [])
    predicted_nontax = {
        row
        for row in blind.categorize(predicted)["nontax"]
        if row[0] != row[2]
    }
    target_set = set(holdout.TARGET_PREDICATES)
    predicted_target = {row for row in predicted_nontax if row[1] in target_set}
    target_gold = {tuple(row) for row in sample["target_gold"]}
    record["metrics"] = blind.evaluate_sample(sample["gold_triples"], predicted)
    record["target_metrics"] = blind.prf(target_gold, predicted_target)
    record["predicate_adherence"] = {
        "target_predictions": len(predicted_target),
        "all_nonself_nontax_predictions": len(predicted_nontax),
        "fraction": (
            len(predicted_target) / len(predicted_nontax)
            if predicted_nontax
            else 1.0
        ),
    }
    record["_target_gold"] = target_gold
    record["_visible_target_gold"] = {
        tuple(row) for row in sample["visible_target_gold"]
    }
    record["_predicted_target"] = predicted_target
    if prediction.get("guard_diagnostics") is not None:
        record["guard_diagnostics"] = prediction["guard_diagnostics"]
    return record


def relation_metrics(
    records: list[dict[str, Any]], *, predicate: str | None = None
) -> dict[str, Any]:
    gold: set[tuple[int, str, str, str]] = set()
    predicted: set[tuple[int, str, str, str]] = set()
    for row in records:
        if row.get("status") != "ok":
            continue
        package_index = int(row["package_index"])
        gold.update(
            (package_index, *triple)
            for triple in row["_target_gold"]
            if predicate is None or triple[1] == predicate
        )
        predicted.update(
            (package_index, *triple)
            for triple in row["_predicted_target"]
            if predicate is None or triple[1] == predicate
        )
    return blind.prf(gold, predicted)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in records if row.get("status") == "ok"]
    target = relation_metrics(successful)
    target["positive_samples"] = sum(
        int(row["target_metrics"]["tp"]) > 0 for row in successful
    )
    target_predictions = sum(
        int(row["predicate_adherence"]["target_predictions"])
        for row in successful
    )
    all_predictions = sum(
        int(row["predicate_adherence"]["all_nonself_nontax_predictions"])
        for row in successful
    )
    visible_support = sum(len(row["_visible_target_gold"]) for row in successful)
    visible_tp = sum(
        len(row["_visible_target_gold"] & row["_predicted_target"])
        for row in successful
    )
    all_support = sum(len(row["_target_gold"]) for row in successful)
    all_tp = sum(
        len(row["_target_gold"] & row["_predicted_target"])
        for row in successful
    )
    output = blind.aggregate_records(successful, len(records))
    output["target_exact"] = target
    output["target_by_gold_predicate"] = {
        predicate: relation_metrics(successful, predicate=predicate)
        for predicate in holdout.TARGET_PREDICATES
    }
    output["target_predicate_adherence"] = {
        "target_predictions": target_predictions,
        "all_nonself_nontax_predictions": all_predictions,
        "fraction": target_predictions / all_predictions if all_predictions else 1.0,
    }
    output["target_recall_by_endpoint_visibility"] = {
        "endpoint_visible": {
            "support": visible_support,
            "true_positives": visible_tp,
            "recall": visible_tp / visible_support if visible_support else 0.0,
        },
        "not_endpoint_visible": {
            "support": all_support - visible_support,
            "true_positives": all_tp - visible_tp,
            "recall": (
                (all_tp - visible_tp) / (all_support - visible_support)
                if all_support > visible_support
                else 0.0
            ),
        },
    }
    return output


def macro_stratum_f1(strata: dict[str, dict[str, Any]]) -> float:
    return sum(float(strata[name]["target_exact"]["f1"]) for name in holdout.STRATA) / len(
        holdout.STRATA
    )


def paired_bootstrap(
    records_by_system: dict[str, list[dict[str, Any]]], *, replicates: int, seed: int
) -> dict[str, Any]:
    by_system = {
        system: {int(row["package_index"]): row for row in records}
        for system, records in records_by_system.items()
    }
    indices_by_stratum = {
        stratum: sorted(
            int(row["package_index"])
            for row in records_by_system[holdout.SYSTEMS[0]]
            if row.get("status") == "ok" and row["assigned_stratum"] == stratum
        )
        for stratum in holdout.STRATA
    }
    if any(not rows for rows in indices_by_stratum.values()):
        return {"status": "incomplete", "valid_replicates": 0}
    rng = random.Random(seed)
    distributions = {system: [] for system in holdout.SYSTEMS}
    deltas: list[float] = []
    for _ in range(replicates):
        sampled = {
            stratum: [rng.choice(indices) for _ in indices]
            for stratum, indices in indices_by_stratum.items()
        }
        metrics = {}
        for system in holdout.SYSTEMS:
            stratum_f1 = []
            for stratum, indices in sampled.items():
                gold = set()
                predicted = set()
                for draw, package_index in enumerate(indices):
                    row = by_system[system][package_index]
                    gold.update((draw, *triple) for triple in row["_target_gold"])
                    predicted.update(
                        (draw, *triple) for triple in row["_predicted_target"]
                    )
                stratum_f1.append(float(blind.prf(gold, predicted)["f1"]))
            metrics[system] = sum(stratum_f1) / len(stratum_f1)
            distributions[system].append(metrics[system])
        deltas.append(metrics[holdout.SYSTEMS[1]] - metrics[holdout.SYSTEMS[0]])
    return {
        "status": "complete",
        "replicates": replicates,
        "seed": seed,
        "macro_target_f1": {
            system: {
                "lower": percentile(values, 0.025),
                "upper": percentile(values, 0.975),
            }
            for system, values in distributions.items()
        },
        "routed_minus_general": {
            "lower": percentile(deltas, 0.025),
            "upper": percentile(deltas, 0.975),
        },
    }


def report(
    records: list[dict[str, Any]],
    package: dict[str, Any],
    evaluation: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    records_by_system = {
        system: [row for row in records if row["system"] == system]
        for system in holdout.SYSTEMS
    }
    summaries = {}
    macro = {}
    for system, system_rows in records_by_system.items():
        strata = {
            stratum: summarize(
                [row for row in system_rows if row["assigned_stratum"] == stratum]
            )
            for stratum in holdout.STRATA
        }
        summaries[system] = {
            "all": summarize(system_rows),
            "by_assigned_stratum": strata,
        }
        macro[system] = macro_stratum_f1(strata)
    successful = [row for row in records if row.get("status") == "ok"]
    complete = len(successful) == len(records)
    gate = evaluation["locked_protocol"]["mechanism_support_gate"]
    routed = summaries[holdout.SYSTEMS[1]]["all"]["target_exact"]
    mechanism_supported = bool(
        complete
        and macro[holdout.SYSTEMS[1]] > macro[holdout.SYSTEMS[0]]
        and int(routed["positive_samples"])
        >= int(gate["minimum_routed_exact_tp_samples"])
    )
    return {
        "meta": {
            "protocol": holdout.EVAL_PROTOCOL,
            "status": "complete" if complete else "incomplete",
            "manifest": str(Path(args.eval_manifest).resolve()),
            "manifest_sha256": holdout.prior.file_sha256(Path(args.eval_manifest)),
            "package_id": package["meta"]["package_id"],
            "model": args.model,
            "attempt_ledger": {
                "planned_units": len(records),
                "successful_units": len(successful),
                "failed_final_units": len(records) - len(successful),
                "recorded_unit_attempts": sum(int(row.get("attempts") or 0) for row in records),
            },
        },
        "primary": {
            "metric": "macro_exact_target_f1_across_assigned_strata",
            "values": macro,
            "routed_minus_general": macro[holdout.SYSTEMS[1]] - macro[holdout.SYSTEMS[0]],
        },
        "summaries": summaries,
        "paired_bootstrap": (
            paired_bootstrap(
                records_by_system,
                replicates=int(evaluation["locked_protocol"]["bootstrap_replicates"]),
                seed=int(evaluation["locked_protocol"]["bootstrap_seed"]),
            )
            if complete
            else {"status": "incomplete", "valid_replicates": 0}
        ),
        "mechanism_gate": {
            "supported": mechanism_supported,
            "minimum_routed_exact_tp_samples": int(
                gate["minimum_routed_exact_tp_samples"]
            ),
            "observed_routed_exact_tp_samples": int(routed["positive_samples"]),
            "routed_macro_f1_strictly_greater": (
                macro[holdout.SYSTEMS[1]] > macro[holdout.SYSTEMS[0]]
            ),
            "retuning_authorized": False,
        },
        "samples": [
            {
                key: value
                for key, value in row.items()
                if not key.startswith("_") and key != "guard_diagnostics"
            }
            for row in records
        ],
        "interpretation": {
            "independent_of_prior_posthoc_audit_samples": True,
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
    package: dict[str, Any],
    dataset: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    llm = build_llm(args.backend, model=args.model, base_url=args.base_url)
    records = []
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
    records.sort(
        key=lambda row: (
            holdout.SYSTEMS.index(row["system"]),
            int(row["package_index"]),
        )
    )
    return report(records, package, evaluation, args)


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
    parser.add_argument(
        "--max-completion-tokens",
        type=int,
        default=holdout.MAX_COMPLETION_TOKENS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    if args.max_completion_tokens != holdout.MAX_COMPLETION_TOKENS:
        raise SystemExit("runtime max completion tokens differ from frozen protocol")
    package, dataset, evaluation = load_frozen(
        Path(args.package_dir), Path(args.eval_manifest), args.model
    )
    hits, replayable = cache_status(dataset, evaluation, args.model)
    planned = int(evaluation["meta"]["planned_units"])
    print(
        f"Verified official predicate-routing units={planned} "
        f"generation_cache_hits={hits} postprocess_cache_hits={replayable} "
        f"external_calls={planned - hits}",
        flush=True,
    )
    if not args.execute:
        print("Audit-only default: add --execute only after separate API authorization.")
        return 0
    os.environ["ONTOBRIDG_MAX_TOKENS"] = str(args.max_completion_tokens)
    payload = execute(args, package, dataset, evaluation)
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
