"""Frozen closed-book contamination probe on LLMs4OL 2026 train samples.

The probe measures benchmark factual probeability, not verified pretraining
membership. Protected gold facts are intentionally exposed only to this probe
and can never be used for generation, calibration, or model selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import run_official_blind_eval as blind
from ontobridg.llm.factory import build_llm


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
PROTOCOL = "llms4ol2026_closed_book_probe_v1"
DEFAULT_SEED = 20260926
DEFAULT_PREFIX = 30
DEFAULT_PER_TASK = 15
DEFAULT_MAX_POSITIVES = 4
DEFAULT_MANIFEST = RESULTS / (
    "official_contamination_probe_manifest_ab_n15_seed20260926.json"
)
DEFAULT_OUTPUT = RESULTS / (
    "official_contamination_probe_openai_ab_n15_seed20260926.json"
)
DEFAULT_BLIND_RESULTS = {
    "a": RESULTS / "official_blind_openai_a_n30_off0.json",
    "b": RESULTS / "official_blind_openai_b_n30_off0.json",
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _token_length(text: str) -> int:
    return len(re.findall(r"[a-z0-9]+", text.lower()))


def _gold(sample: dict[str, Any], task: str) -> list[tuple[str, str, str]]:
    return [
        tuple(row)
        for row in blind.normalize_triples(
            sample.get(blind.TASK_CONFIG[task]["gold_key"]) or []
        )
    ]


def select_indices(*, task: str, prefix: int, count: int, seed: int) -> list[int]:
    if count > prefix:
        raise ValueError("per-task count cannot exceed the frozen prefix")
    rng = random.Random(seed + (0 if task == "a" else 1))
    return sorted(rng.sample(range(prefix), count))


def _object_pool(
    samples: list[dict[str, Any]], task: str
) -> dict[str, list[str]]:
    pool: dict[str, set[str]] = {}
    for sample in samples:
        for _, predicate, obj in _gold(sample, task):
            pool.setdefault(predicate, set()).add(obj)
    return {key: sorted(values) for key, values in pool.items()}


def build_probe_items(
    *,
    sample: dict[str, Any],
    task: str,
    sample_index: int,
    object_pool: dict[str, list[str]],
    seed: int,
    max_positives: int,
) -> list[dict[str, Any]]:
    sample_gold = set(_gold(sample, task))
    eligible: list[tuple[tuple[str, str, str], str]] = []
    for triple in sorted(sample_gold):
        subject, predicate, gold_object = triple
        alternatives = [
            obj
            for obj in object_pool.get(predicate, [])
            if obj != gold_object
            and obj != subject
            and (subject, predicate, obj) not in sample_gold
        ]
        if not alternatives:
            continue
        alternatives.sort(
            key=lambda obj: (
                abs(_token_length(obj) - _token_length(gold_object)),
                obj,
            )
        )
        eligible.append((triple, alternatives[0]))

    rng = random.Random(seed + sample_index + (0 if task == "a" else 100_000))
    rng.shuffle(eligible)
    chosen = eligible[:max_positives]
    items: list[dict[str, Any]] = []
    for triple, negative_object in chosen:
        items.append(
            {
                "triple": list(triple),
                "expected": "entailed",
                "control_type": "official_target_fact",
            }
        )
        items.append(
            {
                "triple": [triple[0], triple[1], negative_object],
                "expected": "not_entailed",
                "control_type": "same_predicate_object_swap_non_gold",
            }
        )
    rng.shuffle(items)
    for position, item in enumerate(items):
        item["item_id"] = f"q{position:02d}"
    return items


def build_prompt(sample_row: dict[str, Any]) -> str:
    statements = [
        {"id": row["item_id"], "triple": row["triple"]}
        for row in sample_row["items"]
    ]
    return (
        "This is a closed-book ontology knowledge probe. Do not use tools, web "
        "search, or external context. For each triple, judge whether it is an "
        "established ontology assertion rather than merely plausible. Return "
        "compact JSON only with exactly this shape: "
        '{"items":[{"id":"q00","label":"entailed|not_entailed",'
        '"confidence":0.0}]}. Confidence is the probability that your chosen '
        "label is correct. Return every supplied id exactly once.\n\n"
        f"TRIPLES:\n{json.dumps(statements, ensure_ascii=False)}"
    )


def probe_cache_path(model: str, row: dict[str, Any]) -> Path:
    sample_hash = _sha256_text(str(row["sample_id"]))[:10]
    return (
        RESULTS
        / "contamination_probe_cache"
        / blind._slug(model)
        / f"task_{row['task']}"
        / f"{int(row['sample_index']):05d}_{sample_hash}.json"
    )


def cache_is_replayable(path: Path, row: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(
        cached.get("protocol") == PROTOCOL
        and cached.get("status") == "ok"
        and cached.get("prompt_sha256") == row["prompt_sha256"]
    )


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    rows = []
    selected_by_task = {}
    query_counts = {"entailed": 0, "not_entailed": 0}
    for task in ("a", "b"):
        all_samples = [sample for _, sample in blind.load_samples(task, 0, 0)]
        selected = select_indices(
            task=task,
            prefix=args.prefix,
            count=args.per_task,
            seed=args.seed,
        )
        selected_by_task[task] = selected
        pool = _object_pool(all_samples, task)
        for index in selected:
            sample = all_samples[index]
            items = build_probe_items(
                sample=sample,
                task=task,
                sample_index=index,
                object_pool=pool,
                seed=args.seed,
                max_positives=args.max_positives,
            )
            if not items:
                raise ValueError(f"task {task} sample {index} has no probe items")
            for item in items:
                query_counts[item["expected"]] += 1
            row = {
                "task": task,
                "sample_index": index,
                "sample_id": str(sample.get("id") or index),
                "items": items,
            }
            row["prompt_sha256"] = _sha256_text(build_prompt(row))
            rows.append(row)

    cache_hits = sum(
        probe_cache_path(args.model, row).exists()
        and cache_is_replayable(probe_cache_path(args.model, row), row)
        for row in rows
    )
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen_not_executed",
            "frozen_before_generation": True,
            "model": args.model,
            "seed": args.seed,
            "prefix": args.prefix,
            "per_task": args.per_task,
            "max_positive_facts_per_sample": args.max_positives,
            "contains_source_text": False,
            "contains_protected_gold": True,
            "gold_visible_to_model": True,
            "gold_visibility_scope": "closed_book_probe_only",
            "membership_labels_available": False,
            "claim_scope": "benchmark_factual_probeability_not_verified_membership",
            "cache_hits": cache_hits,
            "estimated_external_calls": len(rows) - cache_hits,
        },
        "locked_protocol": {
            "one_call_per_sample": True,
            "positive_source": "official_target_triples",
            "negative_source": "same_predicate_object_swap_not_in_sample_gold",
            "primary_probe_metric": "item_level_auroc_from_self_reported_probability",
            "sample_probeability": "balanced_mean_probability_correct",
            "clean_polluted_split": "within_task_bottom_top_rank_tertiles",
            "middle_tertile_reported_but_excluded_from_contrast": True,
            "blind_performance_source": "existing_frozen_context_grounded_results",
            "allow_generation_tuning": False,
            "allow_calibration_tuning": False,
            "allow_deployment_change": False,
        },
        "selected_indices": selected_by_task,
        "query_counts": query_counts,
        "blind_results": {
            task: {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for task, path in DEFAULT_BLIND_RESULTS.items()
        },
        "samples": rows,
    }


def preserve_or_write_manifest(
    path: Path, candidate: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        candidate_meta = dict(candidate["meta"])
        existing_meta = dict(existing["meta"])
        for key in ("cache_hits", "estimated_external_calls"):
            candidate_meta.pop(key, None)
            existing_meta.pop(key, None)
        if existing_meta != candidate_meta:
            raise ValueError("existing contamination manifest metadata differs")
        for key in (
            "locked_protocol",
            "selected_indices",
            "query_counts",
            "blind_results",
            "samples",
        ):
            if existing[key] != candidate[key]:
                raise ValueError(f"existing contamination manifest differs in {key}")
        return existing, False
    blind.atomic_write_json(path, candidate)
    return candidate, True


def validate_manifest(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    meta = manifest["meta"]
    if meta["protocol"] != PROTOCOL or not meta["frozen_before_generation"]:
        raise ValueError("contamination probe manifest is not frozen")
    if meta["model"] != args.model:
        raise ValueError("runtime model differs from frozen probe model")
    if meta["contains_source_text"] or not meta["contains_protected_gold"]:
        raise ValueError("unexpected probe payload-scope declaration")
    if meta["claim_scope"] != "benchmark_factual_probeability_not_verified_membership":
        raise ValueError("probe claim scope drifted")
    for source in manifest["blind_results"].values():
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("frozen blind-result source hash mismatch")
    locked = manifest["locked_protocol"]
    if any(
        locked[key]
        for key in (
            "allow_generation_tuning",
            "allow_calibration_tuning",
            "allow_deployment_change",
        )
    ):
        raise ValueError("contamination probe cannot authorize tuning or deployment")

    expected_counts = {"entailed": 0, "not_entailed": 0}
    seen_samples = set()
    rows_by_task: dict[str, list[int]] = {"a": [], "b": []}
    all_samples = {
        task: dict(blind.load_samples(task, 0, 0)) for task in ("a", "b")
    }
    for row in manifest["samples"]:
        if "context" in row:
            raise ValueError("probe manifest must not contain source context")
        key = (row["task"], int(row["sample_index"]))
        if key in seen_samples:
            raise ValueError("duplicate probe sample")
        seen_samples.add(key)
        rows_by_task[row["task"]].append(int(row["sample_index"]))
        if _sha256_text(build_prompt(row)) != row["prompt_sha256"]:
            raise ValueError("probe prompt hash mismatch")
        sample_gold = set(_gold(all_samples[row["task"]][key[1]], row["task"]))
        item_ids = set()
        row_counts = {"entailed": 0, "not_entailed": 0}
        for item in row["items"]:
            item_id = item["item_id"]
            if item_id in item_ids:
                raise ValueError("duplicate probe item id")
            item_ids.add(item_id)
            expected = item["expected"]
            triple = tuple(item["triple"])
            if expected not in row_counts:
                raise ValueError("unknown probe expected label")
            if expected == "entailed" and triple not in sample_gold:
                raise ValueError("probe positive is not in official sample gold")
            if expected == "not_entailed" and triple in sample_gold:
                raise ValueError("probe negative overlaps official sample gold")
            row_counts[expected] += 1
            expected_counts[expected] += 1
        if row_counts["entailed"] != row_counts["not_entailed"]:
            raise ValueError("probe sample is not class-balanced")
    if expected_counts != manifest["query_counts"]:
        raise ValueError("probe query counts differ from manifest summary")
    for task in ("a", "b"):
        if sorted(rows_by_task[task]) != manifest["selected_indices"][task]:
            raise ValueError("probe selected indices differ from sample rows")


def parse_probe_response(text: str, expected_ids: Iterable[str]) -> list[dict[str, Any]]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.I)
    if fenced:
        cleaned = fenced.group(1).strip()
    payload = json.loads(cleaned)
    rows = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise ValueError("probe response must contain an items list")
    expected = set(expected_ids)
    parsed = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("probe item must be an object")
        item_id = str(row.get("id") or "")
        label = str(row.get("label") or "").strip().lower()
        if item_id not in expected or item_id in seen:
            raise ValueError("probe response has missing, duplicate, or unknown id")
        if label not in {"entailed", "not_entailed"}:
            raise ValueError("probe label must be entailed or not_entailed")
        confidence = float(row.get("confidence"))
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("probe confidence must be in [0,1]")
        parsed.append(
            {"item_id": item_id, "label": label, "confidence": confidence}
        )
        seen.add(item_id)
    if seen != expected:
        raise ValueError("probe response does not cover every id")
    return parsed


def run_one(
    *, llm: Any, row: dict[str, Any], model: str, parse_retries: int
) -> dict[str, Any]:
    path = probe_cache_path(model, row)
    if cache_is_replayable(path, row):
        return json.loads(path.read_text(encoding="utf-8"))
    previous_attempts = 0
    previous_errors: list[str] = []
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("protocol") == PROTOCOL:
                previous_attempts = int(previous.get("attempts") or 0)
                previous_errors = [str(error) for error in previous.get("errors") or []]
        except (OSError, json.JSONDecodeError):
            pass
    prompt = build_prompt(row)
    errors = []
    raw_text = ""
    started = time.perf_counter()
    for attempt in range(1, max(1, parse_retries) + 1):
        try:
            suffix = (
                ""
                if attempt == 1
                else "\n\nYour previous response was invalid. Return the required JSON only."
            )
            response = llm.complete(
                prompt + suffix,
                task=f"contamination_probe_{row['task']}",
                meta={
                    "sample_id": row["sample_id"],
                    "sample_index": row["sample_index"],
                },
            )
            raw_text = response.text
            predictions = parse_probe_response(
                raw_text, [item["item_id"] for item in row["items"]]
            )
            payload = {
                "schema_version": 1,
                "protocol": PROTOCOL,
                "status": "ok",
                "model": model,
                "task": row["task"],
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "prompt_sha256": row["prompt_sha256"],
                "predictions": predictions,
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
        "sample_index": row["sample_index"],
        "sample_id": row["sample_id"],
        "prompt_sha256": row["prompt_sha256"],
        "raw_text": raw_text,
        "errors": previous_errors + errors,
        "attempts": previous_attempts + max(1, parse_retries),
        "attempts_this_run": max(1, parse_retries),
        "latency_sec": round(time.perf_counter() - started, 3),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    blind.atomic_write_json(path, payload)
    return payload


def roc_auc(scores: list[float], labels: list[bool]) -> float | None:
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    if not positives or not negatives:
        return None
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            wins += (positive > negative) + 0.5 * (positive == negative)
    return wins / (len(positives) * len(negatives))


def score_records(
    manifest: dict[str, Any], cached: list[dict[str, Any]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_rows = {
        (row["task"], int(row["sample_index"])): row
        for row in manifest["samples"]
    }
    scored_samples = []
    all_scores = []
    all_labels = []
    for record in cached:
        if record.get("status") != "ok":
            continue
        key = (record["task"], int(record["sample_index"]))
        source = manifest_rows[key]
        expected = {row["item_id"]: row for row in source["items"]}
        item_rows = []
        positive_correctness = []
        negative_correctness = []
        for prediction in record["predictions"]:
            item = expected[prediction["item_id"]]
            label = item["expected"] == "entailed"
            probability = (
                float(prediction["confidence"])
                if prediction["label"] == "entailed"
                else 1.0 - float(prediction["confidence"])
            )
            correct_probability = probability if label else 1.0 - probability
            (positive_correctness if label else negative_correctness).append(
                correct_probability
            )
            all_scores.append(probability)
            all_labels.append(label)
            item_rows.append(
                {
                    "item_id": prediction["item_id"],
                    "expected": item["expected"],
                    "predicted": prediction["label"],
                    "confidence": prediction["confidence"],
                    "probability_entailed": probability,
                    "correct": prediction["label"] == item["expected"],
                }
            )
        positive_mean = statistics.fmean(positive_correctness)
        negative_mean = statistics.fmean(negative_correctness)
        scored_samples.append(
            {
                "task": record["task"],
                "sample_index": int(record["sample_index"]),
                "probeability": (positive_mean + negative_mean) / 2.0,
                "positive_correct_probability": positive_mean,
                "negative_correct_probability": negative_mean,
                "items": item_rows,
            }
        )
    return (
        {
            "n_items": len(all_scores),
            "n_positive": sum(all_labels),
            "n_negative": len(all_labels) - sum(all_labels),
            "auroc": roc_auc(all_scores, all_labels),
            "accuracy": (
                sum(
                    (score >= 0.5) == label
                    for score, label in zip(all_scores, all_labels)
                )
                / len(all_scores)
                if all_scores
                else None
            ),
        },
        scored_samples,
    )


def rank_tertiles(scored_samples: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for task in ("a", "b"):
        rows = sorted(
            (row for row in scored_samples if row["task"] == task),
            key=lambda row: (row["probeability"], row["sample_index"]),
        )
        k = len(rows) // 3
        output[task] = {
            "clean_low_probeability": [row["sample_index"] for row in rows[:k]],
            "middle": [row["sample_index"] for row in rows[k : len(rows) - k]],
            "polluted_high_probeability": [
                row["sample_index"] for row in rows[len(rows) - k :]
            ],
        }
    return output


def _summarize_blind_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    def f1_for(metric: str) -> float:
        tp = sum(int(row["metrics"][metric]["tp"]) for row in records)
        fp = sum(int(row["metrics"][metric]["fp"]) for row in records)
        fn = sum(int(row["metrics"][metric]["fn"]) for row in records)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return (
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

    typing_f1 = f1_for("typing")
    taxonomy_f1 = f1_for("taxonomy")
    nontax_f1 = f1_for("nontax")
    graph = statistics.fmean(
        float(row["metrics"]["official_graph_similarity"]["graph_similarity"])
        for row in records
    )
    return {
        "n": len(records),
        "typing_f1": typing_f1,
        "taxonomy_f1": taxonomy_f1,
        "nontax_f1": nontax_f1,
        "official_graph_similarity_avg": graph,
        "composite_typing_tax_gs": (typing_f1 + taxonomy_f1 + graph) / 3.0,
    }


def stratified_blind_performance(
    manifest: dict[str, Any], splits: dict[str, Any]
) -> dict[str, Any]:
    output = {}
    for task in ("a", "b"):
        source_info = manifest["blind_results"][task]
        source = json.loads(Path(source_info["path"]).read_text(encoding="utf-8"))
        task_output = {}
        for system in ("closed_vocab", "open_vocab"):
            records = source["systems"][system]["samples"]
            by_index = {int(row["sample_index"]): row for row in records}
            system_output = {}
            for split_name, indices in splits[task].items():
                system_output[split_name] = _summarize_blind_records(
                    [by_index[index] for index in indices]
                )
            clean = system_output["clean_low_probeability"]
            polluted = system_output["polluted_high_probeability"]
            system_output["polluted_minus_clean"] = {
                key: polluted[key] - clean[key]
                for key in clean
                if key != "n"
            }
            task_output[system] = system_output
        output[task] = task_output
    return output


def execute(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    llm = build_llm(args.backend, model=args.model, base_url=args.base_url)
    records_by_key = {}

    def run_row(row: dict[str, Any]) -> dict[str, Any]:
        return run_one(
            llm=llm,
            row=row,
            model=args.model,
            parse_retries=args.parse_retries,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_row, row): row for row in manifest["samples"]}
        for future in as_completed(futures):
            row = futures[future]
            record = future.result()
            records_by_key[(row["task"], int(row["sample_index"]))] = record

    records = [records_by_key[key] for key in sorted(records_by_key)]
    overall, scored_samples = score_records(manifest, records)
    splits = rank_tertiles(scored_samples)
    performance = stratified_blind_performance(manifest, splits)
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
                "recorded_unit_attempts": sum(
                    int(row.get("attempts") or 0) for row in records
                ),
                "transport_retries_inside_unit_attempts_observable": False,
            },
            "external_calls_for_scoring": 0,
        },
        "probe_metrics": overall,
        "sample_scores": scored_samples,
        "rank_tertile_splits": splits,
        "context_grounded_blind_performance": performance,
        "interpretation": {
            "contamination_confirmed": False,
            "valid_claim": "benchmark_factual_probeability_and_association_with_task_performance",
            "invalid_claim": "verified_pretraining_membership",
            "negative_controls_may_include_unlabeled_true_facts": True,
            "authorize_generation_or_calibration_tuning": False,
            "authorize_deployment_change": False,
        },
        "confirmatory_eligible": False,
        "scientific_deployable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prefix", type=int, default=DEFAULT_PREFIX)
    parser.add_argument("--per-task", type=int, default=DEFAULT_PER_TASK)
    parser.add_argument("--max-positives", type=int, default=DEFAULT_MAX_POSITIVES)
    parser.add_argument(
        "--backend",
        choices=["openai", "openai_compat", "api"],
        default="openai",
    )
    parser.add_argument(
        "--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini")
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--parse-retries", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.workers <= 4:
        raise SystemExit("--workers must be between 1 and 4")
    candidate = build_manifest(args)
    manifest, created = preserve_or_write_manifest(args.manifest, candidate)
    validate_manifest(manifest, args)
    if args.audit_only:
        hits = sum(
            cache_is_replayable(probe_cache_path(args.model, row), row)
            for row in manifest["samples"]
        )
        print(
            f"{'Frozen' if created else 'Verified'} contamination probe "
            f"samples={len(manifest['samples'])} cache_hits={hits} "
            f"external_calls={len(manifest['samples']) - hits}",
            flush=True,
        )
        print(f"{'Wrote' if created else 'Preserved'} {args.manifest}", flush=True)
        return 0
    report = execute(args, manifest)
    blind.atomic_write_json(args.output, report)
    ledger = report["meta"]["attempt_ledger"]
    auc = report["probe_metrics"]["auroc"]
    print(
        f"ok={ledger['successful_units']}/{ledger['planned_units']} "
        f"attempts={ledger['recorded_unit_attempts']} auroc={auc:.4f}",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0 if ledger["failed_final_units"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
