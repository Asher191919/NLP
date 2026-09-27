"""Analyze the frozen official-train contamination probe without API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable

import run_official_blind_eval as blind
import run_official_contamination_probe as probe


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT = RESULTS / (
    "official_contamination_probe_analysis_ab_n15_seed20260926.json"
)
DEFAULT_REPLICATES = 10_000
DEFAULT_SEED = 20260926


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot take percentile of empty values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _interval(values: list[float], estimate: float) -> dict[str, Any]:
    return {
        "estimate": estimate,
        "lower": percentile(values, 0.025),
        "upper": percentile(values, 0.975),
        "bootstrap_valid_replicates": len(values),
    }


def channel_for_predicate(predicate: str) -> str:
    if predicate in blind.INSTANCE_RELATIONS:
        return "typing"
    if predicate in blind.TAXONOMY_RELATIONS:
        return "taxonomy"
    return "nontax"


def annotated_samples(
    manifest: dict[str, Any], result: dict[str, Any]
) -> list[dict[str, Any]]:
    manifest_rows = {
        (row["task"], int(row["sample_index"])): row
        for row in manifest["samples"]
    }
    output = []
    for sample in result["sample_scores"]:
        key = (sample["task"], int(sample["sample_index"]))
        source_items = {
            row["item_id"]: row for row in manifest_rows[key]["items"]
        }
        items = []
        for scored in sample["items"]:
            source = source_items[scored["item_id"]]
            items.append(
                {
                    **scored,
                    "predicate": source["triple"][1],
                    "channel": channel_for_predicate(source["triple"][1]),
                    "control_type": source["control_type"],
                }
            )
        output.append({**sample, "items": items})
    return output


def summarize_items(items: list[dict[str, Any]]) -> dict[str, Any]:
    labels = [row["expected"] == "entailed" for row in items]
    scores = [float(row["probability_entailed"]) for row in items]
    positives = [score for score, label in zip(scores, labels) if label]
    negatives = [score for score, label in zip(scores, labels) if not label]
    positive_accuracy = sum(score >= 0.5 for score in positives) / len(positives)
    negative_accuracy = sum(score < 0.5 for score in negatives) / len(negatives)
    return {
        "n_items": len(items),
        "n_positive": len(positives),
        "n_negative": len(negatives),
        "auroc": probe.roc_auc(scores, labels),
        "accuracy": sum((score >= 0.5) == label for score, label in zip(scores, labels))
        / len(scores),
        "positive_accuracy": positive_accuracy,
        "negative_accuracy": negative_accuracy,
        "balanced_accuracy": (positive_accuracy + negative_accuracy) / 2.0,
        "mean_probability_entailed_positive": sum(positives) / len(positives),
        "mean_probability_entailed_negative": sum(negatives) / len(negatives),
    }


def grouped_probe_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    all_items = [item for sample in samples for item in sample["items"]]
    return {
        "overall": summarize_items(all_items),
        "by_task": {
            task: summarize_items(
                [
                    item
                    for sample in samples
                    if sample["task"] == task
                    for item in sample["items"]
                ]
            )
            for task in ("a", "b")
        },
        "by_channel": {
            channel: summarize_items(
                [item for item in all_items if item["channel"] == channel]
            )
            for channel in ("typing", "taxonomy", "nontax")
            if any(item["channel"] == channel for item in all_items)
        },
    }


def bootstrap_probe(
    samples: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    rng = random.Random(seed)

    def one_scope(scope: list[dict[str, Any]]) -> dict[str, Any]:
        point = summarize_items([item for sample in scope for item in sample["items"]])
        fields = ("auroc", "accuracy", "positive_accuracy", "negative_accuracy")
        draws = {field: [] for field in fields}
        for _ in range(replicates):
            sampled = [rng.choice(scope) for _ in scope]
            metrics = summarize_items(
                [item for sample in sampled for item in sample["items"]]
            )
            for field in fields:
                value = metrics[field]
                if value is not None:
                    draws[field].append(float(value))
        return {
            field: _interval(draws[field], float(point[field])) for field in fields
        }

    return {
        "protocol": "sample_cluster_bootstrap_probe_v1",
        "resampling_unit": "task_sample",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": 0.95,
        "overall": one_scope(samples),
        "by_task": {
            task: one_scope([sample for sample in samples if sample["task"] == task])
            for task in ("a", "b")
        },
    }


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_ss = sum((a - x_mean) ** 2 for a in x)
    y_ss = sum((b - y_mean) ** 2 for b in y)
    if x_ss == 0.0 or y_ss == 0.0:
        return None
    return numerator / math.sqrt(x_ss * y_ss)


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(_average_ranks(x), _average_ranks(y))


def _blind_records(
    manifest: dict[str, Any], task: str, system: str
) -> dict[int, dict[str, Any]]:
    source = json.loads(
        Path(manifest["blind_results"][task]["path"]).read_text(encoding="utf-8")
    )
    return {
        int(row["sample_index"]): row for row in source["systems"][system]["samples"]
    }


def association_analysis(
    manifest: dict[str, Any],
    samples: list[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    for task in ("a", "b"):
        task_samples = [sample for sample in samples if sample["task"] == task]
        task_output = {}
        for system in ("closed_vocab", "open_vocab"):
            records = _blind_records(manifest, task, system)
            pairs = [
                (
                    float(sample["probeability"]),
                    float(
                        records[int(sample["sample_index"])]["metrics"][
                            "official_graph_similarity"
                        ]["graph_similarity"]
                    ),
                )
                for sample in task_samples
            ]
            point = spearman(
                [row[0] for row in pairs], [row[1] for row in pairs]
            )
            rng = random.Random(seed + (0 if task == "a" else 100) + (0 if system == "closed_vocab" else 10))
            draws = []
            for _ in range(replicates):
                sampled = [rng.choice(pairs) for _ in pairs]
                value = spearman(
                    [row[0] for row in sampled], [row[1] for row in sampled]
                )
                if value is not None:
                    draws.append(value)
            task_output[system] = (
                _interval(draws, float(point))
                if point is not None and draws
                else {
                    "estimate": point,
                    "lower": None,
                    "upper": None,
                    "bootstrap_valid_replicates": len(draws),
                }
            )
        output[task] = task_output
    return {
        "metric": "spearman_probeability_vs_per_sample_official_graph_similarity",
        "conditional_on": "frozen_probe_scores_and_existing_blind_predictions",
        "by_task_system": output,
    }


def split_difference_bootstrap(
    manifest: dict[str, Any],
    result: dict[str, Any],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    fields = (
        "typing_f1",
        "taxonomy_f1",
        "nontax_f1",
        "official_graph_similarity_avg",
        "composite_typing_tax_gs",
    )
    output = {}
    for task in ("a", "b"):
        output[task] = {}
        splits = result["rank_tertile_splits"][task]
        for system in ("closed_vocab", "open_vocab"):
            records = _blind_records(manifest, task, system)
            clean = [records[int(index)] for index in splits["clean_low_probeability"]]
            polluted = [
                records[int(index)] for index in splits["polluted_high_probeability"]
            ]
            clean_point = probe._summarize_blind_records(clean)
            polluted_point = probe._summarize_blind_records(polluted)
            point = {
                field: polluted_point[field] - clean_point[field] for field in fields
            }
            rng = random.Random(seed + (0 if task == "a" else 100) + (0 if system == "closed_vocab" else 10))
            draws = {field: [] for field in fields}
            for _ in range(replicates):
                clean_sample = [rng.choice(clean) for _ in clean]
                polluted_sample = [rng.choice(polluted) for _ in polluted]
                clean_metrics = probe._summarize_blind_records(clean_sample)
                polluted_metrics = probe._summarize_blind_records(polluted_sample)
                for field in fields:
                    draws[field].append(
                        polluted_metrics[field] - clean_metrics[field]
                    )
            output[task][system] = {
                field: _interval(draws[field], point[field]) for field in fields
            }
    return {
        "contrast": "polluted_high_probeability_minus_clean_low_probeability",
        "split_is_fixed_from_observed_probeability_rank_tertiles": True,
        "by_task_system": output,
    }


def build_analysis(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    result_path = Path(args.result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    samples = annotated_samples(manifest, result)
    grouped = grouped_probe_metrics(samples)
    bootstrap = bootstrap_probe(
        samples, replicates=args.bootstrap_replicates, seed=args.bootstrap_seed
    )
    association = association_analysis(
        manifest,
        samples,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    split_bootstrap = split_difference_bootstrap(
        manifest,
        result,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    return {
        "meta": {
            "protocol": "llms4ol2026_closed_book_probe_analysis_v1",
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": file_sha256(manifest_path),
            "source_result": str(result_path.resolve()),
            "source_result_sha256": file_sha256(result_path),
            "api_calls": 0,
            "retuning_performed": False,
        },
        "probe_metrics": grouped,
        "sample_cluster_bootstrap": bootstrap,
        "probeability_task_performance_association": association,
        "rank_tertile_performance_difference_bootstrap": split_bootstrap,
        "interpretation": {
            "contamination_confirmed": False,
            "probe_discriminates_official_gold_from_constructed_non_gold_controls": True,
            "task_performance_association_must_be_consistent_across_tasks_and_systems": False,
            "negative_controls_may_include_unlabeled_true_facts": True,
            "self_reported_confidence_is_not_a_likelihood": True,
            "valid_claim": "closed_book_benchmark_facts_are_highly_probeable_in_this_pilot",
            "invalid_claim": "the_model_training_set_contains_the_benchmark",
        },
        "confirmatory_eligible": False,
        "scientific_deployable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=probe.DEFAULT_MANIFEST)
    parser.add_argument("--result", type=Path, default=probe.DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_SEED)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.bootstrap_replicates < 1:
        raise SystemExit("--bootstrap-replicates must be positive")
    analysis = build_analysis(args)
    blind.atomic_write_json(args.output, analysis)
    overall = analysis["probe_metrics"]["overall"]
    interval = analysis["sample_cluster_bootstrap"]["overall"]["auroc"]
    print(
        f"auroc={overall['auroc']:.4f} "
        f"ci=[{interval['lower']:.4f},{interval['upper']:.4f}]",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
