"""Analyze the frozen endpoint-alias surface stress without API calls."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import analyze_official_contamination_probe as probe_analysis
import run_official_blind_eval as blind
import run_official_contamination_probe as probe
import run_official_contamination_surface_stress as surface


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
DEFAULT_OUTPUT = RESULTS / (
    "official_contamination_surface_analysis_ab_n15_seed20260926.json"
)
DEFAULT_REPLICATES = 10_000
DEFAULT_SEED = 20260926
FIELDS = (
    "typing_f1",
    "taxonomy_f1",
    "nontax_f1",
    "official_graph_similarity_avg",
    "composite_typing_tax_gs",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_blind_records(
    manifest: dict[str, Any], task: str, system: str
) -> dict[int, dict[str, Any]]:
    source_info = manifest["sources"]["blind_results"][task]
    source_path = Path(source_info["path"])
    if file_sha256(source_path) != source_info["sha256"]:
        raise ValueError(f"frozen Task {task.upper()} blind-result hash mismatch")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return {
        int(row["sample_index"]): row
        for row in payload["systems"][system]["samples"]
    }


def paired_records(
    manifest: dict[str, Any], result: dict[str, Any]
) -> dict[str, dict[str, list[dict[str, Any]]]]:
    alias_records = {
        (row["task"], row["system"], int(row["sample_index"])): row
        for row in result["samples"]
        if row.get("status") == "ok"
    }
    output: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for task in ("a", "b"):
        task_rows = sorted(
            (row for row in manifest["samples"] if row["task"] == task),
            key=lambda row: int(row["sample_index"]),
        )
        output[task] = {}
        for system in surface.SYSTEMS:
            originals = _load_blind_records(manifest, task, system)
            pairs = []
            for row in task_rows:
                index = int(row["sample_index"])
                alias_key = (task, system, index)
                if index not in originals or alias_key not in alias_records:
                    raise ValueError(
                        f"missing paired record for Task {task.upper()} {system} index {index}"
                    )
                pairs.append(
                    {
                        "sample_index": index,
                        "input_visible_endpoint_fraction": float(
                            row["input_visible_endpoint_fraction"]
                        ),
                        "original": originals[index],
                        "endpoint_alias": alias_records[alias_key],
                    }
                )
            output[task][system] = pairs
    return output


def summarize_pairs(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    if not pairs:
        raise ValueError("cannot summarize an empty paired sample set")
    original = probe._summarize_blind_records(
        [row["original"] for row in pairs]
    )
    endpoint_alias = probe._summarize_blind_records(
        [row["endpoint_alias"] for row in pairs]
    )
    return {
        "n_samples": len(pairs),
        "sample_indices": [row["sample_index"] for row in pairs],
        "original": original,
        "endpoint_alias": endpoint_alias,
        "original_minus_alias": {
            field: original[field] - endpoint_alias[field] for field in FIELDS
        },
    }


def bootstrap_pairs(
    pairs: list[dict[str, Any]], *, replicates: int, seed: int
) -> dict[str, Any]:
    point = summarize_pairs(pairs)
    rng = random.Random(seed)
    draws = {field: [] for field in FIELDS}
    for _ in range(replicates):
        sampled = [rng.choice(pairs) for _ in pairs]
        contrast = summarize_pairs(sampled)["original_minus_alias"]
        for field in FIELDS:
            draws[field].append(float(contrast[field]))
    intervals = {}
    for field in FIELDS:
        values = draws[field]
        interval = probe_analysis._interval(
            values, float(point["original_minus_alias"][field])
        )
        interval["bootstrap_fraction_gt_zero"] = sum(
            value > 0.0 for value in values
        ) / len(values)
        interval["bootstrap_fraction_lt_zero"] = sum(
            value < 0.0 for value in values
        ) / len(values)
        intervals[field] = interval
    return {**point, "confidence_intervals": intervals}


def analyze_scope(
    pairs: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    fully_visible_only: bool,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    for task_position, task in enumerate(("a", "b")):
        output[task] = {}
        for system_position, system in enumerate(surface.SYSTEMS):
            selected = pairs[task][system]
            if fully_visible_only:
                selected = [
                    row
                    for row in selected
                    if row["input_visible_endpoint_fraction"] == 1.0
                ]
            output[task][system] = bootstrap_pairs(
                selected,
                replicates=replicates,
                seed=seed + task_position * 100 + system_position * 10,
            )
    return output


def validate_inputs(
    manifest_path: Path,
    result_path: Path,
    manifest: dict[str, Any],
    result: dict[str, Any],
) -> None:
    if manifest["meta"]["protocol"] != surface.PROTOCOL:
        raise ValueError("unexpected surface-stress manifest protocol")
    if result["meta"]["protocol"] != surface.PROTOCOL:
        raise ValueError("unexpected surface-stress result protocol")
    if result["meta"]["status"] != "complete":
        raise ValueError("surface-stress result is incomplete")
    if result["meta"]["manifest_sha256"] != file_sha256(manifest_path):
        raise ValueError("surface-stress result does not match manifest hash")
    if Path(result["meta"]["manifest"]).resolve() != manifest_path.resolve():
        raise ValueError("surface-stress result points to another manifest")
    if manifest["meta"]["model"] != result["meta"]["model"]:
        raise ValueError("surface-stress model differs between manifest and result")
    if result.get("confirmatory_eligible") or result.get("scientific_deployable"):
        raise ValueError("surface stress must not authorize confirmation or deployment")
    if manifest["locked_protocol"]["primary_contrast"] != (
        "original_minus_endpoint_alias"
    ):
        raise ValueError("unexpected frozen surface-stress contrast")
    if result_path.resolve() == manifest_path.resolve():
        raise ValueError("manifest and result must be separate files")
    surface.validate_manifest(
        manifest, argparse.Namespace(model=manifest["meta"]["model"])
    )


def validate_point_estimates(
    analysis: dict[str, Any], result: dict[str, Any]
) -> None:
    for task in ("a", "b"):
        for system in surface.SYSTEMS:
            observed = analysis[task][system]
            recorded = result["paired_summaries"][task][system]
            for condition in ("original", "endpoint_alias", "original_minus_alias"):
                for field in FIELDS:
                    if observed[condition][field] != recorded[condition][field]:
                        raise ValueError(
                            "surface analysis point estimate differs from source result: "
                            f"Task {task.upper()} {system} {condition} {field}"
                        )


def build_analysis(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = Path(args.manifest)
    result_path = Path(args.result)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    validate_inputs(manifest_path, result_path, manifest, result)
    pairs = paired_records(manifest, result)
    all_selected = analyze_scope(
        pairs,
        fully_visible_only=False,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
    )
    validate_point_estimates(all_selected, result)
    fully_visible = analyze_scope(
        pairs,
        fully_visible_only=True,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed + 1_000,
    )
    return {
        "meta": {
            "protocol": "llms4ol2026_endpoint_alias_surface_analysis_v1",
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": file_sha256(manifest_path),
            "source_result": str(result_path.resolve()),
            "source_result_sha256": file_sha256(result_path),
            "bootstrap_protocol": "paired_task_sample_cluster_bootstrap_v1",
            "bootstrap_replicates": args.bootstrap_replicates,
            "bootstrap_seed": args.bootstrap_seed,
            "confidence_level": 0.95,
            "api_calls": 0,
            "retuning_performed": False,
        },
        "contrast": "original_minus_endpoint_alias",
        "primary_all_selected_samples": all_selected,
        "sensitivity_fully_input_visible_samples": {
            "restriction": "input_visible_endpoint_fraction == 1.0",
            "scoring_scope": "whole_samples_no_per_triple_filtering",
            "by_task_system": fully_visible,
        },
        "interpretation": {
            "contamination_confirmed": False,
            "surface_dependence_direction_consistent_across_tasks": False,
            "task_a_aliasing_improves_observed_performance": True,
            "task_b_aliasing_reduces_observed_performance": True,
            "nontax_contamination_conclusion_supported": False,
            "valid_claim": "endpoint_aliasing_has_task_dependent_performance_effects_in_this_pilot",
            "invalid_claim": "the_model_training_set_contains_the_benchmark",
            "authorize_tuning_or_deployment_change": False,
        },
        "confirmatory_eligible": False,
        "scientific_deployable": False,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=surface.DEFAULT_MANIFEST)
    parser.add_argument("--result", type=Path, default=surface.DEFAULT_OUTPUT)
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
    for task in ("a", "b"):
        for system in surface.SYSTEMS:
            interval = analysis["primary_all_selected_samples"][task][system][
                "confidence_intervals"
            ]["official_graph_similarity_avg"]
            print(
                f"Task {task.upper()} {system}: original-alias GS="
                f"{interval['estimate']:.4f} "
                f"CI=[{interval['lower']:.4f},{interval['upper']:.4f}]",
                flush=True,
            )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
