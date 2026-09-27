"""Freeze and run the official semantic metric on frozen v5 outputs.

This is a secondary, relation-only analysis. Exact matching remains primary and
the semantic result cannot alter thresholds, rules, gates, or deployment state.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import run_official_blind_eval as blind
import run_v5_confirmatory as confirmatory


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OFFICIAL_SOURCE = (
    ROOT
    / "data"
    / "LLMs4OL-Challenge"
    / "2026"
    / "metrics"
    / "graph_similarity.py"
)
SUPERSEDED_PROTOCOL = RESULTS / (
    "v5_axiom_confirmatory_semantic_protocol_a_n24_seed20260925.json"
)
DEFAULT_PROTOCOL = RESULTS / (
    "v5_axiom_confirmatory_semantic_protocol_v2_a_n24_seed20260925.json"
)
DEFAULT_OUTPUT = RESULTS / (
    "v5_axiom_confirmatory_semantic_secondary_a_n24_seed20260925.json"
)
PROTOCOL = "v5_axiom_confirmatory_semantic_secondary_v2"
MODEL_ID = "nomic-ai/nomic-embed-text-v1.5"
MODEL_REVISION = "e9b6763023c676ca8431644204f50c2b100d9aab"
REMOTE_CODE_MODEL_ID = "nomic-ai/nomic-bert-2048"
REMOTE_CODE_REVISION = "7710840340a098cfb869c4f65e87cf2b1b70caca"
THRESHOLD = 0.90


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    indices = [int(row["sample_index"]) for row in manifest["samples"]]
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "frozen_not_executed",
            "frozen_before_semantic_scoring": True,
            "contains_source_text": False,
            "contains_gold_triples": False,
            "api_calls": 0,
            "supersedes_unexecuted_protocol": (
                {
                    "path": str(SUPERSEDED_PROTOCOL.resolve()),
                    "sha256": file_sha256(SUPERSEDED_PROTOCOL),
                    "reason": "nested_remote_code_revision_was_not_locked",
                }
                if SUPERSEDED_PROTOCOL.exists()
                else None
            ),
        },
        "claim_scope": "task_a_endpoint_visible_axiom_slice_nontax_only",
        "primary_analysis": {
            "metric": "official_exact_nontax_micro",
            "source_score": str(Path(args.score).resolve()),
            "source_score_sha256": file_sha256(Path(args.score)),
            "remains_primary": True,
        },
        "secondary_analysis": {
            "metric": "llms4ol2026_graph_similarity_semantic_match",
            "aggregation": "mean_of_per_sample_nontax_only_scores",
            "threshold": THRESHOLD,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "remote_code_dependency": {
                "model_id": REMOTE_CODE_MODEL_ID,
                "revision": REMOTE_CODE_REVISION,
            },
            "trust_remote_code": True,
            "official_source": str(OFFICIAL_SOURCE.resolve()),
            "official_source_sha256": file_sha256(OFFICIAL_SOURCE),
            "systems": ["base_v5", "frozen_calibrated_v5"],
            "sample_indices": indices,
            "n_samples": len(indices),
        },
        "inputs": {
            "confirmatory_manifest": {
                "path": str(Path(args.manifest).resolve()),
                "sha256": file_sha256(Path(args.manifest)),
            },
            "base_result": {
                "path": str(Path(args.base_result).resolve()),
                "sha256": file_sha256(Path(args.base_result)),
            },
            "calibrated_result": {
                "path": str(Path(args.calibrated_result).resolve()),
                "sha256": file_sha256(Path(args.calibrated_result)),
            },
        },
        "locked_interpretation": {
            "secondary_only": True,
            "authorize_threshold_or_rule_retuning": False,
            "can_change_preregistered_gate": False,
            "can_change_deployment_status": False,
            "exact_match_remains_primary": True,
        },
    }


def preserve_or_write_protocol(
    path: Path, candidate: dict[str, Any]
) -> tuple[dict[str, Any], bool]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != candidate:
            raise ValueError(
                "existing frozen semantic protocol differs from current inputs"
            )
        return existing, False
    blind.atomic_write_json(path, candidate)
    return candidate, True


def validate_protocol(protocol: dict[str, Any]) -> None:
    secondary = protocol["secondary_analysis"]
    interpretation = protocol["locked_interpretation"]
    if protocol["meta"]["protocol"] != PROTOCOL:
        raise ValueError("unexpected semantic protocol ID")
    if float(secondary["threshold"]) != THRESHOLD:
        raise ValueError("semantic threshold differs from frozen 0.90")
    if secondary["model_id"] != MODEL_ID:
        raise ValueError("semantic model differs from frozen model")
    if secondary["model_revision"] != MODEL_REVISION:
        raise ValueError("semantic model revision differs from frozen revision")
    remote_code = secondary.get("remote_code_dependency") or {}
    if remote_code.get("model_id") != REMOTE_CODE_MODEL_ID:
        raise ValueError("nested semantic remote-code model differs")
    if remote_code.get("revision") != REMOTE_CODE_REVISION:
        raise ValueError("nested semantic remote-code revision differs")
    if file_sha256(Path(secondary["official_source"])) != secondary[
        "official_source_sha256"
    ]:
        raise ValueError("official metric source hash mismatch")
    for row in protocol["inputs"].values():
        if file_sha256(Path(row["path"])) != row["sha256"]:
            raise ValueError("frozen semantic input hash mismatch")
    required_true = (
        "secondary_only",
        "can_change_preregistered_gate",
        "can_change_deployment_status",
        "exact_match_remains_primary",
    )
    if not interpretation[required_true[0]]:
        raise ValueError("semantic analysis must remain secondary")
    if interpretation[required_true[1]] or interpretation[required_true[2]]:
        raise ValueError("semantic analysis cannot alter gate or deployment")
    if not interpretation[required_true[3]]:
        raise ValueError("exact match must remain primary")
    if interpretation["authorize_threshold_or_rule_retuning"]:
        raise ValueError("semantic analysis cannot authorize retuning")


def _load_official_metric(protocol: dict[str, Any]):
    """Load the unmodified official source while pinning its model revision."""
    import sentence_transformers

    real_constructor = sentence_transformers.SentenceTransformer
    secondary = protocol["secondary_analysis"]

    def pinned_constructor(model_name_or_path: str, *args, **kwargs):
        if model_name_or_path != secondary["model_id"]:
            raise ValueError(f"unexpected semantic model {model_name_or_path}")
        kwargs["revision"] = secondary["model_revision"]
        kwargs["trust_remote_code"] = True
        remote_revision = secondary["remote_code_dependency"]["revision"]
        model_kwargs = dict(kwargs.get("model_kwargs") or {})
        config_kwargs = dict(kwargs.get("config_kwargs") or {})
        model_kwargs["code_revision"] = remote_revision
        config_kwargs["code_revision"] = remote_revision
        kwargs["model_kwargs"] = model_kwargs
        kwargs["config_kwargs"] = config_kwargs
        return real_constructor(model_name_or_path, *args, **kwargs)

    sentence_transformers.SentenceTransformer = pinned_constructor
    try:
        source = Path(secondary["official_source"])
        spec = importlib.util.spec_from_file_location(
            "llms4ol2026_graph_similarity_semantic_pinned", source
        )
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load official metric source: {source}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sentence_transformers.SentenceTransformer = real_constructor
    return module


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    fields = (
        "edge_f1",
        "neighborhood_similarity",
        "taxonomy_similarity",
        "graph_similarity",
    )
    return {
        field: sum(float(row[field]) for row in rows) / len(rows)
        for field in fields
    }


def _score_system(
    *,
    official,
    predictions: dict[tuple[str, int], set[tuple[str, str, str]]],
    gold: dict[tuple[str, int], set[tuple[str, str, str]]],
    indices: list[int],
    threshold: float,
) -> dict[str, Any]:
    samples = []
    exact_rows = []
    semantic_rows = []
    improved = unchanged = degraded = 0
    for index in indices:
        key = ("a", index)
        gold_triples = sorted(gold.get(key, set()))
        pred_triples = sorted(predictions.get(key, set()))
        exact = official.exact_match(gold_triples, pred_triples)
        semantic = official.semantic_match(
            gold_triples, pred_triples, threshold=threshold
        )
        delta = float(semantic["graph_similarity"]) - float(
            exact["graph_similarity"]
        )
        if delta > 1e-12:
            improved += 1
        elif delta < -1e-12:
            degraded += 1
        else:
            unchanged += 1
        exact_rows.append(exact)
        semantic_rows.append(semantic)
        samples.append(
            {
                "sample_index": index,
                "gold_nontax": len(gold_triples),
                "pred_nontax": len(pred_triples),
                "exact": exact,
                "semantic": semantic,
                "semantic_minus_exact_graph_similarity": delta,
            }
        )
    exact_mean = _mean_metrics(exact_rows)
    semantic_mean = _mean_metrics(semantic_rows)
    return {
        "exact_mean": exact_mean,
        "semantic_mean": semantic_mean,
        "semantic_minus_exact": {
            field: semantic_mean[field] - exact_mean[field]
            for field in exact_mean
        },
        "sample_change_counts": {
            "improved": improved,
            "unchanged": unchanged,
            "degraded": degraded,
        },
        "samples": samples,
    }


def _dependency_versions() -> dict[str, str]:
    names = (
        "sentence-transformers",
        "transformers",
        "torch",
        "rapidfuzz",
        "numpy",
        "scipy",
        "networkx",
        "huggingface-hub",
        "einops",
    )
    output = {}
    for name in names:
        try:
            output[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            output[name] = "not-installed"
    return output


def execute(protocol: dict[str, Any], *, protocol_path: Path) -> dict[str, Any]:
    validate_protocol(protocol)
    secondary = protocol["secondary_analysis"]
    indices = [int(index) for index in secondary["sample_indices"]]
    base = json.loads(
        Path(protocol["inputs"]["base_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    calibrated = json.loads(
        Path(protocol["inputs"]["calibrated_result"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    official = _load_official_metric(protocol)
    gold = confirmatory._gold_by_sample(indices)
    systems = {
        "base_v5": _score_system(
            official=official,
            predictions=confirmatory._accepted_predictions(base),
            gold=gold,
            indices=indices,
            threshold=float(secondary["threshold"]),
        ),
        "frozen_calibrated_v5": _score_system(
            official=official,
            predictions=confirmatory._accepted_predictions(calibrated),
            gold=gold,
            indices=indices,
            threshold=float(secondary["threshold"]),
        ),
    }
    return {
        "meta": {
            "protocol": PROTOCOL,
            "status": "complete_secondary_analysis",
            "protocol_file_sha256": file_sha256(protocol_path),
            "official_source_sha256": secondary["official_source_sha256"],
            "model_id": secondary["model_id"],
            "model_revision": secondary["model_revision"],
            "remote_code_dependency": secondary["remote_code_dependency"],
            "threshold": secondary["threshold"],
            "api_calls": 0,
            "python": sys.version,
            "platform": platform.platform(),
            "dependency_versions": _dependency_versions(),
        },
        "scope": {
            "task": "a",
            "triple_channel": "nontax_only",
            "aggregation": secondary["aggregation"],
            "n_samples": len(indices),
            "selection_is_endpoint_visible_gold_stratified": True,
        },
        "systems": systems,
        "calibrated_minus_base": {
            metric_kind: {
                field: systems["frozen_calibrated_v5"][metric_kind][field]
                - systems["base_v5"][metric_kind][field]
                for field in systems["base_v5"][metric_kind]
            }
            for metric_kind in ("exact_mean", "semantic_mean")
        },
        "locked_interpretation": protocol["locked_interpretation"],
        "scientific_deployable": False,
        "confirmatory_eligible": False,
        "reason": "secondary_metric_cannot_override_failed_exact_confirmatory_gate",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=confirmatory.DEFAULT_MANIFEST)
    parser.add_argument("--base-result", type=Path, default=confirmatory.DEFAULT_BASE_RESULT)
    parser.add_argument(
        "--calibrated-result",
        type=Path,
        default=confirmatory.DEFAULT_CALIBRATED_RESULT,
    )
    parser.add_argument("--score", type=Path, default=confirmatory.DEFAULT_SCORE_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.freeze_only:
        candidate = build_protocol(args)
        protocol, created = preserve_or_write_protocol(args.protocol, candidate)
        validate_protocol(protocol)
        print(
            f"{'Frozen' if created else 'Verified'} semantic protocol "
            f"n={protocol['secondary_analysis']['n_samples']} "
            f"threshold={protocol['secondary_analysis']['threshold']:.2f}",
            flush=True,
        )
        print(f"{'Wrote' if created else 'Preserved'} {args.protocol}", flush=True)
        return 0
    if not args.protocol.exists():
        raise SystemExit("semantic protocol is not frozen; run --freeze-only first")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    report = execute(protocol, protocol_path=args.protocol)
    blind.atomic_write_json(args.output, report)
    base = report["systems"]["base_v5"]["semantic_mean"]["graph_similarity"]
    calibrated = report["systems"]["frozen_calibrated_v5"]["semantic_mean"][
        "graph_similarity"
    ]
    print(
        f"semantic_graph_similarity base={base:.4f} calibrated={calibrated:.4f}",
        flush=True,
    )
    print(f"Wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
