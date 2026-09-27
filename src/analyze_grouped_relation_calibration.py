"""Bootstrap and plot sample-grouped out-of-fold relation calibration results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

from ontobridg.relations.calibration import calibration_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "experiments"
    / "results"
    / "model_weaklabel_relation_calibration_grouped_cv_5b53d848edca6abb58f4_n30.json"
)
DEFAULT_OUTPUT = (
    ROOT
    / "experiments"
    / "results"
    / "model_weaklabel_relation_calibration_grouped_cv_uncertainty_5b53d848edca6abb58f4_n30.json"
)
DEFAULT_FIGURE_STEM = (
    ROOT / "papers" / "figures" / "weaklabel_relation_calibration_reliability"
)
DEFAULT_BOOTSTRAP_REPLICATES = 10_000
DEFAULT_BOOTSTRAP_SEED = 20260924
DEFAULT_BINS = 10


def percentile(values: Iterable[float], probability: float) -> float:
    """Return the linearly interpolated empirical percentile (R type 7)."""
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be between 0 and 1")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metrics(
    predictions: list[dict[str, Any]], score_field: str, *, bins: int
) -> dict[str, float | int]:
    return calibration_metrics(
        [float(row[score_field]) for row in predictions],
        [bool(row["weak_label"]) for row in predictions],
        bins=bins,
    )


def reliability_bins(
    predictions: list[dict[str, Any]], score_field: str, *, bins: int
) -> list[dict[str, Any]]:
    if bins < 1:
        raise ValueError("reliability bins must be positive")
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(bins)]
    for row in predictions:
        score = min(1.0, max(0.0, float(row[score_field])))
        index = min(int(score * bins), bins - 1)
        buckets[index].append(row)
    rows = []
    for index, bucket in enumerate(buckets):
        rows.append(
            {
                "bin": index,
                "lower": index / bins,
                "upper": (index + 1) / bins,
                "n": len(bucket),
                "sample_groups": len(
                    {str(row["sample_key"]) for row in bucket}
                ),
                "mean_confidence": (
                    sum(float(row[score_field]) for row in bucket) / len(bucket)
                    if bucket
                    else None
                ),
                "empirical_accuracy": (
                    sum(bool(row["weak_label"]) for row in bucket) / len(bucket)
                    if bucket
                    else None
                ),
            }
        )
    return rows


def _interval(
    values: list[float], *, estimate: float, confidence_level: float
) -> dict[str, float | int]:
    tail = (1.0 - confidence_level) / 2.0
    return {
        "estimate": float(estimate),
        "lower": percentile(values, tail),
        "upper": percentile(values, 1.0 - tail),
        "bootstrap_valid_replicates": len(values),
    }


def cluster_bootstrap_oof(
    predictions: list[dict[str, Any]],
    *,
    replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    bins: int = DEFAULT_BINS,
    confidence_level: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap frozen OOF predictions with sample_key as the resampling unit."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence level must be between 0 and 1")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        sample_key = str(row.get("sample_key") or "")
        if not sample_key:
            raise ValueError("OOF prediction is missing sample_key")
        for field in ("raw_confidence", "calibrated_confidence", "weak_label"):
            if field not in row:
                raise ValueError(f"OOF prediction is missing {field}")
        grouped.setdefault(sample_key, []).append(row)
    if len(grouped) < 2:
        raise ValueError("cluster bootstrap requires at least two sample groups")

    sample_keys = sorted(grouped)
    raw_point = _metrics(predictions, "raw_confidence", bins=bins)
    calibrated_point = _metrics(predictions, "calibrated_confidence", bins=bins)
    prevalence_point = sum(bool(row["weak_label"]) for row in predictions) / len(
        predictions
    )
    distributions = {
        "raw_brier": [],
        "raw_ece": [],
        "calibrated_brier": [],
        "calibrated_ece": [],
        "delta_brier": [],
        "delta_ece": [],
        "prevalence": [],
    }
    reliability_distributions = {
        "raw": [[] for _ in range(bins)],
        "calibrated": [[] for _ in range(bins)],
    }
    rng = random.Random(seed)
    for _ in range(replicates):
        sampled: list[dict[str, Any]] = []
        for _ in sample_keys:
            sampled.extend(grouped[rng.choice(sample_keys)])
        raw = _metrics(sampled, "raw_confidence", bins=bins)
        calibrated = _metrics(sampled, "calibrated_confidence", bins=bins)
        distributions["raw_brier"].append(float(raw["brier"]))
        distributions["raw_ece"].append(float(raw["ece"]))
        distributions["calibrated_brier"].append(float(calibrated["brier"]))
        distributions["calibrated_ece"].append(float(calibrated["ece"]))
        distributions["delta_brier"].append(
            float(calibrated["brier"]) - float(raw["brier"])
        )
        distributions["delta_ece"].append(
            float(calibrated["ece"]) - float(raw["ece"])
        )
        distributions["prevalence"].append(
            sum(bool(row["weak_label"]) for row in sampled) / len(sampled)
        )
        for label, field in (
            ("raw", "raw_confidence"),
            ("calibrated", "calibrated_confidence"),
        ):
            for row in reliability_bins(sampled, field, bins=bins):
                if row["n"]:
                    reliability_distributions[label][row["bin"]].append(
                        float(row["empirical_accuracy"])
                    )

    metric_intervals = {
        "raw": {
            "brier": _interval(
                distributions["raw_brier"],
                estimate=float(raw_point["brier"]),
                confidence_level=confidence_level,
            ),
            "ece": _interval(
                distributions["raw_ece"],
                estimate=float(raw_point["ece"]),
                confidence_level=confidence_level,
            ),
        },
        "calibrated": {
            "brier": _interval(
                distributions["calibrated_brier"],
                estimate=float(calibrated_point["brier"]),
                confidence_level=confidence_level,
            ),
            "ece": _interval(
                distributions["calibrated_ece"],
                estimate=float(calibrated_point["ece"]),
                confidence_level=confidence_level,
            ),
        },
        "calibrated_minus_raw": {
            "brier": _interval(
                distributions["delta_brier"],
                estimate=float(calibrated_point["brier"])
                - float(raw_point["brier"]),
                confidence_level=confidence_level,
            ),
            "ece": _interval(
                distributions["delta_ece"],
                estimate=float(calibrated_point["ece"])
                - float(raw_point["ece"]),
                confidence_level=confidence_level,
            ),
        },
    }

    reliability: dict[str, list[dict[str, Any]]] = {}
    for label, field in (
        ("raw", "raw_confidence"),
        ("calibrated", "calibrated_confidence"),
    ):
        reliability[label] = []
        for row in reliability_bins(predictions, field, bins=bins):
            values = reliability_distributions[label][row["bin"]]
            enriched = dict(row)
            enriched["accuracy_interval"] = (
                _interval(
                    values,
                    estimate=float(row["empirical_accuracy"]),
                    confidence_level=confidence_level,
                )
                if values
                else None
            )
            reliability[label].append(enriched)

    return {
        "protocol": "sample_cluster_bootstrap_oof_v1",
        "resampling_unit": "sample_key",
        "prediction_refit_per_replicate": False,
        "conditional_on": "frozen_oof_predictions_and_frozen_weak_labels",
        "replicates": replicates,
        "seed": seed,
        "confidence_level": confidence_level,
        "percentile_method": "linear_interpolation_r_type_7",
        "bins": bins,
        "n_candidates": len(predictions),
        "n_sample_groups": len(sample_keys),
        "positive_candidates": sum(bool(row["weak_label"]) for row in predictions),
        "positive_sample_groups": len(
            {
                str(row["sample_key"])
                for row in predictions
                if bool(row["weak_label"])
            }
        ),
        "prevalence": _interval(
            distributions["prevalence"],
            estimate=prevalence_point,
            confidence_level=confidence_level,
        ),
        "metrics": metric_intervals,
        "reliability": reliability,
    }


def render_figure(analysis: dict[str, Any], figure_stem: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    charcoal = "#333333"
    raw_color = "#D55E00"
    calibrated_color = "#0072B2"
    neutral = "#A7A7A7"
    series = (
        ("raw", "Raw confidence", raw_color, "s"),
        ("calibrated", "Calibrated", calibrated_color, "o"),
    )

    width_inches = 180.0 / 25.4
    height_inches = 82.0 / 25.4
    fig, axes = plt.subplots(1, 2, figsize=(width_inches, height_inches))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.20, top=0.88, wspace=0.34)

    ax = axes[0]
    ax.plot([0, 1], [0, 1], color=neutral, linewidth=0.9, linestyle="--", zorder=1)
    for label, display, color, marker in series:
        rows = [row for row in analysis["reliability"][label] if row["n"]]
        x = [float(row["mean_confidence"]) for row in rows]
        y = [float(row["empirical_accuracy"]) for row in rows]
        lower = [
            value - float(row["accuracy_interval"]["lower"])
            for value, row in zip(y, rows)
        ]
        upper = [
            float(row["accuracy_interval"]["upper"]) - value
            for value, row in zip(y, rows)
        ]
        ax.errorbar(
            x,
            y,
            yerr=[lower, upper],
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.1,
            elinewidth=0.9,
            capsize=2.2,
            label=display,
            zorder=3,
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xticks([index / 5 for index in range(6)])
    ax.set_yticks([index / 5 for index in range(6)])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Weak-label positive rate")
    ax.set_title("Reliability (sample-grouped OOF)", loc="left", pad=6)
    ax.legend(frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(direction="out", length=3)

    ax = axes[1]
    metric_rows = (("brier", "Brier"), ("ece", "ECE"))
    y_base = [1.0, 0.0]
    offsets = {"raw": 0.11, "calibrated": -0.11}
    for label, display, color, marker in series:
        values = []
        lower_errors = []
        upper_errors = []
        y_values = []
        for y, (key, _) in zip(y_base, metric_rows):
            interval = analysis["metrics"][label][key]
            estimate = float(interval["estimate"])
            values.append(estimate)
            lower_errors.append(estimate - float(interval["lower"]))
            upper_errors.append(float(interval["upper"]) - estimate)
            y_values.append(y + offsets[label])
        ax.errorbar(
            values,
            y_values,
            xerr=[lower_errors, upper_errors],
            linestyle="none",
            color=color,
            marker=marker,
            markersize=4.5,
            elinewidth=1.0,
            capsize=2.2,
            label=display,
        )
    ax.set_yticks(y_base, [label for _, label in metric_rows])
    ax.set_xlim(-0.02, 0.9)
    ax.set_xticks([index / 10 for index in range(10)])
    ax.set_ylim(-0.45, 1.45)
    ax.set_xlabel("Metric value (lower is better)")
    ax.set_title("95% sample-cluster bootstrap intervals", loc="left", pad=6)
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", direction="out", length=3)

    for label, ax in zip(("a", "b"), axes):
        ax.text(
            -0.17,
            1.09,
            label,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=charcoal,
            va="top",
        )

    figure_stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = [
        figure_stem.with_suffix(".svg"),
        figure_stem.with_suffix(".pdf"),
        figure_stem.with_suffix(".png"),
    ]
    creator = "OntoBridg grouped calibration analysis"
    fig.savefig(
        outputs[0], facecolor="white", metadata={"Date": None, "Creator": creator}
    )
    fig.savefig(
        outputs[1],
        facecolor="white",
        metadata={"CreationDate": None, "ModDate": None, "Creator": creator},
    )
    fig.savefig(
        outputs[2], dpi=300, facecolor="white", metadata={"Software": creator}
    )
    plt.close(fig)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--figure-stem", type=Path, default=DEFAULT_FIGURE_STEM)
    parser.add_argument(
        "--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES
    )
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--bins", type=int, default=DEFAULT_BINS)
    parser.add_argument("--no-figure", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = json.loads(args.input.read_text(encoding="utf-8"))
    cv = source.get("grouped_cross_validation") or {}
    predictions = list(cv.get("out_of_fold_predictions") or [])
    expected = int((cv.get("coverage") or {}).get("out_of_fold_predictions") or 0)
    if not predictions or len(predictions) != expected:
        raise ValueError(
            "source artifact must contain every out_of_fold_predictions record"
        )
    bootstrap = cluster_bootstrap_oof(
        predictions,
        replicates=args.bootstrap_replicates,
        seed=args.bootstrap_seed,
        bins=args.bins,
    )
    analysis = {
        "meta": {
            "protocol": "weaklabel_relation_calibration_uncertainty_v1",
            "source_artifact": str(args.input),
            "source_artifact_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "annotation_source": source.get("meta", {}).get("annotation_source"),
            "claim_label": source.get("meta", {}).get("claim_label"),
            "official_gold_used": False,
            "additional_external_calls": 0,
            "deployable": False,
            "confirmatory_eligible": False,
            "interpretation_scope": (
                "uncertainty conditional on frozen OOF predictions and weak labels"
            ),
            "figure_dimensions": {
                "width_mm": 180,
                "height_mm": 82,
                "raster_dpi": 300,
                "vector_formats": ["svg", "pdf"],
            },
        },
        "bootstrap": bootstrap,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {args.output}")
    if not args.no_figure:
        outputs = render_figure(bootstrap, args.figure_stem)
        for path in outputs:
            print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
