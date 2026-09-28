"""Build the vector-first figures used by the OntoBridg manuscript."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


OUTPUT_DIR = Path(__file__).resolve().parent
ROOT = OUTPUT_DIR.parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
WIDTH_IN = 180 / 25.4
FIXED_DATE = datetime(2026, 9, 28, tzinfo=timezone.utc)

CHARCOAL = "#2B2B2B"
MID_GRAY = "#747474"
LIGHT_GRAY = "#E7E7E7"
BLUE = "#2F6F9F"
LIGHT_BLUE = "#DCEAF3"
VERMILION = "#C44E3B"
LIGHT_RED = "#F5E1DC"
TEAL = "#2A7F75"
WHITE = "#FFFFFF"


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.2,
            "patch.linewidth": 0.8,
            "svg.hashsalt": "ontobridg-submission-v1",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": WHITE,
        }
    )


def load_result(name: str) -> dict:
    with (RESULTS_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def figure_data() -> dict[str, float | int]:
    nonce = load_result("nonce_relation_holdout_openai_n30_seed20260926.json")
    transfer = load_result("official_relation_transfer_openai_a_n30_seed20260927.json")
    routing = load_result("official_predicate_routing_openai_a_n27_seed20260927.json")

    nonce_open = nonce["summaries"]["open_vocab"]["all"]
    nonce_guarded = nonce["summaries"]["open_vocab_guarded_v2"]["all"]
    transfer_open = transfer["summaries"]["open_vocab"]["all"]
    transfer_guarded = transfer["summaries"]["open_vocab_guarded_v2"]["all"]
    routed = routing["summaries"]["open_vocab_guarded_v6_predicate_routed"]["all"]
    gate = routing["mechanism_gate"]

    return {
        "nonce_open_f1": nonce_open["nontax_f1"],
        "nonce_guarded_f1": nonce_guarded["nontax_f1"],
        "nonce_open_gs": nonce_open["official_graph_similarity_avg"],
        "nonce_guarded_gs": nonce_guarded["official_graph_similarity_avg"],
        "transfer_open_f1": transfer_open["nontax_nonself_f1_micro"]["f1"],
        "transfer_guarded_f1": transfer_guarded["nontax_nonself_f1_micro"]["f1"],
        "transfer_open_gs": transfer_open["official_graph_similarity_avg"],
        "transfer_guarded_gs": transfer_guarded["official_graph_similarity_avg"],
        "transfer_gold": transfer_open["nontax_nonself_gold_total"],
        "routing_tp": routed["target_exact"]["tp"],
        "routing_fp": routed["target_exact"]["fp"],
        "routing_positive_samples": gate["observed_routed_exact_tp_samples"],
        "routing_required_samples": gate["minimum_routed_exact_tp_samples"],
    }


def save_all(fig: plt.Figure, stem: str) -> None:
    svg_path = OUTPUT_DIR / f"{stem}.svg"
    fig.savefig(
        svg_path,
        format="svg",
        metadata={"Date": "2026-09-28"},
    )
    svg_text = svg_path.read_text(encoding="utf-8")
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.pdf",
        format="pdf",
        metadata={"CreationDate": FIXED_DATE, "ModDate": FIXED_DATE},
    )
    fig.savefig(
        OUTPUT_DIR / f"{stem}.png",
        format="png",
        dpi=300,
        metadata={"Creation Time": "2026-09-28T00:00:00Z"},
    )


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = WHITE,
    edgecolor: str = MID_GRAY,
    textcolor: str = CHARCOAL,
    linewidth: float = 0.8,
) -> None:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
    )
    ax.add_patch(patch)
    ax.text(x, y, text, ha="center", va="center", color=textcolor, linespacing=1.15)


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={"arrowstyle": "-|>", "color": MID_GRAY, "lw": 0.9},
    )


def build_evidence_chain(data: dict[str, float | int]) -> None:
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 3.5))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.96, bottom=0.08)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.01, 0.965, "A", weight="bold", fontsize=10, va="top")
    ax.text(0.055, 0.95, "Candidate generation and deterministic relation guard", weight="bold", va="top")

    top_nodes = [
        (0.09, "Legal task\ninputs", LIGHT_GRAY, MID_GRAY),
        (0.285, "LLM candidate\ngeneration", LIGHT_BLUE, BLUE),
        (0.48, "Raw triples +\nconfidence + evidence", WHITE, MID_GRAY),
        (0.675, "Relation-only\nguard", LIGHT_BLUE, BLUE),
        (0.88, "Accepted graph\n+ audit trail", WHITE, MID_GRAY),
    ]
    for x, label, face, edge in top_nodes:
        rounded_box(ax, x, 0.75, 0.16, 0.16, label, facecolor=face, edgecolor=edge)
    for left, right in zip(top_nodes, top_nodes[1:]):
        arrow(ax, (left[0] + 0.085, 0.75), (right[0] - 0.085, 0.75))

    ax.text(
        0.675,
        0.615,
        "confidence  |  verbatim evidence  |  canonicalization  |  self-loop / generic predicate  |  max 8",
        ha="center",
        va="top",
        color=MID_GRAY,
        fontsize=6.8,
    )

    ax.plot([0.02, 0.98], [0.51, 0.51], color=LIGHT_GRAY, lw=0.8)
    ax.text(0.01, 0.475, "B", weight="bold", fontsize=10, va="top")
    ax.text(0.055, 0.46, "Frozen evidence chain and stopping decisions", weight="bold", va="top")

    bottom_nodes = [
        (0.075, "Blind prefix\nover-generation", WHITE, MID_GRAY),
        (0.245, "Relation-only guard\nGS improves", WHITE, MID_GRAY),
        (0.415, f"Nonce holdout\nF1 {data['nonce_guarded_f1']:.4f}", LIGHT_BLUE, BLUE),
        (0.595, f"Official transfer\n0 / {data['transfer_gold']} TP", LIGHT_RED, VERMILION),
        (
            0.765,
            "Predicate routing\n"
            f"{data['routing_positive_samples']} / {data['routing_required_samples']} samples",
            LIGHT_RED,
            VERMILION,
        ),
        (0.925, "STOP\nno deployment", VERMILION, VERMILION),
    ]
    for x, label, face, edge in bottom_nodes:
        text_color = WHITE if label.startswith("STOP") else CHARCOAL
        rounded_box(
            ax,
            x,
            0.285,
            0.14,
            0.145,
            label,
            facecolor=face,
            edgecolor=edge,
            textcolor=text_color,
        )
    for left, right in zip(bottom_nodes, bottom_nodes[1:]):
        arrow(ax, (left[0] + 0.075, 0.285), (right[0] - 0.075, 0.285))

    ax.text(
        0.5,
        0.105,
        "Failed gates prohibit same-slice prompt repair, recalibration, n>=100 expansion, and deployment.",
        ha="center",
        va="center",
        color=CHARCOAL,
        fontsize=7.2,
    )

    save_all(fig, "ontobridg_evidence_chain")
    plt.close(fig)


def style_axis(ax: plt.Axes, ylabel: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(MID_GRAY)
    ax.spines["bottom"].set_color(MID_GRAY)
    ax.tick_params(axis="both", colors=CHARCOAL, length=2.5, width=0.7)
    ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6, zorder=0)
    ax.set_ylabel(ylabel)


def label_bars(
    ax: plt.Axes,
    bars,
    *,
    fmt: str = ".4f",
    offset: float = 0.02,
    skip_zero: bool = False,
) -> None:
    for bar in bars:
        value = bar.get_height()
        if skip_zero and value == 0:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            format(value, fmt),
            ha="center",
            va="bottom",
            fontsize=6.6,
            color=CHARCOAL,
        )


def build_main_results(data: dict[str, float | int]) -> None:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(WIDTH_IN, 2.72),
        gridspec_kw={"width_ratios": [1.1, 1.1, 0.9]},
    )
    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.25, wspace=0.52)

    labels = ["Nonce", "Official\ntransfer"]
    positions = [0, 1]
    width = 0.34

    ax = axes[0]
    open_f1 = [data["nonce_open_f1"], data["transfer_open_f1"]]
    guarded_f1 = [data["nonce_guarded_f1"], data["transfer_guarded_f1"]]
    bars_open = ax.bar(
        [x - width / 2 for x in positions],
        open_f1,
        width,
        label="Open",
        color=LIGHT_GRAY,
        edgecolor=CHARCOAL,
        hatch="///",
        zorder=3,
    )
    bars_guarded = ax.bar(
        [x + width / 2 for x in positions],
        guarded_f1,
        width,
        label="Guarded v2",
        color=BLUE,
        edgecolor=BLUE,
        zorder=3,
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    style_axis(ax, "Exact NonTax F1")
    ax.set_title("A   Relation recovery", loc="left", weight="bold")
    label_bars(ax, bars_open, skip_zero=True)
    label_bars(ax, bars_guarded, skip_zero=True)
    ax.text(1, 0.022, "both 0", ha="center", va="bottom", fontsize=6.6)
    ax.legend(frameon=False, loc="upper right", handlelength=1.4)

    ax = axes[1]
    open_gs = [data["nonce_open_gs"], data["transfer_open_gs"]]
    guarded_gs = [data["nonce_guarded_gs"], data["transfer_guarded_gs"]]
    bars_open = ax.bar(
        [x - width / 2 for x in positions],
        open_gs,
        width,
        color=LIGHT_GRAY,
        edgecolor=CHARCOAL,
        hatch="///",
        zorder=3,
    )
    bars_guarded = ax.bar(
        [x + width / 2 for x in positions],
        guarded_gs,
        width,
        color=BLUE,
        edgecolor=BLUE,
        zorder=3,
    )
    ax.set_xticks(positions, labels)
    ax.set_ylim(0, 0.66)
    ax.set_yticks([0, 0.2, 0.4, 0.6])
    style_axis(ax, "Exact Graph Similarity")
    ax.set_title("B   Graph score", loc="left", weight="bold")
    label_bars(ax, bars_open, offset=0.012)
    label_bars(ax, bars_guarded, offset=0.012)

    ax = axes[2]
    values = [data["routing_tp"], data["routing_fp"]]
    bars = ax.bar(
        [0, 1],
        values,
        width=0.58,
        color=[TEAL, VERMILION],
        edgecolor=[TEAL, VERMILION],
        zorder=3,
    )
    ax.set_xticks([0, 1], ["Exact TP", "Exact FP"])
    ax.set_ylim(0, 152)
    ax.set_yticks([0, 50, 100, 150])
    style_axis(ax, "Routed predictions")
    ax.set_title("C   Predicate routing", loc="left", weight="bold")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 4,
            str(value),
            ha="center",
            va="bottom",
            fontsize=7,
            color=CHARCOAL,
            weight="bold",
        )
    ax.text(
        0.04,
        0.96,
        "Gate failed:\n"
        f"{data['routing_positive_samples']} samples < {data['routing_required_samples']}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color=VERMILION,
        fontsize=7,
    )

    save_all(fig, "ontobridg_main_results")
    plt.close(fig)


def main() -> None:
    configure()
    data = figure_data()
    build_evidence_chain(data)
    build_main_results(data)
    print(f"Wrote submission figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
