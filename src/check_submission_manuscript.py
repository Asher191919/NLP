"""Validate the venue-neutral OntoBridg manuscript package."""
from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPERS = ROOT / "papers"
MANUSCRIPT = PAPERS / "paper_draft.md"
SUPPLEMENT = PAPERS / "appendix.md"
BIBLIOGRAPHY = PAPERS / "references.bib"
RESULTS = ROOT / "experiments" / "results"
ANNOTATIONS = ROOT / "experiments" / "annotations"


def words(text: str) -> list[str]:
    return re.findall(r"\b[\w'-]+\b", text)


def section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\s*$\n(?P<body>.*?)(?=^## |\Z)",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        raise ValueError(f"missing section: {heading}")
    return match.group("body").strip()


def numbered_references(text: str) -> list[int]:
    references = section(text, "References")
    return [
        int(value)
        for value in re.findall(r"^(\d+)\.\s", references, flags=re.MULTILINE)
    ]


def cited_numbers(text: str) -> set[int]:
    body = text.split("## References", 1)[0]
    output: set[int] = set()
    for group in re.findall(r"\[((?:\d+\s*,\s*)*\d+)\]", body):
        values = [int(value.strip()) for value in group.split(",")]
        if 0 in values:
            continue
        output.update(values)
    return output


def figure_paths(text: str, source: Path) -> list[Path]:
    return [
        (source.parent / relative).resolve()
        for relative in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    ]


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def clean_table_cell(cell: str) -> str:
    return cell.strip().replace("**", "").replace("`", "")


def table_rows(text: str, number: int) -> list[list[str]]:
    marker = f"**Table {number}."
    if marker not in text:
        return []
    rows: list[list[str]] = []
    started = False
    for line in text.split(marker, 1)[1].splitlines():
        if line.startswith("|"):
            started = True
            cells = [clean_table_cell(cell) for cell in line.strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                rows.append(cells)
        elif started:
            break
    return rows[1:] if rows else []


def metric(value: float) -> str:
    return "0" if abs(value) < 5e-9 else f"{value:.4f}"


def prose_number(value: int) -> str:
    words = {
        0: "zero",
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
    }
    return words.get(value, str(value))


def check_evidence(manuscript: str, supplement: str) -> list[str]:
    errors: list[str] = []

    def compare_table(number: int, expected: list[list[str]]) -> None:
        actual = table_rows(manuscript, number)
        if actual != expected:
            errors.append(f"Table {number} differs from frozen result artifacts")

    def require(fragment: str, label: str) -> None:
        if fragment not in manuscript:
            errors.append(f"missing or stale evidence claim: {label}")

    blind_rows: list[list[str]] = []
    for task, filename in (
        ("A", "official_blind_openai_a_n30_off0.json"),
        ("B", "official_blind_openai_b_n30_off0.json"),
    ):
        result = load_json(RESULTS / filename)
        for system, label in (("closed_vocab", "Closed"), ("open_vocab", "Open")):
            summary = result["systems"][system]["summary"]
            blind_rows.append(
                [
                    task,
                    label,
                    metric(summary["term_typing_f1_micro"]["f1"]),
                    metric(summary["taxonomy_f1_micro"]["f1"]),
                    metric(summary["nontax_f1_micro"]["f1"]),
                    metric(summary["official_graph_similarity_avg"]),
                    f"{summary['nontax_pred_total']} / {summary['nontax_gold_total']}",
                ]
            )
    compare_table(2, blind_rows)

    density = load_json(
        RESULTS / "official_stratified_open_openai_ab_s5_seed20260922.json"
    )
    density_rows: list[list[str]] = []
    systems = (
        ("open_vocab", "Open"),
        ("open_vocab_guarded_hybrid", "Guarded v1"),
        ("open_vocab_guarded_v2_hybrid", "Guarded v2"),
    )
    for task_key, task in (("task_a", "A"), ("task_b", "B")):
        for system, label in systems:
            summary = density["tasks"][task_key]["systems"][system]["summary"]
            density_rows.append(
                [
                    task,
                    label,
                    metric(summary["term_typing_f1_micro"]["f1"]),
                    metric(summary["taxonomy_f1_micro"]["f1"]),
                    metric(summary["nontax_f1_micro"]["f1"]),
                    metric(summary["official_graph_similarity_avg"]),
                    str(summary["nontax_pred_total"]),
                ]
            )
    compare_table(3, density_rows)

    nonce = load_json(RESULTS / "nonce_relation_holdout_openai_n30_seed20260926.json")
    transfer = load_json(
        RESULTS / "official_relation_transfer_openai_a_n30_seed20260927.json"
    )
    transfer_rows: list[list[str]] = []
    for system, label in (("open_vocab", "Open"), ("open_vocab_guarded_v2", "Guarded v2")):
        summary = nonce["summaries"][system]["all"]
        predictions = sum(
            sample["metrics"]["pred_counts"]["nontax"]
            for sample in nonce["samples"]
            if sample["system"] == system and sample["status"] == "ok"
        )
        gold = sum(
            sample["metrics"]["gold_counts"]["nontax"]
            for sample in nonce["samples"]
            if sample["system"] == system and sample["status"] == "ok"
        )
        transfer_rows.append(
            [
                "Nonce",
                label,
                metric(summary["nontax_precision"]),
                metric(summary["nontax_recall"]),
                metric(summary["nontax_f1"]),
                metric(summary["official_graph_similarity_avg"]),
                f"{predictions} / {gold}",
            ]
        )
    for system, label in (("open_vocab", "Open"), ("open_vocab_guarded_v2", "Guarded v2")):
        summary = transfer["summaries"][system]["all"]
        nontax = summary["nontax_nonself_f1_micro"]
        transfer_rows.append(
            [
                "Official transfer",
                label,
                metric(nontax["precision"]),
                metric(nontax["recall"]),
                metric(nontax["f1"]),
                metric(summary["official_graph_similarity_avg"]),
                f"{summary['nontax_nonself_pred_total']} / "
                f"{summary['nontax_nonself_gold_total']}",
            ]
        )
    compare_table(4, transfer_rows)

    routing = load_json(
        RESULTS / "official_predicate_routing_openai_a_n27_seed20260927.json"
    )
    routing_rows: list[list[str]] = []
    for system, label in (
        ("open_vocab_guarded_v3_relations", "General v3"),
        ("open_vocab_guarded_v6_predicate_routed", "Routed v6"),
    ):
        summary = routing["summaries"][system]["all"]
        target = summary["target_exact"]
        macro = routing["primary"]["values"][system]
        routing_rows.append(
            [
                label,
                metric(macro),
                f"{metric(target['precision'])} / {metric(target['recall'])} / "
                f"{metric(target['f1'])}",
                f"{target['tp']} / {target['fp']} / {target['fn']}",
                str(target["positive_samples"]),
                metric(summary["official_graph_similarity_avg"]),
            ]
        )
    compare_table(5, routing_rows)

    calibration = load_json(
        RESULTS / "relation_guarded_v2_independent_calibrator_a27_b12_seed20260923.json"
    )
    raw = calibration["calibration_metrics"]["calibration_raw"]
    calibrated = calibration["calibration_metrics"]["calibration_calibrated"]
    support = calibration["deployment_support"]
    require(
        f"contains {raw['n']} eligible candidates but only "
        f"{prose_number(raw['positives'])} exact positives from "
        f"{prose_number(support['positive_samples'])} samples",
        "independent calibration support",
    )
    require(
        f"Brier score from {raw['brier']:.4f} to {calibrated['brier']:.4f} "
        f"and expected calibration error (ECE) from {raw['ece']:.4f} to "
        f"{calibrated['ece']:.4f}",
        "independent calibration metrics",
    )

    agreement = load_json(
        ANNOTATIONS / "relation_text_model_annotation_v1" / "model_agreement.json"
    )
    weak_cv = load_json(
        RESULTS
        / "model_weaklabel_relation_calibration_grouped_cv_5b53d848edca6abb58f4_n30.json"
    )["grouped_cross_validation"]
    weak_metrics = weak_cv["pooled_out_of_fold_metrics"]["calibrated"]
    require(
        f"micro Jaccard {agreement['micro_relation_set_jaccard']:.4f}",
        "weak-label agreement",
    )
    require(
        f"Brier/ECE to {weak_metrics['brier']:.4f}/{weak_metrics['ece']:.4f}",
        "weak-label grouped calibration",
    )

    confirmatory = load_json(
        RESULTS / "v5_axiom_confirmatory_score_a_n24_seed20260925.json"
    )
    base = confirmatory["official_exact"]["base_v5"]
    frozen = confirmatory["official_exact"]["frozen_calibrated_v5"]
    require(
        f"Base v5 yields {base['tp']} TP in {base['tp'] + base['fp']} predictions "
        f"from {prose_number(base['positive_samples'])} samples "
        f"(F1 {base['f1']:.4f})",
        "confirmatory base result",
    )
    require(
        f"leaving {frozen['tp']} TP in {frozen['tp'] + frozen['fp']} predictions "
        f"from {prose_number(frozen['positive_samples'])} samples "
        f"(F1 {frozen['f1']:.4f})",
        "confirmatory calibrated result",
    )

    probe = load_json(
        RESULTS / "official_contamination_probe_analysis_ab_n15_seed20260926.json"
    )
    probe_metric = probe["probe_metrics"]["overall"]
    probe_interval = probe["sample_cluster_bootstrap"]["overall"]["auroc"]
    require(
        f"AUROC is {probe_metric['auroc']:.4f} with a 10,000-replicate "
        f"sample-cluster 95% interval of [{probe_interval['lower']:.4f}, "
        f"{probe_interval['upper']:.4f}]",
        "contamination probe",
    )

    config = density["meta"]
    for value, label in (
        (config["confidence_threshold"], "explicit confidence threshold"),
        (config["implicit_confidence_threshold"], "cross-sentence confidence threshold"),
        (config["max_implicit_gap_chars"], "cross-sentence gap"),
        (config["max_nontax"], "relation quantity cap"),
    ):
        require(str(value), label)

    semantic = load_json(
        RESULTS / "v5_axiom_confirmatory_semantic_secondary_a_n24_seed20260925.json"
    )
    for system in ("base_v5", "frozen_calibrated_v5"):
        changes = semantic["systems"][system]["sample_change_counts"]
        if changes != {"improved": 0, "unchanged": 24, "degraded": 0}:
            errors.append(f"unexpected semantic-secondary change counts for {system}")
    if "0 of 24 samples improve, 24 remain unchanged, and 0 degrade" not in supplement:
        errors.append("missing or stale evidence claim: semantic secondary analysis")

    return errors


def check() -> list[str]:
    errors: list[str] = []
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    supplement = SUPPLEMENT.read_text(encoding="utf-8")
    bibliography = BIBLIOGRAPHY.read_text(encoding="utf-8")

    for path, text in ((MANUSCRIPT, manuscript), (SUPPLEMENT, supplement)):
        if re.search(r"\b[A-Za-z]:[\\/]", text):
            errors.append(f"{path.name}: contains a local absolute path")
        for marker in ("TODO", "TBD", "写作备注", "投稿前"):
            if marker.lower() in text.lower():
                errors.append(f"{path.name}: forbidden drafting marker {marker!r}")
        for image_path in figure_paths(text, path):
            if not image_path.exists():
                errors.append(f"{path.name}: missing figure {image_path}")

    abstract_count = len(words(section(manuscript, "Abstract")))
    if not 180 <= abstract_count <= 260:
        errors.append(f"abstract word count outside 180-260: {abstract_count}")

    manuscript_count = len(words(manuscript))
    if not 3500 <= manuscript_count <= 8000:
        errors.append(f"manuscript word count outside 3500-8000: {manuscript_count}")

    references = numbered_references(manuscript)
    expected = list(range(1, len(references) + 1))
    if references != expected:
        errors.append(f"reference numbering is not sequential: {references}")
    citations = cited_numbers(manuscript)
    missing = citations - set(references)
    uncited = set(references) - citations
    if missing:
        errors.append(f"citations without references: {sorted(missing)}")
    if uncited:
        errors.append(f"references not cited in body: {sorted(uncited)}")

    bib_entries = len(re.findall(r"^@\w+\{", bibliography, flags=re.MULTILINE))
    if bib_entries < len(references):
        errors.append(
            f"BibTeX has fewer entries than manuscript references: "
            f"{bib_entries} < {len(references)}"
        )

    for label, expected_count in (("Figure", 2), ("Table", 5)):
        found = [
            int(value)
            for value in re.findall(
                rf"\*\*{label} (\d+)\.", manuscript, flags=re.MULTILINE
            )
        ]
        if found != list(range(1, expected_count + 1)):
            errors.append(f"{label.lower()} numbering mismatch: {found}")

    errors.extend(check_evidence(manuscript, supplement))

    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    print(
        "Submission manuscript verified "
        f"words={len(words(manuscript))} "
        f"abstract_words={len(words(section(manuscript, 'Abstract')))} "
        f"references={len(numbered_references(manuscript))} "
        f"figures={len(figure_paths(manuscript, MANUSCRIPT))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
