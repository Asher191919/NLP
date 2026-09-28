"""Build a minimal anonymous, offline-verifiable OntoBridg artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
DEFAULT_OUTPUT = DIST / "ontobridg-anonymous-artifact"
ARCHIVE_ROOT = "ontobridg-anonymous-artifact"
TEXT_SUFFIXES = {".bib", ".json", ".md", ".py", ".svg", ".txt"}
PAPER_FILES = (
    "papers/paper_draft.md",
    "papers/appendix.md",
    "papers/references.bib",
    "papers/submission_checklist.md",
)
FIGURE_FILES = (
    "papers/figures/build_submission_figures.py",
    "papers/figures/ontobridg_evidence_chain.pdf",
    "papers/figures/ontobridg_evidence_chain.png",
    "papers/figures/ontobridg_evidence_chain.svg",
    "papers/figures/ontobridg_main_results.pdf",
    "papers/figures/ontobridg_main_results.png",
    "papers/figures/ontobridg_main_results.svg",
    "papers/figures/weaklabel_relation_calibration_reliability.pdf",
    "papers/figures/weaklabel_relation_calibration_reliability.png",
    "papers/figures/weaklabel_relation_calibration_reliability.svg",
)
MANIFEST_FILES = (
    "experiments/results/nonce_relation_holdout_manifest_n30_seed20260926.json",
    "experiments/results/official_relation_transfer_manifest_a_n30_seed20260927.json",
    "experiments/results/official_predicate_routing_manifest_a_n27_seed20260927.json",
    "experiments/results/official_contamination_probe_manifest_ab_n15_seed20260926.json",
    "experiments/results/official_contamination_surface_manifest_ab_n15_seed20260926.json",
    "experiments/results/v5_axiom_confirmatory_manifest_a24_seed20260925.json",
)
RESULT_FILES = (
    "experiments/results/official_blind_openai_a_n30_off0.json",
    "experiments/results/official_blind_openai_b_n30_off0.json",
    "experiments/results/official_stratified_open_openai_ab_s5_seed20260922.json",
    "experiments/results/nonce_relation_holdout_openai_n30_seed20260926.json",
    "experiments/results/official_relation_transfer_openai_a_n30_seed20260927.json",
    "experiments/results/official_predicate_routing_openai_a_n27_seed20260927.json",
    "experiments/results/relation_guarded_v2_independent_calibrator_a27_b12_seed20260923.json",
    "experiments/results/model_weaklabel_relation_calibration_grouped_cv_5b53d848edca6abb58f4_n30.json",
    "experiments/results/v5_axiom_confirmatory_score_a_n24_seed20260925.json",
    "experiments/results/official_contamination_probe_analysis_ab_n15_seed20260926.json",
    "experiments/results/v5_axiom_confirmatory_semantic_secondary_a_n24_seed20260925.json",
    "experiments/annotations/relation_text_model_annotation_v1/model_agreement.json",
)
TEMPLATE_FILES = {
    "artifact/README.md": "README.md",
    "artifact/requirements.txt": "requirements.txt",
    "artifact/run_portable_tests.py": "run_portable_tests.py",
    "artifact/run_reproduction.py": "run_reproduction.py",
}
EXCLUDED_SOURCE_NAMES = {
    "run_competition_sim.py",
    "run_demo.py",
    "test_api_smoke.py",
}
FORBIDDEN = {
    "Windows absolute path": re.compile(r"\b[A-Za-z]:[\\/]"),
    "Unix home path": re.compile(r"/(?:Users|home)/[^/\s]+/"),
    "credential-like token": re.compile(r"\b(?:ak_|sk-)[A-Za-z0-9]{20,}\b"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def load(relative: str) -> dict[str, Any]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def safe_output(path: Path) -> Path:
    resolved = path.resolve()
    dist = DIST.resolve()
    if dist not in resolved.parents or resolved == dist:
        raise ValueError(f"artifact output must be a child of {dist}")
    return resolved


def sanitize_text(text: str) -> str:
    root = str(ROOT)
    for value in (root, root.replace("\\", "/"), root.replace("\\", "\\\\")):
        text = text.replace(value, ".")
    return text


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix.lower() in TEXT_SUFFIXES:
        text = sanitize_text(source.read_text(encoding="utf-8"))
        destination.write_text(text, encoding="utf-8", newline="\n")
    else:
        shutil.copyfile(source, destination)


def source_files() -> Iterable[tuple[Path, Path]]:
    for relative in (*PAPER_FILES, *FIGURE_FILES, *MANIFEST_FILES):
        yield ROOT / relative, Path(relative)
    for source_relative, destination_relative in TEMPLATE_FILES.items():
        yield ROOT / source_relative, Path(destination_relative)
    for source in sorted((ROOT / "src").rglob("*.py")):
        if "__pycache__" in source.parts:
            continue
        if source.name.startswith("_tmp_") or source.name.startswith("merge_"):
            continue
        if source.name in EXCLUDED_SOURCE_NAMES or source.name == Path(__file__).name:
            continue
        yield source, source.relative_to(ROOT)


def project_blind(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "systems": {
            name: {"summary": result["systems"][name]["summary"]}
            for name in ("closed_vocab", "open_vocab")
        }
    }


def project_density(result: dict[str, Any]) -> dict[str, Any]:
    meta_keys = (
        "confidence_threshold",
        "implicit_confidence_threshold",
        "max_nontax",
        "max_implicit_gap_chars",
    )
    return {
        "meta": {key: result["meta"][key] for key in meta_keys},
        "tasks": {
            task: {
                "systems": {
                    name: {"summary": system["summary"]}
                    for name, system in result["tasks"][task]["systems"].items()
                }
            }
            for task in ("task_a", "task_b")
        },
    }


def project_nonce(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "summaries": result["summaries"],
        "samples": [
            {
                "system": sample["system"],
                "status": sample["status"],
                "metrics": {
                    "gold_counts": sample["metrics"]["gold_counts"],
                    "pred_counts": sample["metrics"]["pred_counts"],
                },
            }
            for sample in result["samples"]
        ],
        "interpretation": result["interpretation"],
        "confirmatory_eligible": result["confirmatory_eligible"],
        "scientific_deployable": result["scientific_deployable"],
    }


def project_transfer(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "summaries": result["summaries"],
        "interpretation": result["interpretation"],
        "confirmatory_eligible": result["confirmatory_eligible"],
        "scientific_deployable": result["scientific_deployable"],
    }


def project_routing(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "primary",
            "summaries",
            "paired_bootstrap",
            "mechanism_gate",
            "interpretation",
            "confirmatory_eligible",
            "scientific_deployable",
        )
    }


def project_calibrator(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_summary": result["candidate_summary"],
        "deployment_support": result["deployment_support"],
        "calibration_metrics": result["calibration_metrics"],
        "interpretation": result["interpretation"],
    }


def project_weak_cv(result: dict[str, Any]) -> dict[str, Any]:
    grouped = result["grouped_cross_validation"]
    return {
        "grouped_cross_validation": {
            key: grouped[key]
            for key in (
                "protocol",
                "fit_scope",
                "evaluation_scope",
                "n_folds",
                "seed",
                "n_sample_groups",
                "n_positive_sample_groups",
                "coverage",
                "pooled_out_of_fold_metrics",
                "fold_status_counts",
                "calibration_shape",
                "deployment_support",
            )
        }
    }


def project_confirmatory(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "official_exact",
            "candidate_calibration",
            "preregistered_gate",
            "confirmatory_evidence_valid",
            "scientific_deployable",
            "confirmatory_eligible",
            "status",
            "reason",
        )
    }


def project_probe(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result[key]
        for key in (
            "probe_metrics",
            "sample_cluster_bootstrap",
            "probeability_task_performance_association",
            "interpretation",
            "confirmatory_eligible",
            "scientific_deployable",
        )
    }


def project_semantic(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "meta": result["meta"],
        "scope": result["scope"],
        "systems": {
            name: {
                key: system[key]
                for key in (
                    "exact_mean",
                    "semantic_mean",
                    "semantic_minus_exact",
                    "sample_change_counts",
                )
            }
            for name, system in result["systems"].items()
        },
        "calibrated_minus_base": result["calibrated_minus_base"],
        "locked_interpretation": result["locked_interpretation"],
        "scientific_deployable": result["scientific_deployable"],
        "confirmatory_eligible": result["confirmatory_eligible"],
        "reason": result["reason"],
    }


def result_projections() -> dict[str, dict[str, Any]]:
    paths = {Path(relative).name: relative for relative in RESULT_FILES}
    return {
        paths["official_blind_openai_a_n30_off0.json"]: project_blind(
            load(paths["official_blind_openai_a_n30_off0.json"])
        ),
        paths["official_blind_openai_b_n30_off0.json"]: project_blind(
            load(paths["official_blind_openai_b_n30_off0.json"])
        ),
        paths["official_stratified_open_openai_ab_s5_seed20260922.json"]: project_density(
            load(paths["official_stratified_open_openai_ab_s5_seed20260922.json"])
        ),
        paths["nonce_relation_holdout_openai_n30_seed20260926.json"]: project_nonce(
            load(paths["nonce_relation_holdout_openai_n30_seed20260926.json"])
        ),
        paths["official_relation_transfer_openai_a_n30_seed20260927.json"]: project_transfer(
            load(paths["official_relation_transfer_openai_a_n30_seed20260927.json"])
        ),
        paths["official_predicate_routing_openai_a_n27_seed20260927.json"]: project_routing(
            load(paths["official_predicate_routing_openai_a_n27_seed20260927.json"])
        ),
        paths["relation_guarded_v2_independent_calibrator_a27_b12_seed20260923.json"]: project_calibrator(
            load(paths["relation_guarded_v2_independent_calibrator_a27_b12_seed20260923.json"])
        ),
        paths["model_weaklabel_relation_calibration_grouped_cv_5b53d848edca6abb58f4_n30.json"]: project_weak_cv(
            load(paths["model_weaklabel_relation_calibration_grouped_cv_5b53d848edca6abb58f4_n30.json"])
        ),
        paths["v5_axiom_confirmatory_score_a_n24_seed20260925.json"]: project_confirmatory(
            load(paths["v5_axiom_confirmatory_score_a_n24_seed20260925.json"])
        ),
        paths["official_contamination_probe_analysis_ab_n15_seed20260926.json"]: project_probe(
            load(paths["official_contamination_probe_analysis_ab_n15_seed20260926.json"])
        ),
        paths["v5_axiom_confirmatory_semantic_secondary_a_n24_seed20260925.json"]: project_semantic(
            load(paths["v5_axiom_confirmatory_semantic_secondary_a_n24_seed20260925.json"])
        ),
        paths["model_agreement.json"]: load(paths["model_agreement.json"]),
    }


def write_projections(output: Path) -> None:
    for relative, payload in result_projections().items():
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def source_hashes() -> str:
    lines = []
    for relative in sorted((*RESULT_FILES, *MANIFEST_FILES)):
        lines.append(f"{digest(ROOT / relative)}  {relative}")
    return "\n".join(lines) + "\n"


def scan(output: Path) -> None:
    failures: list[str] = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(output).as_posix()}: {label}")
    if failures:
        raise RuntimeError("anonymous artifact scan failed:\n" + "\n".join(failures))


def write_manifest(output: Path) -> None:
    lines = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            lines.append(f"{digest(path)}  {path.relative_to(output).as_posix()}")
    (output / "MANIFEST.sha256").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )


def write_zip(output: Path) -> Path:
    archive = output.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as bundle:
        for path in sorted(output.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(output).as_posix()
            info = zipfile.ZipInfo(
                f"{ARCHIVE_ROOT}/{relative}", date_time=(2026, 9, 28, 0, 0, 0)
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, path.read_bytes())
    return archive


def build(output: Path) -> tuple[Path, Path]:
    output = safe_output(output)
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for source, relative in source_files():
        copy_file(source, output / relative)
    write_projections(output)
    (output / "SOURCE_ARTIFACTS.sha256").write_text(
        source_hashes(), encoding="utf-8", newline="\n"
    )
    scan(output)
    write_manifest(output)
    archive = write_zip(output)
    return output, archive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output, archive = build(args.output)
    file_count = sum(1 for path in output.rglob("*") if path.is_file())
    print(f"Anonymous artifact: {output}")
    print(f"Archive: {archive}")
    print(f"Files: {file_count}")
    print(f"Archive SHA256: {digest(archive)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
