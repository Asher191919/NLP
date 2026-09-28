"""Run the anonymous OntoBridg artifact checks without external model calls."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEXT_SUFFIXES = {".bib", ".json", ".md", ".py", ".svg", ".txt"}
FIGURES = (
    "papers/figures/ontobridg_evidence_chain.pdf",
    "papers/figures/ontobridg_evidence_chain.png",
    "papers/figures/ontobridg_evidence_chain.svg",
    "papers/figures/ontobridg_main_results.pdf",
    "papers/figures/ontobridg_main_results.png",
    "papers/figures/ontobridg_main_results.svg",
)
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


def load_manifest() -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in (ROOT / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        entries[relative] = expected
    return entries


def verify_manifest(entries: dict[str, str], paths: tuple[str, ...] | None = None) -> None:
    selected = paths or tuple(entries)
    failures: list[str] = []
    for relative in selected:
        path = ROOT / relative
        expected = entries.get(relative)
        if expected is None:
            failures.append(f"missing manifest entry: {relative}")
        elif not path.is_file():
            failures.append(f"missing file: {relative}")
        elif digest(path) != expected:
            failures.append(f"hash mismatch: {relative}")
    if failures:
        raise RuntimeError("\n".join(failures))


def scan_anonymity() -> None:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        if ".venv" in path.parts or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN.items():
            if pattern.search(text):
                failures.append(f"{path.relative_to(ROOT).as_posix()}: {label}")
    if failures:
        raise RuntimeError("anonymity scan failed:\n" + "\n".join(failures))


def run(*parts: str) -> None:
    environment = os.environ.copy()
    for name in ("ONTOBRIDG_API_KEY", "OPENAI_API_KEY"):
        environment.pop(name, None)
    print("+", sys.executable, *parts, flush=True)
    subprocess.run(
        [sys.executable, *parts],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def main() -> int:
    entries = load_manifest()
    scan_anonymity()
    verify_manifest(entries)
    run("run_portable_tests.py")
    run("src/check_submission_manuscript.py")
    run("papers/figures/build_submission_figures.py")
    verify_manifest(entries, FIGURES)
    scan_anonymity()
    print("Anonymous artifact reproduction verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
