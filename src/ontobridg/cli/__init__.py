"""OntoBridg CLI entrypoints (thin)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ontobridg", description="OntoBridg OL toolkit")
    sub = parser.add_subparsers(dest="cmd")
    p_sim = sub.add_parser("sim", help="Run competition simulation")
    p_sim.add_argument("--out", default=str(Path(__file__).resolve().parents[3] / "experiments" / "results" / "competition_sim_metrics.json"))
    args = parser.parse_args(argv)
    if args.cmd == "sim":
        # delegate to run_competition_sim
        root = Path(__file__).resolve().parents[3]
        sys.path.insert(0, str(root / "src"))
        import run_competition_sim  # type: ignore
        return run_competition_sim.main(["--out", args.out])
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
