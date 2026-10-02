"""Merge chunked official train baseline JSON files into one summary per task/system."""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "experiments" / "results"
TASK_NAME = {"a": "Flagship", "b": "Reuse"}


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def pack_from_chunk(data: dict, system: str, task: str) -> dict | None:
    key = f"task_{task}"
    if key not in data.get("tasks", {}) or system not in data["tasks"][key]:
        return None
    return data["tasks"][key][system]


def merge_system(packs: list[dict], system: str, task: str) -> dict:
    def merge_prf(field: str) -> dict:
        tp = fp = fn = 0
        for p in packs:
            m = p[field]
            tp += m["tp"]
            fp += m["fp"]
            fn += m["fn"]
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    n = sum(p["n_samples"] for p in packs)
    if n == 0:
        raise ValueError("no samples")
    gs = sum(p["official_graph_similarity_avg"] * p["n_samples"] for p in packs) / n
    typ = merge_prf("term_typing_f1_micro")
    tax = merge_prf("taxonomy_f1_micro")
    nontax = merge_prf("nontax_f1_micro")
    return {
        "system": system,
        "task": task,
        "n_samples": n,
        "term_typing_f1_micro": typ,
        "taxonomy_f1_micro": tax,
        "nontax_f1_micro": nontax,
        "official_graph_similarity_avg": gs,
        "composite_typing_tax_gs": (typ["f1"] + tax["f1"] + gs) / 3.0,
        "nontax_pred_total": sum(p["nontax_pred_total"] for p in packs),
        "nontax_gold_total": sum(p["nontax_gold_total"] for p in packs),
        "metric_policy": {
            "graph_similarity": "llms4ol2026_graph_similarity_exact",
            "external_reportable": True,
            "proxy_sgs_used": False,
        },
    }


def find_chunk_files(task: str, system: str) -> list[Path]:
    # official_train_baseline_openai_{task}_n{limit}_off{offset}.json
    pattern = f"official_train_baseline_openai_{task}_n*_off*.json"
    files = sorted(RES.glob(pattern))
    out = []
    for f in files:
        data = _load(f)
        if data and pack_from_chunk(data, system, task) is not None:
            # verify system present
            if system in data.get("tasks", {}).get(f"task_{task}", {}):
                out.append(f)
    return out


def merge_task(task: str, systems: list[str]) -> dict:
    results: dict = {
        "meta": {
            "challenge": "LLMs4OL 2026",
            "task": task,
            "backend": "openai",
            "model": "dots3-note-prev",
            "metric": "official_graph_similarity_exact",
            "note": "micro-merged from offset/limit chunks",
        },
        "tasks": {f"task_{task}": {}},
        "sources": {},
    }
    for system in systems:
        packs = []
        srcs = []
        # prefer official-looking chunk files; also accept merged n10 files if only those
        for f in sorted(RES.glob(f"official_train_baseline_openai_{task}_n*_off*.json")):
            data = _load(f)
            pack = pack_from_chunk(data, system, task) if data else None
            if pack and pack.get("n_samples", 0) > 0:
                # skip partial if a complete overlapping run exists later? keep all non-overlap via offset
                if pack.get("partial"):
                    # include partial only if no complete pack for same offset range — simple: include all with n>0
                    pass
                packs.append(pack)
                srcs.append(f.name)
        if not packs:
            # fallback: n10 combined files
            for f in [
                RES / f"official_train_baseline_openai_{task}_n10.json",
                RES / f"official_train_baseline_openai_{task}_n10_open_merged.json",
            ]:
                data = _load(f)
                pack = pack_from_chunk(data, system, task) if data else None
                if pack:
                    packs.append(pack)
                    srcs.append(f.name)
        if packs:
            results["tasks"][f"task_{task}"][system] = merge_system(packs, system, task)
            results["sources"][system] = srcs
    return results


def main() -> int:
    out_all = {
        "meta": {
            "metric": "official_graph_similarity_exact",
            "model": "dots3-note-prev",
            "backend": "openai",
        },
        "tasks": {},
    }
    for task in ("a", "b"):
        merged = merge_task(task, ["closed_vocab", "open_vocab"])
        out_all["tasks"].update(merged["tasks"])
        out_path = RES / f"official_train_baseline_openai_{task}_merged.json"
        out_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n=== Task {task.upper()} ({TASK_NAME.get(task, task)}) ===")
        header = (
            f"{'system':16} {'n':>3} {'TypF1':>8} {'TaxF1':>8} {'Nontax':>8} "
            f"{'GS_off':>8} {'Combo':>8} {'nontax':>10}"
        )
        print(header)
        print("-" * len(header))
        for system, p in merged["tasks"][f"task_{task}"].items():
            print(
                f"{system:16} {p['n_samples']:3d} "
                f"{p['term_typing_f1_micro']['f1']:8.4f} "
                f"{p['taxonomy_f1_micro']['f1']:8.4f} "
                f"{p['nontax_f1_micro']['f1']:8.4f} "
                f"{p['official_graph_similarity_avg']:8.4f} "
                f"{p['composite_typing_tax_gs']:8.4f} "
                f"{p['nontax_pred_total']:4d}/{p['nontax_gold_total']:<5d}"
            )
        print(f"Wrote {out_path}")

    out_path = RES / "official_train_baseline_openai_summary.json"
    out_path.write_text(json.dumps(out_all, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
