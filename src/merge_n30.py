"""Merge chunked official train baseline JSON files into one n=Total result per system."""
from __future__ import annotations

import json
from pathlib import Path

RES = Path(r"D:\NLP\experiments\results")
CKPT = RES / "checkpoints"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_system(task: str, system: str, offsets: list[int], size: int) -> dict | None:
    """Merge packs for system across [offset, offset+size) chunks.
    Prefer final n{size}_off{offset} result files; fallback to checkpoints.
    """
    packs = []
    for off in offsets:
        cand = [
            RES / f"official_train_baseline_openai_{task}_n{size}_off{off}.json",
            CKPT / f"openai_task{task}_{system}_n{size}_off{off}.json",
            CKPT / f"openai_task{task}_{system}_n{min(size, 10)}_off{off}.json",
        ]
        # also search any n for this offset+system
        for p in sorted(CKPT.glob(f"openai_task{task}_{system}_n*_off{off}.json")):
            cand.append(p)
        pack = None
        for p in cand:
            if not p.exists():
                continue
            data = load_json(p)
            if "tasks" in data:
                # result file shape
                task_pack = data["tasks"].get(f"task_{task}", {})
                pack = task_pack.get(system)
            else:
                # checkpoint shape (already a pack)
                pack = data
            if pack and pack.get("n_samples", pack.get("n_samples_done", 0)) > 0:
                # normalize partial checkpoint
                if "n_samples" not in pack and "n_samples_done" in pack:
                    pack = {**pack, "n_samples": pack["n_samples_done"]}
                if pack.get("system") != system and "system" in pack:
                    if pack.get("system") not in {system, None}:
                        continue
                packs.append(pack)
                break
        if not packs or (packs and packs[-1].get("partial")):
            # incomplete offset — cannot form full Total
            # try any checkpoint for this offset
            for p in sorted(CKPT.glob(f"openai_task{task}_{system}_n*_off{off}.json")):
                ck = load_json(p)
                if ck.get("n_samples_done", 0) > 0:
                    if not packs or packs[-1].get("n_samples", 0) == 0:
                        packs.append({**ck, "n_samples": ck["n_samples_done"], "system": system})
                    break

    if not packs:
        return None
    # If any pack partial and incomplete, still merge what we have but flag
    total_n = sum(p.get("n_samples", 0) for p in packs)
    if total_n == 0:
        return None

    def merge_prf(key: str) -> dict:
        tp = fp = fn = 0
        for p in packs:
            m = p.get(key) or p.get(key.replace("_micro", "")) or {}
            # packs store term_typing_f1_micro as nested dict with tp/fp/fn
            src = p.get(key)
            if src is None:
                # checkpoint uses same keys
                src = p.get(key)
            tp += src.get("tp", 0)
            fp += src.get("fp", 0)
            fn += src.get("fn", 0)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    gs_sum = 0.0
    gs_n = 0
    nontax_pred = 0
    nontax_gold = 0
    for p in packs:
        n = p.get("n_samples", 0)
        gs_sum += p.get("official_graph_similarity_avg", 0.0) * n
        nontax_pred += p.get("nontax_pred_total", 0)
        nontax_gold += p.get("nontax_gold_total", 0)
        gs_n += n

    typ = merge_prf("term_typing_f1_micro")
    tax = merge_prf("taxonomy_f1_micro")
    nontax = merge_prf("nontax_f1_micro")
    gs_avg = gs_sum / gs_n if gs_n else 0.0
    composite = (typ["f1"] + tax["f1"] + gs_avg) / 3.0

    complete = total_n >= len(offsets) * size or all(
        not p.get("partial", False) and p.get("n_samples", 0) == size for p in packs
    )
    # if last chunk smaller, complete when sum matches expected handled outside
    return {
        "system": system,
        "backend": "openai",
        "model": packs[0].get("model", "dots3-note-prev"),
        "n_samples": total_n,
        "n_chunks": len(packs),
        "term_typing_f1_micro": typ,
        "taxonomy_f1_micro": tax,
        "nontax_f1_micro": nontax,
        "official_graph_similarity_avg": gs_avg,
        "composite_typing_tax_gs": composite,
        "nontax_pred_total": nontax_pred,
        "nontax_gold_total": nontax_gold,
        "partial": any(p.get("partial") for p in packs),
        "metric_policy": {
            "graph_similarity": "llms4ol2026_graph_similarity_exact",
            "external_reportable": True,
            "proxy_sgs_used": False,
        },
        "chunk_n": [p.get("n_samples") for p in packs],
    }


def find_systems(task: str) -> list[str]:
    systems = set()
    for p in RES.glob(f"official_train_baseline_openai_{task}_n*_off*.json"):
        data = load_json(p)
        for s in data.get("tasks", {}).get(f"task_{task}", {}):
            systems.add(s)
    for p in CKPT.glob(f"openai_task{task}_*_n*_off*.json"):
        # openai_task{task}_{system}_n{limit}_off{offset}
        name = p.name
        # openai_taska_open_vocab_n5_off0.json
        parts = name.split("_")
        # ['openai','taska','open','vocab','n5','off0.json'] messy
        if "closed_vocab" in name:
            systems.add("closed_vocab")
        elif "open_vocab" in name:
            systems.add("open_vocab")
    return sorted(systems) or ["closed_vocab", "open_vocab"]


def merge_task(task: str, total: int, chunk: int) -> dict:
    offsets = list(range(0, total, chunk))
    # expected size per offset
    expected = []
    for off in offsets:
        expected.append(min(chunk, total - off))

    systems_out = {}
    for system in find_systems(task):
        packs = []
        for off, size in zip(offsets, expected):
            # try result file first
            result_file = RES / f"official_train_baseline_openai_{task}_n{size}_off{off}.json"
            pack = None
            if result_file.exists():
                data = load_json(result_file)
                pack = data.get("tasks", {}).get(f"task_{task}", {}).get(system)
            if pack is None:
                # checkpoint: openai_task{task}_{system}_n{size}_off{off}.json
                # system may contain underscore
                for p in CKPT.glob(f"openai_task{task}_*off{off}.json"):
                    if system.replace("_", "") in p.name.replace("_", "") or system in p.name:
                        ck = load_json(p)
                        if ck.get("system") == system or system in p.name:
                            pack = ck
                            break
            if pack is None:
                for p in CKPT.glob(f"openai_task{task}_*_off{off}.json"):
                    ck = load_json(p)
                    if ck.get("system") == system:
                        pack = ck
                        break
            if pack:
                if "n_samples" not in pack and "n_samples_done" in pack:
                    pack = {**pack, "n_samples": pack["n_samples_done"]}
                pack = {**pack, "system": system, "target_n": size}
                packs.append(pack)

        if not packs:
            continue

        def merge_prf(packs_list, key):
            tp = fp = fn = 0
            for p in packs_list:
                m = p[key]
                tp += m.get("tp", 0)
                fp += m.get("fp", 0)
                fn += m.get("fn", 0)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

        total_n = sum(p.get("n_samples", 0) for p in packs)
        gs = sum(p.get("official_graph_similarity_avg", 0.0) * p.get("n_samples", 0) for p in packs)
        gs = gs / total_n if total_n else 0.0
        typ = merge_prf(packs, "term_typing_f1_micro")
        tax = merge_prf(packs, "taxonomy_f1_micro")
        nontax = merge_prf(packs, "nontax_f1_micro")
        systems_out[system] = {
            "system": system,
            "n_samples": total_n,
            "target_n": total,
            "complete": total_n >= total,
            "term_typing_f1_micro": typ,
            "taxonomy_f1_micro": tax,
            "nontax_f1_micro": nontax,
            "official_graph_similarity_avg": gs,
            "composite_typing_tax_gs": (typ["f1"] + tax["f1"] + gs) / 3.0,
            "nontax_pred_total": sum(p.get("nontax_pred_total", 0) for p in packs),
            "nontax_gold_total": sum(p.get("nontax_gold_total", 0) for p in packs),
            "chunks": [p.get("n_samples") for p in packs],
            "partial": any(p.get("partial") for p in packs),
            "metric_policy": {
                "graph_similarity": "llms4ol2026_graph_similarity_exact",
                "external_reportable": True,
                "proxy_sgs_used": False,
            },
        }

    return {
        "meta": {
            "challenge": "LLMs4OL 2026",
            "task": task,
            "backend": "openai",
            "model": "dots3-note-prev",
            "target_n": total,
            "chunk": chunk,
            "metric": "official_graph_similarity_exact",
            "merged": True,
        },
        "tasks": {f"task_{task}": systems_out},
    }


def main() -> None:
    results = {
        "meta": {
            "metric": "llms4ol2026_graph_similarity_exact",
            "note": "External Graph Similarity only; proxy SGS not used",
        },
        "tasks": {},
    }
    for task, total in (("a", 30), ("b", 30)):
        merged = merge_task(task, total=total, chunk=5)
        results["tasks"].update(merged["tasks"])
        out = RES / f"official_train_baseline_openai_{task}_n{total}_merged.json"
        out.write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n=== Task {task.upper()} merged -> {out.name} ===")
        header = f"{'system':16} {'n':>4} {'TypF1':>8} {'TaxF1':>8} {'Nontax':>8} {'GS_off':>8} {'Combo':>8} {'nontax':>8} {'ok':>4}"
        print(header)
        print("-" * len(header))
        for name, pack in merged["tasks"][f"task_{task}"].items():
            print(
                f"{name:16} {pack['n_samples']:4d} "
                f"{pack['term_typing_f1_micro']['f1']:8.4f} "
                f"{pack['taxonomy_f1_micro']['f1']:8.4f} "
                f"{pack['nontax_f1_micro']['f1']:8.4f} "
                f"{pack['official_graph_similarity_avg']:8.4f} "
                f"{pack['composite_typing_tax_gs']:8.4f} "
                f"{pack['nontax_pred_total']:4d}/{pack['nontax_gold_total']:<3d} "
                f"{'Y' if pack['complete'] else 'N':>4}"
            )

    summary = RES / "official_train_baseline_openai_n30_summary.json"
    summary.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {summary}")


if __name__ == "__main__":
    main()
