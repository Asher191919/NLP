import json
from pathlib import Path

res = Path(r"D:\NLP\experiments\results")
chunks = [
    res / "official_train_baseline_openai_a_n5_off0.json",
    res / "official_train_baseline_openai_a_n4_off5.json",
    res / "official_train_baseline_openai_a_n1_off9.json",
]
closed_path = res / "official_train_baseline_openai_a_n10.json"

packs = []
chunk_names = []
for p in chunks:
    d = json.loads(p.read_text(encoding="utf-8"))
    pack = d["tasks"]["task_a"]["open_vocab"]
    packs.append(pack)
    chunk_names.append(p.name)
    print(
        p.name,
        "n=", pack["n_samples"],
        "TypF1=", round(pack["term_typing_f1_micro"]["f1"], 4),
        "TaxF1=", round(pack["taxonomy_f1_micro"]["f1"], 4),
        "TaxTP=", pack["taxonomy_f1_micro"]["tp"],
        "TaxFN=", pack["taxonomy_f1_micro"]["fn"],
        "GS=", round(pack["official_graph_similarity_avg"], 4),
    )


def merge_prf(key: str) -> dict:
    tp = fp = fn = 0
    for pack in packs:
        m = pack[key]
        tp += m["tp"]
        fp += m["fp"]
        fn += m["fn"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


typ = merge_prf("term_typing_f1_micro")
tax = merge_prf("taxonomy_f1_micro")
nontax = merge_prf("nontax_f1_micro")
n = sum(p["n_samples"] for p in packs)
gs = sum(p["official_graph_similarity_avg"] * p["n_samples"] for p in packs) / n
composite = (typ["f1"] + tax["f1"] + gs) / 3.0
nontax_pred = sum(p["nontax_pred_total"] for p in packs)
nontax_gold = sum(p["nontax_gold_total"] for p in packs)

open_m = {
    "system": "open_vocab",
    "backend": "openai",
    "model": "dots3-note-prev",
    "n_samples": n,
    "term_typing_f1_micro": typ,
    "taxonomy_f1_micro": tax,
    "nontax_f1_micro": nontax,
    "official_graph_similarity_avg": gs,
    "composite_typing_tax_gs": composite,
    "nontax_pred_total": nontax_pred,
    "nontax_gold_total": nontax_gold,
    "metric_policy": {
        "graph_similarity": "llms4ol2026_graph_similarity_exact",
        "external_reportable": True,
        "proxy_sgs_used": False,
    },
    "merged_from_chunks": True,
    "chunks": chunk_names,
}

out = res / "official_train_baseline_openai_a_n10_open_merged.json"
out.write_text(
    json.dumps(
        {
            "meta": {
                "challenge": "LLMs4OL 2026",
                "task": "a",
                "backend": "openai",
                "model": "dots3-note-prev",
                "system": "open_vocab",
                "n_samples": n,
                "metric": "official_graph_similarity_exact",
                "merged": True,
            },
            "tasks": {"task_a": {"open_vocab": open_m}},
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

closed = json.loads(closed_path.read_text(encoding="utf-8"))["tasks"]["task_a"][
    "closed_vocab"
]

print("\n" + "=" * 88)
print("OFFICIAL Task A Flagship — open n=10 COMPLETE vs closed n=10")
print("=" * 88)
header = (
    f"{'system':16} {'n':>3} {'TypF1':>8} {'TaxF1':>8} {'Nontax':>8} "
    f"{'GS_off':>8} {'Combo':>8} {'nontax':>10}"
)
print(header)
print("-" * len(header))
for name, p in (("closed_vocab", closed), ("open_vocab", open_m)):
    print(
        f"{name:16} {p['n_samples']:3d} "
        f"{p['term_typing_f1_micro']['f1']:8.4f} "
        f"{p['taxonomy_f1_micro']['f1']:8.4f} "
        f"{p['nontax_f1_micro']['f1']:8.4f} "
        f"{p['official_graph_similarity_avg']:8.4f} "
        f"{p['composite_typing_tax_gs']:8.4f} "
        f"{p['nontax_pred_total']:4d}/{p['nontax_gold_total']:<5d}"
    )

print("\nDELTA open - closed:")
print(
    f"  TypF1 {open_m['term_typing_f1_micro']['f1'] - closed['term_typing_f1_micro']['f1']:+.4f}"
)
print(
    f"  TaxF1 {open_m['taxonomy_f1_micro']['f1'] - closed['taxonomy_f1_micro']['f1']:+.4f}"
)
print(
    f"  GS    {open_m['official_graph_similarity_avg'] - closed['official_graph_similarity_avg']:+.4f}"
)
print(
    f"  Combo {open_m['composite_typing_tax_gs'] - closed['composite_typing_tax_gs']:+.4f}"
)
print(f"\nWrote {out}")
