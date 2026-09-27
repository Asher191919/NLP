"""Offline evaluation on official LLMs4OL 2026 Task A/B train samples.

Uses mock LLM + OntoBridg vs closed-vocab baseline on a small subset,
with Edge-F1 / Neighborhood / Taxonomy graph-similarity components
(mirrors official formula; embedding mode omitted offline).

Run:
  & $env:MIMO_PYTHON D:\\NLP\\src\\run_official_2026_eval.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ontobridg.baselines.rag_pipeline import RAGPipeline
from ontobridg.data.schema import OntologyDataset, OntologyKnowledge, RelationSample, TaxonomySample, TermSample

from ontobridg.relations.open_vocab import TAXONOMIC_RELATIONS

CHALLENGE = ROOT / "data" / "LLMs4OL-Challenge" / "2026"
TASK_A = CHALLENGE / "TaskA-Flagship" / "train_task_a.json"
TASK_B = CHALLENGE / "TaskB-Reuse" / "train_task_b.json"
OUT = ROOT / "experiments" / "results" / "official_2026_offline_eval.json"


def _norm(x: str) -> str:
    return " ".join(str(x).lower().strip().split())


def triples_to_graph(triples: list[list[str]]) -> dict:
    terms: dict[str, str] = {}
    taxonomy: list[tuple[str, str]] = []
    relations: list[tuple[str, str, str]] = []
    types: set[str] = set()
    for t in triples:
        if len(t) < 3:
            continue
        s, p, o = _norm(t[0]), _norm(t[1]), _norm(t[2])
        if p in {"instance-of", "instanceof", "instance_of"}:
            terms[s] = o
            types.add(o)
        elif p in TAXONOMIC_RELATIONS or p in {"subclass-of", "subClassOf"}:
            taxonomy.append((s, o))
            types.add(s)
            types.add(o)
        else:
            relations.append((s, p, o))
            types.add(s)
            types.add(o)
    return {"terms": terms, "taxonomy": taxonomy, "relations": relations, "types": sorted(types)}


def edge_f1(gold: set, pred: set) -> float:
    """Micro-style edge F1. Empty-empty => 0 for averaging honesty (not 1.0)."""
    if not gold and not pred:
        return 0.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    p = tp / len(pred)
    r = tp / len(gold)
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def neighborhood_similarity(gold_triples: list[tuple], pred_triples: list[tuple]) -> float:
    def neigh(triples):
        d = defaultdict(set)
        for s, p, o in triples:
            d[s].add(("OUT", p, o))
            d[o].add(("IN", p, s))
        return d

    g, p = neigh(gold_triples), neigh(pred_triples)
    nodes = set(g) | set(p)
    scores = []
    for n in nodes:
        gs, ps = g.get(n, set()), p.get(n, set())
        u = gs | ps
        if u:
            scores.append(len(gs & ps) / len(u))
    return sum(scores) / len(scores) if scores else 0.0


def taxonomy_similarity(gold_tax: list[tuple[str, str]], pred_tax: list[tuple[str, str]]) -> float:
    def ancestors(edges):
        parent = defaultdict(set)
        for c, p in edges:
            parent[c].add(p)
        # transitive ancestors
        memo = {}

        def anc(n, seen=None):
            if n in memo:
                return memo[n]
            seen = seen or set()
            if n in seen:
                return set()
            seen = seen | {n}
            out = set(parent.get(n, set()))
            for p in list(parent.get(n, set())):
                out |= anc(p, seen)
            memo[n] = out
            return out

        nodes = set(parent) | {p for _, p in edges}
        return {n: anc(n) for n in nodes}

    g, p = ancestors(gold_tax), ancestors(pred_tax)
    nodes = set(g) | set(p)
    scores = []
    for n in nodes:
        gs, ps = g.get(n, set()), p.get(n, set())
        u = gs | ps
        if u:
            scores.append(len(gs & ps) / len(u))
    return sum(scores) / len(scores) if scores else 0.0


def sgs(gold_terms, gold_tax, gold_rel, pred_terms, pred_tax, pred_rel) -> dict:
    gold_all = [("is-a", c, p) for c, p in gold_tax] + [
        (r, h, t) for h, r, t in gold_rel if r not in TAXONOMIC_RELATIONS
    ] + [("instance-of", t, ty) for t, ty in gold_terms.items()]
    pred_all = [("is-a", c, p) for c, p in pred_tax] + [
        (r, h, t) for h, r, t in pred_rel if r not in TAXONOMIC_RELATIONS
    ] + [("instance-of", t, ty) for t, ty in pred_terms.items()]
    e = edge_f1(set(gold_all), set(pred_all))
    n = neighborhood_similarity(gold_all, pred_all)
    t = taxonomy_similarity(gold_tax, pred_tax)
    return {"edge_f1": e, "neighborhood": n, "taxonomy": t, "sgs": (e + n + t) / 3}


def f1_multilabel(gold: dict[str, str], pred: dict[str, str]) -> float:
    keys = set(gold) | set(pred)
    if not keys:
        return 0.0
    tp = sum(1 for k in keys if k in gold and k in pred and gold[k] == pred[k])
    fp = sum(1 for k in keys if k not in gold and k in pred)
    fn = sum(1 for k in keys if k in gold and (k not in pred or gold[k] != pred[k]))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def f1_triples(gold: set, pred: set) -> float:
    return edge_f1(gold, pred)


def micro_f1_triple_sets(list_gold: list[set], list_pred: list[set]) -> dict:
    """Pool TP/FP/FN across samples — avoids empty-empty F1=1 inflation."""
    tp = fp = fn = 0
    for g, p in zip(list_gold, list_pred):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def load_official(path: Path, limit: int) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data[:limit]


def sample_to_dataset(sample: dict, name: str) -> OntologyDataset:
    g = triples_to_graph(sample.get("primitive-ontology-triples") or sample.get("extended-primitive-ontology-triples") or [])
    terms = [TermSample(term=t, type_label=ty) for t, ty in g["terms"].items()]
    tax = [TaxonomySample(child=c, parent=p) for c, p in g["taxonomy"]]
    rels = [RelationSample(head=h, relation=r, tail=t) for h, r, t in g["relations"]]
    ctx = sample.get("context", "")
    corpus = [ctx] if ctx else []
    return OntologyDataset(
        name=name,
        types=g["types"],
        terms=terms,
        taxonomy=tax,
        relations=rels,
        domain_ranges={},
        metadata={"corpus": corpus, "id": sample.get("id", ""), "gold_triples": sample.get("primitive-ontology-triples") or []},
    )


def knowledge_from_dataset(ds: OntologyDataset, contamination: float = 0.35, seed: int = 0) -> OntologyKnowledge:
    import random

    rng = random.Random(seed)
    know = OntologyKnowledge()
    for t in ds.terms:
        if rng.random() > contamination:
            know.term_to_type[t.term] = t.type_label
    for e in ds.taxonomy:
        if rng.random() > contamination:
            know.parents.setdefault(e.child, set()).add(e.parent)
    for r in ds.relations:
        if rng.random() > contamination:
            know.relations.add((r.head, r.relation, r.tail))
    return know


def eval_system(system: str, datasets: list[OntologyDataset], seed: int) -> list[dict]:
    rows = []
    tax_golds, tax_preds, nontax_golds, nontax_preds = [], [], [], []
    type_tp = type_fp = type_fn = 0
    for ds in datasets:
        know = knowledge_from_dataset(ds, seed=seed)
        pipe = RAGPipeline.mock(
            know,
            closed_vocabulary=(system == "pro_team"),
            use_symbolic_repair=(system == "ontobridg"),
            seed=seed,
        )
        domain_ranges = {} if system == "pro_team" else dict(ds.domain_ranges)
        pred = pipe.run(
            corpus=list(ds.metadata.get("corpus", [])),
            type_inventory=list(ds.types) or sorted({t.type_label for t in ds.terms}),
            domain_ranges=domain_ranges,
        )
        gold_terms = ds.term_map()
        gold_tax = list(ds.taxonomy_pairs())
        gold_rel = list(ds.relation_triples())
        pred_terms = pred.get("terms", {})
        pred_tax = [tuple(x) for x in pred.get("taxonomy", [])]
        pred_rel = [tuple(x) for x in pred.get("relations", [])]
        nontax_gold = {r for r in gold_rel if r[1] not in TAXONOMIC_RELATIONS}
        nontax_pred = {r for r in pred_rel if r[1] not in TAXONOMIC_RELATIONS}
        tax_gold = {tuple(x) for x in gold_tax}
        tax_pred = {tuple(x) for x in pred_tax}
        tax_golds.append(tax_gold)
        tax_preds.append(tax_pred)
        nontax_golds.append(nontax_gold)
        nontax_preds.append(nontax_pred)
        for k in set(gold_terms) | set(pred_terms):
            g, p = gold_terms.get(k), pred_terms.get(k)
            if g and p and g == p:
                type_tp += 1
            elif p and (not g or g != p):
                type_fp += 1
                if g:
                    type_fn += 1
            elif g and not p:
                type_fn += 1
        s = sgs(gold_terms, gold_tax, gold_rel, pred_terms, pred_tax, pred_rel)
        rows.append(
            {
                "dataset": ds.name,
                "system": system,
                "pred_nontax": len(nontax_pred),
                "gold_nontax": len(nontax_gold),
                "pred_tax": len(tax_pred),
                "gold_tax": len(tax_gold),
                **s,
            }
        )
    tax_m = micro_f1_triple_sets(tax_golds, tax_preds)
    nontax_m = micro_f1_triple_sets(nontax_golds, nontax_preds)
    prec = type_tp / (type_tp + type_fp) if type_tp + type_fp else 0.0
    rec = type_tp / (type_tp + type_fn) if type_tp + type_fn else 0.0
    typing_f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    sgs_avg = sum(r["sgs"] for r in rows) / len(rows) if rows else 0.0
    summary_pack = {
        "typing_f1_micro": typing_f1,
        "typing_precision": prec,
        "typing_recall": rec,
        "taxonomy_f1_micro": tax_m["f1"],
        "taxonomy_detail": tax_m,
        "nontax_f1_micro": nontax_m["f1"],
        "nontax_detail": nontax_m,
        "sgs_avg": sgs_avg,
        "combo_official_like": (typing_f1 + tax_m["f1"] + sgs_avg) / 3,
        "pred_nontax": sum(r["pred_nontax"] for r in rows),
        "gold_nontax": sum(r["gold_nontax"] for r in rows),
        "n_samples": len(rows),
    }
    return rows, summary_pack


def main() -> int:
    n = 15
    a_samples = load_official(TASK_A, n)
    b_samples = load_official(TASK_B, n)
    a_sets = [sample_to_dataset(s, f"official_taskA_{i}") for i, s in enumerate(a_samples)]
    b_sets = [sample_to_dataset(s, f"official_taskB_{i}") for i, s in enumerate(b_samples)]

    results = {
        "n_samples_per_task": n,
        "metric_notes": [
            "Micro-averaged F1 across pooled TP/FP/FN (empty-empty not scored as 1.0).",
            "SGS here is offline proxy; official metric needs nomic embeddings + graph_similarity.py.",
            "Mock knowledge is sampled from gold — NOT a fair leaderboard run; only pipeline smoke test.",
            "Official Task A/B gold is dominated by is-a/instance-of; non-tax is rare long-tail.",
        ],
        "summary": {},
    }
    for task, sets in (("task_a", a_sets), ("task_b", b_sets)):
        for system in ("pro_team", "ontobridg"):
            rows, pack = eval_system(system, sets, seed=7)
            results[f"{task}_rows_{system}"] = rows
            results["summary"][f"{task}_{system}"] = pack

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 88)
    print("OFFICIAL 2026 offline eval (subset of train, mock LLM, micro-F1)")
    print("=" * 88)
    for task in ("task_a", "task_b"):
        print(f"\n{task.upper()}:")
        for label, key in (
            ("pro_team_closed", f"{task}_pro_team"),
            ("ontobridg_open", f"{task}_ontobridg"),
        ):
            s = results["summary"].get(key, {})
            if not s:
                continue
            print(
                f"  {label}: TypF1_micro={s['typing_f1_micro']:.4f} "
                f"TaxF1_micro={s['taxonomy_f1_micro']:.4f} "
                f"NontaxF1_micro={s['nontax_f1_micro']:.4f} "
                f"SGS_proxy={s['sgs_avg']:.4f} "
                f"combo={s['combo_official_like']:.4f} "
                f"nontax={s['pred_nontax']}/{s['gold_nontax']} "
                f"tax={s['taxonomy_detail']['tp']}/{s['taxonomy_detail']['fp']}/{s['taxonomy_detail']['fn']}"
            )
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
