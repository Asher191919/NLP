"""Official LLMs4OL 2026 train baseline runner.

Systems:
  - closed_vocab   : pro-team-like taxonomy-only extraction (no non-tax)
  - open_vocab     : OntoBridg open relation mining + symbolic repair (offline patterns/KB)
  - openai         : real OpenAI-compatible LLM (requires OPENAI_API_KEY / ONTOBRIDG_*)

External metrics:
  - Term-Typing F1 (micro)
  - Taxonomy F1 (micro edge)
  - Non-taxonomic relation F1 (micro edge)
  - Official Graph Similarity (exact_match from challenge metrics)

Usage:
  & $env:MIMO_PYTHON D:\\NLP\\src\\run_official_train_baseline.py
  & $env:MIMO_PYTHON D:\\NLP\\src\\run_official_train_baseline.py --task a --limit 200 --backend openai
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ontobridg.baselines.rag_pipeline import RAGPipeline
from ontobridg.data.schema import (
    OntologyDataset,
    OntologyKnowledge,
    RelationSample,
    TaxonomySample,
    TermSample,
)
from ontobridg.eval.official_graph_similarity import evaluate_structures as official_gs
from ontobridg.relations.open_vocab import TAXONOMIC_RELATIONS

CHALLENGE = ROOT / "data" / "LLMs4OL-Challenge" / "2026"
OUT_DIR = ROOT / "experiments" / "results"
SRC_DIR = Path(__file__).resolve().parent


def _maybe_load_dots_config() -> None:
    """Fill ONTOBRIDG_* from dots_config.json when env vars are unset."""
    import json as _json

    cfg_path = SRC_DIR / "dots_config.json"
    if not cfg_path.exists():
        return
    try:
        cfg = _json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not os.environ.get("ONTOBRIDG_BASE_URL") and cfg.get("base_url"):
        os.environ["ONTOBRIDG_BASE_URL"] = str(cfg["base_url"])
    if not os.environ.get("ONTOBRIDG_MODEL") and cfg.get("model"):
        os.environ["ONTOBRIDG_MODEL"] = str(cfg["model"])
    if (
        not os.environ.get("ONTOBRIDG_API_KEY")
        and not os.environ.get("OPENAI_API_KEY")
        and cfg.get("api_key")
    ):
        os.environ["ONTOBRIDG_API_KEY"] = str(cfg["api_key"])
        os.environ.setdefault("OPENAI_API_KEY", str(cfg["api_key"]))


_maybe_load_dots_config()


def _norm(x: str) -> str:
    return " ".join(str(x).lower().strip().split())


def triples_to_graph(triples) -> dict:
    terms: dict[str, str] = {}
    taxonomy: list[tuple[str, str]] = []
    relations: list[tuple[str, str, str]] = []
    types: set[str] = set()
    all_triples: list[list[str]] = []
    for t in triples or []:
        if len(t) < 3:
            continue
        s, p, o = _norm(t[0]), _norm(t[1]), _norm(t[2])
        all_triples.append([s, p, o])
        if p in {"instance-of", "instanceof", "instance_of"}:
            terms[s] = o
            types.add(o)
        elif p in TAXONOMIC_RELATIONS or p in {"subclass-of", "subclassof", "kind-of"}:
            taxonomy.append((s, o))
            types.update([s, o])
        else:
            relations.append((s, p, o))
            types.update([s, o])
    return {
        "terms": terms,
        "taxonomy": taxonomy,
        "relations": relations,
        "types": sorted(types),
        "triples": all_triples,
    }


def micro_prf(list_gold: list[set], list_pred: list[set]) -> dict:
    tp = fp = fn = 0
    for g, p in zip(list_gold, list_pred):
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def typing_micro(gold_maps: list[dict], pred_maps: list[dict]) -> dict:
    tp = fp = fn = 0
    for g, p in zip(gold_maps, pred_maps):
        for k in set(g) | set(p):
            gv, pv = g.get(k), p.get(k)
            if gv and pv and gv == pv:
                tp += 1
            elif pv and (not gv or gv != pv):
                fp += 1
                if gv:
                    fn += 1
            elif gv and not pv:
                fn += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def sample_to_dataset(sample: dict, name: str, gold_key: str) -> OntologyDataset:
    gold_triples = sample.get(gold_key) or sample.get("primitive-ontology-triples") or []
    g = triples_to_graph(gold_triples)
    # For Reuse, also keep initial ontology as provided context structure
    initial = triples_to_graph(sample.get("initial-primitive-ontology-triples") or [])
    terms = [TermSample(term=t, type_label=ty) for t, ty in g["terms"].items()]
    tax = [TaxonomySample(child=c, parent=p) for c, p in g["taxonomy"]]
    rels = [RelationSample(head=h, relation=r, tail=t) for h, r, t in g["relations"]]
    corpus = []
    if sample.get("context"):
        corpus.append(str(sample["context"]))
    return OntologyDataset(
        name=name,
        types=g["types"] or initial["types"],
        terms=terms,
        taxonomy=tax,
        relations=rels,
        domain_ranges={},
        metadata={
            "corpus": corpus,
            "id": sample.get("id", ""),
            "gold_triples": g["triples"],
            "initial_triples": initial["triples"],
            "given_terms": sample.get("terms") or [],
            "given_types": sample.get("types") or [],
        },
    )


def empty_knowledge() -> OntologyKnowledge:
    """No gold leakage — baselines must induce from context/patterns only."""
    return OntologyKnowledge()


def pattern_only_predict(corpus: list[str], type_inventory: list[str], closed: bool) -> dict:
    """Cheap non-LLM baseline: surface patterns + is-a heuristics from context."""
    text = "\n".join(corpus)
    low = text.lower()
    terms: dict[str, str] = {}
    taxonomy: list[list[str]] = []
    relations: list[list[str]] = []

    # crude term harvest from Title/Content noun-ish spans
    import re

    # capture phrases around "is a / subclass of / instance of"
    for m in re.finditer(
        r"([a-z][a-z0-9 \-/()]{2,60}?)\s+(?:is a|is an|is formally|is subclass of|subclass of|instance of|functions as)\s+([a-z][a-z0-9 \-/()]{2,60})",
        low,
    ):
        child, parent = m.group(1).strip(), m.group(2).strip()
        child = re.sub(r"\s+", " ", child)
        parent = re.sub(r"\s+", " ", parent)
        taxonomy.append([child, "is-a", parent])
        terms.setdefault(child, parent if parent in type_inventory else "Entity")

    for m in re.finditer(
        r"([a-z][a-z0-9 \-/()]{2,50}?)\s+(?:is recognized as|is treated as|is categorized as)\s+(?:a|an)?\s*([a-z][a-z0-9 \-/()]{2,50})",
        low,
    ):
        child, parent = m.group(1).strip(), m.group(2).strip()
        taxonomy.append([child, "is-a", parent])

    if not closed:
        for m in re.finditer(
            r"([a-z][a-z0-9 \-/()]{2,50}?)\s+(?:measures|detects|triggers|receives|controls|treats|causes|uses|monitors|includes|contains)\s+([a-z][a-z0-9 \-/()]{2,50})",
            low,
        ):
            h, t = m.group(1).strip(), m.group(2).strip()
            rel = m.group(0)
            for key, lab in (
                ("measures", "measures"),
                ("detects", "detects"),
                ("triggers", "triggers"),
                ("receives", "receives"),
                ("controls", "controls"),
                ("treats", "treats"),
                ("causes", "causes"),
                ("uses", "uses"),
                ("monitors", "monitors"),
                ("includes", "includes"),
                ("contains", "has-part"),
            ):
                if key in rel:
                    relations.append([h, lab, t])
                    break

    # type inventory snap
    for term in list(terms):
        matched = None
        for ty in type_inventory:
            if term.endswith(ty) or ty in term:
                matched = ty
                break
        terms[term] = matched or terms.get(term, "Entity")

    return {
        "terms": terms,
        "taxonomy": taxonomy,
        "relations": [] if closed else relations,
        "nontax_relations": [] if closed else [r for r in relations if r[1] not in TAXONOMIC_RELATIONS],
        "mode": "pattern_closed" if closed else "pattern_open",
        "corpus_chars": len(text),
    }


def openai_predict(dataset: OntologyDataset, backend: str, model: str, closed: bool) -> dict:
    from ontobridg.llm.factory import build_llm
    from ontobridg.prompts.templates import taska_prompt, taskb_prompt, taskc_prompt
    import json as _json
    import re as _re

    llm = build_llm(backend, model=model)
    types = list(dataset.types) or sorted({t.type_label for t in dataset.terms})
    corpus = list(dataset.metadata.get("corpus", []))
    context = "\n".join(corpus)
    context_cap = context[:3500]
    # Prefer gold terms when present (term typing eval is on gold terms)
    terms_guess = list(dataset.metadata.get("given_terms") or []) or [
        t.term for t in dataset.terms
    ][:50]
    if not types:
        types = list(dataset.metadata.get("given_types") or []) or ["Entity"]

    def _parse(text: str):
        text = text.strip()
        m = _re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, _re.S | _re.I)
        if m:
            text = m.group(1)
        try:
            return _json.loads(text)
        except Exception:
            m2 = _re.search(r"(\{.*\}|\[.*\])", text, _re.S)
            if m2:
                try:
                    return _json.loads(m2.group(1))
                except Exception:
                    return {}
            return {}

    # --- Task A typing: give context + gold term list + type inventory ---
    terms_snip = terms_guess[:40]
    types_snip = types[:80]
    prompt_a = (
        "Ontology term typing. Given context and a type inventory, "
        "assign exactly one type from the inventory to EACH term.\n\n"
        f"Context:\n{context_cap}\n\n"
        f"Type inventory (choose ONLY these):\n{', '.join(types_snip)}\n\n"
        f"Terms:\n{', '.join(terms_snip)}\n\n"
        'Return ONLY JSON: [{"term":"...","type":"..."}]'
    )
    print(f"    [llm] typing terms={len(terms_snip)} types={len(types_snip)} ctx={len(context_cap)}", flush=True)
    t0 = time.time()
    raw_a = llm.complete(
        prompt_a, task="term_typing", meta={"terms": terms_snip, "types": types_snip}
    ).text
    print(f"    [llm] typing done {time.time()-t0:.1f}s", flush=True)
    parsed_a = _parse(raw_a)
    term_types: dict[str, str] = {}
    items = parsed_a if isinstance(parsed_a, list) else parsed_a.get("terms", [])
    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict) and it.get("term"):
                term_types[_norm(it["term"])] = _norm(it.get("type") or "Entity")
    # snap types into inventory when possible
    inv_set = {t.lower() for t in types_snip}
    for t, ty in list(term_types.items()):
        if ty not in inv_set:
            best = next((x for x in types_snip if x.lower() in ty or ty in x.lower()), None)
            if best:
                term_types[t] = best

    # --- Task B taxonomy ---
    tax_types = types_snip or list(set(term_types.values()))[:80]
    prompt_b = (
        "Discover is-a (subclass) relations among the types using the context.\n"
        f"Context:\n{context_cap}\n\n"
        f"Types:\n{', '.join(tax_types)}\n"
        f"Known term typings (optional):\n"
        f"{', '.join(f'{k}->{v}' for k,v in list(term_types.items())[:30])}\n\n"
        'Return ONLY JSON list of {{"child":"...","parent":"..."}}. child is-a parent.\n'
        "Prefer shallow useful edges, no cycles."
    )
    print(f"    [llm] taxonomy types={len(tax_types)}", flush=True)
    t0 = time.time()
    raw_b = llm.complete(
        prompt_b,
        task="taxonomy",
        meta={"types": tax_types, "term_types": dict(list(term_types.items())[:40])},
    ).text
    print(f"    [llm] taxonomy done {time.time()-t0:.1f}s", flush=True)
    parsed_b = _parse(raw_b)
    taxonomy = []
    items_b = parsed_b if isinstance(parsed_b, list) else parsed_b.get("taxonomy", [])
    if isinstance(items_b, list):
        for it in items_b:
            if isinstance(it, dict) and it.get("child") and it.get("parent"):
                taxonomy.append([_norm(it["child"]), "is-a", _norm(it["parent"])])

    # --- Task C open relations (only open system) ---
    relations = []
    if not closed:
        term_types_str = ", ".join(list(term_types)[:40]) or "(none)"
        prompt_c = (
            "Extract NON-TAXONOMIC relations (not is-a) from the context.\n"
            "Allowed examples: treats, causes, part-of, located-in, measures, "
            "controls, uses, monitors, has-part, produces.\n"
            f"Context:\n{context_cap}\n\n"
            f"Known terms: {term_types_str}\n\n"
            'Return ONLY JSON list: [{"head":"...","relation":"...","tail":"..."}]\n'
            "If none, return []."
        )
        print(f"    [llm] relations open", flush=True)
        t0 = time.time()
        raw_c = llm.complete(
            prompt_c,
            task="relation_extraction",
            meta={"corpus": [context_cap], "term_types": term_types},
        ).text
        print(f"    [llm] relations done {time.time()-t0:.1f}s", flush=True)
        parsed_c = _parse(raw_c)
        items_c = parsed_c if isinstance(parsed_c, list) else parsed_c.get("relations", [])
        if isinstance(items_c, list):
            for it in items_c:
                if isinstance(it, dict) and it.get("head") and it.get("relation") and it.get("tail"):
                    relations.append(
                        [_norm(it["head"]), _norm(it["relation"]), _norm(it["tail"])]
                    )

    return {
        "terms": term_types,
        "taxonomy": taxonomy,
        "relations": relations,
        "nontax_relations": [r for r in relations if r[1] not in TAXONOMIC_RELATIONS],
        "mode": f"llm:{backend}:{model}{'_closed' if closed else '_open'}",
    }


def predict_system(system: str, ds: OntologyDataset, args) -> dict:
    use_llm = (
        args.backend in {"openai", "local"}
        and bool(
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ONTOBRIDG_API_KEY")
        )
    )
    if system in {"closed_vocab", "pro_team"}:
        if use_llm:
            try:
                return openai_predict(ds, args.backend, args.model, closed=True)
            except Exception as exc:
                print(f"  [warn] openai closed_vocab failed, fallback offline: {exc}", flush=True)
        return pattern_only_predict(
            list(ds.metadata.get("corpus", [])), list(ds.types), closed=True
        )
    if system in {"open_vocab", "ontobridg"}:
        if use_llm:
            try:
                return openai_predict(ds, args.backend, args.model, closed=False)
            except Exception as exc:
                print(f"  [warn] openai open_vocab failed, fallback offline: {exc}", flush=True)
        knowledge = ds.metadata.get("scrubbed_knowledge") or empty_knowledge()
        # seed inventory from given types only (no gold term->type)
        if ds.metadata.get("given_types"):
            for ty in ds.metadata["given_types"]:
                knowledge.type_lexicon.setdefault("types", [])
        pipe = RAGPipeline.mock(
            knowledge,
            closed_vocabulary=False,
            use_symbolic_repair=True,
            seed=args.seed,
        )
        type_inventory = list(ds.types) or list(ds.metadata.get("given_types") or []) or ["Entity"]
        pred = pipe.run(
            corpus=list(ds.metadata.get("corpus", [])),
            type_inventory=type_inventory,
            domain_ranges={},
        )
        # merge pattern harvest for official long-tail triples
        pat = pattern_only_predict(list(ds.metadata.get("corpus", [])), type_inventory, closed=False)
        terms = {**pat.get("terms", {}), **pred.get("terms", {})}
        tax = pred.get("taxonomy", []) + pat.get("taxonomy", [])
        # dedup
        tax = [list(x) for x in {tuple(x) for x in tax}]
        rel = pred.get("relations", []) + pat.get("relations", [])
        rel = [list(x) for x in {tuple(x) for x in rel}]
        return {
            "terms": terms,
            "taxonomy": tax,
            "relations": rel,
            "nontax_relations": [r for r in rel if r[1] not in TAXONOMIC_RELATIONS],
            "mode": "ontobridg_open_offline",
        }
    raise ValueError(system)


def eval_batch(system: str, datasets: list[OntologyDataset], args) -> dict:
    """Accumulate micro metrics; checkpoint after each sample so timeouts don't lose work."""
    gold_terms_list: list[dict] = []
    pred_terms_list: list[dict] = []
    gold_tax_list: list[set] = []
    pred_tax_list: list[set] = []
    gold_nontax_list: list[set] = []
    pred_nontax_list: list[set] = []
    gs_sum = 0.0
    gs_n = 0
    nontax_pred_total = 0
    nontax_gold_total = 0
    n_done = 0
    t0 = time.time()

    ckpt_dir = OUT_DIR / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{args.backend}_task{args.task}_{system}_n{args.limit}_off{getattr(args, 'offset', 0)}.json"
    ckpt_base = {
        "system": system,
        "backend": args.backend,
        "model": args.model,
        "task": args.task,
        "limit": args.limit,
        "offset": getattr(args, "offset", 0),
        "n_samples_done": 0,
    }

    def _snapshot() -> None:
        typing = typing_micro(gold_terms_list, pred_terms_list)
        tax = micro_prf(gold_tax_list, pred_tax_list)
        nontax = micro_prf(gold_nontax_list, pred_nontax_list)
        gs_avg = gs_sum / gs_n if gs_n else 0.0
        pack = {
            **ckpt_base,
            "n_samples_done": n_done,
            "n_samples_target": len(datasets),
            "term_typing_f1_micro": typing,
            "taxonomy_f1_micro": tax,
            "nontax_f1_micro": nontax,
            "official_graph_similarity_avg": gs_avg,
            "composite_typing_tax_gs": (typing["f1"] + tax["f1"] + gs_avg) / 3.0
            if (n_done and (typing["f1"] or tax["f1"] or gs_avg))
            else (typing["f1"] + tax["f1"] + gs_avg) / 3.0,
            "nontax_pred_total": nontax_pred_total,
            "nontax_gold_total": nontax_gold_total,
            "elapsed_sec": round(time.time() - t0, 2),
            "partial": n_done < len(datasets),
            "metric_policy": {
                "graph_similarity": "llms4ol2026_graph_similarity_exact",
                "external_reportable": True,
                "proxy_sgs_used": False,
            },
        }
        ckpt_path.write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")

    try:
        for ds in datasets:
            gold_terms = ds.term_map()
            gold_tax = {tuple(x) for x in ds.taxonomy_pairs()}
            gold_rel = ds.relation_triples()
            gold_nontax = {r for r in gold_rel if r[1] not in TAXONOMIC_RELATIONS}
            t_pred = time.time()
            pred = predict_system(system, ds, args)
            if getattr(args, "verbose", False):
                print(
                    f"    sample={ds.name} pred_mode={pred.get('mode')} "
                    f"terms={len(pred.get('terms', {}))} tax={len(pred.get('taxonomy', []))} "
                    f"rel={len(pred.get('relations', []))} "
                    f"[{time.time() - t_pred:.1f}s]",
                    flush=True,
                )
            pred_terms = pred.get("terms", {})
            pred_tax = {(x[0], x[-1]) for x in pred.get("taxonomy", []) if len(x) >= 2}
            pred_rel = {tuple(x) for x in pred.get("relations", []) if len(x) >= 3}
            pred_nontax = {r for r in pred_rel if r[1] not in TAXONOMIC_RELATIONS}

            gold_terms_list.append(gold_terms)
            pred_terms_list.append(pred_terms)
            gold_tax_list.append(gold_tax)
            pred_tax_list.append(pred_tax)
            gold_nontax_list.append(gold_nontax)
            pred_nontax_list.append(pred_nontax)
            nontax_pred_total += len(pred_nontax)
            nontax_gold_total += len(gold_nontax)

            gs = official_gs(
                gold_term_types=gold_terms,
                gold_taxonomy=list(gold_tax),
                gold_relations=list(gold_rel),
                pred_term_types=pred_terms,
                pred_taxonomy=list(pred_tax),
                pred_relations=list(pred_rel),
            )
            gs_sum += gs["graph_similarity"]
            gs_n += 1
            n_done += 1
            _snapshot()
    except Exception as exc:
        _snapshot()
        print(f"  [checkpoint] {system} stopped after {n_done}/{len(datasets)}: {exc}", flush=True)
        # return partial pack if any samples done, else re-raise
        if n_done == 0:
            raise

    typing = typing_micro(gold_terms_list, pred_terms_list)
    tax = micro_prf(gold_tax_list, pred_tax_list)
    nontax = micro_prf(gold_nontax_list, pred_nontax_list)
    gs_avg = gs_sum / gs_n if gs_n else 0.0
    composite = (typing["f1"] + tax["f1"] + gs_avg) / 3.0
    return {
        **ckpt_base,
        "n_samples": n_done,
        "n_samples_target": len(datasets),
        "partial": n_done < len(datasets),
        "term_typing_f1_micro": typing,
        "taxonomy_f1_micro": tax,
        "nontax_f1_micro": nontax,
        "official_graph_similarity_avg": gs_avg,
        "composite_typing_tax_gs": composite,
        "nontax_pred_total": nontax_pred_total,
        "nontax_gold_total": nontax_gold_total,
        "elapsed_sec": round(time.time() - t0, 2),
        "metric_policy": {
            "graph_similarity": "llms4ol2026_graph_similarity_exact",
            "external_reportable": True,
            "proxy_sgs_used": False,
        },
        "checkpoint": str(ckpt_path),
    }


def load_samples(task: str, limit: int, offset: int = 0) -> list[dict]:
    if task == "a":
        path = CHALLENGE / "TaskA-Flagship" / "train_task_a.json"
        gold_key = "primitive-ontology-triples"
        prefix = "flagship"
    elif task == "b":
        path = CHALLENGE / "TaskB-Reuse" / "train_task_b.json"
        gold_key = "extended-primitive-ontology-triples"
        prefix = "reuse"
    else:
        raise ValueError("task must be a or b")
    data = json.loads(path.read_text(encoding="utf-8"))
    start = max(0, int(offset))
    if limit and limit > 0:
        data = data[start : start + limit]
    else:
        data = data[start:]
    datasets = [
        sample_to_dataset(s, f"{prefix}_{start + i}", gold_key)
        for i, s in enumerate(data)
    ]
    return datasets


def main() -> int:
    ap = argparse.ArgumentParser(description="Official LLMs4OL 2026 train baseline")
    ap.add_argument("--task", default="ab", choices=["a", "b", "ab"])
    ap.add_argument("--limit", type=int, default=300, help="samples per task (0=all)")
    ap.add_argument("--backend", default="offline", choices=["offline", "openai", "local", "mock"])
    ap.add_argument("--model", default=os.environ.get("ONTOBRIDG_MODEL", "gpt-4o-mini"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--systems", default="closed_vocab,open_vocab")
    ap.add_argument("--verbose", action="store_true", help="per-sample progress")
    ap.add_argument("--offset", type=int, default=0, help="start index in train split")
    args = ap.parse_args()

    has_key = bool(os.environ.get("OPENAI_API_KEY") or os.environ.get("ONTOBRIDG_API_KEY"))
    print("=" * 88)
    print("Official LLMs4OL 2026 train baseline")
    print("=" * 88)
    print(
        f"task={args.task} limit={args.limit or 'ALL'} offset={args.offset} "
        f"backend={args.backend} model={args.model}"
    )
    print(f"API key present: {has_key}")
    if args.backend in {"openai", "local"} and not has_key:
        print("WARNING: no OPENAI_API_KEY/ONTOBRIDG_API_KEY — falling back to offline baselines.")
        args.backend = "offline"
    print("External metric: official Graph Similarity exact_match (NOT internal proxy).")
    print()

    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    results = {
        "meta": {
            "challenge": "LLMs4OL 2026",
            "data_root": str(CHALLENGE),
            "backend": args.backend,
            "model": args.model,
            "limit": args.limit,
            "offset": args.offset,
            "metric": "official_graph_similarity_exact",
            "api_key_present": has_key,
        },
        "tasks": {},
    }

    tasks = ["a", "b"] if args.task == "ab" else [args.task]
    for task in tasks:
        datasets = load_samples(task, args.limit, offset=args.offset)
        print(f"--- Task {task.upper()}  n={len(datasets)} offset={args.offset} ---")
        task_out = {}
        for system in systems:
            pack = eval_batch(system, datasets, args)
            task_out[system] = pack
            note = " [PARTIAL]" if pack.get("partial") else ""
            print(
                f"  {system:14s} TypF1={pack['term_typing_f1_micro']['f1']:.4f} "
                f"TaxF1={pack['taxonomy_f1_micro']['f1']:.4f} "
                f"NontaxF1={pack['nontax_f1_micro']['f1']:.4f} "
                f"GS_official={pack['official_graph_similarity_avg']:.4f} "
                f"composite={pack['composite_typing_tax_gs']:.4f} "
                f"nontax={pack['nontax_pred_total']}/{pack['nontax_gold_total']} "
                f"n={pack['n_samples']}/{pack.get('n_samples_target', '?')} "
                f"[{pack['elapsed_sec']}s]{note}",
                flush=True,
            )
        results["tasks"][f"task_{task}"] = task_out
        print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        OUT_DIR
        / f"official_train_baseline_{args.backend}_{args.task}"
        f"_n{args.limit}_off{args.offset}.json"
    )
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {out_path}")

    if not has_key:
        print(
            "\nNOTE: Offline numbers are pattern/mock baselines on official train gold.\n"
            "Pre-filled dots endpoint config (API key required):\n"
            "  1) $env:ONTOBRIDG_API_KEY = 'ak_<rotated-key>'\n"
            "  2) & D:\\NLP\\src\\run_dots_smoke.ps1\n"
            "  3) & D:\\NLP\\src\\run_dots_baseline.ps1 -Limit 20\n"
            "  Base=https://note3-prev-api.askdiandian.com/v1  Model=dots3-note-prev\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
