from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ontobridg.contamination.protocol import run_contamination_protocol, scrub_knowledge_for_holdout
from ontobridg.data.synthetic import (
    ALL_DOMAINS,
    default_knowledge_from_dataset,
    make_multi_domain_dataset,
    make_synthetic_dataset,
)
from ontobridg.pipeline.orchestrator import OntoBridgPipeline
from ontobridg.symbolic.constraints import detect_cycles, transitive_closure
from ontobridg.tasks.task_a_flagship import TaskAFlagship


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _fmt_f1(x: float) -> str:
    return f"{x * 100:5.1f}%"


def main() -> int:
    seed = 13
    _section("OntoBridg demo — synthetic LLMs4OL offline run")

    dataset = make_synthetic_dataset()
    print(
        f"dataset: {dataset.name}\n"
        f"  terms={len(dataset.terms)}  types={len(dataset.types)}  "
        f"taxonomy_edges={len(dataset.taxonomy)}  relations={len(dataset.relations)}"
    )

    knowledge = default_knowledge_from_dataset(dataset, seed=seed)
    print(
        f"mock LLM knowledge: typed_terms={len(knowledge.term_to_type)}  "
        f"parent_links={sum(len(v) for v in knowledge.parents.values())}  "
        f"relations={len(knowledge.relations)}"
    )

    # Pipeline path
    pipeline = OntoBridgPipeline(knowledge, use_symbolic_repair=True, seed=seed)
    pipe_result = pipeline.run_dataset(dataset)
    m = pipe_result.metrics
    print(f"\npipeline system={pipe_result.system}")
    if m:
        print(f"  TermTyping F1 = {_fmt_f1(float(m.term_typing.get('f1', 0)))}")
        print(f"  Taxonomy   F1 = {_fmt_f1(float(m.taxonomy.get('f1', 0)))}")
        print(f"  NonTax     F1 = {_fmt_f1(float(m.nontax_relation.get('f1', 0)))}")
        print(f"  Official GraphSim = {m.official_graph_similarity:.4f}  (external_reportable=True)")
        print(f"  Proxy SGS internal = {m.proxy_sgs_score:.4f}  (external_reportable=False)")
        print(f"  Composite (GS+Typ+Tax)/3 = {m.official_like_score:.4f}")
    pred = pipe_result.prediction
    print(f"  pred terms/tax/rel/nontax = "
          f"{len(pred.get('terms', {}))}/{len(pred.get('taxonomy', []))}/"
          f"{len(pred.get('relations', []))}/{len(pred.get('nontax_relations', []))}")
    print(f"  repair mode: {pred.get('repair', {})}")

    _section("Symbolic sanity")
    gold_tax = dataset.taxonomy_pairs()
    closure = transitive_closure(gold_tax)
    cycles = detect_cycles([tuple(x) for x in pred.get("taxonomy", [])])
    print(f"  gold taxonomy direct={len(gold_tax)} closure={len(closure)}")
    print(f"  predicted cycles={len(cycles)}")

    _section("Closed-vocab baseline vs OntoBridg (Task A)")
    closed = TaskAFlagship(knowledge, system="pro_team", seed=seed)
    open_sys = TaskAFlagship(knowledge, system="ontobridg", seed=seed)
    c_res = closed.run(dataset)
    o_res = open_sys.run(dataset)
    for label, res in (("pro_team_closed", c_res), ("ontobridg_open", o_res)):
        mm = res.metrics
        print(
            f"  {label}: TypF1={float(mm.term_typing.get('f1',0)):.4f} "
            f"TaxF1={float(mm.taxonomy.get('f1',0)):.4f} "
            f"NontaxF1={float(mm.nontax_relation.get('f1',0)):.4f} "
            f"GS_official={mm.official_graph_similarity:.4f} "
            f"proxy_internal={mm.proxy_sgs_score:.4f} "
            f"composite={mm.official_like_score:.4f} "
            f"pred_nontax={res.extras.get('pred_nontax_count', 0)}"
        )

    _section("Contamination protocol (multi-domain, knowledge-scrubbed holdout)")
    multi = make_multi_domain_dataset(seed=seed)
    knowledge_m = default_knowledge_from_dataset(multi, contamination=0.55, seed=seed)
    runner_sys = TaskAFlagship(knowledge_m, system="ontobridg", seed=seed)

    def predict(dset, corpus):
        # Use scrubbed knowledge when holdout blindness is requested
        know = dset.metadata.get("scrubbed_knowledge") or knowledge_m
        sys_local = TaskAFlagship(know, system="ontobridg", seed=seed)
        pred = sys_local.run(dset).prediction
        return pred

    report = run_contamination_protocol(
        multi,
        predict,
        known_term_types=knowledge_m.term_to_type,
        known_relations=knowledge_m.relations,
        holdout_fraction=0.25,
        seed=seed,
        knowledge=knowledge_m,
    )
    rd = report.to_dict()
    cont = rd.get("contamination", {})
    drops = rd.get("drops", {})
    print(f"  contamination_level={cont.get('level')} "
          f"term_probe={cont.get('term_probe_rate', 0):.2f} "
          f"rel_probe={cont.get('relation_probe_rate', 0):.2f}")
    print(f"  drops: {drops}")

    out = {
        "dataset": dataset.name,
        "pipeline_metrics": m.to_dict() if m else {},
        "closed_vs_open": {
            "pro_team": c_res.metrics.to_dict() if c_res.metrics else {},
            "ontobridg": o_res.metrics.to_dict() if o_res.metrics else {},
        },
        "contamination": rd,
    }
    out_path = Path(r"D:\NLP\experiments\results\demo_metrics.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
