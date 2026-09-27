from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ontobridg.contamination.utils import (
    build_novel_synthetic_ontology,
    evaluate_contamination_slices,
    holdout_terms,
)
from ontobridg.data.loaders import load_competition_folder, load_json_ontology, save_json_ontology
from ontobridg.data.synthetic import default_knowledge_from_dataset, make_synthetic_dataset
from ontobridg.eval.metrics import evaluate_pipeline_result
from ontobridg.llm.factory import build_llm
from ontobridg.pipeline.orchestrator import OntoBridgPipeline
from ontobridg.prompts.templates import PromptStyle


def _print_json(payload: object) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_dataset(args: argparse.Namespace):
    if args.data:
        path = Path(args.data)
        if path.is_dir():
            return load_competition_folder(path)
        return load_json_ontology(path)
    return make_synthetic_dataset(args.dataset_name)


def cmd_run(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args)
    if args.backend == "mock":
        knowledge = default_knowledge_from_dataset(dataset, seed=args.seed) if args.synthetic_knowledge else None
        llm = build_llm("mock", knowledge=knowledge)
    else:
        llm = build_llm(args.backend, model=args.model, base_url=args.base_url)

    pipeline = OntoBridgPipeline(
        llm,
        prompt_style=PromptStyle(args.prompt_style),
        symbolic_repair=not args.no_symbolic,
        seed=args.seed,
    )
    result = pipeline.run(dataset, run_task0=args.task0)

    payload = {
        "dataset": dataset.name,
        "n_terms_typed": len(result.term_types),
        "n_taxonomy_edges": len(result.final_taxonomy()),
        "n_relations": len(result.final_relations()),
        "symbolic_notes": result.symbolic.notes if result.symbolic else [],
        "term_types": result.term_types,
        "taxonomy": sorted(list(result.final_taxonomy())),
        "relations": [list(x) for x in sorted(result.final_relations())],
    }
    _print_json(payload)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"saved: {out}", file=sys.stderr)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args)
    if args.pred:
        pred = load_json_ontology(args.pred)
        pred_types = pred.term_map()
        pred_tax = pred.taxonomy_pairs()
        pred_rel = pred.relation_triples()
    else:
        knowledge = default_knowledge_from_dataset(dataset, seed=args.seed)
        llm = build_llm("mock", knowledge=knowledge)
        pipeline = OntoBridgPipeline(llm, seed=args.seed, symbolic_repair=not args.no_symbolic)
        result = pipeline.run(dataset)
        pred_types = result.final_term_types()
        pred_tax = result.final_taxonomy()
        pred_rel = result.final_relations()

    report = evaluate_pipeline_result(
        dataset.term_map(),
        pred_types,
        dataset.taxonomy_pairs(),
        pred_tax,
        gold_relations=dataset.relation_triples(),
        pred_relations=pred_rel,
    )
    _print_json(report)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    dataset = make_synthetic_dataset()
    knowledge = default_knowledge_from_dataset(dataset, seed=args.seed)
    llm = build_llm("mock", knowledge=knowledge)
    pipeline = OntoBridgPipeline(llm, seed=args.seed)
    result = pipeline.run(dataset)
    report = evaluate_pipeline_result(
        dataset.term_map(),
        result.final_term_types(),
        dataset.taxonomy_pairs(),
        result.final_taxonomy(),
        gold_relations=dataset.relation_triples(),
        pred_relations=result.final_relations(),
    )
    seen, heldout = holdout_terms(dataset, seed=args.seed)

    def predict(term: str) -> str:
        return knowledge.term_to_type.get(term, llm._type_term(term, dataset.types))

    novel = build_novel_synthetic_ontology(seed=args.seed + 3)

    def novel_predict(term: str) -> str:
        return knowledge.term_to_type.get(term, llm._type_term(term, novel.types))

    contam = evaluate_contamination_slices(
        predict,
        dataset,
        seed=args.seed,
        novel_dataset=novel,
        novel_predict_fn=novel_predict,
    )
    summary = {
        "backend": "mock",
        "task_a_micro_f1": report["task_a"]["micro"]["f1"],
        "task_a_macro_f1": report["task_a"]["macro"]["f1"],
        "task_b_f1": report["task_b"]["f1"],
        "task_c_f1": (report["task_c"] or {}).get("f1"),
        "graph_composite": report["graph_similarity"]["composite_similarity"],
        "contamination": contam.to_dict(),
        "n_seen_terms": len(seen),
        "n_heldout_terms": len(heldout),
        "symbolic_notes": result.symbolic.notes if result.symbolic else [],
    }
    _print_json(summary)
    return 0


def cmd_contamination(args: argparse.Namespace) -> int:
    dataset = _load_dataset(args)
    knowledge = default_knowledge_from_dataset(dataset, seed=args.seed)
    llm = build_llm("mock", knowledge=knowledge)

    def predict(term: str) -> str:
        return knowledge.term_to_type.get(term, llm._type_term(term, dataset.types))

    novel = build_novel_synthetic_ontology(seed=args.seed + 3)

    def novel_predict(term: str) -> str:
        return knowledge.term_to_type.get(term, llm._type_term(term, novel.types))

    report = evaluate_contamination_slices(
        predict,
        dataset,
        holdout_frac=args.holdout,
        seed=args.seed,
        novel_dataset=novel,
        novel_predict_fn=novel_predict,
    )
    _print_json(report.to_dict())
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    dataset = make_synthetic_dataset(args.dataset_name)
    save_json_ontology(dataset, args.output)
    print(f"exported {dataset.name} -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ontobridg",
        description="OntoBridg: hybrid neuro-symbolic ontology learning (LLMs4OL)",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", type=str, default=None, help="JSON ontology file or competition folder")
        p.add_argument("--dataset-name", type=str, default="synthetic_llms4ol")
        p.add_argument("--backend", type=str, default="mock", choices=["mock", "openai", "openai_compat", "api", "local", "local_http"])
        p.add_argument("--model", type=str, default="gpt-4o-mini")
        p.add_argument("--base-url", type=str, default=None)
        p.add_argument("--prompt-style", type=str, default="instruction", choices=[s.value for s in PromptStyle])
        p.add_argument("--seed", type=int, default=13)
        p.add_argument("--no-symbolic", action="store_true")
        p.add_argument("--output", type=str, default=None)
        p.add_argument("--synthetic-knowledge", action="store_true", default=True)
        p.add_argument("--task0", action="store_true")

    p_run = sub.add_parser("run", help="Run pipeline and print predictions")
    add_common(p_run)
    p_run.set_defaults(func=cmd_run)

    p_eval = sub.add_parser("eval", help="Run/evaluate predictions")
    add_common(p_eval)
    p_eval.add_argument("--pred", type=str, default=None, help="Predictions JSON from `run --output`")
    p_eval.set_defaults(func=cmd_eval)

    p_demo = sub.add_parser("demo", help="Offline mock demo on synthetic data")
    p_demo.add_argument("--seed", type=int, default=13)
    p_demo.set_defaults(func=cmd_demo)

    p_cont = sub.add_parser("contamination", help="Contamination-aware evaluation")
    add_common(p_cont)
    p_cont.add_argument("--holdout", type=float, default=0.3)
    p_cont.set_defaults(func=cmd_contamination)

    p_exp = sub.add_parser("export-synthetic", help="Export synthetic dataset JSON")
    p_exp.add_argument("--dataset-name", type=str, default="synthetic_llms4ol")
    p_exp.add_argument("--output", type=str, required=True)
    p_exp.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
