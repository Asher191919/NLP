"""Pipeline orchestrator — dual API for RAG open-vocab and prompt/LLM paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ontobridg.data.schema import OntologyDataset, OntologyKnowledge
from ontobridg.eval.semantic_graph import CompetitionMetrics, evaluate_prediction
from ontobridg.symbolic.constraints import SymbolicRepairResult, repair_all


@dataclass
class PipelineResult:
    prediction: dict[str, Any] = field(default_factory=dict)
    metrics: CompetitionMetrics | None = None
    system: str = "ontobridg"
    dataset: str = ""
    term_types: dict[str, str] = field(default_factory=dict)
    taxonomy_edges: set[tuple[str, str]] = field(default_factory=set)
    relations: set[tuple[str, str, str]] = field(default_factory=set)
    extracted_terms: list[dict[str, Any]] = field(default_factory=list)
    raw_task_a: list[dict[str, Any]] = field(default_factory=list)
    raw_task_b: list[dict[str, Any]] = field(default_factory=list)
    raw_task_c: list[dict[str, Any]] = field(default_factory=list)
    symbolic: SymbolicRepairResult | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def final_taxonomy(self) -> set[tuple[str, str]]:
        if self.symbolic is not None:
            return set(self.symbolic.taxonomy)
        if self.taxonomy_edges:
            return set(self.taxonomy_edges)
        return {tuple(x) for x in self.prediction.get("taxonomy", [])}  # type: ignore[misc]

    def final_relations(self) -> set[tuple[str, str, str]]:
        if self.symbolic is not None:
            return set(self.symbolic.relations)
        if self.relations:
            return set(self.relations)
        return {tuple(x) for x in self.prediction.get("relations", [])}  # type: ignore[misc]

    def final_term_types(self) -> dict[str, str]:
        if self.term_types:
            return dict(self.term_types)
        terms = self.prediction.get("terms", {})
        return dict(terms) if isinstance(terms, dict) else {}

    def to_dict(self) -> dict:
        return {
            "system": self.system,
            "dataset": self.dataset,
            "prediction": {
                "n_terms": len(self.prediction.get("terms", {}) or self.final_term_types()),
                "n_taxonomy": len(self.final_taxonomy()),
                "n_relations": len(self.final_relations()),
                "n_nontax": len(self.prediction.get("nontax_relations", [])),
            },
            "metrics": self.metrics.to_dict() if self.metrics else {},
        }


def _is_knowledge(obj: Any) -> bool:
    return isinstance(obj, OntologyKnowledge) or (
        obj is not None and hasattr(obj, "term_to_type") and hasattr(obj, "parents")
    )


class OntoBridgPipeline:
    """High-level orchestrator.

    Modes
    -----
    1. RAG open-vocab (knowledge first arg):
         OntoBridgPipeline(knowledge, use_symbolic_repair=True).run_dataset(ds)
    2. Prompt/LLM (LLM backend first arg):
         OntoBridgPipeline(llm, symbolic_repair=True).run(ds)
    """

    def __init__(
        self,
        llm_or_knowledge: Any = None,
        *,
        knowledge: OntologyKnowledge | None = None,
        closed_vocabulary: bool = False,
        use_symbolic_repair: bool = True,
        symbolic_repair: bool | None = None,
        prompt_style: Any = None,
        negative_pairs_per_type: int = 2,
        max_taxonomy_pairs: int | None = None,
        seed: int = 0,
    ) -> None:
        if symbolic_repair is not None:
            use_symbolic_repair = bool(symbolic_repair)

        self.closed_vocabulary = closed_vocabulary
        self.use_symbolic_repair = use_symbolic_repair
        self.symbolic_repair = use_symbolic_repair
        self.prompt_style = prompt_style
        self.negative_pairs_per_type = negative_pairs_per_type
        self.max_taxonomy_pairs = max_taxonomy_pairs
        self.seed = seed
        self.knowledge: OntologyKnowledge | None = None
        self.llm: Any = None
        self.rag: Any = None
        self.mode = "prompt"

        if _is_knowledge(llm_or_knowledge) or knowledge is not None:
            self.knowledge = knowledge or llm_or_knowledge
            self.mode = "rag"
            try:
                from ontobridg.baselines.rag_pipeline import RAGPipeline

                self.rag = RAGPipeline.mock(
                    self.knowledge,
                    closed_vocabulary=closed_vocabulary,
                    use_symbolic_repair=use_symbolic_repair,
                    seed=seed,
                )
            except Exception:
                self.rag = None
                self.mode = "knowledge_prompt"
                from ontobridg.llm.mock import MockLLM

                self.llm = MockLLM(knowledge=self.knowledge, seed=seed)
        else:
            self.llm = llm_or_knowledge
            if self.llm is None:
                from ontobridg.llm.mock import MockLLM

                self.llm = MockLLM(knowledge=OntologyKnowledge(), seed=seed)
            if _is_knowledge(self.llm):
                self.knowledge = self.llm
                from ontobridg.llm.mock import MockLLM

                self.llm = MockLLM(knowledge=self.knowledge, seed=seed)

        if prompt_style is not None:
            try:
                from ontobridg.prompts.templates import PromptStyle

                self.prompt_style = (
                    prompt_style
                    if isinstance(prompt_style, PromptStyle)
                    else PromptStyle(prompt_style)
                )
            except Exception:
                self.prompt_style = prompt_style
        else:
            try:
                from ontobridg.prompts.templates import PromptStyle

                self.prompt_style = PromptStyle.INSTRUCTION
            except Exception:
                self.prompt_style = None

    # ------------------------------------------------------------------ RAG path
    def run_dataset(self, dataset: OntologyDataset) -> PipelineResult:
        if self.rag is None:
            return self.run(dataset)
        corpus = list(dataset.metadata.get("corpus", []))
        domain_ranges = {} if self.closed_vocabulary else dict(dataset.domain_ranges)
        pred = self.rag.run(
            corpus=corpus,
            type_inventory=list(dataset.types),
            domain_ranges=domain_ranges,
        )
        metrics = evaluate_prediction(
            dataset.term_map(),
            list(dataset.taxonomy_pairs()),
            list(dataset.relation_triples()),
            pred.get("terms", {}),
            [tuple(x) for x in pred.get("taxonomy", [])],
            [tuple(x) for x in pred.get("relations", [])],
        )
        return PipelineResult(
            prediction=pred,
            metrics=metrics,
            system="closed_vocab" if self.closed_vocabulary else "ontobridg",
            dataset=dataset.name,
            term_types=dict(pred.get("terms", {})),
            taxonomy_edges={tuple(x) for x in pred.get("taxonomy", [])},  # type: ignore[misc]
            relations={tuple(x) for x in pred.get("relations", [])},  # type: ignore[misc]
            meta={"mode": "rag", "repair": pred.get("repair", {})},
        )

    # -------------------------------------------------------------- prompt path
    def run(
        self,
        dataset: OntologyDataset,
        *,
        texts: Sequence[str] | None = None,
        run_task0: bool = False,
    ) -> PipelineResult:
        if self.mode == "rag" and self.rag is not None:
            return self.run_dataset(dataset)

        llm = self.llm
        if llm is None:
            raise RuntimeError("OntoBridgPipeline.run requires an LLM backend")

        result = PipelineResult(dataset_name_ := dataset.name)
        result.dataset = dataset.name
        result.system = "ontobridg_prompt"
        types = list(dataset.types)

        if run_task0:
            result.extracted_terms = self.run_task0(dataset, texts=texts)

        result.term_types = self.run_taska(dataset, types=types)
        candidate_pairs = self._candidate_taxonomy_pairs(dataset)
        pred_edges = self.run_taskb(candidate_pairs)
        result.taxonomy_edges = pred_edges
        result.relations = self.run_taskc(
            dataset, term_types=result.term_types, taxonomy=pred_edges
        )

        if self.symbolic_repair:
            result.symbolic = repair_all(
                result.taxonomy_edges,
                result.relations,
                result.term_types,
                domain_ranges=dataset.domain_ranges,
            )
            result.taxonomy_edges = result.symbolic.taxonomy
            result.relations = result.symbolic.relations
            result.meta["symbolic_notes"] = result.symbolic.notes
        result.meta["mode"] = "prompt"
        result.meta["n_types"] = len(types)
        result.meta["n_taxonomy_candidates"] = len(candidate_pairs)
        result.prediction = {
            "terms": result.final_term_types(),
            "taxonomy": [list(x) for x in sorted(result.final_taxonomy())],
            "relations": [list(x) for x in sorted(result.final_relations())],
            "nontax_relations": [
                list(x)
                for x in sorted(result.final_relations())
                if x[1] not in {"is-a", "isa", "type-of", "subclass-of"}
            ],
        }
        return result

    def run_task0(
        self,
        dataset: OntologyDataset,
        *,
        texts: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        from ontobridg.prompts.templates import task0_prompt

        candidates = [t.term for t in dataset.terms]
        outputs: list[dict[str, Any]] = []
        if not texts:
            texts = [f"{t.term} is a {t.type_label}." for t in dataset.terms[:30]]
        for text in texts:
            prompt = task0_prompt(text, candidates=candidates)
            response = self.llm.complete(
                prompt,
                task="term_extraction",
                meta={"text": text, "candidates": candidates},
            )
            payload = response.raw if isinstance(response.raw, dict) else {}
            for item in payload.get("terms", []):
                if isinstance(item, dict) and item.get("term"):
                    outputs.append(
                        {"term": item["term"], "type": item.get("type", "Entity")}
                    )
        return outputs

    def run_taska(
        self,
        dataset: OntologyDataset,
        *,
        types: Sequence[str] | None = None,
        terms: Sequence[str] | None = None,
    ) -> dict[str, str]:
        from ontobridg.prompts.templates import taska_prompt

        type_list = list(types) if types is not None else list(dataset.types)
        term_list = list(terms) if terms is not None else [t.term for t in dataset.terms]
        predictions: dict[str, str] = {}
        raw_store = getattr(self, "_raw_a", None)
        if raw_store is None:
            self._raw_a = []
            raw_store = self._raw_a
        for term in term_list:
            prompt = taska_prompt(term, types=type_list, style=self.prompt_style)
            response = self.llm.complete(
                prompt,
                task="term_typing",
                meta={"term": term, "types": type_list},
            )
            payload = response.raw if isinstance(response.raw, dict) else {}
            predicted = payload.get("type") or ""
            predictions[term] = predicted or "Entity"
            raw_store.append({"term": term, "pred": predictions[term], "raw": payload})
        return predictions

    def _candidate_taxonomy_pairs(self, dataset: OntologyDataset) -> list[tuple[str, str]]:
        import random

        gold = dataset.taxonomy_pairs()
        types = list(dataset.types)
        positives = [(c, p) for c, p in gold if c in types and p in types]
        negatives: list[tuple[str, str]] = []
        rng = random.Random(self.seed)
        by_type_count: dict[str, int] = {t: 0 for t in types}
        attempts = 0
        max_attempts = max(50, len(types) * self.negative_pairs_per_type * 20)
        target_neg = max(2, len(types) * self.negative_pairs_per_type)
        while len(negatives) < target_neg and attempts < max_attempts:
            attempts += 1
            child = rng.choice(types)
            parent = rng.choice(types)
            if child == parent:
                continue
            if (child, parent) in gold or (child, parent) in negatives:
                continue
            if by_type_count.get(child, 0) >= self.negative_pairs_per_type:
                continue
            negatives.append((child, parent))
            by_type_count[child] = by_type_count.get(child, 0) + 1
        pairs = positives + negatives
        rng.shuffle(pairs)
        if self.max_taxonomy_pairs is not None:
            pairs = pairs[: self.max_taxonomy_pairs]
        return pairs

    def run_taskb(self, pairs: Sequence[tuple[str, str]]) -> set[tuple[str, str]]:
        from ontobridg.prompts.templates import taskb_prompt

        predicted: set[tuple[str, str]] = set()
        raw_store = getattr(self, "_raw_b", None)
        if raw_store is None:
            self._raw_b = []
            raw_store = self._raw_b
        for child, parent in pairs:
            prompt = taskb_prompt(child, parent, style=self.prompt_style)
            response = self.llm.complete(
                prompt,
                task="taxonomy",
                meta={"child": child, "parent": parent},
            )
            payload = response.raw if isinstance(response.raw, dict) else {}
            is_a = payload.get("is_a")
            if is_a is None:
                text_l = response.text.lower()
                is_a = "true" in text_l and "false" not in text_l
            raw_store.append(
                {"child": child, "parent": parent, "is_a": bool(is_a), "raw": payload}
            )
            if is_a:
                predicted.add((child, parent))
        return predicted

    def run_taskc(
        self,
        dataset: OntologyDataset,
        *,
        term_types: dict[str, str] | None = None,
        taxonomy: set[tuple[str, str]] | None = None,
        mode: str = "classify",
    ) -> set[tuple[str, str, str]]:
        import random

        from ontobridg.prompts.templates import taskc_prompt

        _ = term_types, taxonomy
        rel_names = sorted(dataset.domain_ranges.keys()) or sorted(
            {r.relation for r in dataset.relations}
        )
        known_heads = sorted({r.head for r in dataset.relations})
        known_tails = sorted({r.tail for r in dataset.relations})
        terms = [t.term for t in dataset.terms]
        all_entities = sorted(set(terms + known_heads + known_tails))
        gold_triples = dataset.relation_triples()

        rng = random.Random(self.seed + 1)
        negatives: list[tuple[str, str, str]] = []
        for _ in range(min(20, max(5, len(gold_triples)))):
            if not all_entities or not rel_names:
                break
            h = rng.choice(all_entities)
            t = rng.choice(all_entities)
            r = rng.choice(rel_names)
            if h != t and (h, r, t) not in gold_triples:
                negatives.append((h, r, t))
        classify_set = list(dict.fromkeys(list(gold_triples) + negatives))

        predicted: set[tuple[str, str, str]] = set()
        raw_store = getattr(self, "_raw_c", None)
        if raw_store is None:
            self._raw_c = []
            raw_store = self._raw_c
        if mode == "extract":
            prompt = taskc_prompt(
                "", "", relations=rel_names, extract=True, text=str(dataset.metadata)
            )
            response = self.llm.complete(
                prompt,
                task="relation",
                meta={"mode": "extract", "relations": rel_names},
            )
            payload = response.raw if isinstance(response.raw, dict) else {}
            for item in payload.get("relations", []):
                if (
                    isinstance(item, dict)
                    and item.get("head")
                    and item.get("relation")
                    and item.get("tail")
                ):
                    predicted.add((item["head"], item["relation"], item["tail"]))
            return predicted

        for head, rel, tail in classify_set:
            prompt = taskc_prompt(head, tail, relations=rel_names, style=self.prompt_style)
            response = self.llm.complete(
                prompt,
                task="relation",
                meta={
                    "head": head,
                    "tail": tail,
                    "relation": rel,
                    "relations": rel_names,
                    "mode": "classify",
                },
            )
            payload = response.raw if isinstance(response.raw, dict) else {}
            pred_rel = payload.get("relation") or ""
            valid = bool(payload.get("valid", pred_rel != ""))
            raw_store.append(
                {
                    "head": head,
                    "gold_rel": rel,
                    "pred_rel": pred_rel,
                    "tail": tail,
                    "valid": valid,
                    "raw": payload,
                }
            )
            if valid and pred_rel:
                predicted.add((head, pred_rel, tail))
        return predicted
