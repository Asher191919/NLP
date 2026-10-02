"""Official LLMs4OL 2026 Graph Similarity — pure-Python port of challenge metrics.

Source of truth:
  D:\\NLP\\data\\LLMs4OL-Challenge\\2026\\metrics\\graph_similarity.py

Official formula (exact_match mode):
  graph_similarity = (edge_f1 + neighborhood_similarity + taxonomy_similarity) / 3

Notes:
- Official edge_f1 returns 1.0 when both gold and pred are empty.
- fuzzy/semantic modes require rapidfuzz / sentence-transformers and are optional.
- External reports MUST use this module, not ontobridg.eval.semantic_graph.proxy SGS.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable, Sequence

Triple = tuple[str, str, str]
TaxonomyRelations = frozenset({"is-a"})

OFFICIAL_SOURCE = "data/LLMs4OL-Challenge/2026/metrics/graph_similarity.py"
OFFICIAL_FORMULA = "(edge_f1 + neighborhood_similarity + taxonomy_similarity) / 3"
METRIC_ID = "llms4ol2026_graph_similarity_exact"
EXTERNAL_REPORTABLE = True
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def normalize(text: str) -> str:
    return " ".join(str(text).lower().strip().split())


def normalize_triples(triples: Iterable[Sequence[str]]) -> list[Triple]:
    out: list[Triple] = []
    for t in triples:
        if len(t) < 3:
            continue
        out.append((normalize(t[0]), normalize(t[1]), normalize(t[2])))
    return out


def _to_triples(
    term_types: dict[str, str] | None,
    taxonomy: Iterable[tuple[str, str]] | Iterable[Sequence[str]] | None,
    relations: Iterable[tuple[str, str, str]] | Iterable[Sequence[str]] | None,
) -> list[Triple]:
    """Convert OntoBridg structures to official (s, p, o) triples."""
    triples: list[Triple] = []
    if term_types:
        for term, ty in term_types.items():
            triples.append((normalize(term), "instance-of", normalize(ty)))
    if taxonomy:
        for e in taxonomy:
            if len(e) < 2:
                continue
            child, parent = normalize(e[0]), normalize(e[1])
            # Official taxonomy graph uses predicate == "is-a"
            triples.append((child, "is-a", parent))
    if relations:
        for r in relations:
            if len(r) < 3:
                continue
            h, p, o = normalize(r[0]), normalize(r[1]), normalize(r[2])
            if p in {"isa", "type-of", "subclass-of", "subClassOf", "kind-of"}:
                p = "is-a"
            triples.append((h, p, o))
    return triples


def edge_f1(gold_edges: Iterable[Triple], pred_edges: Iterable[Triple]) -> float:
    gold_set = set(gold_edges)
    pred_set = set(pred_edges)
    intersection = gold_set & pred_set
    if len(gold_set) == 0 and len(pred_set) == 0:
        return 1.0
    if len(pred_set) == 0 or len(gold_set) == 0:
        return 0.0
    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gold_set)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def get_neighborhood(triples: Iterable[Triple]) -> dict[str, set]:
    neigh: dict[str, set] = defaultdict(set)
    for s, p, o in triples:
        neigh[s].add(("OUT", p, o))
        neigh[o].add(("IN", p, s))
    return neigh


def neighborhood_similarity(gold_triples: list[Triple], pred_triples: list[Triple]) -> float:
    gold_neigh = get_neighborhood(gold_triples)
    pred_neigh = get_neighborhood(pred_triples)
    all_nodes = set(gold_neigh.keys()) | set(pred_neigh.keys())
    scores: list[float] = []
    for n in all_nodes:
        g = gold_neigh.get(n, set())
        p = pred_neigh.get(n, set())
        union = g | p
        if len(union) == 0:
            continue
        scores.append(len(g & p) / len(union))
    return sum(scores) / len(scores) if scores else 0.0


def _taxonomy_graph(triples: list[Triple]) -> dict[str, set[str]]:
    """child -> parents for official TAXONOMY_RELATIONS only."""
    parent_of: dict[str, set[str]] = defaultdict(set)
    nodes: set[str] = set()
    for s, p, o in triples:
        nodes.add(s)
        nodes.add(o)
        if p in TaxonomyRelations:
            parent_of[s].add(o)
    return {n: parent_of.get(n, set()) for n in nodes}


def _ancestors(parent_of: dict[str, set[str]], node: str) -> set[str]:
    seen: set[str] = set()
    stack = list(parent_of.get(node, set()))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(parent_of.get(cur, set()))
    return seen


def _descendants(parent_of: dict[str, set[str]], node: str) -> set[str]:
    children: dict[str, set[str]] = defaultdict(set)
    for child, parents in parent_of.items():
        for p in parents:
            children[p].add(child)
    seen: set[str] = set()
    stack = list(children.get(node, set()))
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(children.get(cur, set()))
    return seen


def taxonomy_similarity(gold_triples: list[Triple], pred_triples: list[Triple]) -> float:
    g_parent = _taxonomy_graph(gold_triples)
    p_parent = _taxonomy_graph(pred_triples)
    all_nodes = set(g_parent) | set(p_parent)
    scores: list[float] = []
    for n in all_nodes:
        gold_anc = _ancestors(g_parent, n) if n in g_parent else set()
        pred_anc = _ancestors(p_parent, n) if n in p_parent else set()
        gold_desc = _descendants(g_parent, n) if n in g_parent else set()
        pred_desc = _descendants(p_parent, n) if n in p_parent else set()
        gold_rel = gold_anc | gold_desc
        pred_rel = pred_anc | pred_desc
        union = gold_rel | pred_rel
        if len(union) == 0:
            continue
        scores.append(len(gold_rel & pred_rel) / len(union))
    return sum(scores) / len(scores) if scores else 0.0


def exact_match(gold_triples: Sequence[Sequence[str]], pred_triples: Sequence[Sequence[str]]) -> dict[str, float]:
    gold = normalize_triples(gold_triples)
    pred = normalize_triples(pred_triples)
    f1_edges = edge_f1(gold, pred)
    neigh_sim = neighborhood_similarity(gold, pred)
    tax_sim = taxonomy_similarity(gold, pred)
    final_score = (f1_edges + neigh_sim + tax_sim) / 3
    return {
        "edge_f1": f1_edges,
        "neighborhood_similarity": neigh_sim,
        "taxonomy_similarity": tax_sim,
        "graph_similarity": final_score,
        "metric_id": METRIC_ID,
        "external_reportable": True,
        "official_formula": OFFICIAL_FORMULA,
    }


def evaluate_structures(
    *,
    gold_term_types: dict[str, str] | None = None,
    gold_taxonomy: Iterable | None = None,
    gold_relations: Iterable | None = None,
    pred_term_types: dict[str, str] | None = None,
    pred_taxonomy: Iterable | None = None,
    pred_relations: Iterable | None = None,
) -> dict[str, float]:
    """Official exact_match Graph Similarity on OntoBridg structure fields."""
    gold = _to_triples(gold_term_types, gold_taxonomy, gold_relations)
    pred = _to_triples(pred_term_types, pred_taxonomy, pred_relations)
    return exact_match(gold, pred)


def load_official_module():
    """Optionally import the challenge script (needs networkx/rapidfuzz/ST)."""
    import importlib.util

    path = PROJECT_ROOT / "data" / "LLMs4OL-Challenge" / "2026" / "metrics" / "graph_similarity.py"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location("llms4ol_official_graph_similarity", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


__all__ = [
    "METRIC_ID",
    "OFFICIAL_FORMULA",
    "OFFICIAL_SOURCE",
    "EXTERNAL_REPORTABLE",
    "edge_f1",
    "exact_match",
    "evaluate_structures",
    "load_official_module",
    "neighborhood_similarity",
    "normalize",
    "normalize_triples",
    "taxonomy_similarity",
]
