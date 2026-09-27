from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class SymbolicRepairResult:
    taxonomy: set[tuple[str, str]]
    relations: set[tuple[str, str, str]]
    term_types: dict[str, str]
    removed_edges: list[tuple[str, str]] = field(default_factory=list)
    removed_relations: list[tuple[str, str, str]] = field(default_factory=list)
    added_edges: list[tuple[str, str]] = field(default_factory=list)
    cycles_found: list[list[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def transitive_closure(edges: Iterable[tuple[str, str]]) -> set[tuple[str, str]]:
    closure: set[tuple[str, str]] = set(edges)
    changed = True
    while changed:
        changed = False
        by_child: dict[str, set[str]] = {}
        for c, p in closure:
            by_child.setdefault(c, set()).add(p)
        additions: set[tuple[str, str]] = set()
        for c, parents in by_child.items():
            for p in parents:
                for gp in by_child.get(p, set()):
                    if (c, gp) not in closure:
                        additions.add((c, gp))
        if additions:
            closure |= additions
            changed = True
    return closure


def detect_cycles(edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    graph: dict[str, set[str]] = {}
    for child, parent in edges:
        graph.setdefault(child, set()).add(parent)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {n: WHITE for n in graph}
    cycles: list[list[str]] = []
    stack: list[str] = []

    def dfs(node: str) -> None:
        color[node] = GRAY
        stack.append(node)
        for nxt in graph.get(node, set()):
            if color.get(nxt, WHITE) == GRAY:
                if nxt in stack:
                    idx = stack.index(nxt)
                    cycles.append(stack[idx:] + [nxt])
            elif color.get(nxt, WHITE) == WHITE:
                if nxt not in color:
                    color[nxt] = WHITE
                dfs(nxt)
        stack.pop()
        color[node] = BLACK

    for node in list(graph.keys()):
        if color.get(node, WHITE) == WHITE:
            dfs(node)
    return cycles


def _remove_edge_from_cycles(
    edges: set[tuple[str, str]],
    cycles: list[list[str]],
) -> tuple[set[tuple[str, str]], list[tuple[str, str]]]:
    working = set(edges)
    removed: list[tuple[str, str]] = []
    for cycle in cycles:
        if len(cycle) < 2:
            continue
        pairs = list(zip(cycle[:-1], cycle[1:]))
        # break weakest-looking edge: prefer removing non-topmost domain edge
        victim = None
        for c, p in pairs:
            if (c, p) in working:
                victim = (c, p)
                break
        if victim is None and pairs:
            victim = pairs[0]
        if victim and victim in working:
            working.discard(victim)
            removed.append(victim)
    return working, removed


def repair_taxonomy(
    edges: Iterable[tuple[str, str]],
    *,
    allow_transitive_fill: bool = True,
    max_fill: int = 200,
) -> tuple[set[tuple[str, str]], SymbolicRepairResult]:
    working = set(edges)
    working = {(c, p) for c, p in working if c and p and c != p}
    cycles = detect_cycles(working)
    working, removed = _remove_edge_from_cycles(working, cycles)
    added: list[tuple[str, str]] = []
    if allow_transitive_fill:
        closure = transitive_closure(working)
        missing = closure - working
        # do not materialize full closure; only fill short missing edges
        by_child: dict[str, set[str]] = {}
        for c, p in working:
            by_child.setdefault(c, set()).add(p)
        for c, p in sorted(missing):
            if len(added) >= max_fill:
                break
            # fill if there is a 2-hop path only (grandparent)
            two_hop = any(gp in by_child.get(parent, set()) for parent in by_child.get(c, set()) for gp in [p])
            if two_hop:
                added.append((c, p))
        working |= set(added)
    result = SymbolicRepairResult(
        taxonomy=working,
        relations=set(),
        term_types={},
        removed_edges=removed,
        added_edges=added,
        cycles_found=cycles,
        notes=[f"cycles_detected={len(cycles)}", f"edges_removed={len(removed)}", f"edges_added={len(added)}"],
    )
    return working, result


def _type_ancestors(type_name: str, edges: set[tuple[str, str]]) -> set[str]:
    parents: dict[str, set[str]] = {}
    for c, p in edges:
        parents.setdefault(c, set()).add(p)
    seen: set[str] = set()
    stack = list(parents.get(type_name, set()))
    stack.append(type_name)
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(parents.get(cur, set()))
    return seen


def apply_type_range_filter(
    relations: Iterable[tuple[str, str, str]],
    term_types: dict[str, str],
    domain_ranges: dict[str, tuple[str, str]],
    taxonomy_edges: set[tuple[str, str]] | None = None,
) -> tuple[set[tuple[str, str, str]], list[tuple[str, str, str]]]:
    taxonomy_edges = taxonomy_edges or set()
    kept: set[tuple[str, str, str]] = set()
    dropped: list[tuple[str, str, str]] = []
    for head, rel, tail in relations:
        if rel not in domain_ranges:
            kept.add((head, rel, tail))
            continue
        domain, range_ = domain_ranges[rel]
        h_type = term_types.get(head, "")
        t_type = term_types.get(tail, "")
        if not h_type or not t_type:
            kept.add((head, rel, tail))
            continue
        h_ok = domain == "*" or h_type == domain or domain in _type_ancestors(h_type, taxonomy_edges)
        t_ok = range_ == "*" or t_type == range_ or range_ in _type_ancestors(t_type, taxonomy_edges)
        if h_ok and t_ok:
            kept.add((head, rel, tail))
        else:
            dropped.append((head, rel, tail))
    return kept, dropped


_STOP_RELATIONS = {"relatedTo", "associatedWith", "linkedTo"}
_BAD_TAIL_MARKERS = {"transportation", "concept", "entity", "thing"}


def lexical_relation_filter(
    relations: Iterable[tuple[str, str, str]],
    *,
    min_head_len: int = 2,
    min_tail_len: int = 2,
) -> tuple[set[tuple[str, str, str]], list[tuple[str, str, str]]]:
    kept: set[tuple[str, str, str]] = set()
    dropped: list[tuple[str, str, str]] = []
    for h, r, t in relations:
        reasons_ok = True
        if not h or not t or not r:
            reasons_ok = False
        if len(h.strip()) < min_head_len or len(t.strip()) < min_tail_len:
            reasons_ok = False
        if h.strip().lower() == t.strip().lower():
            reasons_ok = False
        if r.strip().lower() in {x.lower() for x in _STOP_RELATIONS} and t.strip().lower() in _BAD_TAIL_MARKERS:
            reasons_ok = False
        if reasons_ok:
            kept.add((h, r, t))
        else:
            dropped.append((h, r, t))
    return kept, dropped


def repair_all(
    taxonomy_edges: Iterable[tuple[str, str]],
    relations: Iterable[tuple[str, str, str]],
    term_types: dict[str, str],
    domain_ranges: dict[str, tuple[str, str]] | None = None,
) -> SymbolicRepairResult:
    tax, tax_result = repair_taxonomy(taxonomy_edges)
    rel_set = set(relations)
    rel_set, lex_dropped = lexical_relation_filter(rel_set)
    rel_set, range_dropped = apply_type_range_filter(
        rel_set,
        term_types,
        domain_ranges or {},
        taxonomy_edges=tax,
    )
    notes = list(tax_result.notes)
    notes.append(f"relations_lexical_drop={len(lex_dropped)}")
    notes.append(f"relations_range_drop={len(range_dropped)}")
    return SymbolicRepairResult(
        taxonomy=tax,
        relations=rel_set,
        term_types=dict(term_types),
        removed_edges=tax_result.removed_edges,
        removed_relations=lex_dropped + range_dropped,
        added_edges=tax_result.added_edges,
        cycles_found=tax_result.cycles_found,
        notes=notes,
    )
