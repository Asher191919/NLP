"""Leakage-controlled predicate and entity canonicalization for relation triples."""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable


CANONICALIZATION_VERSION = "relation_canonicalization_v2"


@dataclass(frozen=True)
class PredicateAlias:
    label: str
    reverse: bool = False


def _surface(value: Any) -> str:
    return " ".join(str(value).lower().strip().split())


def _key(value: Any) -> str:
    text = _surface(value).replace("_", " ").replace("-", " ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


_PREDICATE_ALIASES = {
    "is a": PredicateAlias("is-a"),
    "isa": PredicateAlias("is-a"),
    "subclass of": PredicateAlias("is-a"),
    "subclassof": PredicateAlias("is-a"),
    "kind of": PredicateAlias("is-a"),
    "instance of": PredicateAlias("instance-of"),
    "instanceof": PredicateAlias("instance-of"),
    "part of": PredicateAlias("part_of"),
    "is part of": PredicateAlias("part_of"),
    "has part": PredicateAlias("has part"),
    "contains as part": PredicateAlias("has part"),
    "develops from": PredicateAlias("develops_from"),
    "developed from": PredicateAlias("develops_from"),
    "arises from": PredicateAlias("develops_from"),
    "derives from": PredicateAlias("derives from"),
    "derived from": PredicateAlias("derives from"),
    "gives rise to": PredicateAlias("develops_from", reverse=True),
    "is located in": PredicateAlias("located in"),
    "located in": PredicateAlias("located in"),
    "is defined by": PredicateAlias("is defined by"),
    "equivalent to": PredicateAlias("equivalent class"),
    "equivalent class": PredicateAlias("equivalent class"),
    "equivalentclass": PredicateAlias("equivalent class"),
    "disjoint from": PredicateAlias("disjoint with"),
    "disjoint with": PredicateAlias("disjoint with"),
    "disjointwith": PredicateAlias("disjoint with"),
    "same as": PredicateAlias("same as"),
    "exact match": PredicateAlias("exact match"),
    "tree view": PredicateAlias("tree view"),
    "database cross reference": PredicateAlias("database_cross_reference"),
    "databasecrossreference": PredicateAlias("database_cross_reference"),
    "see also": PredicateAlias("see also"),
    "seealso": PredicateAlias("see also"),
    "has role": PredicateAlias("has role"),
    "positively regulates": PredicateAlias("positively regulates"),
    "regulates": PredicateAlias("regulates"),
    "domain": PredicateAlias("domain"),
    "range": PredicateAlias("range"),
    "type": PredicateAlias("type"),
}


def canonicalize_predicate(predicate: Any) -> tuple[str, bool, str | None]:
    original = _surface(predicate)
    alias = _PREDICATE_ALIASES.get(_key(predicate))
    if alias is None:
        return original, False, None
    changed_from = original if alias.label != original or alias.reverse else None
    return alias.label, alias.reverse, changed_from


def build_entity_inventory(sample: dict[str, Any], *, task: str) -> list[str]:
    """Build an entity inventory using only fields that are legal model inputs."""
    values: list[Any] = []
    if task == "b":
        values.extend(sample.get("terms") or [])
        values.extend(sample.get("types") or [])
        for triple in sample.get("initial-primitive-ontology-triples") or []:
            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                values.extend((triple[0], triple[2]))
    inventory: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = _surface(value)
        if normalized and normalized not in seen:
            inventory.append(normalized)
            seen.add(normalized)
    return inventory


def _context_form(entity: str, context: str) -> str | None:
    context_norm = _surface(context)
    variants = [entity, _key(entity)]
    for variant in variants:
        if not variant:
            continue
        match = re.search(rf"(?<![a-z0-9]){re.escape(variant)}(?![a-z0-9])", context_norm)
        if match:
            return context_norm[match.start() : match.end()]
    return None


def canonicalize_entity(
    entity: Any,
    *,
    context: str,
    inventory: Iterable[str] = (),
    fuzzy_threshold: float = 0.94,
) -> tuple[str, str | None, str]:
    original = _surface(entity)
    if not original:
        return original, None, "empty"

    by_key: dict[str, str] = {}
    for value in inventory:
        normalized = _surface(value)
        if normalized:
            by_key.setdefault(_key(normalized), normalized)
    entity_key = _key(original)
    if entity_key in by_key:
        canonical = by_key[entity_key]
        return canonical, original if canonical != original else None, "inventory_exact"

    context_match = _context_form(original, context)
    if context_match is not None:
        return context_match, original if context_match != original else None, "context_exact"

    best: tuple[float, str] | None = None
    entity_tokens = set(entity_key.split())
    for candidate_key, candidate in by_key.items():
        candidate_tokens = set(candidate_key.split())
        if not entity_tokens or not candidate_tokens or not (entity_tokens & candidate_tokens):
            continue
        score = SequenceMatcher(None, entity_key, candidate_key).ratio()
        if score >= fuzzy_threshold and (best is None or score > best[0]):
            best = (score, candidate)
    if best is not None:
        canonical = best[1]
        return canonical, original if canonical != original else None, "inventory_fuzzy"
    return original, None, "unchanged"


def canonicalize_triple(
    triple: Iterable[Any],
    *,
    context: str,
    entity_inventory: Iterable[str] = (),
) -> tuple[list[str], dict[str, Any]]:
    subject_raw, predicate_raw, object_raw = list(triple)[:3]
    subject, subject_from, subject_method = canonicalize_entity(
        subject_raw, context=context, inventory=entity_inventory
    )
    obj, object_from, object_method = canonicalize_entity(
        object_raw, context=context, inventory=entity_inventory
    )
    predicate, reverse, predicate_from = canonicalize_predicate(predicate_raw)
    if reverse:
        subject, obj = obj, subject
    canonical = [subject, predicate, obj]
    changes = {
        "version": CANONICALIZATION_VERSION,
        "changed": any(value is not None for value in (subject_from, predicate_from, object_from)),
        "reversed": reverse,
        "subject_from": subject_from,
        "predicate_from": predicate_from,
        "object_from": object_from,
        "subject_method": subject_method,
        "object_method": object_method,
    }
    return canonical, changes
