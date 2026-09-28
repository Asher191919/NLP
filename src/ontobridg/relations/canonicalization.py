"""Leakage-controlled predicate and entity canonicalization for relation triples.

The predicate alias table is derived from
`ontobridg.relations.predicate_inventory`, which is the single versioned source
of truth for canonical predicate labels, families, and routes. Adding, removing,
or redirecting an alias is a canonicalization behaviour change, so it requires
bumping both `CANONICALIZATION_VERSION` and the inventory version.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from ontobridg.relations.predicate_inventory import (
    alias_table,
    normalize_key,
    surface as predicate_surface,
)


CANONICALIZATION_VERSION = "relation_canonicalization_v2"


@dataclass(frozen=True)
class PredicateAlias:
    label: str
    reverse: bool = False


def _surface(value: Any) -> str:
    return predicate_surface(value)


def _key(value: Any) -> str:
    return normalize_key(value)


_PREDICATE_ALIASES = {
    key: PredicateAlias(label=label, reverse=reverse)
    for key, (label, reverse) in alias_table().items()
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
