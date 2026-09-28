"""Versioned predicate inventory, route registry, and predicate projection.

Context (docs/STRATEGY.md section 4, item 2 "predicate system alignment and
dynamic routing"): the completed `official_relation_transfer_v1` and
`official_predicate_routing_v1` packets showed that the dominant official
failure is predicate-inventory mismatch and broad `type` over-generation rather
than missing textual evidence. The predicate families were therefore sedimented
here, so that prompt inventories, alias canonicalization, and offline adherence
diagnostics share one versioned source of truth instead of duplicating literal
predicate lists inside runners.

Two-stage mechanism supported by this module:

1. Candidate proposal stays open (the LLM emits whatever predicate it wants).
2. Predicate projection maps the proposal onto a canonical inventory label
   (`project_predicate`) and reports route adherence (`route_adherence`).

Scope limits: this module is schema-level infrastructure. It contains no gold
instances, performs no network or filesystem access, changes no official score,
and cannot reopen the failed mechanism gate documented in
`experiments/holdouts/official_predicate_routing_v1/README.md`.

Versioning rule: adding, removing, or redirecting any alias changes relation
canonicalization behaviour, so it requires bumping both
`PREDICATE_INVENTORY_VERSION` and
`ontobridg.relations.canonicalization.CANONICALIZATION_VERSION`.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable


PREDICATE_INVENTORY_VERSION = "predicate_inventory_v1"

FAMILY_TAXONOMIC = "taxonomic"
FAMILY_INSTANCE = "instance"
FAMILY_AXIOM = "axiom"
FAMILY_TYPING = "typing"
FAMILY_METADATA = "metadata"
FAMILY_IDENTITY = "identity"
FAMILY_CHEMISTRY = "chemistry"
FAMILY_MERONYMIC = "meronymic"
FAMILY_DERIVATION = "derivation"
FAMILY_LOCATION = "location"
FAMILY_REGULATION = "regulation"
FAMILY_SCHEMA = "schema"
FAMILY_ANNOTATION = "annotation"

FAMILY_ORDER = (
    FAMILY_TAXONOMIC,
    FAMILY_INSTANCE,
    FAMILY_AXIOM,
    FAMILY_TYPING,
    FAMILY_METADATA,
    FAMILY_IDENTITY,
    FAMILY_CHEMISTRY,
    FAMILY_MERONYMIC,
    FAMILY_DERIVATION,
    FAMILY_LOCATION,
    FAMILY_REGULATION,
    FAMILY_SCHEMA,
    FAMILY_ANNOTATION,
)


@dataclass(frozen=True)
class PredicateEntry:
    """One canonical predicate label with its surface aliases.

    `reversed_aliases` are surfaces whose direction is the inverse of the
    canonical label (for example "gives rise to" for `develops_from`), so a
    triple using them must have subject and object swapped.
    """

    label: str
    family: str
    aliases: tuple[str, ...] = ()
    reversed_aliases: tuple[str, ...] = ()



# Official-family predicate census. Alias coverage is frozen to the
# canonicalization table it replaced; adding an alias is a version bump.
PREDICATE_ENTRIES: tuple[PredicateEntry, ...] = (
    PredicateEntry(
        "is-a",
        FAMILY_TAXONOMIC,
        aliases=("is a", "isa", "subclass of", "subclassof", "kind of"),
    ),
    PredicateEntry("instance-of", FAMILY_INSTANCE, aliases=("instance of", "instanceof")),
    PredicateEntry("type", FAMILY_TYPING, aliases=("type",)),
    PredicateEntry(
        "equivalent class",
        FAMILY_AXIOM,
        aliases=("equivalent to", "equivalent class", "equivalentclass"),
    ),
    PredicateEntry(
        "disjoint with",
        FAMILY_AXIOM,
        aliases=("disjoint from", "disjoint with", "disjointwith"),
    ),
    PredicateEntry(
        "database_cross_reference",
        FAMILY_METADATA,
        aliases=("database cross reference", "databasecrossreference"),
    ),
    PredicateEntry("has role", FAMILY_METADATA, aliases=("has role",)),
    PredicateEntry("same as", FAMILY_IDENTITY, aliases=("same as",)),
    PredicateEntry("exact match", FAMILY_IDENTITY, aliases=("exact match",)),
    PredicateEntry("is_conjugate_acid_of", FAMILY_CHEMISTRY),
    PredicateEntry("is_conjugate_base_of", FAMILY_CHEMISTRY),
    PredicateEntry("part_of", FAMILY_MERONYMIC, aliases=("part of", "is part of")),
    PredicateEntry("has part", FAMILY_MERONYMIC, aliases=("has part", "contains as part")),
    PredicateEntry(
        "develops_from",
        FAMILY_DERIVATION,
        aliases=("develops from", "developed from", "arises from"),
        reversed_aliases=("gives rise to",),
    ),
    PredicateEntry(
        "derives from",
        FAMILY_DERIVATION,
        aliases=("derives from", "derived from"),
    ),
    PredicateEntry("located in", FAMILY_LOCATION, aliases=("is located in", "located in")),
    PredicateEntry("regulates", FAMILY_REGULATION, aliases=("regulates",)),
    PredicateEntry(
        "positively regulates",
        FAMILY_REGULATION,
        aliases=("positively regulates",),
    ),
    PredicateEntry("domain", FAMILY_SCHEMA, aliases=("domain",)),
    PredicateEntry("range", FAMILY_SCHEMA, aliases=("range",)),
    PredicateEntry("is defined by", FAMILY_SCHEMA, aliases=("is defined by",)),
    PredicateEntry("tree view", FAMILY_ANNOTATION, aliases=("tree view",)),
    PredicateEntry("see also", FAMILY_ANNOTATION, aliases=("see also", "seealso")),
)


def surface(value: Any) -> str:
    """Lowercase and collapse whitespace without changing other characters."""
    return " ".join(str(value).lower().strip().split())


def normalize_key(value: Any) -> str:
    """Normalize a predicate surface into the alias lookup key."""
    text = surface(value).replace("_", " ").replace("-", " ")
    return " ".join(re.findall(r"[a-z0-9]+", text))


def _build_alias_table() -> dict[str, tuple[str, bool]]:
    """Build the alias map from the declared surfaces only.

    Labels are deliberately *not* auto-inserted as aliases: entries such as the
    chemistry labels would then gain surfaces that the frozen
    `relation_canonicalization_v2` table never redirected, silently changing
    canonicalization behaviour without a version bump. Every label still stays a
    canonicalization fixed point because both the projector and
    `canonicalize_predicate` fall back to the raw surface.
    """
    table: dict[str, tuple[str, bool]] = {}
    for entry in PREDICATE_ENTRIES:
        for alias in entry.aliases:
            table[normalize_key(alias)] = (entry.label, False)
        for alias in entry.reversed_aliases:
            table[normalize_key(alias)] = (entry.label, True)
    return table


_ALIAS_TABLE: dict[str, tuple[str, bool]] = _build_alias_table()
_FAMILY_BY_LABEL: dict[str, str] = {entry.label: entry.family for entry in PREDICATE_ENTRIES}


def alias_table() -> dict[str, tuple[str, bool]]:
    """Return a copy of the normalized alias -> (canonical label, reverse) map."""
    return dict(_ALIAS_TABLE)


def canonical_labels() -> tuple[str, ...]:
    return tuple(entry.label for entry in PREDICATE_ENTRIES)


def family_of(label: Any) -> str | None:
    """Family of a canonical label, resolved through projection when needed."""
    direct = _FAMILY_BY_LABEL.get(str(label))
    if direct is not None:
        return direct
    hit = _ALIAS_TABLE.get(normalize_key(label))
    return _FAMILY_BY_LABEL.get(hit[0]) if hit else None


STYLE_COMMA_AND = "comma_and"
STYLE_AND = "and"
STYLE_OR = "or"
STYLE_SEMICOLON = "semicolon"
RENDER_STYLES = (STYLE_COMMA_AND, STYLE_AND, STYLE_OR, STYLE_SEMICOLON)


@dataclass(frozen=True)
class PredicateRoute:
    """A frozen inventory of canonical labels plus its prompt rendering style."""

    key: str
    system: str
    labels: tuple[str, ...]
    style: str
    description: str


PREDICATE_ROUTES: dict[str, PredicateRoute] = {
    "general_relation_v1": PredicateRoute(
        key="general_relation_v1",
        system="open_vocab_guarded_v2",
        labels=(
            "part_of",
            "has part",
            "develops_from",
            "derives from",
            "located in",
            "regulates",
        ),
        style=STYLE_COMMA_AND,
        description=(
            "Open relation-only anchors shared by guarded v2, v3, and v4; only the "
            "family anchors are listed and other stable labels stay allowed."
        ),
    ),
    "axiom_v1": PredicateRoute(
        key="axiom_v1",
        system="open_vocab_guarded_v5_axioms_only",
        labels=("equivalent class", "disjoint with"),
        style=STYLE_AND,
        description=(
            "Class-axiom inventory (equivalence and disjointness) used by the "
            "axiom-only pass."
        ),
    ),
    "official_task_ab_inventory_v1": PredicateRoute(
        key="official_task_ab_inventory_v1",
        system="open_vocab_guarded_v6_predicate_routed",
        labels=(
            "equivalent class",
            "disjoint with",
            "type",
            "database_cross_reference",
            "has role",
            "same as",
            "is_conjugate_acid_of",
            "is_conjugate_base_of",
        ),
        style=STYLE_COMMA_AND,
        description=(
            "Frozen official Task A/B predicate-family inventory routed by the v6 arm: "
            "axiom family (equivalent class, disjoint with), type/metadata family "
            "(type, database_cross_reference, has role, same as), and chemistry family "
            "(is_conjugate_acid_of, is_conjugate_base_of)."
        ),
    ),
    "taxonomic_core_v1": PredicateRoute(
        key="taxonomic_core_v1",
        system="closed_vocab",
        labels=("is-a", "instance-of"),
        style=STYLE_AND,
        description=(
            "Primitive-ontology core emitted by the closed-vocabulary baseline and by "
            "the taxonomy and typing stages."
        ),
    ),
}

# Prompt systems that consume a frozen route. Versions v2, v3, and v4 share the
# general relation list: the v4 axiom pass only names the axiom predicates in
# prose, so it keeps the general relation route here.
SYSTEM_ROUTE_KEYS: dict[str, str] = {
    "open_vocab_guarded_v2": "general_relation_v1",
    "open_vocab_guarded_v3_relations": "general_relation_v1",
    "open_vocab_guarded_v4_axioms": "general_relation_v1",
    "open_vocab_guarded_v5_axioms_only": "axiom_v1",
    "open_vocab_guarded_v6_predicate_routed": "official_task_ab_inventory_v1",
    "closed_vocab": "taxonomic_core_v1",
}

_ROUTES_BY_LABEL: dict[str, tuple[str, ...]] = {}
for _route in PREDICATE_ROUTES.values():
    for _label in _route.labels:
        _ROUTES_BY_LABEL.setdefault(_label, ())
        _ROUTES_BY_LABEL[_label] += (_route.key,)
del _route, _label


def route_for_system(system: str) -> PredicateRoute:
    """Frozen route consumed by a prompt system, if it has one."""
    key = SYSTEM_ROUTE_KEYS.get(str(system))
    if key is None:
        raise ValueError(f"no predicate route registered for system: {system}")
    return PREDICATE_ROUTES[key]


def inventory_labels(route_key: str) -> tuple[str, ...]:
    """Ordered canonical labels of a frozen route."""
    route = PREDICATE_ROUTES.get(str(route_key))
    if route is None:
        raise ValueError(f"unknown predicate route: {route_key}")
    return route.labels


def routes_containing(label: Any) -> tuple[str, ...]:
    """Route keys whose frozen inventory contains a canonical label."""
    direct = _ROUTES_BY_LABEL.get(str(label))
    if direct is not None:
        return direct
    hit = _ALIAS_TABLE.get(normalize_key(label))
    return _ROUTES_BY_LABEL.get(hit[0], ()) if hit else ()


def render_labels(labels: Iterable[str], *, style: str) -> str:
    """Deterministic prompt rendering of an ordered label list."""
    if style not in RENDER_STYLES:
        raise ValueError(f"unsupported label style: {style}")
    values = [str(value) for value in labels]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if style == STYLE_COMMA_AND:
        return f"{', '.join(values[:-1])}, and {values[-1]}"
    if style == STYLE_AND:
        return " and ".join(values)
    if style == STYLE_OR:
        return " or ".join(values)
    if style == STYLE_SEMICOLON:
        return "; ".join(values)
    raise ValueError(f"unreachable label style: {style}")


@dataclass(frozen=True)
class PredicateProjection:
    """Result of projecting one raw predicate surface onto the inventory."""

    raw: str
    key: str
    label: str
    family: str | None
    reverse: bool
    changed: bool
    routes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": PREDICATE_INVENTORY_VERSION,
            "raw": self.raw,
            "key": self.key,
            "label": self.label,
            "family": self.family,
            "reverse": self.reverse,
            "changed": self.changed,
            "routes": list(self.routes),
        }


def project_predicate(predicate: Any) -> PredicateProjection:
    """Project a raw predicate onto its canonical label, family, and routes.

    Unknown surfaces are preserved verbatim: the projector never invents or
    repairs a label, so it stays usable as an open-vocabulary diagnostic.
    """
    raw = surface(predicate)
    key = normalize_key(predicate)
    hit = _ALIAS_TABLE.get(key)
    label, reverse = hit if hit is not None else (raw, False)
    return PredicateProjection(
        raw=raw,
        key=key,
        label=label,
        family=_FAMILY_BY_LABEL.get(label),
        reverse=reverse,
        changed=label != raw or reverse,
        routes=routes_containing(label),
    )


def route_adherence(predicates: Iterable[Any], route_key: str) -> dict[str, Any]:
    """Zero-call adherence diagnostic of predicted predicates against a route."""
    route = PREDICATE_ROUTES.get(str(route_key))
    if route is None:
        raise ValueError(f"unknown predicate route: {route_key}")
    inventory = set(route.labels)
    total = 0
    in_route = 0
    projected = 0
    families: Counter[str] = Counter()
    off_route: Counter[str] = Counter()
    for predicate in predicates:
        total += 1
        projection = project_predicate(predicate)
        families[projection.family or "unknown"] += 1
        if projection.changed:
            projected += 1
        if projection.label in inventory:
            in_route += 1
        else:
            off_route[projection.label] += 1
    return {
        "version": PREDICATE_INVENTORY_VERSION,
        "route": route.key,
        "system": route.system,
        "inventory": list(route.labels),
        "total_predictions": total,
        "in_route": in_route,
        "in_route_rate": round(in_route / total, 4) if total else 0.0,
        "projected": projected,
        "projection_rate": round(projected / total, 4) if total else 0.0,
        "family_counts": dict(sorted(families.items())),
        "off_route_labels": dict(off_route.most_common(10)),
    }