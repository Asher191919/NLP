"""Frozen-contract tests for the versioned predicate inventory library.

These tests pin three contracts:

1. `ontobridg.relations.predicate_inventory` reproduces the frozen
   `relation_canonicalization_v2` alias behaviour exactly.
2. The route registry stays in sync with the systems and prompt literals used by
   `run_official_blind_eval`, so the frozen holdout artifacts keep rebuilding
   byte-identically (`build_*_holdout.py --audit-only`).
3. Projection/adherence diagnostics stay descriptive: unknown surfaces are never
   rewritten into inventory labels.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_official_blind_eval as blind
from ontobridg.relations import canonicalization
from ontobridg.relations.predicate_inventory import (
    PREDICATE_INVENTORY_VERSION,
    PREDICATE_ROUTES,
    STYLE_AND,
    STYLE_COMMA_AND,
    STYLE_OR,
    STYLE_SEMICOLON,
    SYSTEM_ROUTE_KEYS,
    alias_table,
    canonical_labels,
    family_of,
    inventory_labels,
    project_predicate,
    render_labels,
    route_adherence,
    route_for_system,
    routes_containing,
    surface as predicate_surface,
)


# Literal copy of the frozen relation_canonicalization_v2 alias table that the
# inventory replaced. A diff here is a canonicalization behaviour change, which
# requires bumping CANONICALIZATION_VERSION and re-verifying every frozen packet.
LEGACY_ALIAS_TABLE = {
    "is a": ("is-a", False),
    "isa": ("is-a", False),
    "subclass of": ("is-a", False),
    "subclassof": ("is-a", False),
    "kind of": ("is-a", False),
    "instance of": ("instance-of", False),
    "instanceof": ("instance-of", False),
    "part of": ("part_of", False),
    "is part of": ("part_of", False),
    "has part": ("has part", False),
    "contains as part": ("has part", False),
    "develops from": ("develops_from", False),
    "developed from": ("develops_from", False),
    "arises from": ("develops_from", False),
    "derives from": ("derives from", False),
    "derived from": ("derives from", False),
    "gives rise to": ("develops_from", True),
    "is located in": ("located in", False),
    "located in": ("located in", False),
    "is defined by": ("is defined by", False),
    "equivalent to": ("equivalent class", False),
    "equivalent class": ("equivalent class", False),
    "equivalentclass": ("equivalent class", False),
    "disjoint from": ("disjoint with", False),
    "disjoint with": ("disjoint with", False),
    "disjointwith": ("disjoint with", False),
    "same as": ("same as", False),
    "exact match": ("exact match", False),
    "tree view": ("tree view", False),
    "database cross reference": ("database_cross_reference", False),
    "databasecrossreference": ("database_cross_reference", False),
    "see also": ("see also", False),
    "seealso": ("see also", False),
    "has role": ("has role", False),
    "positively regulates": ("positively regulates", False),
    "regulates": ("regulates", False),
    "domain": ("domain", False),
    "range": ("range", False),
    "type": ("type", False),
}


class PredicateInventoryContractTests(unittest.TestCase):
    def test_alias_table_is_exactly_the_frozen_legacy_table(self) -> None:
        self.assertEqual(alias_table(), LEGACY_ALIAS_TABLE)
        self.assertEqual(canonicalization.CANONICALIZATION_VERSION, "relation_canonicalization_v2")
        self.assertEqual(PREDICATE_INVENTORY_VERSION, "predicate_inventory_v1")

    def test_projection_agrees_with_canonicalize_predicate(self) -> None:
        surfaces = [
            *LEGACY_ALIAS_TABLE,
            *canonical_labels(),
            "Is-A",
            "Sub Class Of",
            "is_conjugate_acid_of",
            "is conjugate acid of",
            "part-of",
            "unregistered narrative verb",
            "",
        ]
        for surface in surfaces:
            with self.subTest(surface=surface):
                label, reverse, changed_from = canonicalization.canonicalize_predicate(surface)
                projection = project_predicate(surface)
                self.assertEqual(projection.label, label)
                self.assertEqual(projection.reverse, reverse)
                self.assertEqual(projection.changed, changed_from is not None)

    def test_every_inventory_label_is_a_canonicalization_fixed_point(self) -> None:
        for label in canonical_labels():
            with self.subTest(label=label):
                self.assertEqual(
                    canonicalization.canonicalize_predicate(label),
                    (label, False, None),
                )
                projection = project_predicate(label)
                self.assertEqual(projection.label, label)
                self.assertFalse(projection.changed)
                self.assertEqual(projection.family, family_of(label))

    def test_projection_preserves_unknown_surfaces_verbatim(self) -> None:
        for raw in ("causes", "associated with", "  Mixed Case Verb  ", ""):
            with self.subTest(raw=raw):
                projection = project_predicate(raw)
                self.assertEqual(projection.label, predicate_surface(raw))
                self.assertIsNone(projection.family)
                self.assertEqual(projection.routes, ())
                self.assertFalse(projection.changed)

    def test_reverse_alias_reports_the_direction_flip(self) -> None:
        projection = project_predicate("Gives Rise To")
        self.assertEqual(projection.label, "develops_from")
        self.assertTrue(projection.reverse)
        self.assertTrue(projection.changed)
        self.assertEqual(
            canonicalization.canonicalize_predicate("gives rise to"),
            ("develops_from", True, "gives rise to"),
        )

    def test_chemistry_labels_need_a_version_bump_to_gain_surface_aliases(self) -> None:
        """Documents the frozen-table limitation instead of silently fixing it."""
        self.assertIsNone(alias_table().get("is conjugate acid of"))
        self.assertEqual(
            canonicalization.canonicalize_predicate("is conjugate acid of"),
            ("is conjugate acid of", False, None),
        )
        projection = project_predicate("is conjugate acid of")
        self.assertEqual(projection.label, "is conjugate acid of")
        self.assertIsNone(projection.family)
        self.assertEqual(projection.routes, ())
        canonical = project_predicate("is_conjugate_acid_of")
        self.assertEqual(canonical.label, "is_conjugate_acid_of")
        self.assertEqual(canonical.family, "chemistry")
        self.assertEqual(canonical.routes, ("official_task_ab_inventory_v1",))

    def test_route_registry_covers_every_guarded_prompt_system(self) -> None:
        for system in sorted(blind.GUARDED_V2_SYSTEMS):
            with self.subTest(system=system):
                self.assertIn(system, SYSTEM_ROUTE_KEYS)
                route = route_for_system(system)
                self.assertTrue(route.labels)
                self.assertIn(route.style, (STYLE_AND, STYLE_COMMA_AND))
                self.assertEqual(inventory_labels(route.key), route.labels)
        for key, route in PREDICATE_ROUTES.items():
            with self.subTest(route=key):
                self.assertEqual(SYSTEM_ROUTE_KEYS[route.system], key)

    def test_route_label_order_and_rendering_are_frozen(self) -> None:
        general = inventory_labels("general_relation_v1")
        official = inventory_labels("official_task_ab_inventory_v1")
        axiom = inventory_labels("axiom_v1")
        self.assertEqual(
            general,
            ("part_of", "has part", "develops_from", "derives from", "located in", "regulates"),
        )
        self.assertEqual(
            official,
            (
                "equivalent class",
                "disjoint with",
                "type",
                "database_cross_reference",
                "has role",
                "same as",
                "is_conjugate_acid_of",
                "is_conjugate_base_of",
            ),
        )
        self.assertEqual(axiom, ("equivalent class", "disjoint with"))
        self.assertEqual(inventory_labels("taxonomic_core_v1"), ("is-a", "instance-of"))
        self.assertEqual(
            render_labels(general, style=STYLE_COMMA_AND),
            "part_of, has part, develops_from, derives from, located in, and regulates",
        )
        self.assertEqual(
            render_labels(official, style=STYLE_COMMA_AND),
            "equivalent class, disjoint with, type, database_cross_reference, has role, "
            "same as, is_conjugate_acid_of, and is_conjugate_base_of",
        )
        self.assertEqual(
            render_labels(official, style=STYLE_SEMICOLON),
            "equivalent class; disjoint with; type; database_cross_reference; has role; "
            "same as; is_conjugate_acid_of; is_conjugate_base_of",
        )
        self.assertEqual(
            render_labels(axiom, style=STYLE_AND), "equivalent class and disjoint with"
        )
        self.assertEqual(
            render_labels(axiom, style=STYLE_OR), "equivalent class or disjoint with"
        )

    def test_route_inventories_appear_in_the_frozen_prompt_passes(self) -> None:
        sample = {"id": "route-contract-test", "context": "Alpha Beta Gamma."}
        prompts = {
            system: blind.build_prompt(sample, "a", system, 6000)
            for system in sorted(blind.GUARDED_V2_SYSTEMS)
        }
        for system, prompt in prompts.items():
            route = route_for_system(system)
            rendered = render_labels(inventory_labels(route.key), style=route.style)
            with self.subTest(system=system):
                self.assertIn(f"such as {rendered}", prompt)

        general = render_labels(inventory_labels("general_relation_v1"), style=STYLE_COMMA_AND)
        official = inventory_labels("official_task_ab_inventory_v1")
        axiom = inventory_labels("axiom_v1")
        routed = prompts["open_vocab_guarded_v6_predicate_routed"]
        self.assertIn(
            f"in this inventory: {render_labels(official, style=STYLE_SEMICOLON)}.",
            routed,
        )
        self.assertNotIn(general, routed)
        axiom_only = prompts["open_vocab_guarded_v5_axioms_only"]
        self.assertIn(f"emit only {render_labels(axiom, style=STYLE_OR)} triples;", axiom_only)
        self.assertNotIn(general, axiom_only)
        general_prompt = prompts["open_vocab_guarded_v3_relations"]
        self.assertIn(general, general_prompt)
        self.assertNotIn(render_labels(official, style=STYLE_SEMICOLON), general_prompt)

    def test_route_adherence_reports_in_route_and_off_route_rates(self) -> None:
        report = route_adherence(
            ["equivalent class", "Equivalent To", "part of", "causes"],
            "official_task_ab_inventory_v1",
        )
        self.assertEqual(report["version"], PREDICATE_INVENTORY_VERSION)
        self.assertEqual(report["route"], "official_task_ab_inventory_v1")
        self.assertEqual(report["system"], "open_vocab_guarded_v6_predicate_routed")
        self.assertEqual(report["inventory"], list(inventory_labels("official_task_ab_inventory_v1")))
        self.assertEqual(report["total_predictions"], 4)
        self.assertEqual(report["in_route"], 2)
        self.assertEqual(report["in_route_rate"], 0.5)
        self.assertEqual(report["projected"], 2)
        self.assertEqual(report["projection_rate"], 0.5)
        self.assertEqual(report["family_counts"], {"axiom": 2, "meronymic": 1, "unknown": 1})
        self.assertEqual(report["off_route_labels"], {"part_of": 1, "causes": 1})

    def test_route_adherence_on_empty_input_is_zero_not_undefined(self) -> None:
        report = route_adherence([], "axiom_v1")
        self.assertEqual(report["total_predictions"], 0)
        self.assertEqual(report["in_route_rate"], 0.0)
        self.assertEqual(report["projection_rate"], 0.0)
        self.assertEqual(report["off_route_labels"], {})

    def test_routes_containing_reports_membership(self) -> None:
        self.assertEqual(routes_containing("part_of"), ("general_relation_v1",))
        self.assertEqual(routes_containing("Part Of"), ("general_relation_v1",))
        self.assertEqual(routes_containing("equivalent class"), ("axiom_v1", "official_task_ab_inventory_v1"))
        self.assertEqual(routes_containing("type"), ("official_task_ab_inventory_v1",))
        self.assertEqual(routes_containing("narrative verb"), ())
        self.assertEqual(routes_containing("is conjugate acid of"), ())

    def test_unknown_route_or_system_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            inventory_labels("unknown_route")
        with self.assertRaises(ValueError):
            route_adherence([], "unknown_route")
        with self.assertRaises(ValueError):
            route_for_system("open_vocab_guarded")
        with self.assertRaises(ValueError):
            route_for_system("not_a_system")

    def test_render_labels_styles_are_deterministic(self) -> None:
        self.assertEqual(render_labels(("a", "b", "c"), style=STYLE_COMMA_AND), "a, b, and c")
        self.assertEqual(render_labels(("a", "b"), style=STYLE_COMMA_AND), "a, and b")
        self.assertEqual(render_labels(("a", "b"), style=STYLE_AND), "a and b")
        self.assertEqual(render_labels(("a", "b"), style=STYLE_OR), "a or b")
        self.assertEqual(render_labels(("a", "b"), style=STYLE_SEMICOLON), "a; b")
        self.assertEqual(render_labels(("a",), style=STYLE_COMMA_AND), "a")
        self.assertEqual(render_labels(("a",), style=STYLE_SEMICOLON), "a")
        self.assertEqual(render_labels((), style=STYLE_SEMICOLON), "")
        with self.assertRaises(ValueError):
            render_labels(("a", "b"), style="unknown_style")


if __name__ == "__main__":
    unittest.main()

