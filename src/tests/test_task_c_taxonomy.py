"""Contract tests for the offline Task C taxonomy stack.

These tests pin four contracts:

1. `ontobridg.eval.official_taxonomy_match` reproduces the official challenge
   metric (`data/LLMs4OL-Challenge/2026/metrics/taxonomy.py`) bit-for-bit on
   exact and fuzzy match (data-bound: requires the challenge repository).
2. The zero-API surface/transfer inducer is deterministic and its symbolic
   repair stage always emits a self-loop-free forest under the parent cap.
3. `evaluate_task_c` reports the official metric id, keeps the local surrogate
   closed under `deployment_gate`, and never claims external reportability.
4. `OntoLearnerTaxonomyLearner` keeps the OntoLearner `AutoLearner` contract
   (`load`/`fit`/`predict`/`fit_predict`, `task="taxonomy-discovery"`) for
   extract()-shaped objects, dict rows, and plain term lists.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ontobridg.eval.official_taxonomy_match import evaluate as official_evaluate
from ontobridg.tasks.task_c_taxonomy import (
    MIN_CLAIM_EXACT_F1,
    TASK_C_VERSION,
    OntoLearnerTaxonomyLearner,
    SourceTaxonomyRules,
    SurfaceTaxonomyInducer,
    TaskCDomain,
    TaskCTaxonomySystem,
    deployment_gate,
    edge_pairs,
    evaluate_task_c,
    head_token,
    mine_source_rules,
    normalize,
    rules_sha256,
    singularize,
)

CHALLENGE = SRC.parent / "data" / "LLMs4OL-Challenge" / "2026"
OFFICIAL_METRIC_PATH = CHALLENGE / "metrics" / "taxonomy.py"
ARCHON_INPUT = CHALLENGE / "TaskC-Taxonomy" / "Archeon_test_input.txt"
ARCHON_GOLD = CHALLENGE / "TaskC-Taxonomy" / "Archeon_gold.json"


def edge(parent: str, child: str, index: int = 0) -> dict[str, str]:
    return {"id": f"TR_g{index:04d}", "parent": parent, "child": child}


def load_official_metric_module():
    """Load the official challenge metric without any network/model import."""
    cached = getattr(load_official_metric_module, "_module", None)
    if cached is not None:
        return cached
    if "sentence_transformers" not in sys.modules:
        stub = types.ModuleType("sentence_transformers")
        stub.SentenceTransformer = lambda *args, **kwargs: None  # type: ignore[attr-defined]
        sys.modules["sentence_transformers"] = stub
    spec = importlib.util.spec_from_file_location(
        "official_task_c_taxonomy_metric", OFFICIAL_METRIC_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    load_official_metric_module._module = module  # type: ignore[attr-defined]
    return module


class TaskCSurfaceInducerTests(unittest.TestCase):
    """Deterministic surface + transfer scoring and symbolic repair."""

    def test_head_token_and_singularize(self) -> None:
        self.assertEqual(head_token("centrally registered identifier"), "registered identifier")
        self.assertEqual(head_token("site"), "site")
        self.assertEqual(head_token(""), "")
        self.assertEqual(singularize("categories"), "category")
        self.assertEqual(singularize("boxes"), "box")
        self.assertEqual(singularize("bus"), "bus")
        self.assertEqual(singularize("analysis"), "analysis")

    def test_normalize_collapses_whitespace(self) -> None:
        self.assertEqual(normalize("  Foo   BAR "), "foo bar")

    def test_edge_pairs_accepts_mappings_attributes_and_sequences(self) -> None:
        rows = [
            {"parent": "P", "child": "C"},
            SimpleNamespace(parent="P2", child="C2"),
            ("P3", "C3"),
            object(),
            "ab",  # strings must not be treated as positional rows
        ]
        self.assertEqual(
            edge_pairs(rows),
            [("p", "c"), ("p2", "c2"), ("p3", "c3")],
        )

    def test_mine_source_rules_mines_child_and_head_maps(self) -> None:
        rules = mine_source_rules([("metal", "steel"), ("metal", "steel"), ("steel", "alloy")])
        self.assertEqual(rules.n_edges, 3)
        self.assertEqual(rules.parents_for_child("steel"), ["metal"])
        self.assertEqual(rules.parents_for_child("alloy"), ["steel"])
        self.assertEqual(rules.parent_frequency["metal"], 2)
        # head key is the term's own last-two-token head
        self.assertEqual(rules.parents_for_head("steel"), ["metal"])
        self.assertEqual(rules.parents_for_head("alloy"), ["steel"])
        self.assertEqual(rules.parents_for_head("hard steel"), [])

    def test_rules_json_round_trip_and_sha_format(self) -> None:
        rules = mine_source_rules([("metal", "steel plate"), ("alloy", "steel plate")])
        restored = SourceTaxonomyRules.from_json(rules.to_json())
        self.assertEqual(restored.to_json(), rules.to_json())
        self.assertEqual(restored.n_edges, rules.n_edges)
        expected = hashlib.sha256(b"metal\tsteel plate").hexdigest()
        self.assertEqual(rules_sha256([("metal", "steel plate")]), expected)
        self.assertEqual(len(rules.edges_sha256), 64)

    def test_suffix_containment_outranks_substring(self) -> None:
        inducer = SurfaceTaxonomyInducer()
        pool = ["carbon steel alloy", "alloy", "steel"]
        found = {c.parent: c for c in inducer.candidates("carbon steel alloy", pool)}
        self.assertEqual(found["alloy"].reason, "suffix_containment")
        self.assertEqual(found["alloy"].score, 0.80)
        self.assertEqual(found["steel"].reason, "substring_containment")
        self.assertEqual(found["steel"].score, 0.45)
        selected = inducer.select("carbon steel alloy", pool)
        self.assertEqual([c.parent for c in selected], ["alloy"])

    def test_morphology_singular_beats_substring_for_plural_child(self) -> None:
        inducer = SurfaceTaxonomyInducer()
        found = inducer.candidates(
            "data storages", ["data storages", "data storage"]
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].reason, "morphology_singular")
        self.assertEqual(found[0].score, 0.55)
        self.assertEqual(found[0].parent, "data storage")

    def test_source_child_match_outranks_surface_and_skips_head_transfer(self) -> None:
        rules = mine_source_rules([("metal", "steel plate")])
        inducer = SurfaceTaxonomyInducer()
        found = inducer.candidates("steel plate", ["steel plate", "metal"], rules)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].reason, "source_child_match")
        self.assertEqual(found[0].score, 1.0)
        self.assertEqual(found[0].parent, "metal")

    def test_source_head_transfer_matches_last_two_token_head(self) -> None:
        rules = mine_source_rules([("metal", "steel plate")])
        inducer = SurfaceTaxonomyInducer()
        found = inducer.candidates("hard steel plate", ["hard steel plate"], rules)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].reason, "source_head_transfer")
        self.assertAlmostEqual(found[0].score, 0.62)

    def test_predict_is_deterministic_with_stable_tr_ids(self) -> None:
        terms = ["carbon steel alloy", "alloy", "steel", "steel plate"]
        inducer = SurfaceTaxonomyInducer()
        first = inducer.predict(terms)
        second = SurfaceTaxonomyInducer().predict(terms)
        self.assertEqual(first, second)
        self.assertTrue(first)
        for row in first:
            self.assertRegex(row["id"], r"^TR_[0-9a-f]{8}$")
            expected = "TR_" + hashlib.sha256(
                f'{row["child"]}|{row["parent"]}'.encode("utf-8")
            ).hexdigest()[:8]
            self.assertEqual(row["id"], expected)
        self.assertEqual(inducer.last_stats["terms"], len(terms))
        self.assertEqual(inducer.last_stats["predicted_edges"], len(first))

    def test_repair_drops_self_loops_and_duplicates(self) -> None:
        inducer = SurfaceTaxonomyInducer()
        kept, dropped = inducer._repair(
            [("a", "a", 0.9), ("a", "b", 0.5), ("a", "b", 0.9), ("", "c", 0.8)]
        )
        # self-loops and empty parents are filtered up front, duplicates keep
        # only the best score, and nothing needed cycle-repair.
        self.assertEqual(kept, [("a", "b", 0.9)])
        self.assertEqual(dropped, [])

    def test_repair_blocks_cycles_into_a_forest(self) -> None:
        inducer = SurfaceTaxonomyInducer()
        kept, dropped = inducer._repair(
            [("a", "b", 0.9), ("b", "a", 0.8), ("b", "c", 0.7), ("c", "a", 0.6)]
        )
        kept_pairs = {(c, p) for c, p, _ in kept}
        self.assertIn(("a", "b"), kept_pairs)
        self.assertIn(("b", "c"), kept_pairs)
        self.assertNotIn(("b", "a"), kept_pairs)
        self.assertNotIn(("c", "a"), kept_pairs)
        self.assertEqual(len(kept) + len(dropped), 4)
        # every node has at most one parent in the kept set
        children = [c for c, _ in kept_pairs]
        self.assertEqual(len(children), len(set(children)))

    def test_select_respects_parent_cap(self) -> None:
        pool = ["carbon steel alloy", "alloy", "steel"]
        capped = SurfaceTaxonomyInducer(max_parents_per_child=1)
        self.assertEqual(len(capped.select("carbon steel alloy", pool)), 1)
        generous = SurfaceTaxonomyInducer(max_parents_per_child=3)
        self.assertEqual(len(generous.select("carbon steel alloy", pool)), 2)


class TaskCSystemEvaluationTests(unittest.TestCase):
    """Modes, official report shape, and the honest deployment gate."""

    @staticmethod
    def _domains() -> dict[str, TaskCDomain]:
        alpha = TaskCDomain(
            name="Alpha",
            terms=["metal", "steel"],
            gold_edges=[edge("metal", "steel", 0)],
        )
        beta = TaskCDomain(
            name="Beta",
            terms=["steel", "alloy steel"],
            gold_edges=[edge("metal", "steel", 1), edge("steel", "alloy steel", 2)],
        )
        return {"Alpha": alpha, "Beta": beta}

    def test_unknown_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            TaskCTaxonomySystem(mode="telepathy")

    def test_zero_shot_mode_never_mines_source_rules(self) -> None:
        system = TaskCTaxonomySystem(mode="zero_shot_surface")
        rules = system.fit({"Alpha": self._domains()["Alpha"]})
        self.assertEqual(rules.n_edges, 0)
        predictions = system.predict(["carbon steel alloy", "alloy", "steel"])
        self.assertTrue(all(row["parent"] != row["child"] for row in predictions))

    def test_evaluate_task_c_requires_sources_or_rules_for_transfer(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_task_c(self._domains(), mode="source_transfer")

    def test_evaluate_task_c_end_to_end_on_synthetic_domains(self) -> None:
        domains = self._domains()
        report = evaluate_task_c(domains, mode="source_transfer", source_domains=domains)
        self.assertEqual(report["protocol"], TASK_C_VERSION)
        self.assertEqual(report["metric_id"], "llms4ol2026_task_c_taxonomy")
        self.assertEqual(set(report["domains"]), {"Alpha", "Beta"})
        self.assertEqual(report["domains"]["Alpha"]["predicted_edges"], 1)
        self.assertEqual(report["domains"]["Beta"]["predicted_edges"], 2)
        self.assertEqual(report["domains"]["Beta"]["coverage"], 1.0)
        self.assertEqual(report["micro"]["gold_edges"], 3)
        self.assertEqual(report["micro"]["predicted_edges"], 3)
        self.assertAlmostEqual(report["micro"]["exact_match"]["f1_score"], 1.0)
        self.assertAlmostEqual(report["macro_exact_f1"], 1.0)
        self.assertTrue(report["metric_engines"])
        self.assertEqual(
            report["gate"]["reasons"], ["protocol_fidelity_local_surrogate"]
        )
        self.assertFalse(report["gate"]["passed"])
        self.assertFalse(report["scientific_deployable"])
        self.assertFalse(report["external_reportable"])

    def test_semantic_match_stays_unavailable_without_explicit_embedder(self) -> None:
        report = official_evaluate(
            [edge("metal", "steel")], [edge("metal", "steel")], semantic=True
        )
        self.assertEqual(report["semantic_match"]["status"], "unavailable")
        self.assertNotIn(
            "semantic_match",
            official_evaluate([edge("metal", "steel")], [edge("metal", "steel")]),
        )

    def test_deployment_gate_passes_only_on_official_fidelity_above_floor(self) -> None:
        metrics = {
            "micro": {
                "exact_match": {"f1_score": 0.5},
                "fuzzy_match": {"f1_score": 0.5},
            },
            "domains": {"Alpha": {"predicted_edges": 3}},
            "protocol_fidelity": "official",
        }
        gate = deployment_gate(metrics)
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["scientific_deployable"])
        self.assertFalse(gate["confirmatory_eligible"])
        self.assertFalse(gate["external_reportable"])
        self.assertEqual(gate["min_claim_exact_f1"], MIN_CLAIM_EXACT_F1)

    def test_deployment_gate_reports_floor_empty_domain_and_fidelity_reasons(self) -> None:
        metrics = {
            "micro": {
                "exact_match": {"f1_score": 0.05},
                "fuzzy_match": {"f1_score": 0.0},
            },
            "domains": {"Alpha": {"predicted_edges": 0}},
        }
        gate = deployment_gate(metrics)
        self.assertFalse(gate["passed"])
        self.assertIn(f"micro_exact_f1_below_{MIN_CLAIM_EXACT_F1}", gate["reasons"])
        self.assertIn("domains_with_zero_predictions:Alpha", gate["reasons"])
        self.assertIn("protocol_fidelity_local_surrogate", gate["reasons"])


class OntoLearnerTaxonomyLearnerTests(unittest.TestCase):
    """`AutoLearner` signature parity without importing the heavy package."""

    ROWS = [
        {"id": "r1", "parent": "metal", "child": "steel"},
        {"id": "r2", "parent": "steel", "child": "alloy steel"},
    ]

    @classmethod
    def _extract(cls) -> SimpleNamespace:
        rows = [
            SimpleNamespace(parent=row["parent"], child=row["child"])
            for row in cls.ROWS
        ]
        return SimpleNamespace(type_taxonomies=SimpleNamespace(taxonomies=rows))

    def test_load_records_configuration(self) -> None:
        learner = OntoLearnerTaxonomyLearner()
        learner.load(model_id="offline-inducer", seed=42)
        self.assertEqual(learner.config["model_id"], "offline-inducer")
        self.assertEqual(learner.config["seed"], 42)

    def test_fit_rejects_non_taxonomy_tasks(self) -> None:
        learner = OntoLearnerTaxonomyLearner()
        with self.assertRaises(ValueError) as ctx:
            learner.fit(self.ROWS, task="term-typing")
        self.assertEqual(str(ctx.exception), "term-typing is not a valid task.")
        with self.assertRaises(ValueError):
            learner.predict(["steel"], task="non-taxonomic-re")

    def test_fit_predict_contract_on_dict_rows(self) -> None:
        learner = OntoLearnerTaxonomyLearner()
        rules = learner.fit(self.ROWS, task="taxonomy-discovery")
        self.assertEqual(rules.n_edges, 2)
        predictions = learner.predict(
            ["steel", "alloy steel"], task="taxonomy-discovery"
        )
        self.assertEqual(
            {(row["parent"], row["child"]) for row in predictions},
            {("metal", "steel"), ("steel", "alloy steel")},
        )
        for row in predictions:
            self.assertRegex(row["id"], r"^TR_[0-9a-f]{8}$")
        self.assertEqual(learner.last_stats["terms"], 2)

    def test_fit_accepts_extract_shaped_objects_with_attribute_rows(self) -> None:
        from_extract = OntoLearnerTaxonomyLearner()
        from_dict = OntoLearnerTaxonomyLearner()
        rules_a = from_extract.fit(self._extract(), task="taxonomy-discovery")
        rules_b = from_dict.fit(self.ROWS, task="taxonomy-discovery")
        self.assertEqual(rules_a.edges_sha256, rules_b.edges_sha256)

    def test_predict_derives_terms_from_extract_shaped_eval_data(self) -> None:
        learner = OntoLearnerTaxonomyLearner()
        learner.fit(self.ROWS, task="taxonomy-discovery")
        predictions = learner.predict(self._extract(), task="taxonomy-discovery")
        # tasks_data_former semantics: union of parent/child surfaces
        self.assertEqual(learner.last_stats["terms"], 3)
        children = {row["child"] for row in predictions}
        self.assertEqual(children, {"steel", "alloy steel"})

    def test_fit_predict_supports_multi_source_domain_mapping(self) -> None:
        source = TaskCDomain(
            name="Alpha",
            terms=["metal", "steel"],
            gold_edges=[{"id": "TR_g0", "parent": "metal", "child": "steel"}],
        )
        learner = OntoLearnerTaxonomyLearner()
        predictions = learner.fit_predict(
            {"Alpha": source}, ["steel", "hard steel"], task="taxonomy-discovery"
        )
        self.assertIn(
            ("metal", "steel"), {(r["parent"], r["child"]) for r in predictions}
        )

    def test_predict_rejects_unsupported_eval_shapes(self) -> None:
        learner = OntoLearnerTaxonomyLearner()
        with self.assertRaises(TypeError):
            learner.predict(42, task="taxonomy-discovery")
        with self.assertRaises(TypeError):
            learner.fit(object(), task="taxonomy-discovery")


@unittest.skipUnless(
    OFFICIAL_METRIC_PATH.exists(),
    "third-party challenge data (LLMs4OL-Challenge) not present",
)
class TaskCOfficialMetricMirrorTests(unittest.TestCase):
    """Bit-exact mirror of the official Task C metric (data-bound)."""

    TERMS = [
        "owl format",
        "rdf format",
        "fiat point",
        "site",
        "written name",
        "independent continuant",
        "ontology module",
        "data storage process",
        "programming language",
        "centrally registered identifier",
    ]

    @classmethod
    def _edges(cls, n: int, prefix: str) -> list[dict[str, str]]:
        import random

        rows: list[dict[str, str]] = []
        for i in range(n):
            child = random.choice(cls.TERMS)
            parent = random.choice(cls.TERMS)
            if child == parent:
                continue
            rows.append({"id": f"{prefix}{i}", "parent": parent, "child": child})
        return rows

    def test_local_mirror_matches_official_challenge_module(self) -> None:
        import random

        from ontobridg.eval import official_taxonomy_match as local

        official = load_official_metric_module()
        random.seed(0)
        checked = 0
        for n in (0, 1, 5, 12):
            for k in (0, 1, 5, 12):
                gold = self._edges(n, "g")
                pred = self._edges(k, "p")
                exact_official = official.exact_match(gold, pred)
                exact_local = local.exact_match(gold, pred)
                fuzzy_official = official.fuzzy_match(gold, pred)
                fuzzy_local = local.fuzzy_match(gold, pred)
                for key in exact_official:
                    self.assertAlmostEqual(
                        exact_official[key], exact_local[key], places=12
                    )
                for key in fuzzy_official:
                    self.assertAlmostEqual(
                        fuzzy_official[key], fuzzy_local[key], places=12
                    )
                report = local.evaluate(gold, pred)
                self.assertAlmostEqual(
                    report["exact_match"]["f1_score"],
                    exact_official["f1_score"],
                    places=12,
                )
                checked += 1
        self.assertEqual(checked, 16)


@unittest.skipUnless(
    ARCHON_INPUT.exists() and ARCHON_GOLD.exists(),
    "third-party challenge data (LLMs4OL-Challenge) not present",
)
class TaskCOfficialDataTests(unittest.TestCase):
    """Official held-out files end to end (data-bound)."""

    def test_archeon_domain_loads_expected_counts_and_hashes(self) -> None:
        from ontobridg.tasks.task_c_taxonomy import load_task_c_domain

        domain = load_task_c_domain("Archeon", root=CHALLENGE)
        self.assertEqual(len(domain.terms), 194)
        self.assertEqual(len(domain.gold_edges), 237)
        self.assertEqual(domain.gold_edge_count_total, 237)
        self.assertEqual(len(domain.input_sha256), 64)
        self.assertEqual(len(domain.gold_sha256), 64)
        limited = load_task_c_domain("Archeon", root=CHALLENGE, term_limit=50)
        self.assertEqual(len(limited.terms), 50)
        self.assertLessEqual(len(limited.gold_edges), 237)
        term_keys = {normalize(term) for term in limited.terms}
        self.assertTrue(all(edge["child"] in term_keys for edge in limited.gold_edges))

    def test_evaluate_task_c_on_archeon_prefix_closes_gate(self) -> None:
        from ontobridg.tasks.task_c_taxonomy import load_task_c_domain

        domain = load_task_c_domain("Archeon", root=CHALLENGE, term_limit=120)
        reference = TaskCTaxonomySystem(mode="zero_shot_surface").predict(domain.terms)
        report = evaluate_task_c({"Archeon": domain}, mode="zero_shot_surface")
        row = report["domains"]["Archeon"]
        self.assertEqual(row["terms"], 120)
        self.assertGreaterEqual(row["gold_edges"], 1)
        # the evaluation path is deterministic: same predictions as a direct run
        self.assertEqual(row["predicted_edges"], len(reference))
        self.assertEqual(row["stats"]["predicted_edges"], len(reference))
        self.assertAlmostEqual(
            row["coverage"],
            len({edge["child"] for edge in reference}) / len(domain.terms),
        )
        self.assertIn(
            "protocol_fidelity_local_surrogate", report["gate"]["reasons"]
        )
        self.assertFalse(report["gate"]["passed"])
        self.assertFalse(report["scientific_deployable"])
        self.assertFalse(report["external_reportable"])
        self.assertGreater(report["micro"]["gold_edges"], 0)


if __name__ == "__main__":
    unittest.main()