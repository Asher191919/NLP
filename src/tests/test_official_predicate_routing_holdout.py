from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_official_predicate_routing_holdout as holdout
import run_official_predicate_routing as runner


class OfficialPredicateRoutingHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, inputs, manifest, evaluation = holdout.build_package()
        cls.dataset_bytes = dataset
        cls.inputs_bytes = inputs
        cls.manifest_bytes = manifest
        cls.dataset = json.loads(dataset)
        cls.inputs = [json.loads(line) for line in inputs.splitlines()]
        cls.manifest = json.loads(manifest)
        cls.evaluation = evaluation

    def test_capacity_aware_strata_and_target_support_are_frozen(self) -> None:
        audit = self.manifest["balance_audit"]
        self.assertEqual(audit["samples"], 27)
        self.assertEqual(
            audit["assigned_strata"],
            {
                "disjoint_with": 8,
                "domain_specific": 8,
                "equivalent_class": 8,
                "type": 3,
            },
        )
        self.assertEqual(audit["visible_target_gold"], 64)
        self.assertEqual(audit["all_target_gold"], 69)
        self.assertEqual(
            audit["visible_target_predicate_counts"],
            {
                "database_cross_reference": 8,
                "disjoint with": 30,
                "equivalent class": 23,
                "type": 3,
            },
        )

    def test_packet_is_disjoint_from_every_prior_selection(self) -> None:
        audit = self.manifest["independence_audit"]
        for key, value in audit.items():
            if key == "selected_indices":
                continue
            self.assertFalse(value, key)

    def test_model_inputs_are_exact_id_context_projection(self) -> None:
        self.assertTrue(all(set(row) == {"id", "context"} for row in self.inputs))
        self.assertEqual(
            self.inputs,
            [
                {"id": row["id"], "context": row["context"]}
                for row in self.dataset["samples"]
            ],
        )

    def test_public_manifests_hide_text_and_gold(self) -> None:
        for payload in (self.manifest, self.evaluation):
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertFalse(payload["meta"]["contains_source_text"])
            self.assertFalse(payload["meta"]["contains_protected_gold"])
            self.assertNotIn('"context":', serialized)
            self.assertNotIn('"gold_triples":', serialized)
            self.assertNotIn('"target_gold":', serialized)

    def test_evaluation_freezes_zero_cache_fifty_four_unit_plan(self) -> None:
        meta = self.evaluation["meta"]
        locked = self.evaluation["locked_protocol"]
        self.assertEqual(meta["planned_units"], 54)
        self.assertEqual(meta["generation_cache_hits"], 0)
        self.assertEqual(meta["estimated_external_calls"], 54)
        self.assertEqual(meta["systems"], list(holdout.SYSTEMS))
        self.assertEqual(
            locked["primary_metric"],
            "macro_exact_target_f1_across_assigned_strata",
        )
        self.assertEqual(locked["max_completion_tokens"], 16384)
        self.assertFalse(locked["allow_same_slice_repair"])
        self.assertFalse(meta["scientific_deployable"])

    def test_prompt_identity_separates_general_and_routed_systems(self) -> None:
        row = self.evaluation["samples"][0]
        self.assertNotEqual(
            row["prompt_sha256"][holdout.SYSTEMS[0]],
            row["prompt_sha256"][holdout.SYSTEMS[1]],
        )
        prompt = holdout.build_prompt(
            self.dataset["samples"][0],
            holdout.SYSTEMS[1],
            holdout.DEFAULT_MAX_CONTEXT_CHARS,
        )
        self.assertIn("FROZEN PREDICATE-FAMILY ROUTER", prompt)

    def test_build_is_byte_deterministic(self) -> None:
        dataset, inputs, manifest, evaluation = holdout.build_package()
        self.assertEqual(dataset, self.dataset_bytes)
        self.assertEqual(inputs, self.inputs_bytes)
        self.assertEqual(manifest, self.manifest_bytes)
        self.assertEqual(evaluation, self.evaluation)

    def test_runner_audit_finds_no_prefrozen_cache(self) -> None:
        hits, replayable = runner.cache_status(
            self.dataset, self.evaluation, "predicate-routing-test-no-cache"
        )
        self.assertEqual((hits, replayable), (0, 0))


if __name__ == "__main__":
    unittest.main()
