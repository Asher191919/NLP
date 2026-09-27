from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_official_relation_transfer_holdout as holdout
import analyze_official_relation_transfer as error_audit
import run_official_blind_eval as blind
import run_official_relation_transfer as runner


class OfficialRelationTransferHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dataset, inputs, manifest, eval_manifest = holdout.build_package()
        cls.dataset_bytes = dataset
        cls.inputs_bytes = inputs
        cls.manifest_bytes = manifest
        cls.dataset = json.loads(dataset)
        cls.inputs = [json.loads(line) for line in inputs.splitlines()]
        cls.manifest = json.loads(manifest)
        cls.eval_manifest = eval_manifest

    def test_balanced_relation_rich_selection(self) -> None:
        audit = self.manifest["balance_audit"]
        self.assertEqual(audit["samples"], 30)
        self.assertEqual(set(audit["assigned_strata"].values()), {10})
        self.assertGreaterEqual(audit["visible_relation_count_min"], 2)
        self.assertLessEqual(audit["visible_relation_count_max"], 8)
        self.assertGreaterEqual(audit["visible_nontax_gold"], 60)

    def test_independence_audit_is_zero(self) -> None:
        audit = self.manifest["independence_audit"]
        self.assertEqual(audit["selected_index_overlap_with_prior_task_a"], [])
        for key, value in audit.items():
            if key in {
                "selected_indices",
                "selected_index_overlap_with_prior_task_a",
            }:
                continue
            self.assertEqual(value, 0, key)

    def test_model_inputs_are_exact_id_context_projection(self) -> None:
        self.assertTrue(all(set(row) == {"id", "context"} for row in self.inputs))
        expected = [
            {"id": row["id"], "context": row["context"]}
            for row in self.dataset["samples"]
        ]
        self.assertEqual(self.inputs, expected)

    def test_public_manifests_hide_text_and_triples(self) -> None:
        for payload in (self.manifest, self.eval_manifest):
            serialized = json.dumps(payload, ensure_ascii=False)
            self.assertFalse(payload["meta"]["contains_source_text"])
            self.assertFalse(payload["meta"]["contains_protected_gold"])
            self.assertNotIn('"context":', serialized)
            self.assertNotIn('"gold_triples":', serialized)

    def test_evaluation_is_frozen_fail_closed_without_calibration(self) -> None:
        meta = self.eval_manifest["meta"]
        locked = self.eval_manifest["locked_protocol"]
        self.assertEqual(meta["planned_units"], 60)
        self.assertEqual(meta["estimated_external_calls"], 60)
        self.assertFalse(meta["confirmatory_eligible"])
        self.assertFalse(meta["scientific_deployable"])
        self.assertIsNone(locked["calibrator"])
        self.assertFalse(locked["allow_prompt_tuning"])
        self.assertFalse(locked["allow_threshold_tuning"])
        self.assertFalse(locked["allow_calibration"])

    def test_build_is_byte_deterministic(self) -> None:
        dataset, inputs, manifest, eval_manifest = holdout.build_package()
        self.assertEqual(dataset, self.dataset_bytes)
        self.assertEqual(inputs, self.inputs_bytes)
        self.assertEqual(manifest, self.manifest_bytes)
        self.assertEqual(eval_manifest, self.eval_manifest)

    def test_runner_audit_finds_no_prefrozen_cache(self) -> None:
        hits, replayable = runner.cache_status(
            self.dataset, self.eval_manifest, "official-transfer-test-no-cache"
        )
        self.assertEqual((hits, replayable), (0, 0))

    def test_official_retry_state_accumulates_only_for_same_prompt(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol": blind.PROTOCOL,
                        "prompt_sha256": "prompt-a",
                        "attempts": 2,
                        "errors": ["first", "second"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                blind.previous_attempt_state(path, prompt_sha256="prompt-a"),
                (2, ["first", "second"]),
            )
            self.assertEqual(
                blind.previous_attempt_state(path, prompt_sha256="prompt-b"),
                (0, []),
            )

    def test_official_successful_retry_retains_attempt_errors(self) -> None:
        import tempfile
        from unittest.mock import patch

        class RetryLLM:
            def __init__(self) -> None:
                self.calls = 0

            def complete(self, prompt, *, task, meta):
                self.calls += 1
                text = "not-json" if self.calls == 1 else '{"triples":[]}'
                return type("Response", (), {"text": text})()

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "cache.json"
            with patch.object(blind, "cache_path", return_value=cache):
                result = blind.predict_one(
                    llm=RetryLLM(),
                    sample={"id": "sample", "context": "context"},
                    index=1,
                    task="a",
                    system="open_vocab",
                    model="test-model",
                    max_context_chars=6000,
                    confidence_threshold=0.85,
                    implicit_confidence_threshold=0.95,
                    calibrated_confidence_threshold=0.5,
                    max_nontax=8,
                    max_implicit_gap_chars=600,
                    calibrator=None,
                    parse_retries=2,
                    refresh_cache=False,
                )
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(result["attempts_this_run"], 2)
            self.assertEqual(len(result["attempt_errors_this_run"]), 1)
            self.assertEqual(result["errors"], result["attempt_errors_this_run"])

    def test_error_audit_entity_key_is_conservative_and_versioned(self) -> None:
        self.assertEqual(
            error_audit.diagnostic_entity_key(
                "Delivery-Mode Freights (Delivery Method)"
            ),
            "delivery mode freight",
        )
        self.assertEqual(
            error_audit.offline.SURFACE_MORPHOLOGY_VERSION,
            "relation_surface_morphology_v1",
        )

    def test_error_audit_separates_predicate_surface_and_endpoint_failures(self) -> None:
        gold = {
            ("leaf item", "disjoint with", "root node"),
            ("acid", "has role", "donor"),
        }
        cases = {
            ("leaf item", "disjoint with", "root node"): "exact",
            ("root node", "disjoint with", "leaf item"): (
                "symmetric_endpoints_reversed"
            ),
            ("leaf item", "part_of", "root node"): (
                "endpoints_exact_predicate_mismatch"
            ),
            ("leaf items", "disjoint with", "root node"): (
                "surface_endpoint_variant_same_predicate"
            ),
            ("acid", "related to", "unknown"): "one_endpoint_exact",
            ("unknown", "related to", "other"): "no_local_gold_overlap",
        }
        for triple, expected in cases.items():
            with self.subTest(triple=triple):
                self.assertEqual(
                    error_audit.classify_prediction(triple, gold)[0], expected
                )

    def test_gold_coverage_distinguishes_missing_predicate_from_missing_entities(self) -> None:
        gold = ("leaf node", "disjoint with", "root node")
        self.assertEqual(
            error_audit.classify_gold_coverage(
                gold, {("leaf node", "part_of", "root node")}
            ),
            "exact_endpoints_wrong_predicate",
        )
        self.assertEqual(
            error_audit.classify_gold_coverage(
                gold, {("leaf node", "part_of", "unrelated")}
            ),
            "one_endpoint_generated",
        )


if __name__ == "__main__":
    unittest.main()
