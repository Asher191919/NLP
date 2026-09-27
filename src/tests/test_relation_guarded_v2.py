from __future__ import annotations

import json
import sys
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_official_blind_eval as runner
from ontobridg.relations.calibration import BetaIsotonicCalibrator
from ontobridg.relations.canonicalization import (
    build_entity_inventory,
    canonicalize_triple,
)


class CanonicalizationTests(unittest.TestCase):
    def test_canonicalizes_entities_and_predicate_without_gold(self) -> None:
        sample = {
            "context": "A car wheel is part of a motor car.",
            "terms": ["car wheel"],
            "types": ["motor car"],
            "initial-primitive-ontology-triples": [],
            "extended-primitive-ontology-triples": [
                ["SECRET_GOLD", "SECRET_RELATION", "SECRET_OBJECT"]
            ],
        }
        inventory = build_entity_inventory(sample, task="b")
        canonical, trace = canonicalize_triple(
            ["car_wheel", "is part-of", "motor car"],
            context=sample["context"],
            entity_inventory=inventory,
        )
        self.assertEqual(canonical, ["car wheel", "part_of", "motor car"])
        self.assertTrue(trace["changed"])
        self.assertNotIn("secret_gold", inventory)

    def test_directional_alias_reverses_endpoints(self) -> None:
        canonical, trace = canonicalize_triple(
            ["primordium", "gives rise to", "flower"],
            context="A primordium gives rise to a flower.",
        )
        self.assertEqual(canonical, ["flower", "develops_from", "primordium"])
        self.assertTrue(trace["reversed"])

    def test_taxonomy_and_typing_aliases_use_official_core_predicates(self) -> None:
        for raw, expected in (
            ("is_a", "is-a"),
            ("subclass_of", "is-a"),
            ("kind_of", "is-a"),
            ("instance_of", "instance-of"),
        ):
            with self.subTest(raw=raw):
                canonical, trace = canonicalize_triple(
                    ["child", raw, "parent"],
                    context="A child and parent are defined here.",
                )
                self.assertEqual(canonical, ["child", expected, "parent"])
                self.assertTrue(trace["changed"])

    def test_camel_case_ontology_predicates_are_canonicalized(self) -> None:
        for raw, expected in (
            ("equivalentClass", "equivalent class"),
            ("disjointWith", "disjoint with"),
            ("seeAlso", "see also"),
            ("databaseCrossReference", "database_cross_reference"),
        ):
            with self.subTest(raw=raw):
                canonical, trace = canonicalize_triple(
                    ["left", raw, "right"],
                    context="left and right",
                )
                self.assertEqual(canonical, ["left", expected, "right"])
                self.assertTrue(trace["changed"])


class GuardedV2Tests(unittest.TestCase):
    def test_prompt_requests_two_span_implicit_evidence(self) -> None:
        prompt = runner.build_prompt(
            {"context": "Alpha occurs. Beta occurs."},
            "a",
            "open_vocab_guarded_v2",
            6000,
        )
        self.assertIn('"evidence_mode":"explicit|cross_sentence"', prompt)
        self.assertIn('"evidence":["verbatim quote 1"', prompt)
        self.assertIn("cross-sentence candidates require", prompt)

    def test_relation_only_prompt_excludes_core_triples(self) -> None:
        prompt = runner.build_prompt(
            {"context": "Soybean substance derives from soybean."},
            "a",
            "open_vocab_guarded_v3_relations",
            6000,
        )
        self.assertIn("RELATION-ONLY PASS", prompt)
        self.assertIn("never emit instance-of", prompt)
        self.assertIn("origin/derivation", prompt)

    def test_axiom_relation_prompt_covers_equivalence_and_disjointness(self) -> None:
        prompt = runner.build_prompt(
            {"context": "Alpha and beta are mutually exclusive categories."},
            "a",
            "open_vocab_guarded_v4_axioms",
            6000,
        )
        self.assertIn("RELATION-ONLY AXIOM PASS", prompt)
        self.assertIn("equivalence, disjointness", prompt)
        self.assertIn("equivalent class and disjoint with", prompt)
        self.assertIn("never emit instance-of", prompt)

    def test_axiom_inventory_prompt_excludes_other_relation_families(self) -> None:
        prompt = runner.build_prompt(
            {"context": "Alpha and beta are mutually exclusive categories."},
            "a",
            "open_vocab_guarded_v5_axioms_only",
            6000,
        )
        self.assertIn("RELATION-ONLY AXIOM-INVENTORY PASS", prompt)
        self.assertIn("emit only equivalent class or disjoint with", prompt)
        self.assertIn("never emit any other predicate", prompt)

    def test_predicate_router_prompt_locks_all_prespecified_families(self) -> None:
        prompt = runner.build_prompt(
            {"context": "Hydroxide is the conjugate base of water."},
            "a",
            "open_vocab_guarded_v6_predicate_routed",
            6000,
        )
        self.assertIn("RELATION-ONLY FROZEN PREDICATE-FAMILY ROUTER", prompt)
        for predicate in (
            "equivalent class",
            "disjoint with",
            "type",
            "database_cross_reference",
            "has role",
            "same as",
            "is_conjugate_acid_of",
            "is_conjugate_base_of",
        ):
            self.assertIn(predicate, prompt)
        self.assertIn("Never emit taxonomy, instance-of, part-whole", prompt)
        self.assertIn('Return {"triples":[]}', prompt)

    def test_accepts_explicit_relation_after_canonicalization(self) -> None:
        context = "A car wheel is part of a motor car."
        payload = {
            "triples": [
                {
                    "subject": "car_wheel",
                    "predicate": "is part of",
                    "object": "motor car",
                    "confidence": 0.90,
                    "evidence_mode": "explicit",
                    "evidence": ["car wheel is part of a motor car"],
                }
            ]
        }
        triples, diagnostics = runner.parse_guarded_v2_model_payload(
            json.dumps(payload),
            context=context,
            sample={"context": context},
            task="a",
        )
        self.assertEqual(triples, [["car wheel", "part_of", "motor car"]])
        self.assertEqual(diagnostics["canonicalized_candidates"], 1)
        self.assertEqual(diagnostics["accepted_nontax"], 1)

    def test_taxonomy_alias_is_core_and_does_not_consume_nontax_cap(self) -> None:
        context = "A sedan is a car. A wheel is part of the car."
        payload = {
            "triples": [
                {
                    "subject": "sedan",
                    "predicate": "subclass_of",
                    "object": "car",
                    "confidence": 0.99,
                    "evidence_mode": "explicit",
                    "evidence": ["A sedan is a car"],
                },
                {
                    "subject": "wheel",
                    "predicate": "part_of",
                    "object": "car",
                    "confidence": 0.99,
                    "evidence_mode": "explicit",
                    "evidence": ["A wheel is part of the car"],
                },
            ]
        }
        triples, diagnostics = runner.parse_guarded_v2_model_payload(
            json.dumps(payload),
            context=context,
            sample={"context": context},
            task="a",
            max_nontax=1,
        )
        self.assertEqual(
            triples,
            [["sedan", "is-a", "car"], ["wheel", "part_of", "car"]],
        )
        self.assertEqual(diagnostics["accepted_core"], 1)
        self.assertEqual(diagnostics["raw_nontax"], 1)
        self.assertEqual(diagnostics["accepted_nontax"], 1)

    def test_cross_sentence_evidence_uses_stricter_threshold(self) -> None:
        context = (
            "The infructescence axis supports developing fruit clusters. "
            "The mature fruit structure is known as the infructescence."
        )

        def run(confidence: float) -> tuple[list[list[str]], dict]:
            payload = {
                "triples": [
                    {
                        "subject": "infructescence axis",
                        "predicate": "part_of",
                        "object": "infructescence",
                        "confidence": confidence,
                        "evidence_mode": "cross_sentence",
                        "evidence": [
                            "The infructescence axis supports developing fruit clusters",
                            "The mature fruit structure is known as the infructescence",
                        ],
                    }
                ]
            }
            return runner.parse_guarded_v2_model_payload(
                json.dumps(payload),
                context=context,
                sample={"context": context},
                task="a",
                implicit_confidence_threshold=0.95,
            )

        accepted, accepted_diagnostics = run(0.96)
        rejected, rejected_diagnostics = run(0.94)
        self.assertEqual(
            accepted,
            [["infructescence axis", "part_of", "infructescence"]],
        )
        self.assertEqual(accepted_diagnostics["accepted_nontax"], 1)
        self.assertIsNotNone(
            accepted_diagnostics["accepted_nontax_candidates"][0]["evidence_gap_chars"]
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            rejected_diagnostics["rejection_reason_counts"]["low_confidence"], 1
        )

    def test_postprocess_hash_includes_parser_version(self) -> None:
        kwargs = {
            "confidence_threshold": 0.85,
            "implicit_confidence_threshold": 0.95,
            "calibrated_confidence_threshold": 0.5,
            "max_nontax": 8,
            "max_implicit_gap_chars": 600,
            "calibrator": None,
        }
        original = runner.postprocess_sha256(**kwargs)
        with patch.object(
            runner,
            "GUARDED_V2_POSTPROCESS_VERSION",
            "guarded_v2_postprocess_test_change",
        ):
            changed = runner.postprocess_sha256(**kwargs)
        self.assertNotEqual(original, changed)

    def test_calibration_predicate_allowlist_rejects_unsupported_relation(self) -> None:
        context = "A wheel is part of a car. A driver operates the car."
        payload = {
            "triples": [
                {
                    "subject": "wheel",
                    "predicate": "part_of",
                    "object": "car",
                    "confidence": 0.99,
                    "evidence_mode": "explicit",
                    "evidence": ["wheel is part of a car"],
                },
                {
                    "subject": "driver",
                    "predicate": "operates",
                    "object": "car",
                    "confidence": 0.99,
                    "evidence_mode": "explicit",
                    "evidence": ["driver operates the car"],
                },
            ]
        }
        triples, diagnostics = runner.parse_guarded_v2_model_payload(
            json.dumps(payload),
            context=context,
            sample={"context": context},
            task="a",
            allowed_nontax_predicates="is part of,develops_from",
        )
        self.assertEqual(triples, [["wheel", "part_of", "car"]])
        self.assertEqual(
            diagnostics["rejection_reason_counts"][
                "predicate_not_calibration_supported"
            ],
            1,
        )
        self.assertEqual(
            diagnostics["allowed_nontax_predicates"],
            ["develops_from", "part_of"],
        )

    def test_postprocess_hash_includes_predicate_allowlist(self) -> None:
        kwargs = {
            "confidence_threshold": 0.85,
            "implicit_confidence_threshold": 0.95,
            "calibrated_confidence_threshold": 0.5,
            "max_nontax": 8,
            "max_implicit_gap_chars": 600,
            "calibrator": None,
        }
        unrestricted = runner.postprocess_sha256(**kwargs)
        restricted = runner.postprocess_sha256(
            **kwargs, allowed_nontax_predicates=["part_of"]
        )
        self.assertNotEqual(unrestricted, restricted)

    def test_deployable_calibrator_can_reject_overconfident_candidate(self) -> None:
        calibrator = BetaIsotonicCalibrator().fit(
            [0.80, 0.90, 0.95, 1.00],
            [0, 0, 1, 1],
        )
        context = "A car wheel is part of a motor car."
        payload = {
            "triples": [
                {
                    "subject": "car wheel",
                    "predicate": "part_of",
                    "object": "motor car",
                    "confidence": 0.90,
                    "evidence_mode": "explicit",
                    "evidence": ["car wheel is part of a motor car"],
                }
            ]
        }
        triples, diagnostics = runner.parse_guarded_v2_model_payload(
            json.dumps(payload),
            context=context,
            sample={"context": context},
            task="a",
            calibrated_confidence_threshold=0.50,
            calibrator=calibrator,
        )
        self.assertEqual(triples, [])
        self.assertTrue(diagnostics["calibration_applied"])
        self.assertEqual(
            diagnostics["rejection_reason_counts"]["low_calibrated_confidence"], 1
        )

    def test_degenerate_calibrator_is_not_applied(self) -> None:
        calibrator = BetaIsotonicCalibrator().fit([0.9, 1.0], [0, 0])
        context = "A car wheel is part of a motor car."
        payload = {
            "triples": [
                {
                    "subject": "car wheel",
                    "predicate": "part_of",
                    "object": "motor car",
                    "confidence": 0.90,
                    "evidence_mode": "explicit",
                    "evidence": ["car wheel is part of a motor car"],
                }
            ]
        }
        triples, diagnostics = runner.parse_guarded_v2_model_payload(
            json.dumps(payload),
            context=context,
            sample={"context": context},
            task="a",
            calibrator=calibrator,
        )
        self.assertEqual(triples, [["car wheel", "part_of", "motor car"]])
        self.assertFalse(diagnostics["calibration_applied"])
        self.assertEqual(diagnostics["calibration_status"], "degenerate_too_few_positives")

    def test_postprocess_change_reuses_raw_generation_without_api_call(self) -> None:
        context = "A car wheel is part of a motor car."
        sample = {"id": "cache-test", "context": context}
        raw_text = json.dumps(
            {
                "triples": [
                    {
                        "subject": "car wheel",
                        "predicate": "part_of",
                        "object": "motor car",
                        "confidence": 0.90,
                        "evidence_mode": "explicit",
                        "evidence": ["car wheel is part of a motor car"],
                    }
                ]
            }
        )
        prompt = runner.build_prompt(sample, "a", "open_vocab_guarded_v2", 6000)

        class NoCallLLM:
            def complete(self, *args, **kwargs):  # pragma: no cover - failure path
                raise AssertionError("cached raw generation should be reprocessed locally")

        cache = SRC / "tests" / f"cache_test_{uuid.uuid4().hex}.json"
        try:
            cache.write_text(
                json.dumps(
                    {
                        "protocol": runner.PROTOCOL,
                        "status": "ok",
                        "prompt_sha256": runner._sha256(prompt),
                        "postprocess_sha256": "old-signature",
                        "raw_text": raw_text,
                        "generation_config": {},
                        "triples": [],
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(runner, "cache_path", return_value=cache):
                record = runner.predict_one(
                    llm=NoCallLLM(),
                    sample=sample,
                    index=0,
                    task="a",
                    system="open_vocab_guarded_v2",
                    model="test-model",
                    max_context_chars=6000,
                    confidence_threshold=0.85,
                    implicit_confidence_threshold=0.95,
                    calibrated_confidence_threshold=0.5,
                    max_nontax=8,
                    max_implicit_gap_chars=600,
                    calibrator=None,
                    parse_retries=1,
                    refresh_cache=False,
                )
            self.assertEqual(record["triples"], [["car wheel", "part_of", "motor car"]])
            self.assertIn("locally_reprocessed_at", record)
            self.assertNotEqual(record["postprocess_sha256"], "old-signature")
        finally:
            cache.unlink(missing_ok=True)


class CalibrationTests(unittest.TestCase):
    def test_report_level_sample_support_gate_disables_calibrator(self) -> None:
        calibrator = BetaIsotonicCalibrator().fit(
            [0.80, 0.90, 0.95, 1.00],
            [0, 0, 1, 1],
        )
        restored = runner.calibrator_from_artifact_payload(
            {
                "calibrator": calibrator.to_dict(),
                "deployment_support": {
                    "deployable": False,
                    "status": "insufficient_independent_positive_samples",
                },
            }
        )
        self.assertFalse(restored.deployable)
        self.assertEqual(
            restored.status, "insufficient_independent_positive_samples"
        )

    def test_calibrator_is_monotonic_and_round_trips(self) -> None:
        calibrator = BetaIsotonicCalibrator().fit(
            [0.1, 0.2, 0.8, 0.9],
            [0, 0, 1, 1],
        )
        self.assertTrue(calibrator.deployable)
        probabilities = [calibrator.calibrate(score) for score in (0.1, 0.5, 0.9)]
        self.assertEqual(probabilities, sorted(probabilities))

        restored = BetaIsotonicCalibrator.from_dict(calibrator.to_dict())
        self.assertEqual(restored.status, "ok")
        self.assertAlmostEqual(restored.calibrate(0.9), calibrator.calibrate(0.9))

    def test_calibrator_refuses_no_positive_dataset(self) -> None:
        calibrator = BetaIsotonicCalibrator().fit([0.9, 0.95, 1.0], [0, 0, 0])
        self.assertFalse(calibrator.deployable)
        self.assertEqual(calibrator.status, "degenerate_too_few_positives")


if __name__ == "__main__":
    unittest.main()
