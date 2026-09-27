from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import build_nonce_relation_holdout as holdout
import run_nonce_relation_holdout as holdout_runner
import run_official_blind_eval as blind


class NonceRelationHoldoutTests(unittest.TestCase):
    def test_holdout_is_balanced_and_disjoint_from_official_2026(self) -> None:
        _, _, _, manifest = holdout.build_package(holdout.DEFAULT_SEED)
        balance = manifest["balance_audit"]
        self.assertEqual(balance["samples"], 30)
        self.assertEqual(balance["gold_relations"], 120)
        self.assertEqual(balance["hard_negative_controls"], 60)
        self.assertEqual(set(balance["predicate_counts"].values()), {20})
        self.assertEqual(
            balance["evidence_mode_counts"],
            {"cross_sentence": 60, "explicit": 60},
        )
        overlap = manifest["official_overlap_audit"]
        self.assertEqual(overlap["official_2026_entity_string_overlap_count"], 0)
        self.assertEqual(overlap["official_2026_exact_gold_triple_overlap_count"], 0)

    def test_manifest_excludes_text_and_gold(self) -> None:
        _, _, _, manifest = holdout.build_package(holdout.DEFAULT_SEED)
        serialized = json.dumps(manifest, ensure_ascii=False)
        self.assertFalse(manifest["meta"]["contains_source_text"])
        self.assertFalse(manifest["meta"]["contains_protected_gold"])
        self.assertNotIn("Controlled fictional relation record", serialized)
        self.assertNotIn('"context":', serialized)
        self.assertNotIn('"triple":', serialized)
        self.assertNotIn(
            holdout.entity_name(holdout.DEFAULT_SEED, 0, 0), serialized
        )

    def test_package_is_byte_deterministic(self) -> None:
        first = holdout.build_package(holdout.DEFAULT_SEED)
        second = holdout.build_package(holdout.DEFAULT_SEED)
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        self.assertEqual(first[3], second[3])

    def test_evaluation_manifest_hides_text_gold_and_freezes_sixty_units(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            package_dir = Path(directory) / "package"
            _, dataset_bytes, inputs_bytes, package_manifest = holdout.build_package(
                holdout.DEFAULT_SEED
            )
            package_dir.mkdir(parents=True)
            (package_dir / "dataset.json").write_bytes(dataset_bytes)
            (package_dir / "model_inputs.jsonl").write_bytes(inputs_bytes)
            (package_dir / "manifest.json").write_bytes(
                holdout._json_bytes(package_manifest)
            )
            args = holdout_runner.build_parser().parse_args(
                [
                    "--audit-only",
                    "--package-dir",
                    str(package_dir),
                    "--model",
                    "nonce-test-model-no-cache",
                ]
            )
            manifest = holdout_runner.build_manifest(args)
        serialized = json.dumps(manifest, ensure_ascii=False)
        self.assertEqual(manifest["meta"]["planned_units"], 60)
        self.assertEqual(manifest["meta"]["estimated_external_calls"], 60)
        self.assertFalse(manifest["meta"]["contains_source_text"])
        self.assertFalse(manifest["meta"]["contains_protected_gold"])
        self.assertNotIn("Controlled fictional relation record", serialized)
        self.assertNotIn(
            holdout.entity_name(holdout.DEFAULT_SEED, 0, 0), serialized
        )
        self.assertIsNone(manifest["locked_protocol"]["calibrator"])

    def test_summary_precommits_predicate_and_evidence_diagnostics(self) -> None:
        gold = [
            ["entity_a", "part_of", "entity_b"],
            ["entity_c", "regulates", "entity_d"],
        ]
        predicted = [["entity_a", "part_of", "entity_b"]]
        record = {
            "status": "ok",
            "latency_sec": 0.1,
            "attempts": 1,
            "triples": predicted,
            "metrics": blind.evaluate_sample(gold, predicted),
            "hard_negative_controls": [["entity_e", "regulates", "entity_f"]],
            "_gold_relation_annotations": [
                {
                    "triple": gold[0],
                    "evidence_mode": "explicit",
                    "evidence": ["span"],
                },
                {
                    "triple": gold[1],
                    "evidence_mode": "cross_sentence",
                    "evidence": ["span one", "span two"],
                },
            ],
        }
        summary = holdout_runner.summarize([record])
        self.assertEqual(summary["by_predicate"]["part_of"]["tp"], 1)
        self.assertEqual(summary["by_predicate"]["regulates"]["fn"], 1)
        self.assertEqual(
            summary["gold_recall_by_evidence_mode"]["explicit"]["recall"], 1.0
        )
        self.assertEqual(
            summary["gold_recall_by_evidence_mode"]["cross_sentence"]["recall"],
            0.0,
        )

    def test_retry_state_accumulates_only_for_matching_frozen_prompt(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            path.write_text(
                json.dumps(
                    {
                        "protocol": holdout_runner.PROTOCOL,
                        "prompt_sha256": "prompt-a",
                        "attempts": 2,
                        "errors": ["first", "second"],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                holdout_runner.previous_attempt_state(path, "prompt-a"),
                (2, ["first", "second"]),
            )
            self.assertEqual(
                holdout_runner.previous_attempt_state(path, "prompt-b"),
                (0, []),
            )


if __name__ == "__main__":
    unittest.main()
