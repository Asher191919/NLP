from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_relation_calibration as calibration
import run_cross_year_relation_calibration as cross_year
import run_relation_v2_offline as offline
import run_annotated_relation_calibration as annotated_calibration
import run_gold_relation_grouped_cv as gold_grouped_cv
import run_v5_confirmatory as v5_confirmatory
import analyze_grouped_relation_calibration as uncertainty
import analyze_v5_confirmatory as v5_analysis
import run_v5_semantic_secondary as v5_semantic
import relation_annotation_workflow as annotation_workflow
from ontobridg.relations.calibration import BetaIsotonicCalibrator


class CalibrationSelectionTests(unittest.TestCase):
    def test_v5_confirmatory_selection_is_balanced_and_disjoint(self) -> None:
        args = v5_confirmatory.build_parser().parse_args(["--audit-only"])
        selected, metadata, pool, excluded, _ = v5_confirmatory.selected_for_args(args)
        indices = {index for index, _ in selected}
        self.assertEqual(len(indices), 24)
        self.assertFalse(indices & excluded)
        self.assertEqual(
            pool["selected_assigned_strata"],
            {"equivalent class": 12, "disjoint with": 12},
        )
        self.assertTrue(
            all(metadata[index]["selection_purpose"] == "untouched_v5_confirmatory" for index in indices)
        )

    def test_v5_confirmatory_manifest_contains_no_source_text(self) -> None:
        args = v5_confirmatory.build_parser().parse_args(["--audit-only"])
        manifest = v5_confirmatory.build_manifest(args)
        serialized = json.dumps(manifest, ensure_ascii=False).lower()
        self.assertFalse(manifest["meta"]["contains_source_text"])
        self.assertTrue(manifest["meta"]["disjoint_from_all_exclusions"])
        self.assertFalse(
            manifest["locked_configuration"]["surface_canonicalization"]
        )
        self.assertFalse(
            manifest["locked_configuration"]["bidirectional_serialization"]
        )
        self.assertNotIn("primitive-ontology-triples\": [[", serialized)

    def test_v5_confirmatory_runtime_rejects_locked_config_drift(self) -> None:
        args = v5_confirmatory.build_parser().parse_args(["--audit-only"])
        manifest = v5_confirmatory.build_manifest(args)
        args.max_nontax += 1
        with self.assertRaisesRegex(ValueError, "does not match frozen manifest"):
            v5_confirmatory.validate_manifest(manifest, args)

    def test_v5_confirmatory_manifest_is_write_once(self) -> None:
        args = v5_confirmatory.build_parser().parse_args(["--audit-only"])
        manifest = v5_confirmatory.build_manifest(args)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            first, created = v5_confirmatory.preserve_or_write_frozen_manifest(
                path, manifest
            )
            self.assertTrue(created)
            changed = json.loads(json.dumps(manifest))
            original_hits = manifest["meta"]["estimated_generation_cache_hits"]
            changed["meta"]["estimated_generation_cache_hits"] = (
                0 if original_hits else 24
            )
            changed["meta"]["estimated_external_calls"] = (
                24 if original_hits else 0
            )
            second, created = v5_confirmatory.preserve_or_write_frozen_manifest(
                path, changed
            )
            self.assertFalse(created)
            self.assertEqual(first, second)
            self.assertEqual(
                second["meta"]["estimated_generation_cache_hits"], original_hits
            )

    def test_cross_year_pool_is_strictly_disjoint_and_deduplicated(self) -> None:
        selected, _, pool = cross_year.build_cross_year_samples(seed=20260923)
        gold = [
            tuple(triple)
            for _, sample in selected
            for triple in sample["primitive-ontology-triples"]
        ]
        self.assertEqual(pool["gold_triple_overlap_with_2026_train"], 0)
        self.assertFalse(pool["duplicate_gold_across_selected_documents"])
        self.assertEqual(len(gold), len(set(gold)))
        self.assertEqual(len(selected), pool["selected_documents"])
        self.assertEqual(len(gold), pool["selected_unique_gold_triples"])
        for _, sample in selected:
            context = blind_compact = calibration.blind._compact(sample["context"])
            self.assertTrue(context)
            for subject, _, obj in sample["primitive-ontology-triples"]:
                self.assertIn(calibration.blind._compact(subject), blind_compact)
                self.assertIn(calibration.blind._compact(obj), blind_compact)

    def test_cross_year_overlap_key_normalizes_symmetric_direction(self) -> None:
        forward = ("alpha", "disjoint with", "beta")
        reverse = ("beta", "disjoint with", "alpha")
        self.assertEqual(
            cross_year._overlap_key(forward),
            cross_year._overlap_key(reverse),
        )

    def test_cross_year_audit_manifest_contains_no_source_text(self) -> None:
        args = cross_year.build_parser().parse_args(["--audit-only"])
        manifest = cross_year.build_audit_manifest(args)
        serialized = json.dumps(manifest, ensure_ascii=False).lower()
        self.assertFalse(manifest["meta"]["contains_source_text"])
        self.assertTrue(manifest["meta"]["disjoint_from_all_2026_train_gold"])
        self.assertNotIn("layers of the atmosphere", serialized)

    def test_dedicated_runner_disables_calibrator_during_generation(self) -> None:
        args = calibration.build_parser().parse_args([])
        args.calibration_file = None
        self.assertIsNone(args.calibration_file)
        self.assertEqual(args.prior_failed_connection_attempts, 0)
        self.assertEqual(args.prior_failed_unit_attempts, 0)
        self.assertEqual(args.prior_failed_low_level_attempts, 0)

    def test_parse_sample_indices_is_sorted_and_deduplicated(self) -> None:
        self.assertEqual(
            calibration.parse_sample_indices("2424, 938,2077,938"),
            [938, 2077, 2424],
        )

    def test_selection_is_disjoint_and_prioritizes_extractable_supported_gold(self) -> None:
        samples = [
            (
                31,
                {
                    "id": "excluded",
                    "context": "A wheel is part of a car.",
                    "primitive-ontology-triples": [["wheel", "part_of", "car"]],
                },
            ),
            (
                32,
                {
                    "id": "supported",
                    "context": "A petal is part of a flower.",
                    "primitive-ontology-triples": [["petal", "part_of", "flower"]],
                },
            ),
            (
                33,
                {
                    "id": "unsupported",
                    "context": "A bee pollinates a flower.",
                    "primitive-ontology-triples": [["bee", "pollinates", "flower"]],
                },
            ),
        ]
        selected, metadata, pool = calibration.select_calibration_samples(
            samples,
            task="a",
            count=1,
            seed=7,
            excluded_indices={31},
            max_context_chars=6000,
        )
        self.assertEqual([index for index, _ in selected], [32])
        self.assertEqual(metadata[32]["visible_supported_gold"], 1)
        self.assertEqual(pool["eligible_samples"], 2)

    def test_prompt_window_controls_endpoint_visibility(self) -> None:
        sample = {
            "context": "alpha appears first; beta appears later",
            "primitive-ontology-triples": [["alpha", "part_of", "beta"]],
        }
        short = calibration.sample_relation_features(
            sample, task="a", max_context_chars=20
        )
        full = calibration.sample_relation_features(
            sample, task="a", max_context_chars=6000
        )
        self.assertEqual(short["visible_nontax_gold"], 0)
        self.assertEqual(full["visible_supported_gold"], 1)

    def test_exclusion_ledger_unions_results_manifests_and_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result_path = root / "result.json"
            manifest_path = root / "manifest.json"
            result_path.write_text(
                json.dumps(
                    {
                        "meta": {"protocol": "result-protocol"},
                        "tasks": {
                            "task_a": {"meta": {"selected_indices": [2, 10]}},
                            "task_b": {"meta": {"selected_indices": [11]}},
                        },
                    }
                ),
                encoding="utf-8",
            )
            manifest_path.write_text(
                json.dumps(
                    {
                        "meta": {"protocol": "annotation-protocol"},
                        "samples": [
                            {"task": "a", "source_index": 12},
                            {"task": "b", "source_index": 13},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            excluded, ledger = calibration.load_exclusion_ledger(
                result_paths=[result_path],
                annotation_manifest_paths=[manifest_path],
                prefix=3,
            )
            self.assertEqual(excluded["a"], {0, 1, 2, 10, 12})
            self.assertEqual(excluded["b"], {0, 1, 2, 11, 13})
            self.assertEqual([row["kind"] for row in ledger], [
                "prefix",
                "result",
                "annotation_manifest",
            ])
            self.assertEqual(ledger[1]["new_task_a_indices"], 1)
            self.assertEqual(ledger[2]["new_task_b_indices"], 1)

    def test_open_exact_predicate_is_distinguished_from_known_supported_set(self) -> None:
        features = calibration.sample_relation_features(
            {
                "context": "A bee pollinates a flower.",
                "primitive-ontology-triples": [["bee", "pollinates", "flower"]],
            },
            task="a",
            max_context_chars=6000,
        )
        self.assertEqual(features["visible_supported_gold"], 0)
        self.assertEqual(features["visible_guard_compatible_exact_gold"], 1)
        self.assertEqual(features["visible_lexically_explicit_gold"], 1)
        self.assertEqual(features["visible_nontax_predicates"], ["pollinates"])


class OfflineCalibrationSplitTests(unittest.TestCase):
    def test_structural_projection_rules_are_gold_blind(self) -> None:
        projections = dict(
            offline.structural_projections_for_triple(
                ["soybean (whole)", "is-a", "soybean substance"],
                evidence=["soybean (whole) is classified as soybean substance"],
            )
        )
        self.assertEqual(
            projections["whole_substance_derivation"],
            ["soybean substance", "derives from", "soybean (whole)"],
        )

        nested = dict(
            offline.structural_projections_for_triple(
                ["astronomical body", "is-a", "astronomical object"],
                evidence=["occupying a nested position within that hierarchy"],
            )
        )
        self.assertEqual(
            nested["nested_taxonomy_inverse_has_part"],
            ["astronomical object", "has part", "astronomical body"],
        )

    def test_projection_audit_requires_cross_sample_replication(self) -> None:
        candidates = [
            {
                "task": "a",
                "sample_index": 1,
                "split": "calibration",
                "rule": "replicated",
                "projected_triple": ["a", "derives from", "b"],
                "correct": True,
            },
            {
                "task": "a",
                "sample_index": 2,
                "split": "calibration",
                "rule": "replicated",
                "projected_triple": ["c", "derives from", "d"],
                "correct": True,
            },
            {
                "task": "a",
                "sample_index": 3,
                "split": "calibration",
                "rule": "single_sample",
                "projected_triple": ["e", "has part", "f"],
                "correct": True,
            },
        ]
        report = offline.audit_structural_projections(
            candidates,
            sample_keys={("a", 1), ("a", 2), ("a", 3)},
            base_candidates=[],
        )
        self.assertEqual(report["replicated_rules"], ["replicated"])
        self.assertEqual(report["selected_projection_summary"]["tp"], 2)
        self.assertFalse(report["deployment_support"]["deployable"])

    def test_merge_source_results_unions_and_deduplicates_samples(self) -> None:
        def payload(task: str, *indices: int) -> dict:
            return {
                "tasks": {
                    f"task_{task}": {
                        "systems": {
                            "open_vocab_guarded_v2": {
                                "samples": [
                                    {
                                        "sample_index": index,
                                        "cache": f"task-{task}-{index}.json",
                                    }
                                    for index in indices
                                ]
                            }
                        }
                    }
                }
            }

        merged = offline.merge_source_results(
            [
                (Path("a.json"), payload("a", 1, 2)),
                (Path("ab.json"), payload("a", 2)),
                (Path("ab.json"), payload("b", 3)),
            ],
            source_system="open_vocab_guarded_v2",
        )
        self.assertEqual(
            merged["tasks"]["task_a"]["meta"]["selected_indices"], [1, 2]
        )
        self.assertEqual(
            merged["tasks"]["task_b"]["meta"]["selected_indices"], [3]
        )

    def test_near_match_classifies_predicate_and_direction_mismatches(self) -> None:
        gold = {("leaf", "part_of", "plant")}
        predicate_type, predicate_gold = offline.classify_near_match(
            ["leaf", "belongs to", "plant"], gold
        )
        direction_type, direction_gold = offline.classify_near_match(
            ["plant", "part_of", "leaf"], gold
        )
        self.assertEqual(predicate_type, "endpoints_exact_predicate_mismatch")
        self.assertEqual(predicate_gold, [["leaf", "part_of", "plant"]])
        self.assertEqual(direction_type, "endpoints_reversed")
        self.assertEqual(direction_gold, [["leaf", "part_of", "plant"]])

    def test_symmetric_near_match_is_separate_from_directed_mismatch(self) -> None:
        gold = {("p wave", "equivalent class", "primary wave")}
        match_type, matched_gold = offline.classify_near_match(
            ["primary wave", "equivalent class", "p wave"], gold
        )
        self.assertEqual(match_type, "symmetric_endpoints_reversed")
        self.assertEqual(
            matched_gold,
            [["p wave", "equivalent class", "primary wave"]],
        )

    def test_relation_metrics_can_ignore_only_symmetric_endpoint_order(self) -> None:
        predicted = {
            ("a", 1): {
                ("primary wave", "equivalent class", "p wave"),
                ("plant", "part_of", "leaf"),
            }
        }
        gold = {
            ("a", 1): {
                ("p wave", "equivalent class", "primary wave"),
                ("leaf", "part_of", "plant"),
            }
        }
        exact = offline.relation_set_metrics(predicted, gold)
        semantic = offline.relation_set_metrics(
            predicted, gold, symmetric_order_invariant=True
        )
        self.assertEqual(exact["tp"], 0)
        self.assertEqual(semantic["tp"], 1)
        self.assertEqual(semantic["fp"], 1)
        self.assertEqual(semantic["fn"], 1)

    def test_surface_morphology_is_conservative_and_versioned(self) -> None:
        self.assertEqual(
            offline.singularize_entity_surface("ambient air quality standards"),
            "ambient air quality standard",
        )
        self.assertEqual(
            offline.singularize_entity_surface("marine ecosystems"),
            "marine ecosystem",
        )
        for invariant in (
            "nsps",
            "species",
            "physics",
            "process",
            "viruses",
            "classes",
            "houses",
        ):
            self.assertEqual(offline.singularize_entity_surface(invariant), invariant)
        self.assertEqual(
            offline.SURFACE_MORPHOLOGY_VERSION,
            "relation_surface_morphology_v1",
        )

    def test_cap_aware_expansion_preserves_base_before_adding_reversals(self) -> None:
        key = ("a", 1)
        ranked = {
            key: [
                (f"left {index}", "equivalent class", f"right {index}")
                for index in range(6)
            ]
        }
        expanded, added = offline.cap_aware_bidirectional_predictions(
            ranked,
            max_per_sample=8,
            surface_canonicalization=False,
        )
        self.assertEqual(len(expanded[key]), 8)
        self.assertEqual(added, 2)
        for triple in ranked[key]:
            self.assertIn(triple, expanded[key])
        self.assertIn(("right 0", "equivalent class", "left 0"), expanded[key])
        self.assertIn(("right 1", "equivalent class", "left 1"), expanded[key])

    def test_exact_transition_reports_recovered_and_lost_edges(self) -> None:
        key = ("a", 7)
        gold = {
            key: {
                ("alpha", "equivalent class", "beta"),
                ("gamma", "part_of", "delta"),
            }
        }
        baseline = {key: {("gamma", "part_of", "delta")}}
        variant = {key: {("alpha", "equivalent class", "beta")}}
        transition = offline.exact_match_transition(baseline, variant, gold)
        self.assertEqual(transition["recovered_exact"], 1)
        self.assertEqual(transition["lost_exact"], 1)
        self.assertEqual(
            transition["recovered"][0]["triple"],
            ["alpha", "equivalent class", "beta"],
        )

    def test_all_successful_samples_are_used_for_dedicated_calibration(self) -> None:
        result = {
            "tasks": {
                "task_a": {
                    "systems": {
                        "open_vocab_guarded_v2": {
                            "samples": [
                                {"sample_index": 3, "status": "ok"},
                                {"sample_index": 4, "status": "error"},
                            ]
                        }
                    }
                }
            }
        }
        chosen = offline.choose_all_successful_indices(
            result, source_system="open_vocab_guarded_v2"
        )
        self.assertEqual(chosen, {"a": {3}})

    def test_threshold_sweep_uses_only_accepted_calibration_candidates(self) -> None:
        calibrator = BetaIsotonicCalibrator().fit(
            [0.90, 0.90, 0.96, 0.97], [0, 1, 0, 1]
        )
        candidates = [
            {
                "confidence": 0.90,
                "canonical_correct": False,
                "postprocess_status": "accepted",
            },
            {
                "confidence": 0.90,
                "canonical_correct": True,
                "postprocess_status": "accepted",
            },
            {
                "confidence": 0.96,
                "canonical_correct": False,
                "postprocess_status": "accepted",
            },
            {
                "confidence": 0.97,
                "canonical_correct": True,
                "postprocess_status": "accepted",
            },
            {
                "confidence": 1.0,
                "canonical_correct": False,
                "postprocess_status": "rejected",
            },
        ]
        sweep = offline.calibration_threshold_sweep(candidates, calibrator)
        self.assertEqual(sweep["n_candidates"], 4)
        self.assertEqual(sweep["n_positives"], 2)
        self.assertIsNotNone(sweep["recommended_threshold"])

    def test_candidate_group_summary_tracks_accepted_positives(self) -> None:
        candidates = [
            {
                "task": "a",
                "canonical_triple": ["x", "part_of", "y"],
                "canonical_correct": True,
                "postprocess_status": "accepted",
            },
            {
                "task": "a",
                "canonical_triple": ["u", "part_of", "v"],
                "canonical_correct": False,
                "postprocess_status": "rejected",
            },
        ]
        grouped = offline.summarize_candidate_groups(candidates, field="predicate")
        self.assertEqual(
            grouped["part_of"],
            {"n": 2, "positives": 1, "accepted": 1, "accepted_positives": 1},
        )

    def test_predicate_policy_keeps_predicates_with_calibration_positives(self) -> None:
        candidates = [
            {
                "canonical_triple": ["x", "part_of", "y"],
                "canonical_correct": True,
                "postprocess_status": "accepted",
            },
            {
                "canonical_triple": ["u", "part_of", "v"],
                "canonical_correct": False,
                "postprocess_status": "accepted",
            },
            {
                "canonical_triple": ["a", "related to", "b"],
                "canonical_correct": False,
                "postprocess_status": "accepted",
            },
        ]
        policy = offline.calibration_predicate_policy(candidates)
        self.assertEqual(policy["allowed_predicates"], ["part_of"])
        self.assertEqual(policy["tp"], 1)
        self.assertEqual(policy["fp"], 1)

    def test_quantity_limited_candidate_is_in_calibration_population(self) -> None:
        candidates = [
            {
                "postprocess_status": "accepted",
                "rejection_reasons": [],
            },
            {
                "postprocess_status": "rejected",
                "rejection_reasons": ["quantity_limit"],
            },
            {
                "postprocess_status": "rejected",
                "rejection_reasons": ["evidence_not_verbatim"],
            },
        ]
        eligible = offline.deployment_eligible_candidates(candidates)
        self.assertEqual(eligible, candidates[:2])


class GroupedCalibrationCrossValidationTests(unittest.TestCase):
    @staticmethod
    def candidates() -> list[dict]:
        rows = []
        for sample_index in range(15):
            for candidate_index in range(2):
                rows.append(
                    {
                        "sample_key": f"sample-{sample_index:02d}",
                        "task": "a" if sample_index % 2 == 0 else "b",
                        "confidence": 0.85 + 0.05 * candidate_index,
                        "canonical_correct": (
                            sample_index < 10 and candidate_index == 0
                        ),
                        "raw_correct": sample_index < 10 and candidate_index == 0,
                        "canonicalization_changed": False,
                        "canonical_triple": [
                            f"subject-{sample_index}",
                            "part_of",
                            f"object-{candidate_index}",
                        ],
                        "postprocess_status": "accepted",
                        "rejection_reasons": [],
                    }
                )
        return rows

    def test_sample_group_never_crosses_fold_boundary(self) -> None:
        report = annotated_calibration.grouped_cross_validation_report(
            self.candidates(), n_folds=5, seed=11
        )
        for fold in report["folds"]:
            train = set(fold["train_sample_keys"])
            validation = set(fold["validation_sample_keys"])
            self.assertFalse(train & validation)
            self.assertEqual(fold["sample_overlap_count"], 0)

    def test_fold_assignment_is_deterministic_for_seed(self) -> None:
        first = annotated_calibration.assign_grouped_calibration_folds(
            self.candidates(), n_folds=5, seed=17
        )
        second = annotated_calibration.assign_grouped_calibration_folds(
            list(reversed(self.candidates())), n_folds=5, seed=17
        )
        self.assertEqual(first, second)

    def test_every_fold_has_validation_samples(self) -> None:
        report = annotated_calibration.grouped_cross_validation_report(
            self.candidates(), n_folds=5, seed=23
        )
        self.assertTrue(
            all(fold["validation_samples"] > 0 for fold in report["folds"])
        )

    def test_positive_samples_are_distributed_across_folds(self) -> None:
        report = annotated_calibration.grouped_cross_validation_report(
            self.candidates(), n_folds=5, seed=29
        )
        positive_counts = [
            fold["validation_positive_samples"] for fold in report["folds"]
        ]
        self.assertTrue(all(count > 0 for count in positive_counts))
        self.assertLessEqual(max(positive_counts) - min(positive_counts), 1)

    def test_oof_coverage_equals_eligible_candidate_count(self) -> None:
        report = annotated_calibration.grouped_cross_validation_report(
            self.candidates(), n_folds=5, seed=31
        )
        self.assertEqual(report["coverage"]["eligible_candidates"], 30)
        self.assertEqual(report["coverage"]["out_of_fold_predictions"], 30)
        self.assertTrue(report["coverage"]["each_candidate_scored_once"])
        self.assertEqual(len(report["out_of_fold_predictions"]), 30)
        self.assertTrue(
            all(
                "context" not in row and "canonical_triple" not in row
                for row in report["out_of_fold_predictions"]
            )
        )

    def test_each_fold_calibrator_is_fit_only_on_training_candidates(self) -> None:
        report = annotated_calibration.grouped_cross_validation_report(
            self.candidates(), n_folds=5, seed=37
        )
        for fold in report["folds"]:
            self.assertEqual(
                fold["calibrator"]["n_samples"], fold["train_candidates"]
            )
            self.assertEqual(
                fold["calibrator"]["n_positives"], fold["train_positives"]
            )
            self.assertEqual(
                fold["train_candidates"] + fold["validation_candidates"], 30
            )

    def test_exploratory_cv_cannot_open_deployment_gate(self) -> None:
        candidates = self.candidates()
        cv = annotated_calibration.grouped_cross_validation_report(
            candidates, n_folds=5, seed=41, exploratory=True
        )
        report = annotated_calibration.build_calibration_report(
            candidates,
            exploratory=True,
            grouped_cv_folds=5,
            grouped_cv_seed=41,
        )
        self.assertFalse(cv["deployment_support"]["deployable"])
        self.assertFalse(cv["deployment_support"]["confirmatory_eligible"])
        self.assertFalse(cv["calibration_shape"]["learned_confidence_ranking"])
        self.assertFalse(report["deployment_support"]["deployable"])
        self.assertFalse(report["deployment_support"]["confirmatory_eligible"])

    def test_reused_validation_scientific_gate_overrides_numeric_gate(self) -> None:
        cv = annotated_calibration.grouped_cross_validation_report(
            self.candidates(), n_folds=5, seed=41
        )
        self.assertTrue(cv["deployment_support"]["deployable"])
        gated, numeric = gold_grouped_cv.apply_scientific_gate(cv)
        self.assertTrue(numeric["deployable"])
        self.assertTrue(gated["deployment_support"]["numeric_gate_passed"])
        self.assertFalse(gated["deployment_support"]["deployable"])
        self.assertFalse(gated["deployment_support"]["confirmatory_eligible"])
        self.assertEqual(
            gated["deployment_support"]["reason"],
            gold_grouped_cv.SCIENTIFIC_GATE_REASON,
        )


class GroupedCalibrationUncertaintyTests(unittest.TestCase):
    @staticmethod
    def predictions() -> list[dict]:
        report = annotated_calibration.grouped_cross_validation_report(
            GroupedCalibrationCrossValidationTests.candidates(),
            n_folds=5,
            seed=43,
            exploratory=True,
        )
        return report["out_of_fold_predictions"]

    def test_percentile_uses_linear_interpolation(self) -> None:
        self.assertEqual(uncertainty.percentile([0.0, 10.0], 0.25), 2.5)
        self.assertEqual(uncertainty.percentile([3.0], 0.95), 3.0)

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        first = uncertainty.cluster_bootstrap_oof(
            self.predictions(), replicates=100, seed=47, bins=10
        )
        second = uncertainty.cluster_bootstrap_oof(
            self.predictions(), replicates=100, seed=47, bins=10
        )
        self.assertEqual(first, second)
        self.assertEqual(first["resampling_unit"], "sample_key")
        self.assertFalse(first["prediction_refit_per_replicate"])

    def test_reliability_bins_cover_every_prediction(self) -> None:
        predictions = self.predictions()
        for field in ("raw_confidence", "calibrated_confidence"):
            rows = uncertainty.reliability_bins(predictions, field, bins=10)
            self.assertEqual(sum(row["n"] for row in rows), len(predictions))
            self.assertEqual(len(rows), 10)

    def test_bootstrap_intervals_report_all_replicates(self) -> None:
        result = uncertainty.cluster_bootstrap_oof(
            self.predictions(), replicates=120, seed=53, bins=10
        )
        for stage in ("raw", "calibrated", "calibrated_minus_raw"):
            for metric in ("brier", "ece"):
                interval = result["metrics"][stage][metric]
                self.assertEqual(interval["bootstrap_valid_replicates"], 120)
                self.assertLessEqual(interval["lower"], interval["upper"])
        self.assertEqual(result["n_candidates"], 30)
        self.assertEqual(result["n_sample_groups"], 15)

    def test_v5_estimation_scale_is_for_planning_not_tuning(self) -> None:
        result = v5_analysis.estimation_scale(
            precision=4 / 15, predictions=15, samples=24
        )
        self.assertGreater(result["estimated_required_samples"], 100)
        self.assertTrue(result["not_a_hypothesis_test_or_tuning_budget"])

    def test_v5_stratum_scale_exposes_sparse_prediction_cost(self) -> None:
        result = v5_analysis.estimation_scale(
            precision=0.5, predictions=6, samples=12
        )
        self.assertEqual(result["required_predictions"], 97)
        self.assertEqual(result["estimated_required_samples"], 194)

    def test_v5_count_metrics_handles_empty_predictions(self) -> None:
        result = v5_analysis.count_metrics(tp=0, fp=0, fn=5)
        self.assertEqual(result["precision"], 0.0)
        self.assertEqual(result["recall"], 0.0)
        self.assertEqual(result["f1"], 0.0)

    def test_v5_error_audit_normalizes_string_none(self) -> None:
        candidates = [
            {
                "postprocess_status": "accepted",
                "canonical_correct": False,
                "symmetric_semantic_correct": False,
                "near_match_type": "none",
                "canonical_triple": ["alpha", "equivalent class", "beta"],
                "confidence": 0.95,
                "sample_index": 1,
                "task": "a",
            }
        ]
        result = v5_analysis.candidate_error_audit(candidates)
        self.assertEqual(
            result["error_type_counts"],
            {"no_exact_endpoint_or_predicate_match": 1},
        )

    def test_v5_semantic_protocol_is_frozen_secondary_only(self) -> None:
        args = v5_semantic.build_parser().parse_args(["--freeze-only"])
        protocol = v5_semantic.build_protocol(args)
        v5_semantic.validate_protocol(protocol)
        self.assertEqual(protocol["secondary_analysis"]["threshold"], 0.90)
        self.assertEqual(
            protocol["secondary_analysis"]["model_revision"],
            v5_semantic.MODEL_REVISION,
        )
        self.assertEqual(
            protocol["secondary_analysis"]["remote_code_dependency"][
                "revision"
            ],
            v5_semantic.REMOTE_CODE_REVISION,
        )
        self.assertTrue(protocol["primary_analysis"]["remains_primary"])
        self.assertFalse(
            protocol["locked_interpretation"][
                "authorize_threshold_or_rule_retuning"
            ]
        )
        self.assertFalse(protocol["meta"]["contains_source_text"])
        self.assertFalse(protocol["meta"]["contains_gold_triples"])


class RelationAnnotationWorkflowTests(unittest.TestCase):
    def test_text_trigger_selection_excludes_used_samples_without_reading_gold(self) -> None:
        samples = [
            (
                10,
                {
                    "id": "used",
                    "context": "A petal is part of a flower.",
                    "primitive-ontology-triples": [["SECRET", "SECRET", "SECRET"]],
                },
            ),
            (
                11,
                {
                    "id": "selected",
                    "context": "A wheel is part of a car and is located in its chassis.",
                    "primitive-ontology-triples": [["SECRET", "SECRET", "SECRET"]],
                },
            ),
            (12, {"id": "no-trigger", "context": "A neutral description."}),
        ]
        selected, metadata, pool = annotation_workflow.select_annotation_samples(
            samples,
            task="a",
            count=1,
            seed=7,
            excluded_indices={10},
        )
        self.assertEqual([index for index, _ in selected], [11])
        self.assertEqual(
            metadata[11]["trigger_families"], ["location", "part_whole"]
        )
        self.assertEqual(pool["eligible_text_triggered_samples"], 1)
        self.assertNotIn("primitive-ontology-triples", metadata[11])

    def test_complete_annotation_requires_human_provenance_and_verbatim_evidence(self) -> None:
        manifest = {"samples": [{"sample_key": "s1"}]}
        row = {
            "sample_key": "s1",
            "task": "a",
            "inputs": {"context": "A wheel is part of a car."},
            "annotation": {
                "annotator_id": "ann-a",
                "annotator_type": "human",
                "completed_at": "2026-09-23T12:00:00+08:00",
                "status": "complete",
                "relations": [
                    {
                        "subject": "wheel",
                        "predicate": "part_of",
                        "object": "car",
                        "evidence_mode": "explicit",
                        "evidence": ["A wheel is part of a car."],
                    }
                ],
            },
        }
        self.assertEqual(
            annotation_workflow.validate_annotation_rows([row], manifest), []
        )
        row["annotation"]["annotator_type"] = "model"
        errors = annotation_workflow.validate_annotation_rows([row], manifest)
        self.assertTrue(any("annotator_type='human'" in error for error in errors))
        row["annotation"]["annotator_type"] = "human"
        row["annotation"]["relations"][0]["evidence"] = ["not in context"]
        errors = annotation_workflow.validate_annotation_rows([row], manifest)
        self.assertTrue(any("not verbatim" in error for error in errors))

    def test_model_provenance_is_explicit_and_mutually_exclusive(self) -> None:
        context = "A wheel is part of a car."
        manifest = {
            "samples": [
                {
                    "sample_key": "s1",
                    "task": "a",
                    "legal_input_fields": ["context"],
                    "context_sha256": annotation_workflow._sha256_text(context),
                }
            ]
        }
        row = {
            "sample_key": "s1",
            "task": "a",
            "inputs": {"context": context},
            "annotation": {
                "annotator_id": "model-a",
                "annotator_type": "model",
                "model_id": "provider:model-a",
                "prompt_version": "relation_model_annotation_v1",
                "completed_at": "2026-09-24T12:00:00+08:00",
                "status": "complete",
                "relations": [],
            },
        }
        self.assertEqual(
            annotation_workflow.validate_annotation_rows(
                [row], manifest, annotation_source="model"
            ),
            [],
        )
        human_errors = annotation_workflow.validate_annotation_rows([row], manifest)
        self.assertTrue(any("annotator_type='human'" in error for error in human_errors))
        del row["annotation"]["model_id"]
        model_errors = annotation_workflow.validate_annotation_rows(
            [row], manifest, annotation_source="model"
        )
        self.assertTrue(any("requires model_id" in error for error in model_errors))

    def test_annotation_validation_rejects_manifest_input_drift(self) -> None:
        manifest = {
            "samples": [
                {
                    "sample_key": "s1",
                    "task": "a",
                    "legal_input_fields": ["context"],
                    "context_sha256": annotation_workflow._sha256_text("Original text."),
                }
            ]
        }
        row = {
            "sample_key": "s1",
            "task": "b",
            "inputs": {"context": "Changed text.", "gold": []},
            "annotation": {"status": "pending", "relations": []},
        }
        errors = annotation_workflow.validate_annotation_rows([row], manifest)
        self.assertTrue(any("task does not match" in error for error in errors))
        self.assertTrue(any("input fields do not match" in error for error in errors))
        self.assertTrue(any("context hash does not match" in error for error in errors))

    def test_agreement_reports_jaccard_and_standard_positive_agreement(self) -> None:
        rows = [
            {
                "task": "a",
                "source_annotation_status": {
                    "annotator_a": "complete",
                    "annotator_b": "complete",
                },
                "agreed_relations": [{"triple": ["x", "part_of", "y"]}],
                "disagreements": [
                    {
                        "triple": ["a", "located in", "b"],
                        "annotator_a": {"asserted": True},
                        "annotator_b": None,
                    },
                    {
                        "triple": ["c", "regulates", "d"],
                        "annotator_a": None,
                        "annotator_b": {"asserted": True},
                    },
                ],
            },
            {
                "task": "a",
                "source_annotation_status": {
                    "annotator_a": "complete",
                    "annotator_b": "complete",
                },
                "agreed_relations": [],
                "disagreements": [],
            },
        ]
        summary = annotation_workflow.summarize_agreement(rows)
        self.assertAlmostEqual(summary["relation_set_jaccard_mean"], 2 / 3)
        self.assertAlmostEqual(summary["micro_relation_set_jaccard"], 1 / 3)
        self.assertAlmostEqual(summary["positive_agreement_rate"], 0.5)
        self.assertEqual(summary["annotator_a_only_relations"], 1)
        self.assertEqual(summary["annotator_b_only_relations"], 1)

    def test_model_freeze_remains_exploratory_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            adjudication = root / "adjudication.jsonl"
            output = root / "frozen.json"
            manifest.write_text(
                json.dumps(
                    {
                        "meta": {"packet_id": "packet"},
                        "samples": [
                            {"sample_key": "s1", "task": "a", "source_index": 1}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "sample_key": "s1",
                "source_annotation_provenance": {
                    "annotator_a": {
                        "id": "model-a",
                        "type": "model",
                        "model_id": "provider:model-a",
                    },
                    "annotator_b": {
                        "id": "model-b",
                        "type": "model",
                        "model_id": "provider:model-b",
                    },
                },
                "agreed_relations": [],
                "disagreements": [],
                "adjudication": {
                    "status": "complete",
                    "adjudicator_id": "model-c",
                    "adjudicator_type": "model",
                    "adjudicator_model_id": "provider:model-c",
                    "adjudicator_prompt_version": "relation_model_annotation_v1",
                    "completed_at": "2026-09-24T12:00:00+08:00",
                    "accept_agreed_relations": True,
                },
            }
            adjudication.write_text(json.dumps(row) + "\n", encoding="utf-8")
            payload = annotation_workflow.freeze_adjudication(
                manifest_path=manifest,
                adjudication_path=adjudication,
                output_path=output,
                annotation_source="model",
            )
            self.assertFalse(payload["meta"]["human_annotated"])
            self.assertFalse(payload["meta"]["confirmatory_eligible"])
            self.assertFalse(payload["meta"]["calibration_ready"])
            self.assertTrue(payload["meta"]["exploratory_calibration_ready"])

    def test_task_b_initial_ontology_cannot_supply_text_evidence(self) -> None:
        manifest = {"samples": [{"sample_key": "s1"}]}
        row = {
            "sample_key": "s1",
            "task": "b",
            "inputs": {
                "context": "The visible text has no asserted relation.",
                "terms": ["wheel", "car"],
                "types": [],
                "initial-primitive-ontology-triples": [
                    ["wheel", "part_of", "car"]
                ],
            },
            "annotation": {
                "annotator_id": "ann-b",
                "annotator_type": "human",
                "completed_at": "2026-09-23T12:00:00+08:00",
                "status": "complete",
                "relations": [
                    {
                        "subject": "wheel",
                        "predicate": "part_of",
                        "object": "car",
                        "evidence_mode": "explicit",
                        "evidence": ["wheel"],
                    }
                ],
            },
        }
        errors = annotation_workflow.validate_annotation_rows([row], manifest)
        self.assertTrue(any("not verbatim" in error for error in errors))

    def test_compare_requires_distinct_human_annotators(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest_path = root / "manifest.json"
            annotation_a = root / "a.jsonl"
            annotation_b = root / "b.jsonl"
            output = root / "adjudication.jsonl"
            manifest_path.write_text(
                json.dumps(
                    {
                        "meta": {"packet_id": "packet"},
                        "samples": [{"sample_key": "s1"}],
                    }
                ),
                encoding="utf-8",
            )
            row = {
                "sample_key": "s1",
                "task": "a",
                "inputs": {"context": "A wheel is part of a car."},
                "annotation": {
                    "annotator_id": "same-person",
                    "annotator_type": "human",
                    "completed_at": "2026-09-23T12:00:00+08:00",
                    "status": "complete",
                    "relations": [],
                },
            }
            text = json.dumps(row) + "\n"
            annotation_a.write_text(text, encoding="utf-8")
            annotation_b.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "different annotator IDs"):
                annotation_workflow.compare_annotations(
                    manifest_path=manifest_path,
                    annotation_a_path=annotation_a,
                    annotation_b_path=annotation_b,
                    output_path=output,
                )

    def test_freeze_rejects_pending_adjudication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root / "manifest.json"
            adjudication = root / "adjudication.jsonl"
            output = root / "frozen.json"
            manifest.write_text(
                json.dumps(
                    {
                        "meta": {"packet_id": "packet"},
                        "samples": [
                            {"sample_key": "s1", "task": "a", "source_index": 1}
                        ],
                    }
                ),
                encoding="utf-8",
            )
            adjudication.write_text(
                json.dumps(
                    {
                        "sample_key": "s1",
                        "adjudication": {"status": "pending"},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not complete"):
                annotation_workflow.freeze_adjudication(
                    manifest_path=manifest,
                    adjudication_path=adjudication,
                    output_path=output,
                )

    def test_annotated_calibration_audit_contains_no_source_text(self) -> None:
        args = annotated_calibration.build_parser().parse_args(["--audit-only"])
        audit = annotated_calibration.build_audit_manifest(args)
        serialized = json.dumps(audit, ensure_ascii=False).lower()
        self.assertFalse(audit["meta"]["contains_source_text"])
        self.assertFalse(audit["meta"]["protected_gold_visible_to_model"])
        self.assertEqual(audit["meta"]["estimated_calls"], 30)
        self.assertNotIn("classification of plant structures", serialized)

    def test_model_calibration_audit_does_not_claim_pending_human_annotation(self) -> None:
        args = annotated_calibration.build_parser().parse_args(
            ["--audit-only", "--annotation-source", "model"]
        )
        audit = annotated_calibration.build_audit_manifest(args)
        self.assertEqual(
            audit["meta"]["annotation_status"],
            "not_applicable_dual_model_weak_labels",
        )
        self.assertFalse(audit["meta"]["confirmatory_eligible"])

    def test_frozen_annotation_loader_rejects_wrong_packet(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frozen.json"
            path.write_text(
                json.dumps(
                    {
                        "meta": {
                            "packet_id": "wrong",
                            "calibration_ready": True,
                        },
                        "samples": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not ready for this packet"):
                annotated_calibration.load_frozen_annotations(
                    path, packet_id="expected", expected_keys={"s1"}
                )


if __name__ == "__main__":
    unittest.main()
