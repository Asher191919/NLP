from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import run_official_blind_eval as runner
import run_official_contamination_probe as contamination_probe
import analyze_official_contamination_probe as contamination_analysis
import analyze_official_contamination_surface_stress as surface_analysis
import run_official_contamination_surface_stress as contamination_surface
import run_stratified_open_eval as stratified


class ContaminationProbeTests(unittest.TestCase):
    def test_probe_controls_are_balanced_and_not_sample_gold(self) -> None:
        sample = {
            "primitive-ontology-triples": [
                ["alpha", "is-a", "parent one"],
                ["beta", "is-a", "parent two"],
            ]
        }
        items = contamination_probe.build_probe_items(
            sample=sample,
            task="a",
            sample_index=3,
            object_pool={"is-a": ["parent one", "parent two", "parent three"]},
            seed=7,
            max_positives=2,
        )
        positives = [row for row in items if row["expected"] == "entailed"]
        negatives = [row for row in items if row["expected"] == "not_entailed"]
        gold = {tuple(row) for row in sample["primitive-ontology-triples"]}
        self.assertEqual(len(positives), len(negatives))
        self.assertTrue(all(tuple(row["triple"]) not in gold for row in negatives))

    def test_probe_parser_requires_complete_unique_ids(self) -> None:
        parsed = contamination_probe.parse_probe_response(
            '{"items":[{"id":"q00","label":"entailed","confidence":0.8},'
            '{"id":"q01","label":"not_entailed","confidence":0.7}]}',
            ["q00", "q01"],
        )
        self.assertEqual(len(parsed), 2)
        with self.assertRaisesRegex(ValueError, "cover every id"):
            contamination_probe.parse_probe_response(
                '{"items":[{"id":"q00","label":"entailed","confidence":0.8}]}',
                ["q00", "q01"],
            )

    def test_probe_auc_handles_ties(self) -> None:
        self.assertEqual(
            contamination_probe.roc_auc([0.9, 0.5, 0.5, 0.1], [True, True, False, False]),
            0.875,
        )

    def test_probe_manifest_contains_no_source_text_and_validates(self) -> None:
        args = contamination_probe.build_parser().parse_args(
            ["--audit-only", "--model", "dots3-note-prev"]
        )
        manifest = contamination_probe.build_manifest(args)
        contamination_probe.validate_manifest(manifest, args)
        self.assertFalse(manifest["meta"]["contains_source_text"])
        self.assertTrue(manifest["meta"]["contains_protected_gold"])
        self.assertNotIn("Content:", json.dumps(manifest, ensure_ascii=False))

    def test_probe_split_summary_accepts_legacy_blind_metrics(self) -> None:
        record = {
            "metrics": {
                "typing": {"tp": 1, "fp": 0, "fn": 0},
                "taxonomy": {"tp": 1, "fp": 1, "fn": 1},
                "nontax": {"tp": 0, "fp": 0, "fn": 1},
                "official_graph_similarity": {"graph_similarity": 0.25},
            }
        }
        summary = contamination_probe._summarize_blind_records([record])
        self.assertEqual(summary["typing_f1"], 1.0)
        self.assertEqual(summary["taxonomy_f1"], 0.5)
        self.assertEqual(summary["official_graph_similarity_avg"], 0.25)

    def test_probe_analysis_statistics(self) -> None:
        self.assertEqual(
            contamination_analysis.percentile([0.0, 1.0], 0.5), 0.5
        )
        self.assertAlmostEqual(
            contamination_analysis.spearman([1, 2, 3], [3, 2, 1]), -1.0
        )
        self.assertEqual(
            contamination_analysis.channel_for_predicate("is-a"), "taxonomy"
        )

    def test_surface_alias_transform_preserves_structure_and_hides_endpoints(self) -> None:
        sample = {
            "id": "sample",
            "context": "A red car is a car.",
            "primitive-ontology-triples": [["red car", "is-a", "car"]],
        }
        transformed, metadata = contamination_surface.transform_sample(
            sample, task="a", sample_index=1, seed=7
        )
        self.assertNotIn("red car", transformed["context"].lower())
        self.assertNotIn(" car", transformed["context"].lower())
        self.assertEqual(metadata["input_visible_endpoint_fraction"], 1.0)
        self.assertEqual(
            transformed["primitive-ontology-triples"][0][1], "is-a"
        )

    def test_surface_manifest_comparison_does_not_mutate_inputs(self) -> None:
        import tempfile

        candidate = {
            "meta": {
                "cache_hits": 0,
                "estimated_external_calls": 2,
                "fixed": "value",
            },
            "samples": [],
        }
        existing = {
            "meta": {
                "cache_hits": 2,
                "estimated_external_calls": 0,
                "fixed": "value",
            },
            "samples": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(existing), encoding="utf-8")
            loaded, created = contamination_surface.preserve_or_write_manifest(
                path, candidate
            )
        self.assertFalse(created)
        self.assertEqual(loaded["meta"]["cache_hits"], 2)
        self.assertEqual(candidate["meta"]["cache_hits"], 0)
        self.assertEqual(candidate["meta"]["estimated_external_calls"], 2)

    def test_surface_paired_bootstrap_uses_original_minus_alias(self) -> None:
        def record(index: int, tp: int, fp: int, fn: int, graph: float) -> dict:
            metric = {"tp": tp, "fp": fp, "fn": fn}
            return {
                "sample_index": index,
                "metrics": {
                    "typing": metric,
                    "taxonomy": metric,
                    "nontax": {"tp": 0, "fp": 0, "fn": 0},
                    "official_graph_similarity": {"graph_similarity": graph},
                },
            }

        pairs = [
            {
                "sample_index": 1,
                "original": record(1, 1, 0, 0, 1.0),
                "endpoint_alias": record(1, 0, 1, 1, 0.0),
            },
            {
                "sample_index": 2,
                "original": record(2, 1, 0, 0, 0.5),
                "endpoint_alias": record(2, 1, 0, 0, 0.25),
            },
        ]
        analysis = surface_analysis.bootstrap_pairs(
            pairs, replicates=100, seed=7
        )
        self.assertGreater(
            analysis["original_minus_alias"]["official_graph_similarity_avg"],
            0.0,
        )
        self.assertEqual(
            analysis["confidence_intervals"]["official_graph_similarity_avg"][
                "bootstrap_valid_replicates"
            ],
            100,
        )


class PromptBoundaryTests(unittest.TestCase):
    def test_task_a_uses_context_only(self) -> None:
        sample = {
            "id": "a1",
            "context": "Visible context about a turbine.",
            "primitive-ontology-triples": [["SECRET_GOLD", "is-a", "HiddenType"]],
            "terms": ["SECRET_TERM"],
            "types": ["SECRET_TYPE"],
        }
        prompt = runner.build_prompt(sample, "a", "open_vocab", 6000)
        self.assertIn("Visible context", prompt)
        self.assertNotIn("SECRET_GOLD", prompt)
        self.assertNotIn("SECRET_TERM", prompt)
        self.assertNotIn("SECRET_TYPE", prompt)

    def test_audit_manifest_contains_no_source_text(self) -> None:
        args = runner.build_parser().parse_args(["--task", "a", "--limit", "1"])
        manifest = runner.build_audit_manifest(args, ["a"], ["open_vocab"])
        serialized = str(manifest)
        self.assertFalse(manifest["meta"]["contains_source_text"])
        self.assertNotIn("CONTEXT:", serialized)
        self.assertEqual(manifest["meta"]["estimated_calls"], 1)

    def test_task_b_uses_initial_but_not_extension_gold(self) -> None:
        sample = {
            "id": "b1",
            "context": "Visible reuse context.",
            "initial-primitive-ontology-triples": [["seed", "is-a", "entity"]],
            "extended-primitive-ontology-triples": [["SECRET_GOLD", "is-a", "HiddenType"]],
            "terms": ["given term"],
            "types": ["given type"],
        }
        prompt = runner.build_prompt(sample, "b", "closed_vocab", 6000)
        self.assertIn("seed", prompt)
        self.assertIn("given term", prompt)
        self.assertIn("given type", prompt)
        self.assertNotIn("SECRET_GOLD", prompt)

    def test_guarded_prompt_requires_confidence_evidence_and_limits(self) -> None:
        sample = {
            "id": "a2",
            "context": "Alpha contains Beta.",
            "primitive-ontology-triples": [["SECRET_GOLD", "part_of", "HiddenType"]],
        }
        prompt = runner.build_prompt(
            sample,
            "a",
            "open_vocab_guarded",
            6000,
            confidence_threshold=0.85,
            max_nontax=8,
        )
        self.assertIn('"confidence":0.0', prompt)
        self.assertIn('"evidence":"verbatim quote from CONTEXT"', prompt)
        self.assertIn("at most 8 non-taxonomic relations", prompt)
        self.assertIn("below confidence 0.85", prompt)
        self.assertNotIn("SECRET_GOLD", prompt)


class ParsingAndMetricsTests(unittest.TestCase):
    def test_parses_fenced_json_and_deduplicates(self) -> None:
        text = '```json\n{"triples":[["A","is-a","B"],["a","is-a","b"]]}\n```'
        self.assertEqual(runner.parse_model_payload(text), [["a", "is-a", "b"]])

    def test_parses_dict_triples(self) -> None:
        text = '[{"subject":"A","predicate":"part-of","object":"B"}]'
        self.assertEqual(runner.parse_model_payload(text), [["a", "part-of", "b"]])

    def test_aggregate_uses_pooled_micro_counts(self) -> None:
        records = [
            {
                "status": "ok",
                "attempts": 1,
                "latency_sec": 1.0,
                "metrics": {
                    "typing": {"tp": 1, "fp": 0, "fn": 0},
                    "taxonomy": {"tp": 1, "fp": 1, "fn": 0},
                    "nontax": {"tp": 0, "fp": 0, "fn": 1},
                    "nontax_nonself": {"tp": 0, "fp": 0, "fn": 1},
                    "official_graph_similarity": {"graph_similarity": 0.5},
                    "gold_counts": {"typing": 1, "taxonomy": 1, "nontax": 1},
                    "pred_counts": {"typing": 1, "taxonomy": 2, "nontax": 0},
                },
            },
            {
                "status": "ok",
                "attempts": 2,
                "latency_sec": 3.0,
                "metrics": {
                    "typing": {"tp": 0, "fp": 1, "fn": 1},
                    "taxonomy": {"tp": 0, "fp": 0, "fn": 1},
                    "nontax": {"tp": 1, "fp": 0, "fn": 0},
                    "nontax_nonself": {"tp": 1, "fp": 0, "fn": 0},
                    "official_graph_similarity": {"graph_similarity": 0.25},
                    "gold_counts": {"typing": 1, "taxonomy": 1, "nontax": 1},
                    "pred_counts": {"typing": 1, "taxonomy": 0, "nontax": 1},
                },
            },
        ]
        summary = runner.aggregate_records(records, 2)
        self.assertTrue(summary["complete"])
        self.assertAlmostEqual(summary["term_typing_f1_micro"]["f1"], 0.5)
        self.assertAlmostEqual(summary["official_graph_similarity_avg"], 0.375)
        self.assertEqual(summary["successful_latency_sec"]["median"], 2.0)
        self.assertEqual(summary["successful_latency_sec"]["p95"], 3.0)
        self.assertEqual(summary["successful_runner_attempts_total"], 3)
        self.assertEqual(summary["successful_samples_with_retry"], 1)

    def test_guarded_parser_filters_and_ranks_nontax_candidates(self) -> None:
        context = "Alpha contains Beta. Alpha contains Gamma. Alpha contains Delta."
        payload = {
            "triples": [
                {
                    "subject": "alpha",
                    "predicate": "is-a",
                    "object": "entity",
                    "confidence": 0.7,
                    "evidence": "",
                },
                {
                    "subject": "alpha",
                    "predicate": "contains",
                    "object": "beta",
                    "confidence": 0.90,
                    "evidence": "Alpha contains Beta",
                },
                {
                    "subject": "alpha",
                    "predicate": "contains",
                    "object": "gamma",
                    "confidence": 0.99,
                    "evidence": "Alpha contains Gamma",
                },
                {
                    "subject": "alpha",
                    "predicate": "contains",
                    "object": "delta",
                    "confidence": 0.95,
                    "evidence": "Alpha contains Delta",
                },
                {
                    "subject": "alpha",
                    "predicate": "contains",
                    "object": "epsilon",
                    "confidence": 0.99,
                    "evidence": "Alpha contains Epsilon",
                },
                {
                    "subject": "alpha",
                    "predicate": "related to",
                    "object": "beta",
                    "confidence": 0.99,
                    "evidence": "Alpha contains Beta",
                },
            ]
        }
        triples, diagnostics = runner.parse_guarded_model_payload(
            json.dumps(payload),
            context=context,
            confidence_threshold=0.85,
            max_nontax=2,
        )
        self.assertIn(["alpha", "is-a", "entity"], triples)
        self.assertIn(["alpha", "contains", "gamma"], triples)
        self.assertIn(["alpha", "contains", "delta"], triples)
        self.assertNotIn(["alpha", "contains", "beta"], triples)
        self.assertEqual(diagnostics["accepted_nontax"], 2)
        self.assertEqual(diagnostics["rejection_reason_counts"]["quantity_limit"], 1)
        self.assertEqual(diagnostics["rejection_reason_counts"]["evidence_not_verbatim"], 1)
        self.assertEqual(diagnostics["rejection_reason_counts"]["generic_predicate"], 1)


class StratifiedSelectionTests(unittest.TestCase):
    @staticmethod
    def _sample(sample_id: str, nontax: int) -> dict:
        triples = [[f"s{i}", f"relation-{i}", f"o{i}"] for i in range(nontax)]
        return {
            "id": sample_id,
            "context": f"Context for {sample_id}",
            "primitive-ontology-triples": triples,
        }

    def test_selects_equal_deterministic_density_strata(self) -> None:
        samples = [
            (0, self._sample("zero", 0)),
            (1, self._sample("low-1", 1)),
            (2, self._sample("low-2", 1)),
            (3, self._sample("medium-1", 2)),
            (4, self._sample("medium-2", 4)),
            (5, self._sample("high-1", 5)),
            (6, self._sample("high-2", 8)),
        ]
        selected_a, metadata_a = stratified.select_density_strata(
            samples,
            task="a",
            samples_per_stratum=2,
            seed=7,
        )
        selected_b, metadata_b = stratified.select_density_strata(
            samples,
            task="a",
            samples_per_stratum=2,
            seed=7,
        )
        self.assertEqual([index for index, _ in selected_a], [index for index, _ in selected_b])
        self.assertEqual(metadata_a, metadata_b)
        self.assertEqual(len(selected_a), 6)
        counts = {name: 0 for name in stratified.STRATA}
        for value in metadata_a.values():
            counts[value["stratum"]] += 1
        self.assertEqual(counts, {"low": 2, "medium": 2, "high": 2})

    def test_density_strata_ignore_nontax_self_loops(self) -> None:
        self_only = self._sample("self-only", 0)
        self_only["primitive-ontology-triples"] = [
            [f"node-{i}", "is defined by", f"node-{i}"] for i in range(6)
        ]
        low = self._sample("low", 1)
        low["primitive-ontology-triples"].extend(
            [[f"meta-{i}", "is defined by", f"meta-{i}"] for i in range(5)]
        )
        samples = [
            (0, self_only),
            (1, low),
            (2, self._sample("medium", 2)),
            (3, self._sample("high", 5)),
        ]

        selected, metadata = stratified.select_density_strata(
            samples,
            task="a",
            samples_per_stratum=1,
            seed=11,
        )

        self.assertEqual([index for index, _ in selected], [1, 2, 3])
        self.assertEqual(metadata[1]["stratum"], "low")
        self.assertEqual(metadata[1]["official_nontax_gold"], 6)
        self.assertEqual(metadata[1]["selection_nontax_gold_nonself"], 1)

    def test_hybrid_keeps_legacy_core_and_guarded_relations(self) -> None:
        legacy = [
            ["a", "instance-of", "type-a"],
            ["type-a", "is-a", "entity"],
            ["a", "legacy relation", "b"],
        ]
        guarded = [
            ["wrong", "is-a", "core"],
            ["a", "guarded relation", "b"],
        ]
        merged = stratified.merge_core_with_guarded_relations(legacy, guarded)
        self.assertEqual(
            merged,
            [
                ["a", "instance-of", "type-a"],
                ["type-a", "is-a", "entity"],
                ["a", "guarded relation", "b"],
            ],
        )


if __name__ == "__main__":
    unittest.main()
