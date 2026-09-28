"""Run the data-free unit-test subset shipped in the anonymous artifact."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TESTS = ROOT / "src" / "tests"
if str(TESTS) not in sys.path:
    sys.path.insert(0, str(TESTS))

MODULES = (
    "test_nonce_relation_holdout",
    "test_official_blind_eval",
    "test_predicate_inventory",
    "test_relation_calibration",
    "test_relation_guarded_v2",
    "test_task_c_taxonomy",
)

# These tests require resources intentionally omitted from the public artifact:
# third-party challenge data, private annotation packets, or internal ledgers.
DATA_BOUND_TESTS = {
    "test_nonce_relation_holdout.NonceRelationHoldoutTests.test_holdout_is_balanced_and_disjoint_from_official_2026",
    "test_nonce_relation_holdout.NonceRelationHoldoutTests.test_manifest_excludes_text_and_gold",
    "test_nonce_relation_holdout.NonceRelationHoldoutTests.test_package_is_byte_deterministic",
    "test_nonce_relation_holdout.NonceRelationHoldoutTests.test_evaluation_manifest_hides_text_gold_and_freezes_sixty_units",
    "test_official_blind_eval.ContaminationProbeTests.test_probe_manifest_contains_no_source_text_and_validates",
    "test_official_blind_eval.PromptBoundaryTests.test_audit_manifest_contains_no_source_text",
    "test_relation_calibration.CalibrationSelectionTests.test_v5_confirmatory_selection_is_balanced_and_disjoint",
    "test_relation_calibration.CalibrationSelectionTests.test_v5_confirmatory_manifest_contains_no_source_text",
    "test_relation_calibration.CalibrationSelectionTests.test_v5_confirmatory_runtime_rejects_locked_config_drift",
    "test_relation_calibration.CalibrationSelectionTests.test_v5_confirmatory_manifest_is_write_once",
    "test_relation_calibration.CalibrationSelectionTests.test_cross_year_pool_is_strictly_disjoint_and_deduplicated",
    "test_relation_calibration.CalibrationSelectionTests.test_cross_year_audit_manifest_contains_no_source_text",
    "test_relation_calibration.GroupedCalibrationUncertaintyTests.test_v5_semantic_protocol_is_frozen_secondary_only",
    "test_relation_calibration.RelationAnnotationWorkflowTests.test_annotated_calibration_audit_contains_no_source_text",
    "test_relation_calibration.RelationAnnotationWorkflowTests.test_model_calibration_audit_does_not_claim_pending_human_annotation",
    "test_task_c_taxonomy.TaskCOfficialMetricMirrorTests.test_local_mirror_matches_official_challenge_module",
    "test_task_c_taxonomy.TaskCOfficialDataTests.test_archeon_domain_loads_expected_counts_and_hashes",
    "test_task_c_taxonomy.TaskCOfficialDataTests.test_evaluate_task_c_on_archeon_prefix_closes_gate",
}
EXPECTED_PORTABLE_TESTS = 129


def iter_tests(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from iter_tests(item)
        else:
            yield item


def portable_suite() -> unittest.TestSuite:
    loader = unittest.defaultTestLoader
    discovered = unittest.TestSuite(loader.loadTestsFromName(name) for name in MODULES)
    selected = unittest.TestSuite()
    excluded: set[str] = set()
    for test in iter_tests(discovered):
        test_id = test.id()
        if test_id in DATA_BOUND_TESTS:
            excluded.add(test_id)
        else:
            selected.addTest(test)

    missing_exclusions = DATA_BOUND_TESTS - excluded
    if missing_exclusions:
        raise RuntimeError(
            "portable-test exclusion list drifted; missing IDs: "
            + ", ".join(sorted(missing_exclusions))
        )
    if selected.countTestCases() != EXPECTED_PORTABLE_TESTS:
        raise RuntimeError(
            "portable-test count drifted: "
            f"expected {EXPECTED_PORTABLE_TESTS}, got {selected.countTestCases()}"
        )
    return selected


def main() -> int:
    suite = portable_suite()
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
