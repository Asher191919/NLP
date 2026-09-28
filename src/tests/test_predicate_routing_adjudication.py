import json
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_predicate_routing_adjudication import (
    audit_off_gold_predictions,
    DEFAULT_ERROR_AUDIT,
)


class TestPredicateRoutingAdjudication(unittest.TestCase):
    def test_adjudication_metrics(self) -> None:
        self.assertTrue(DEFAULT_ERROR_AUDIT.exists())
        data = json.loads(DEFAULT_ERROR_AUDIT.read_text(encoding="utf-8"))
        summary = audit_off_gold_predictions(data)

        self.assertEqual(summary["official_exact_fp"], 138)
        self.assertEqual(summary["official_exact_tp"], 2)
        self.assertEqual(summary["evidence_grounded_fp"], 138)
        self.assertEqual(summary["evidence_grounded_fp_rate"], 1.0)
        self.assertEqual(summary["both_endpoints_verbatim_fp"], 133)
        self.assertAlmostEqual(summary["both_endpoints_verbatim_fp_rate"], 0.9638, places=4)
        self.assertEqual(summary["predicate_distribution"].get("type"), 120)


if __name__ == "__main__":
    unittest.main()

