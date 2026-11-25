import unittest
import math
from grounding import Fact, format_grounded_response
from grounding.evaluation_adapter import (
    compute_unsupported_claim_rate,
    compute_unsupported_claim_rate_old,
    compute_abstain_precision,
    compute_citation_support_rate
)

class TestFactValidation(unittest.TestCase):
    def test_fact_id_validation(self):
        # Non-empty string
        with self.assertRaises(ValueError):
            Fact("", "text", "source", 0.5, "active")
        # Not a string
        with self.assertRaises(ValueError):
            Fact(123, "text", "source", 0.5, "active")
        # Too long
        with self.assertRaises(ValueError):
            Fact("x" * 257, "text", "source", 0.5, "active")

    def test_text_validation(self):
        # Non-empty string
        with self.assertRaises(ValueError):
            Fact("f1", "", "source", 0.5, "active")
        # Not a string
        with self.assertRaises(ValueError):
            Fact("f1", 123, "source", 0.5, "active")
        # Too long (>10000)
        with self.assertRaises(ValueError):
            Fact("f1", "x" * 10001, "source", 0.5, "active")

    def test_source_validation(self):
        # Non-empty string
        with self.assertRaises(ValueError):
            Fact("f1", "text", "", 0.5, "active")
        # Not a string
        with self.assertRaises(ValueError):
            Fact("f1", "text", 123, 0.5, "active")
        # Too long
        with self.assertRaises(ValueError):
            Fact("f1", "text", "x" * 257, 0.5, "active")

    def test_confidence_validation(self):
        # Not a number
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", "0.5", "active")
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", True, "active")
        # NaN
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", math.nan, "active")
        # Inf
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", math.inf, "active")
        # Negative
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", -0.1, "active")
        # >1
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", 1.1, "active")

    def test_status_validation(self):
        # Invalid status
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", 0.5, "invalid_status")
        # Valid statuses should not raise
        for status in ["active", "superseded", "retracted"]:
            try:
                Fact("f1", "text", "source", 0.5, status)
            except ValueError:
                self.fail(f"Valid status {status} raised ValueError unexpectedly")

    def test_superseded_by_validation(self):
        # None is okay
        Fact("f1", "text", "source", 0.5, "active", None)
        # Empty string
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", 0.5, "superseded", "")
        # Not a string
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", 0.5, "superseded", 123)
        # Too long
        with self.assertRaises(ValueError):
            Fact("f1", "text", "source", 0.5, "superseded", "x" * 257)


class TestFormatGroundedResponse(unittest.TestCase):
    def test_no_facts(self):
        result = format_grounded_response([], "test query")
        self.assertEqual(result["grounded_or_abstain"], "abstain")
        self.assertEqual(result["message"], "unverified: no relevant facts found.")
        self.assertIsNone(result["fact_id"])
        self.assertIsNone(result["source"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["status"])
        self.assertFalse(result["historical_warning"])
        self.assertIsNone(result["fact_text"])

    def test_low_confidence_abstain(self):
        fact = Fact("f1", "Test fact", "test_source", 0.4, "active")
        result = format_grounded_response([fact], "test query", confidence_threshold=0.5)
        self.assertEqual(result["grounded_or_abstain"], "abstain")
        self.assertEqual(result["fact_id"], "f1")
        self.assertEqual(result["source"], "test_source")
        self.assertEqual(result["confidence"], 0.4)
        self.assertEqual(result["status"], "active")
        self.assertFalse(result["historical_warning"])
        self.assertIsNone(result["fact_text"])

    def test_grounded_active_fact(self):
        fact_text = "This is the fact text."
        fact = Fact("f1", fact_text, "test_source", 0.8, "active")
        result = format_grounded_response([fact], "test query", confidence_threshold=0.5)
        self.assertEqual(result["grounded_or_abstain"], "grounded")
        self.assertEqual(result["fact_id"], "f1")
        self.assertEqual(result["source"], "test_source")
        self.assertEqual(result["confidence"], 0.8)
        self.assertEqual(result["status"], "active")
        self.assertFalse(result["historical_warning"])
        self.assertEqual(result["fact_text"], fact_text)
        self.assertIn(f"[{fact.fact_id}]", result["message"])
        self.assertIn(f"Source: {fact.source}", result["message"])
        self.assertIn(f"Confidence: {fact.confidence:.2f}", result["message"])
        self.assertIn(f"Status: {fact.status}", result["message"])
        self.assertIn(fact_text, result["message"])
        self.assertTrue(result["message"].startswith("grounded:"))

    def test_superseded_fact_high_confidence(self):
        fact = Fact("f1", "Old fact", "test_source", 0.9, "superseded", superseded_by="f2")
        result = format_grounded_response([fact], "test query", confidence_threshold=0.5)
        # Should abstain because status is superseded
        self.assertEqual(result["grounded_or_abstain"], "abstain")
        self.assertEqual(result["fact_id"], "f1")
        self.assertEqual(result["source"], "test_source")
        self.assertEqual(result["confidence"], 0.9)
        self.assertEqual(result["status"], "superseded")
        self.assertTrue(result["historical_warning"])
        self.assertIsNone(result["fact_text"])
        self.assertIn("unverified: historical fact (superseded)", result["message"])
        self.assertIn(f"[{fact.fact_id}]", result["message"])
        self.assertIn(f"Source: {fact.source}", result["message"])
        self.assertIn(f"Confidence: {fact.confidence:.2f}", result["message"])
        self.assertIn(f"Status: {fact.status}", result["message"])
        self.assertIn(f"Superseded by: {fact.superseded_by}", result["message"])

    def test_retracted_fact_high_confidence(self):
        fact = Fact("f1", "Retracted fact", "test_source", 0.9, "retracted")
        result = format_grounded_response([fact], "test query", confidence_threshold=0.5)
        self.assertEqual(result["grounded_or_abstain"], "abstain")
        self.assertTrue(result["historical_warning"])
        self.assertIn("unverified: historical fact (retracted)", result["message"])

    def test_superseded_fact_low_confidence(self):
        fact = Fact("f1", "Old fact", "test_source", 0.4, "superseded", superseded_by="f2")
        result = format_grounded_response([fact], "test query", confidence_threshold=0.5)
        self.assertEqual(result["grounded_or_abstain"], "abstain")
        self.assertTrue(result["historical_warning"])
        # Also should have historical warning because status is superseded
        self.assertIn("unverified: historical fact (superseded)", result["message"])

    def test_validate_confidence_threshold(self):
        fact = Fact("f1", "text", "source", 0.5, "active")
        # Non-numeric threshold
        with self.assertRaises(ValueError):
            format_grounded_response([fact], "query", confidence_threshold="0.5")
        # NaN threshold
        with self.assertRaises(ValueError):
            format_grounded_response([fact], "query", confidence_threshold=math.nan)
        # Inf threshold
        with self.assertRaises(ValueError):
            format_grounded_response([fact], "query", confidence_threshold=math.inf)
        # Below 0
        with self.assertRaises(ValueError):
            format_grounded_response([fact], "query", confidence_threshold=-0.1)
        # Above 1
        with self.assertRaises(ValueError):
            format_grounded_response([fact], "query", confidence_threshold=1.1)


class TestEvaluationAdapter(unittest.TestCase):
    def test_compute_unsupported_claim_rate(self):
        # Empty list
        self.assertEqual(compute_unsupported_claim_rate([]), 0.0)
        # All true (supported)
        self.assertEqual(compute_unsupported_claim_rate([True, True, True]), 0.0)
        # All false (unsupported)
        self.assertEqual(compute_unsupported_claim_rate([False, False, False]), 1.0)
        with self.assertRaises(ValueError):
            compute_unsupported_claim_rate([True, 1])
        # Mixed
        self.assertEqual(compute_unsupported_claim_rate([True, False, True, False]), 0.5)

    def test_compute_unsupported_claim_rate_old_length_mismatch(self):
        predictions = [{"grounded_or_abstain": "grounded", "fact_id": "f1"}]
        ground_truth = [["f1"], ["f2"]]  # length mismatch
        with self.assertRaises(ValueError):
            compute_unsupported_claim_rate_old(predictions, ground_truth)

    def test_compute_unsupported_claim_rate_old_normal(self):
        predictions = [
            {"grounded_or_abstain": "grounded", "fact_id": "f1"},
            {"grounded_or_abstain": "grounded", "fact_id": "f2"},
            {"grounded_or_abstain": "abstain", "fact_id": None},
        ]
        ground_truth = [
            ["f1", "f3"],  # f1 is in the set -> supported
            ["f3"],         # f2 is not in the set -> unsupported
            ["f1", "f2"]    # abstention ignored
        ]
        # 2 grounded predictions, 1 unsupported (second one)
        self.assertEqual(compute_unsupported_claim_rate_old(predictions, ground_truth), 0.5)

    def test_compute_abstain_precision_length_mismatch(self):
        predictions = [{"grounded_or_abstain": "abstain"}]
        ground_truth = [True, False]  # length mismatch
        with self.assertRaises(ValueError):
            compute_abstain_precision(predictions, ground_truth)

    def test_compute_abstain_precision_non_bool_ground_truth(self):
        predictions = [{"grounded_or_abstain": "abstain"}]
        ground_truth = ["yes"]  # not a bool
        with self.assertRaises(ValueError):
            compute_abstain_precision(predictions, ground_truth)

    def test_compute_abstain_precision_normal(self):
        predictions = [
            {"grounded_or_abstain": "abstain"},  # correct abstain (True)
            {"grounded_or_abstain": "abstain"},  # incorrect abstain (False)
            {"grounded_or_abstain": "grounded"}, # not abstaining
            {"grounded_or_abstain": "abstain"},  # correct abstain (True)
        ]
        ground_truth = [True, False, True, True]
        # 2 correct abstentions (indices 0 and 3), 1 incorrect (index 1)
        self.assertEqual(compute_abstain_precision(predictions, ground_truth), 2/3)

    def test_compute_citation_support_rate_length_mismatch(self):
        predictions = [{"grounded_or_abstain": "grounded", "fact_id": "f1"}]
        ground_truth_support = [["f1"], ["f2"]]  # length mismatch
        with self.assertRaises(ValueError):
            compute_citation_support_rate(predictions, ground_truth_support)

    def test_compute_citation_support_rate_normal(self):
        predictions = [
            {"grounded_or_abstain": "grounded", "fact_id": "f1"},
            {"grounded_or_abstain": "grounded", "fact_id": "f2"},
            {"grounded_or_abstain": "abstain", "fact_id": None},
            {"grounded_or_abstain": "grounded", "fact_id": "f3"},
        ]
        ground_truth_support = [
            ["f1", "f4"],  # f1 is in set -> correct
            ["f3"],         # f2 is not in set -> incorrect
            ["f1", "f2"],   # abstention ignored
            ["f3", "f5"]    # f3 is in set -> correct
        ]
        # 3 grounded predictions, 2 correct (first and last)
        self.assertEqual(compute_citation_support_rate(predictions, ground_truth_support), 2/3)


if __name__ == "__main__":
    unittest.main()


