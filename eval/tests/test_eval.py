import unittest

from eval.eval_harness import evaluate_retrieval, load_jsonl, validate_corpus


class EvalHarnessTests(unittest.TestCase):
    def test_repo_root_corpus_validates(self):
        counts = validate_corpus()
        self.assertGreaterEqual(counts["golden"], 150)
        self.assertGreaterEqual(counts["abstain"], 40)
        golden = load_jsonl(__import__("eval.eval_harness", fromlist=["GOLDEN_PATH"]).GOLDEN_PATH)
        self.assertEqual(len({row["question"] for row in golden}), len(golden))
        self.assertEqual(len({row["id"] for row in golden}), len(golden))

    def test_retrieval_metrics_known_ranks_and_optional_metrics(self):
        golden = [
            {"question": "q1", "answer": "a1", "source": "s1"},
            {"question": "q2", "answer": "a2", "source": "s2"},
        ]
        def retrieve(query, top_k):
            if query == "q1":
                return [{"source": "s1", "text": "a1"}, {"source": "s2"}][:top_k]
            return [{"source": "s1"}, {"source": "s2", "text": "a2"}][:top_k]
        result = evaluate_retrieval(
            golden,
            retrieve,
            answerer=lambda q, docs: ["claim"],
            claim_supported=lambda claim, docs: q_truth(docs),
        )
        self.assertEqual(result["recall_at_1"], 0.5)
        self.assertEqual(result["recall_at_5"], 1.0)
        self.assertEqual(result["mrr"], 0.75)
        self.assertEqual(result["unsupported_claim_rate"], 0.5)
        self.assertGreaterEqual(result["latency_p95_ms"], result["latency_p50_ms"])

    def test_source_match_alone_is_not_a_hit_and_complete_number_token_is(self):
        row = {"question": "q", "answer": "3", "source": "s.md"}
        # Appendix A permits a complete number token, with boundaries preventing 3 from matching 30.
        hit = evaluate_retrieval([row], lambda q, k: [{"source": "s.md", "text": "retried 3 times"}])
        self.assertEqual(hit["recall_at_5"], 1.0)
        miss = evaluate_retrieval([row], lambda q, k: [{"source": "s.md", "text": "retried 30 times"}])
        self.assertEqual(miss["recall_at_5"], 0.0)
        long_row = {"question": "q", "answer": "reserve it for shorter dispatches", "source": "s.md"}
        self.assertEqual(evaluate_retrieval([long_row], lambda q, k: [{"source": "s.md", "text": "other"}])["mrr"], 0.0)
        hit = evaluate_retrieval([long_row], lambda q, k: [{"source": "x.md", "text": "Please RESERVE it for shorter  dispatches."}])
        self.assertEqual(hit["mrr"], 1.0)

    def test_abstain_precision_uses_explicit_labels(self):
        probes = [
            {"question": "q1", "reason": "none", "expected_abstain": True},
            {"question": "q2", "reason": "known", "expected_abstain": False},
            {"question": "q3", "reason": "known", "expected_abstain": False},
        ]
        retriever = lambda q, k: [] if q in {"q1", "q2"} else [{"id": 1}]
        result = evaluate_retrieval([], retriever, abstain_probes=probes, should_abstain=lambda docs: not docs)
        self.assertEqual(result["abstain_precision"], 0.5)

    def test_claim_metric_requires_paired_callbacks_and_bool_labels(self):
        row = {"question": "q", "answer": "a", "source": "s"}
        with self.assertRaises(ValueError):
            evaluate_retrieval([row], lambda q, k: [], answerer=lambda q, d: [])
        with self.assertRaises(ValueError):
            evaluate_retrieval([row], lambda q, k: [], answerer=lambda q, d: ["x"], claim_supported=lambda c, d: 1)

    def test_unavailable_metrics_are_null_and_lengths_not_truncated(self):
        result = evaluate_retrieval([], lambda q, k: [])
        self.assertIsNone(result["unsupported_claim_rate"])
        self.assertIsNone(result["abstain_precision"])
        with self.assertRaises(ValueError):
            evaluate_retrieval([], lambda q, k: [], top_k=0)


def q_truth(docs):
    return docs[0].get("text") == "a1"


if __name__ == "__main__":
    unittest.main()
