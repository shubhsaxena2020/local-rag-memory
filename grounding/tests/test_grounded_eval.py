import json
import random
import unittest

from grounding.grounded_eval import build_split, golden_key, probe_key, run_grounded_eval, run_item
from grounding.grounded_metrics import (
    abstain_confusion, accuracy_metrics, calibrate_threshold, claim_metrics, evaluate_abstention, top_score,
)
from grounding.relevance import answer_in_text, is_relevant, normalize_text

GOLD = [{"id": i, "question": f"question {i} alpha", "answer": f"answer number {i} blue", "source": f"s{i % 3}.md"} for i in range(1, 21)]
PROBES = [{"question": f"unknown topic {i}", "reason": "none", "expected_abstain": True} for i in range(10)]
ANSWERER, JUDGE = "cc/claude-sonnet-5-5", "ag/gemini-3.8-flash"


def verdict_json(n, abstained=False, correct=True, verdict="supported"):
    return json.dumps({"verdicts": [{"claim_id": i, "verdict": verdict, "basis": "retrieved"} for i in range(1, n + 1)],
                       "abstained": abstained, "answer_correct": None if abstained else correct})


class StrictMatchingTests(unittest.TestCase):
    def test_source_alone_is_not_a_hit(self):
        self.assertFalse(is_relevant({"source": "a.md", "text": "unrelated snippet"}, {"source": "a.md", "answer": "the real answer"}))
        self.assertFalse(is_relevant({"source": "a.md"}, {"source": "a.md", "answer": "the real answer"}))

    def test_query_id_is_not_a_gold_passage_identifier(self):
        result = {"original_ids": [17], "text": "unrelated passage"}
        self.assertFalse(is_relevant(result, {"query_id": 17, "answer": "the real answer"}))
        self.assertTrue(is_relevant(result, {"target_fact_id": 17, "answer": "the real answer"}))

    def test_appendix_a_relevance_rules(self):
        # 1. short ordinary substring false
        self.assertFalse(answer_in_text("banana", "I ate a banana today"))
        self.assertFalse(answer_in_text("short", "A short phrase"))
        self.assertFalse(answer_in_text("(15", "timeout (15 minutes)"))

        # 2. exact short answer true
        self.assertTrue(answer_in_text("3", " 3 "))
        self.assertTrue(answer_in_text("Codex", "codex"))
        self.assertTrue(answer_in_text("banana", "banana"))

        # 3. complete entity token true
        self.assertTrue(answer_in_text("Codex", "Codex is a CLI"))

        # 4. complete number token true
        self.assertTrue(answer_in_text("3", "retried 3 times in 2026"))
        self.assertTrue(answer_in_text("42", "The answer is 42."))

        # 5. partial entity/number token false
        self.assertFalse(answer_in_text("3", "retried 30 times in 2026"))
        self.assertFalse(answer_in_text("3", "file_30_test"))
        self.assertFalse(answer_in_text("Codex", "CodexCLI is running"))
        self.assertFalse(answer_in_text("Codex", "Codexes are CLI tools"))

        # 6. >=12-char full phrase true
        self.assertTrue(answer_in_text("Timeout  After 15 minutes", "The job hit a TIMEOUT after 15\nminutes."))
        self.assertFalse(answer_in_text("timeout after 15 minutes", "timeout after 15"))  # partial answer
        self.assertFalse(answer_in_text("......", "a ...... b"))  # not substantive

        # 7. same source alone false
        self.assertFalse(is_relevant({"source": "a.md", "text": "unrelated snippet"}, {"source": "a.md", "answer": "the real answer"}))

    def test_invalid_types_are_misses(self):
        self.assertFalse(is_relevant("x", {"answer": "abcdefg"}))
        self.assertFalse(is_relevant({"text": 5}, {"answer": "abcdefg"}))
        self.assertFalse(is_relevant({"text": "abcdefg"}, {"answer": ""}))

    def test_eval_harness_uses_strict_rule(self):
        from eval.eval_harness import evaluate_retrieval
        # Short ordinary substring is not a hit
        row = {"question": "q", "answer": "banana", "source": "s.md"}
        r = evaluate_retrieval([row], lambda q, k: [{"source": "s.md", "text": "I ate a banana today"}])
        self.assertEqual(r["recall_at_5"], 0.0)
        # Exact match is a hit
        r = evaluate_retrieval([row], lambda q, k: [{"source": "other.md", "text": "banana"}])
        self.assertEqual(r["recall_at_5"], 1.0)


class SplitTests(unittest.TestCase):
    def test_reproducible_and_order_independent(self):
        a = build_split(GOLD, PROBES, seed=7)
        b = build_split(GOLD, PROBES, seed=7)
        sg, sp = GOLD[:], PROBES[:]
        random.Random(1).shuffle(sg)
        random.Random(2).shuffle(sp)
        self.assertEqual(a, b)
        self.assertEqual(a, build_split(sg, sp, seed=7))
        self.assertNotEqual(a["dev"], build_split(GOLD, PROBES, seed=8)["dev"])

    def test_both_kinds_in_both_parts_and_disjoint_complete(self):
        m = build_split(GOLD, PROBES, seed=7)
        for part in ("dev", "heldout"):
            self.assertTrue(any(k.startswith("golden:") for k in m[part]))
            self.assertTrue(any(k.startswith("probe:") for k in m[part]))
        self.assertFalse(set(m["dev"]) & set(m["heldout"]))
        self.assertEqual(len(m["dev"]) + len(m["heldout"]), len(GOLD) + len(PROBES))

    def test_real_corpus_split_has_both_kinds(self):
        from eval.eval_harness import ABSTAIN_PATH, GOLDEN_PATH, load_jsonl
        g, p = load_jsonl(GOLDEN_PATH), load_jsonl(ABSTAIN_PATH)
        m = build_split(g, p)
        self.assertEqual(sum(k.startswith("probe:") for k in m["dev"] + m["heldout"]), 40)
        self.assertEqual(m, build_split(g, p))


class MetricTests(unittest.TestCase):
    def test_claim_level_unsupported_rate(self):
        items = [
            {"verdicts": ["supported", "unsupported", "contradicted"], "bases": ["retrieved", None, None], "abstained": False, "errored": False},
            {"verdicts": ["supported"], "bases": ["reference"], "abstained": False, "errored": False},
            {"verdicts": [], "abstained": True, "errored": False},
            {"verdicts": [], "abstained": None, "errored": True},
        ]
        m = claim_metrics(items)
        self.assertEqual((m["claims"], m["unsupported"], m["contradicted"]), (4, 2, 1))
        self.assertEqual(m["unsupported_claim_rate"]["rate"], 0.5)
        self.assertEqual(m["errored_responses"], 1)
        self.assertEqual(m["not_retrieval_grounded"]["rate"], 0.5)
        self.assertIsNone(claim_metrics([])["unsupported_claim_rate"]["rate"])

    def test_accuracy_counts_abstain_and_errors_as_not_correct(self):
        items = [{"answer_correct": True, "abstained": False, "errored": False, "string_match": True},
                 {"answer_correct": False, "abstained": False, "errored": False, "string_match": False},
                 {"answer_correct": None, "abstained": True, "errored": False},
                 {"answer_correct": None, "abstained": None, "errored": True}]
        m = accuracy_metrics(items)
        self.assertEqual(m["accuracy"]["rate"], 0.25)
        self.assertEqual(m["accuracy_when_answered"]["rate"], 0.5)

    def test_abstain_precision_recall(self):
        m = abstain_confusion([True, True, False, False], [True, False, True, False])
        self.assertEqual((m["tp"], m["fp"], m["fn"], m["tn"]), (1, 1, 1, 1))
        self.assertEqual((m["precision"], m["recall"]), (0.5, 0.5))
        self.assertIsNone(abstain_confusion([False], [False])["precision"])
        with self.assertRaises(ValueError):
            abstain_confusion([True], [1])
        with self.assertRaises(ValueError):
            abstain_confusion([True], [])


class ThresholdTests(unittest.TestCase):
    @staticmethod
    def items(pairs):
        return [{"should_abstain": s, "score": sc, "abstained": False} for s, sc in pairs]

    def test_separable_dev_picks_separating_threshold(self):
        c = calibrate_threshold(self.items([(True, 0.1), (True, 0.2), (False, 0.7), (False, 0.9)]))
        self.assertTrue(c["target_met"])
        self.assertGreater(c["threshold"], 0.2)
        self.assertLessEqual(c["threshold"], 0.7)
        self.assertEqual((c["dev"]["precision"], c["dev"]["recall"]), (1.0, 1.0))

    def test_deterministic_regardless_of_order(self):
        dev = self.items([(True, 0.1), (True, 0.3), (False, 0.8), (False, 0.9)])
        self.assertEqual(calibrate_threshold(dev), calibrate_threshold(list(reversed(dev))))

    def test_unreachable_target_is_flagged(self):
        dev = self.items([(True, 0.5), (False, 0.5), (True, 0.5), (False, 0.5)])
        self.assertFalse(calibrate_threshold(dev, target_precision=0.9)["target_met"])
        self.assertIsNone(calibrate_threshold([])["threshold"])

    def test_gate_fires_below_threshold_only(self):
        m = evaluate_abstention(self.items([(True, 0.1), (False, 0.9)]), 0.5)
        self.assertEqual((m["tp"], m["fp"], m["tn"]), (1, 0, 1))

    def test_retrieval_rank_score_takes_precedence_over_source_confidence(self):
        self.assertEqual(top_score([{"score": 0.03, "confidence": 0.25},
                                    {"score": 0.01, "confidence": 0.95}]), 0.03)

    def test_answerability_score_preferred_over_rrf_score_and_confidence(self):
        facts = [{"answerability_score": 0.31, "score": 0.0328, "confidence": 0.99},
                 {"answerability_score": 0.31, "score": 0.0161, "confidence": 0.10}]
        self.assertEqual(top_score(facts), 0.31)
        # a larger RRF score must not win over a smaller answerability score
        self.assertEqual(top_score([{"answerability_score": 0.05, "score": 0.9}]), 0.05)
        self.assertEqual(top_score([{"answerability_score": 0, "score": 0.9}]), 0.0)

    def test_answerability_score_does_not_mix_scales_with_fallback_facts(self):
        facts = [{"score": 0.9}, {"answerability_score": 0.2, "score": 0.01}]
        self.assertEqual(top_score(facts), 0.2)
        self.assertEqual(top_score([{"answerability_score": None, "score": 0.03}]), 0.03)
        self.assertEqual(top_score([{"answerability_score": None, "confidence": 0.4}]), 0.4)

    def test_answerability_score_invalid_values_raise_and_do_not_fall_back(self):
        for bad in (float("nan"), float("inf"), True, "0.3"):
            with self.assertRaises(ValueError):
                top_score([{"answerability_score": bad, "score": 0.03}])

    def test_top_score_fallback_without_answerability_unchanged(self):
        self.assertEqual(top_score([]), 0.0)
        self.assertIsNone(top_score([{"text": "x"}]))
        self.assertEqual(top_score([{"confidence": 0.4}, {"confidence": 0.7}]), 0.7)
        self.assertEqual(top_score([{"score": 0.0, "confidence": 0.9}]), 0.0)
        for bad in (float("nan"), True, "0.3"):
            with self.assertRaises(ValueError):
                top_score([{"score": bad}])


def retriever(question, top_k):
    for r in GOLD:
        if r["question"] == question:
            return [{"text": r["answer"], "source": r["source"], "score": 0.9}]
    return [{"text": "nothing relevant", "source": "x.md", "score": 0.1}]


def good_answerer(request):
    if request["condition"] == "no_memory":
        return "A guess about something."
    if "nothing relevant" in request["prompt"]:
        return "unverified: no relevant facts found."
    facts = [ln for ln in request["prompt"].splitlines() if ln.startswith("[")]
    return facts[0].split("|", 3)[-1].strip()


def good_judge(prompt):
    claims = prompt.split("NUMBERED CLAIMS")[1]
    n = sum(1 for ln in claims.splitlines() if ln[:1].isdigit() and ". " in ln)
    abstain = "unverified" in prompt.split("ANSWER (untrusted):")[1][:40]
    closed_book = "closed-book" in prompt
    answer = prompt.split("ANSWER (untrusted):\n<<<\n")[1].split("\n>>>")[0]
    gold = prompt.split("GROUND-TRUTH ANSWER (for answer_correct only): ")[1].split("\n")[0] if "GROUND-TRUTH" in prompt else ""
    return verdict_json(n, abstained=abstain, correct=bool(gold) and gold in answer,
                        verdict="unsupported" if closed_book else "supported")


class InvalidCallbackTests(unittest.TestCase):
    def run_one(self, answerer, judge, gold="answer number 1 blue"):
        return run_item(condition="grounded", question="q", facts=[{"text": "t", "score": 0.5}], gold=gold, expect_abstain=False,
                        answerer=answerer, judge=judge, show_memory=True)

    def test_non_string_empty_and_raising_answerer_are_errored(self):
        for bad in (123, None, "   ", ["x"]):
            r = self.run_one(lambda req, b=bad: b, good_judge)
            self.assertTrue(r["errored"], bad)
            self.assertIsNone(r["abstained"])

        def boom(req):
            raise RuntimeError("secret-token-123")
        r = self.run_one(boom, good_judge)
        self.assertTrue(r["errored"])
        self.assertNotIn("secret-token", r["error"])

    def test_invalid_judge_output_is_errored_not_supported(self):
        bad_outputs = [
            "not json", "", None, 5, "{}",
            json.dumps({"abstained": False, "answer_correct": True, "verdicts": []}),
            json.dumps({"abstained": False, "answer_correct": True, "verdicts": [{"claim_id": 1, "verdict": "maybe"}]}),
            json.dumps({"abstained": False, "answer_correct": None, "verdicts": [{"claim_id": 1, "verdict": "supported"}]}),
        ]
        for bad in bad_outputs:
            r = self.run_one(lambda req: "The answer is blue.", lambda p, b=bad: b)
            self.assertTrue(r["errored"], bad)
            self.assertEqual(r["verdicts"], [])

        def boom(p):
            raise TimeoutError("x")
        self.assertTrue(self.run_one(lambda req: "The answer is blue.", boom)["errored"])

    def test_errors_never_counted_as_supported_or_correct(self):
        r = self.run_one(lambda req: 5, good_judge)
        self.assertEqual(claim_metrics([r])["claims"], 0)
        self.assertEqual(claim_metrics([r])["errored_responses"], 1)
        self.assertEqual(accuracy_metrics([r])["accuracy"]["k"], 0)

    def test_bad_retriever_output_and_same_family_rejected(self):
        with self.assertRaises(ValueError):
            run_grounded_eval(GOLD, PROBES, retriever=lambda q, k: "bad", answerer=good_answerer, judge=good_judge,
                              answerer_model=ANSWERER, judge_model=JUDGE)
        with self.assertRaises(ValueError):
            run_grounded_eval(GOLD, PROBES, retriever=retriever, answerer=good_answerer, judge=good_judge,
                              answerer_model=ANSWERER, judge_model="cc/claude-opus-5-5")
        with self.assertRaises(ValueError):
            run_grounded_eval(GOLD, PROBES, retriever=retriever, answerer="x", judge=good_judge,
                              answerer_model=ANSWERER, judge_model=JUDGE)


class EndToEndTests(unittest.TestCase):
    def run_eval(self, answerer=good_answerer, judge=good_judge):
        return run_grounded_eval(GOLD, PROBES, retriever=retriever, answerer=answerer, judge=judge,
                                 answerer_model=ANSWERER, judge_model=JUDGE, seed=7)

    def test_conditions_scored_on_heldout_only(self):
        out = self.run_eval()
        h = out["heldout"]
        n_gold, n_probe = h["split_sizes"]["heldout"]["golden"], h["split_sizes"]["heldout"]["probe"]
        self.assertEqual(h["no_memory"]["n"], n_gold)
        self.assertEqual(h["grounded"]["n"], n_gold)
        self.assertEqual(h["probe"]["n"], n_probe)
        self.assertEqual(h["grounded"]["accuracy"]["accuracy"]["rate"], 1.0)
        self.assertEqual(h["grounded"]["answerer"]["unsupported_claim_rate"]["rate"], 0.0)
        self.assertEqual(h["no_memory"]["answerer"]["unsupported_claim_rate"]["rate"], 1.0)
        self.assertEqual(h["abstention"]["precision"], 1.0)
        self.assertEqual(h["abstention"]["recall"], 1.0)
        self.assertEqual(h["abstention"]["n"], n_gold + n_probe)

    def test_threshold_tuned_on_dev_only(self):
        base = self.run_eval()
        dev_keys = set(base["manifest"]["dev"])
        held = {r["question"] for r in GOLD if golden_key(r) not in dev_keys}
        held |= {r["question"] for r in PROBES if probe_key(r) not in dev_keys}

        def held_only_wrong(request):
            if request["condition"] != "no_memory" and any(q in request["prompt"] for q in held):
                return "A confident invented answer."
            return good_answerer(request)
        out = self.run_eval(answerer=held_only_wrong)
        self.assertEqual(out["calibration_dev"], base["calibration_dev"])  # held-out behaviour cannot move the dev threshold
        self.assertEqual(out["heldout"]["threshold"], base["heldout"]["threshold"])
        self.assertLess(out["heldout"]["grounded"]["accuracy"]["accuracy"]["rate"], 1.0)  # but it does move held-out scores

    def test_reproducible_run(self):
        a, b = self.run_eval(), self.run_eval()
        self.assertEqual(a["heldout"], b["heldout"])
        self.assertEqual(a["manifest"], b["manifest"])

    def test_errors_reported_not_hidden(self):
        h = self.run_eval(judge=lambda p: "garbage")["heldout"]
        self.assertEqual(h["grounded"]["errored"], h["grounded"]["n"])
        self.assertIsNone(h["grounded"]["answerer"]["unsupported_claim_rate"]["rate"])


if __name__ == "__main__":
    unittest.main()
