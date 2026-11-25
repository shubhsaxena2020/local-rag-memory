import unittest

from grounding.answer_protocol import (
    build_answer_request, extract_claims, format_facts, parse_response,
)

FACTS = [{"fact_id": "F1", "source": "notes/deploy.md", "confidence": 1.0, "status": "active",
          "text": "The staging server runs on port 8443."}]


LOW_CONF = [{"fact_id": "F3", "source": "notes/deploy.md", "confidence": 0.3, "status": "active",
             "score": 0.9, "text": "The staging server runs on port 8443."}]


class GroundedAnswerPromptTests(unittest.TestCase):
    def test_active_high_confidence_fact_may_be_paraphrased_or_filled_in(self):
        req = build_answer_request("grounded", "Which port does staging listen on?", FACTS)
        system, prompt = req["system"], req["prompt"]
        # The stored confidence (not the retrieval score) is what the answerer sees.
        self.assertIn("[F1] Source: notes/deploy.md | Confidence: 1.00 | Status: active", prompt)
        self.assertIn("QUESTION: Which port does staging listen on?", prompt)
        low = system.lower()
        self.assertIn("concise paraphrase or fill-in", low)
        self.assertIn("active retrieved fact with sufficient confidence (confidence >= 0.5)", low)
        self.assertIn("actually states the information the question requests", low)
        self.assertIn("give the value the fact supplies for the blank", low)
        self.assertIn("use only those facts, cite their ids and sources, and add no detail", low)

    def test_topical_only_evidence_abstains(self):
        facts = [{"fact_id": "F9", "source": "notes/acme.md", "confidence": 1.0, "status": "active",
                  "text": "Acme was founded in 2011."}]
        req = build_answer_request("probe", "Who is the CEO of Acme?", facts)
        self.assertIn("[F9]", req["prompt"])
        low = req["system"].lower()
        self.assertIn("reply with `unverified: <short reason>`", low)
        self.assertIn("overlap by topic, name or keyword without containing the answer", low)
        self.assertIn("do not guess", low)
        self.assertIn("general knowledge", low)

    def test_low_confidence_only_evidence_abstains(self):
        req = build_answer_request("grounded", "Which port does staging listen on?", LOW_CONF)
        # Low stored confidence is shown (RRF score 0.9 is not substituted) and the safeguard is kept.
        self.assertIn("Confidence: 0.30", req["prompt"])
        self.assertNotIn("Confidence: 0.90", req["prompt"])
        low = req["system"].lower()
        self.assertIn("the only supporting evidence has confidence below 0.5", low)
        self.assertNotIn("not a reason to abstain", low)
    def test_partial_multi_part_question_answers_supported_subparts(self):
        low = build_answer_request("grounded", "Q?", FACTS)["system"].lower()
        self.assertIn("if the facts support only part of a multi-part question", low)
        self.assertIn("answer each supported subpart with its citation", low)
        self.assertIn("clearly mark missing subparts unverified", low)
        self.assertIn("do not present a partial answer as if it were complete", low)

    def test_partial_answer_format_instruction_included(self):
        low = build_answer_request("grounded", "Q?", FACTS)["system"].lower()
        self.assertIn("state the supported answer first with citations, then mark unverified subparts", low)

    def test_total_abstention_safeguard_preserved(self):
        low = build_answer_request("grounded", "Q?", FACTS)["system"].lower()
        self.assertIn("when abstaining on an unanswerable question, do not state any answer", low)
        self.assertIn("superseded or retracted, begin with `unverified:`, give the fact, and warn that it is historical", low)

    def test_cited_answer_is_classified_as_answer_with_clean_claim(self):
        raw = "[F1] Staging runs on port 8443. (Source: notes/deploy.md)"
        parsed = parse_response(raw)
        self.assertEqual((parsed["status"], parsed["kind"]), ("ok", "answer"))
        self.assertEqual(parsed["cited_ids"], ["F1"])
        claims, _ = extract_claims(raw)
        self.assertEqual(len(claims), 1)
        self.assertIn("Staging runs on port 8443.", claims[0])
        self.assertNotIn("notes/deploy.md", claims[0])  # the source citation is not judged as a claim

    def test_no_evidence_case_keeps_abstention_protocol(self):
        req = build_answer_request("grounded", "Who is the CEO of Acme?", [])
        self.assertIn("(no facts retrieved)", req["prompt"])
        low = req["system"].lower()
        self.assertIn("unverified: <short reason>", low)
        self.assertIn("no active retrieved fact that states what the question asks", low)
        self.assertIn("do not guess", low)

    def test_abstention_output_is_still_parsed_as_abstain(self):
        for raw in ("unverified: no retrieved fact names a CEO.", "Unverified: the facts do not say."):
            self.assertEqual(parse_response(raw)["kind"], "abstain")

    def test_no_memory_and_unknown_condition_unchanged(self):
        req = build_answer_request("no_memory", "Q?", FACTS)
        self.assertEqual(req["prompt"], "Q?")
        self.assertNotIn("RETRIEVED FACTS", req["system"])
        with self.assertRaises(ValueError):
            build_answer_request("bogus", "Q?", FACTS)

    def test_format_facts_unchanged(self):
        self.assertEqual(format_facts([]), "(no facts retrieved)")


if __name__ == "__main__":
    unittest.main()
