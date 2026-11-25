import json
import re
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import MagicMock, patch

from judge.judge_harness import (
    ALLOWED_LOOPBACK_HOSTS,
    LABEL_NAMES,
    MUTATION_LABELS,
    JudgeTimeout,
    _NoRedirectHandler,
    _http_generate,
    _validate_endpoint,
    build_payload,
    call_ollama_judge,
    compute_full_accuracy,
    compute_mutation_metrics,
    compute_report,
    dataset_summary,
    load_benchmark_data,
    ollama_judge_prompt,
    parse_judge_response,
    parse_label,
    run_benchmark_on_file,
    run_with_deadline,
)

DATA = Path(__file__).resolve().parents[1] / "benchmark_data.jsonl"


class JudgeHarnessTests(unittest.TestCase):
    def write_jsonl(self, records):
        handle = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", encoding="utf-8", delete=False)
        self.addCleanup(Path(handle.name).unlink, missing_ok=True)
        with handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        return handle.name

    def record(self, **overrides):
        record = {"id": 1, "fact1": "one", "fact2": "two", "label": 0, "category": LABEL_NAMES[0]}
        record.update(overrides)
        return record

    # ---- loader -----------------------------------------------------------
    def test_load_valid_data_and_category_mismatch(self):
        path = self.write_jsonl([self.record()])
        self.assertEqual(len(load_benchmark_data(path)), 1)
        bad_path = self.write_jsonl([self.record(category="wrong")])
        with self.assertRaisesRegex(ValueError, "Category mismatch"):
            load_benchmark_data(bad_path)

    def test_duplicate_id_and_pair_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate id"):
            load_benchmark_data(self.write_jsonl([self.record(), self.record(fact1="x")]))
        with self.assertRaisesRegex(ValueError, "Duplicate fact pair"):
            load_benchmark_data(self.write_jsonl([self.record(), self.record(id=2)]))

    def test_loader_rejects_bad_types_and_ranges(self):
        for bad in (self.record(label=7, category="x"), self.record(label=True), self.record(id="1"),
                    self.record(fact1="  "), self.record(subtype=""), self.record(label=-1, category="x")):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                load_benchmark_data(self.write_jsonl([bad]))
        with self.assertRaisesRegex(ValueError, "Missing required field"):
            load_benchmark_data(self.write_jsonl([{"id": 1}]))

    # ---- parsers ----------------------------------------------------------
    def test_parse_label_accepts_only_ascii_single_digit(self):
        for expected in range(7):
            self.assertEqual(parse_label(f" {expected} \n"), expected)
        for invalid in ("", "7", "10", "2 words", "label 2", "2 3", "\u0663", None, 3):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_label(invalid)

    def test_parse_judge_response_strict_json(self):
        self.assertEqual(parse_judge_response('{"label": 1}'), 1)
        self.assertEqual(parse_judge_response('  {"label":6}\n'), 6)
        self.assertEqual(parse_judge_response("3"), 3)  # lone-digit fallback
        bad = ['{"label": 7}', '{"label": "1"}', '{"label": 1.0}', '{"label": true}', '{"label": null}',
               '{"label": 1, "why": "x"}', '{"lbl": 1}', '[1]', '{"label": 1', '', '   ',
               'The answer is 1', '```json\n{"label": 1}\n```', '1\n2', None]
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_judge_response(text)

    # ---- metrics ----------------------------------------------------------
    def test_metrics_confusion_matrix_and_accuracy(self):
        metrics = compute_mutation_metrics([0, 1, 2, 3, 4, 5, 6], [0, 1, 1, 3, 4, 5, 6])
        self.assertEqual(metrics["confusion_matrix"], {"tp": 2, "tn": 5, "fp": 0, "fn": 0})
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(compute_full_accuracy([0, 1], [0, 2]), 0.5)

    def test_binary_accuracy_can_exceed_exact_accuracy(self):
        # 1<->2 confusion is a correct mutation decision but a wrong class.
        y_true, y_pred = [1, 2, 0, 6], [2, 1, 6, 0]
        self.assertEqual(compute_mutation_metrics(y_true, y_pred)["accuracy"], 1.0)
        self.assertEqual(compute_full_accuracy(y_true, y_pred), 0.0)

    def test_metrics_precision_recall_false_rates(self):
        m = compute_mutation_metrics([1, 1, 0, 0], [0, 1, 1, 0])
        self.assertEqual(m["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["false_mutation_rate"], 0.5)
        self.assertAlmostEqual(m["missed_mutation_rate"], 0.5)

    def test_metric_lengths_and_labels(self):
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            compute_mutation_metrics([0], [])
        for labels in ([True], [-1], [7], ["1"]):
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                compute_mutation_metrics(labels, [0])
        with self.assertRaises(ValueError):
            compute_full_accuracy([0], [False])

    def test_compute_report_scores_none_as_wrong(self):
        y_true = [1, 0, 3, 2]
        y_pred = [1, None, 3, None]
        r = compute_report(y_true, y_pred, ["negation", "a", "b", "w"])
        self.assertEqual(r["n"], 4)
        self.assertEqual(r["n_answered"], 2)
        self.assertAlmostEqual(r["exact_accuracy"], 0.5)
        self.assertAlmostEqual(r["exact_accuracy_on_answered"], 1.0)
        # None on a mutation -> fn, None on a non-mutation -> fp
        self.assertEqual(r["mutation_metrics"]["confusion_matrix"], {"tp": 1, "tn": 1, "fp": 1, "fn": 1})
        self.assertAlmostEqual(r["mutation_metrics"]["accuracy"], 0.5)
        self.assertEqual(r["per_subtype"]["1/negation"], {"correct": 1, "n": 1, "accuracy": 1.0})
        self.assertEqual(r["per_subtype"]["2/w"]["correct"], 0)
        self.assertEqual(r["confusion_matrix_rows_true_cols_pred"][1][1], 1)
        self.assertEqual(r["per_class"]["1"]["recall"], 1.0)

    def test_compute_report_all_none_is_zero_not_crash(self):
        r = compute_report([1, 0], [None, None])
        self.assertEqual(r["exact_accuracy"], 0.0)
        self.assertEqual(r["mutation_metrics"]["accuracy"], 0.0)
        self.assertEqual(r["exact_accuracy_on_answered"], 0.0)

    # ---- request / prompt shape ------------------------------------------
    def test_prompt_is_short_and_lists_all_labels(self):
        p = ollama_judge_prompt("A fact.", "B fact.")
        self.assertLess(len(p), 800)
        for digit in "0123456":
            self.assertRegex(p, rf"(?m)^{digit} ")
        self.assertIn("A fact.", p)
        self.assertIn("B fact.", p)

    def test_payload_is_constrained_and_capped(self):
        pl = build_payload("a", "b", "m")
        self.assertIs(pl["stream"], False)
        self.assertEqual(pl["format"]["required"], ["label"])
        self.assertEqual(pl["format"]["properties"]["label"]["enum"], list(range(7)))
        self.assertLessEqual(pl["options"]["num_predict"], 32)
        self.assertEqual(pl["options"]["temperature"], 0.0)
        self.assertIs(pl["think"], False)
        self.assertNotIn("think", build_payload("a", "b", "m", think=None))

    # ---- bounded call -----------------------------------------------------
    def test_call_uses_injected_transport(self):
        seen = {}

        def transport(url, payload, timeout):
            seen.update(url=url, payload=payload, timeout=timeout)
            return {"response": '{"label": 2}'}

        self.assertEqual(call_ollama_judge("one", "two", model="m", timeout_s=3, transport=transport), 2)
        self.assertEqual(seen["timeout"], 3)
        self.assertEqual(seen["payload"]["model"], "m")

    def test_call_is_loopback_only_and_bounded(self):
        with self.assertRaisesRegex(ValueError, "loopback"):
            call_ollama_judge("one", "two", base_url="http://127.0.0.1.evil.invalid")
        with self.assertRaisesRegex(ValueError, "at most"):
            call_ollama_judge("x" * 10001, "two")
        with self.assertRaisesRegex(ValueError, "timeout_s"):
            call_ollama_judge("one", "two", timeout_s=0, transport=lambda *a: {})

    def test_validate_endpoint_valid_loopback_urls(self):
        valid_urls = [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://127.0.0.1",
            "http://localhost:8080",
            "http://[::1]:11434",
            "http://127.0.0.1:11434/",
        ]
        for url in valid_urls:
            try:
                _validate_endpoint(url)
            except Exception as e:
                self.fail(f"_validate_endpoint unexpectedly raised on valid URL '{url}': {e}")

    def test_validate_endpoint_rejects_non_http_and_remote(self):
        invalid_urls = [
            "https://127.0.0.1:11434",
            "ftp://localhost:11434",
            "file:///etc/passwd",
            "http://192.168.1.100:11434",
            "http://evil.com:11434",
            "http://127.0.0.1.attacker.com:11434",
            "http://0.0.0.0:11434",
            "http://user@localhost:11434",
            "http://user:pass@127.0.0.1:11434",
            "http://localhost:11434/?token=x",
            "http://localhost:11434/#fragment",
            "http://localhost:notaport",
            "http://localhost:0",
            "http://127.0.0.1:65536",
            "",
            "   ",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    _validate_endpoint(url)

    def test_validate_endpoint_rejects_non_string(self):
        for bad in (None, 123, ["http://127.0.0.1:11434"]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                _validate_endpoint(bad)  # type: ignore[arg-type]

    def test_no_redirect_handler_blocks_redirects(self):
        handler = _NoRedirectHandler()
        self.assertIsNone(
            handler.redirect_request(
                None, None, 301, "Moved Permanently", None, "http://127.0.0.1:11434/redirect"
            )
        )
        self.assertIsNone(
            handler.redirect_request(
                None, None, 302, "Found", None, "http://127.0.0.1:11434/redirect"
            )
        )
        self.assertIsNone(
            handler.redirect_request(
                None, None, 307, "Temporary Redirect", None, "http://127.0.0.1:11434/redirect"
            )
        )

    def test_http_generate_validates_endpoint(self):
        with self.assertRaises(ValueError):
            _http_generate("http://evil.com:11434", {}, 1.0)
        with self.assertRaises(ValueError):
            _http_generate("https://127.0.0.1:11434", {}, 1.0)

    def test_http_generate_uses_no_redirect_and_empty_proxies(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_resp = MagicMock()
            mock_resp.read.return_value = b'{"response": "1"}'
            mock_build_opener.return_value.open.return_value.__enter__.return_value = mock_resp

            result = _http_generate("http://127.0.0.1:11434/", {"test": "payload"}, 2.0)
            self.assertEqual(result, {"response": "1"})
            mock_build_opener.assert_called_once()
            args = mock_build_opener.call_args[0]
            # Verify ProxyHandler with empty proxies is passed to prevent environment proxies
            has_empty_proxy_handler = any(
                isinstance(arg, urllib.request.ProxyHandler) and arg.proxies == {}
                for arg in args
            )
            has_no_redirect_handler = any(
                isinstance(arg, _NoRedirectHandler) or arg is _NoRedirectHandler
                for arg in args
            )
            self.assertTrue(has_empty_proxy_handler)
            self.assertTrue(has_no_redirect_handler)

    def test_http_generate_handles_urlerror(self):
        with patch("urllib.request.build_opener") as mock_build_opener:
            mock_build_opener.return_value.open.side_effect = urllib.error.URLError("connection refused")
            with self.assertRaises(ConnectionError):
                _http_generate("http://127.0.0.1:11434", {}, 2.0)

    def test_call_hard_timeout_does_not_hang(self):
        def slow(url, payload, timeout):
            time.sleep(3)
            return {"response": "1"}

        t0 = time.perf_counter()
        with self.assertRaises(JudgeTimeout):
            call_ollama_judge("one", "two", timeout_s=0.2, transport=slow)
        self.assertLess(time.perf_counter() - t0, 2.0)

    def test_run_with_deadline_propagates_errors_and_values(self):
        self.assertEqual(run_with_deadline(lambda: 5, 1), 5)
        with self.assertRaises(KeyError):
            run_with_deadline(lambda: {}["x"], 1)

    def test_call_rejects_unparsable_model_output(self):
        for text in ("I think 1", "", '{"label": 99}'):
            with self.subTest(text=text), self.assertRaises(ValueError):
                call_ollama_judge("one", "two", transport=lambda *a, t=text: {"response": t})

    # ---- bounded batch ----------------------------------------------------
    def test_batch_perfect_judge_via_injection(self):
        data = load_benchmark_data(DATA)
        truth = {(r["fact1"], r["fact2"]): r["label"] for r in data}
        res = run_benchmark_on_file(DATA, judge=lambda a, b: truth[(a, b)])
        self.assertEqual(res["exact_accuracy"], 1.0)
        self.assertEqual(res["mutation_metrics"]["accuracy"], 1.0)
        self.assertTrue(res["complete"])
        self.assertEqual(res["failed"], 0)
        self.assertEqual(len(res["predictions"]), len(data))
        self.assertIsNone(res["aborted_reason"])

    def test_batch_aborts_after_consecutive_failures(self):
        calls = []

        def bad(a, b):
            calls.append(1)
            raise ConnectionError("down")

        res = run_benchmark_on_file(DATA, judge=bad, max_consecutive_failures=3)
        self.assertEqual(len(calls), 3)
        self.assertEqual(res["aborted_reason"], "3_consecutive_failures")
        self.assertEqual(res["failed"], 3)
        self.assertEqual(res["skipped"], res["num_examples"] - 3)
        self.assertEqual(res["exact_accuracy"], 0.0)
        self.assertFalse(res["complete"])

    def test_batch_hung_judge_is_bounded_by_budget(self):
        def hang(a, b):
            time.sleep(3)
            return 0

        t0 = time.perf_counter()
        res = run_benchmark_on_file(DATA, judge=hang, per_call_timeout_s=0.1, batch_budget_s=0.6,
                                    max_consecutive_failures=100)
        elapsed = time.perf_counter() - t0
        self.assertLess(elapsed, 3.0)
        self.assertEqual(res["failures_by_type"].get("timeout", 0), res["failed"])
        self.assertGreater(res["failed"], 0)
        self.assertEqual(res["aborted_reason"], "batch_budget_exhausted")
        self.assertEqual(res["mutation_metrics"]["accuracy"], 0.0)

    def test_batch_classifies_socket_timeout_as_timeout(self):
        def sock(a, b):
            raise TimeoutError("timed out")

        res = run_benchmark_on_file(DATA, judge=sock, max_consecutive_failures=2)
        self.assertEqual(res["failures_by_type"], {"timeout": 2})

    def test_batch_counts_invalid_judge_result(self):
        res = run_benchmark_on_file(DATA, judge=lambda a, b: 99, max_consecutive_failures=2, limit=10)
        self.assertEqual(res["failures_by_type"], {"invalid_output": 2})

    def test_batch_limit(self):
        res = run_benchmark_on_file(DATA, judge=lambda a, b: 0, limit=7)
        self.assertEqual(res["num_examples"], 7)


class BenchmarkDatasetTests(unittest.TestCase):
    """Audits the shipped benchmark_data.jsonl (labels, coverage, structure)."""

    @classmethod
    def setUpClass(cls):
        cls.rows = load_benchmark_data(DATA)

    def sub(self, label, subtype):
        return [r for r in self.rows if r["label"] == label and r.get("subtype") == subtype]

    def test_at_least_60_pairs(self):
        self.assertGreaterEqual(len(self.rows), 60)

    def test_required_coverage_per_class(self):
        counts = dataset_summary(self.rows)["by_label"]
        # 5 (ambiguous) is an abstain label and intentionally has no rows.
        for label in (0, 1, 2, 3, 4, 6):
            self.assertGreaterEqual(counts.get(label, 0), 8, f"label {label} under-covered")
        self.assertNotIn(5, counts)

    def test_required_subtypes_present(self):
        for label, subtype, minimum in ((1, "negation", 5), (1, "numeric", 5), (1, "unit", 5),
                                        (0, "different_attribute", 10), (6, "reworded", 8),
                                        (6, "unit_equivalent", 3), (2, "withdrawal", 8),
                                        (4, "explicit_time_scopes", 8),
                                        (3, "different_entity_same_attribute", 5)):
            self.assertGreaterEqual(len(self.sub(label, subtype)), minimum, (label, subtype))

    def test_every_row_has_subtype(self):
        self.assertTrue(all(r.get("subtype") for r in self.rows))

    def test_mutation_vs_non_mutation_balance(self):
        s = dataset_summary(self.rows)
        self.assertGreaterEqual(s["mutation_rows"], 20)
        self.assertGreaterEqual(s["non_mutation_rows"], 30)
        self.assertEqual(s["mutation_rows"], sum(1 for r in self.rows if r["label"] in MUTATION_LABELS))

    def test_no_symmetric_duplicates(self):
        seen = set()
        for r in self.rows:
            key = frozenset((r["fact1"], r["fact2"]))
            self.assertNotIn(key, seen, r)
            seen.add(key)

    def test_negation_rows_have_exactly_one_negated_side(self):
        neg = re.compile(r"\b(not|does not|is not)\b|n't", re.I)
        for r in self.sub(1, "negation"):
            self.assertNotEqual(bool(neg.search(r["fact1"])), bool(neg.search(r["fact2"])), r)

    def test_retraction_rows_use_withdrawal_language(self):
        kw = re.compile(r"disregard|retract|forget|scratch that|withdraw|cancel that|delete|remove", re.I)
        for r in self.sub(2, "withdrawal"):
            self.assertRegex(r["fact2"], kw)

    def test_non_retraction_rows_do_not_use_withdrawal_language(self):
        kw = re.compile(r"disregard|retract|forget|scratch that|withdraw|cancel that", re.I)
        for r in self.rows:
            if r["label"] != 2:
                self.assertIsNone(kw.search(r["fact2"]), r)

    def test_temporal_rows_time_scope_both_facts(self):
        marker = re.compile(r"\b(\d{4}|until|since|before|during|monday|friday|january)\b", re.I)
        for r in self.sub(4, "explicit_time_scopes"):
            self.assertTrue(marker.search(r["fact1"]) and marker.search(r["fact2"]), r)

    def test_numeric_and_unit_rows_contain_numbers(self):
        for sub in ("numeric", "unit"):
            for r in self.sub(1, sub):
                self.assertNotEqual(r["fact1"], r["fact2"])
                self.assertTrue(re.search(r"\d", r["fact1"]) and re.search(r"\d", r["fact2"]), r)

    def test_unit_rows_keep_the_number_but_change_the_unit(self):
        num = re.compile(r"\d[\d,.:]*")
        for r in self.sub(1, "unit"):
            self.assertEqual(num.findall(r["fact1"]), num.findall(r["fact2"]), r)

    def test_unit_equivalent_paraphrases_differ_textually(self):
        for r in self.sub(6, "unit_equivalent"):
            self.assertNotEqual(r["fact1"], r["fact2"])

    def test_paraphrases_are_not_mutations(self):
        self.assertNotIn(6, MUTATION_LABELS)
        self.assertTrue(all(r["label"] not in MUTATION_LABELS for r in self.rows if r["subtype"] in ("reworded", "unit_equivalent")))

    def test_all_prompts_fit_under_cap(self):
        for r in self.rows:
            self.assertLess(len(ollama_judge_prompt(r["fact1"], r["fact2"])), 1200)


if __name__ == "__main__":
    unittest.main()
