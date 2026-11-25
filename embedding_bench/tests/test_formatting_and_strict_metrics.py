"""Offline tests (no Ollama, no network): per-model formatters, dedup, strict relevance,
variant plan and resident-model preservation in the corrected dense benchmark."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedding_bench.client import OllamaEmbeddingClient
from embedding_bench.dataset import CorpusRecord, GoldenQuery, load_golden_set, normalize_passage
from embedding_bench.formatting import (
    FORMATTERS, VARIANTS, format_document, format_query, get_formatter, resolve_variants,
)
from embedding_bench.harness import benchmark_single_candidate, run_embedding_benchmark, select_winner
from embedding_bench.index import SearchResult
from embedding_bench.metrics import evaluate_single_query, is_relevant_hit

Q = "How do I fix the timeout?"
P = "Raise the limit to 600s."
QWEN_Q = ("Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
          "Query:")


class TestFormatters(unittest.TestCase):
    def test_nomic_prefixed(self):
        self.assertEqual(format_query("nomic", Q), "search_query: How do I fix the timeout?")
        self.assertEqual(format_document("nomic", P, "a.md"), "search_document: Raise the limit to 600s.")

    def test_nomic_raw_ablation_is_unprefixed(self):
        self.assertEqual(format_query("raw", Q), Q)
        self.assertEqual(format_document("raw", P, "a.md"), P)

    def test_qwen3_query_instruction_only(self):
        self.assertEqual(format_query("qwen3", Q), QWEN_Q + Q)
        self.assertEqual(format_document("qwen3", P, "a.md"), P)  # documents unprefixed

    def test_gemma_query_and_title_document(self):
        self.assertEqual(format_query("gemma", Q), "task: search result | query: How do I fix the timeout?")
        self.assertEqual(format_document("gemma", P, "a.md"), "title: a.md | text: Raise the limit to 600s.")

    def test_gemma_missing_source_uses_none(self):
        self.assertEqual(format_document("gemma", P, ""), "title: none | text: Raise the limit to 600s.")
        self.assertEqual(format_document("gemma", P, "   "), "title: none | text: Raise the limit to 600s.")

    def test_bge_m3_raw_both_sides(self):
        self.assertEqual(format_query("bge-m3", Q), Q)
        self.assertEqual(format_document("bge-m3", P, "a.md"), P)

    def test_whitespace_stripped_and_unknown_rejected(self):
        self.assertEqual(format_query("nomic", "  hi \n"), "search_query: hi")
        with self.assertRaises(ValueError):
            get_formatter("nope")

    def test_every_non_raw_formatter_has_source_link(self):
        for name, f in FORMATTERS.items():
            if name != "raw":
                self.assertTrue(f.doc_url.startswith("https://"), name)


class TestVariantPlan(unittest.TestCase):
    def test_default_plan(self):
        keys = [v.key for v in VARIANTS]
        self.assertEqual(keys, ["nomic-embed-text", "nomic-embed-text@raw", "qwen3-embedding:0.6b",
                                "embeddinggemma", "bge-m3"])
        fm = {v.key: v.formatter for v in VARIANTS}
        self.assertEqual(fm["nomic-embed-text"], "nomic")
        self.assertEqual(fm["nomic-embed-text@raw"], "raw")
        abl = [v for v in VARIANTS if v.ablation_of]
        self.assertEqual([(v.key, v.model) for v in abl], [("nomic-embed-text@raw", "nomic-embed-text")])

    def test_resolve_custom_token_is_raw(self):
        v = resolve_variants(["bge-m3", "some-model:1b"])
        self.assertEqual([(x.model, x.formatter) for x in v], [("bge-m3", "bge-m3"), ("some-model:1b", "raw")])


class TestDeduplication(unittest.TestCase):
    def _load(self, rows):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        try:
            return load_golden_set(f.name)
        finally:
            Path(f.name).unlink()

    def test_normalized_dedup_links_all_ids(self):
        corpus, queries = self._load([
            {"id": 1, "question": "q1", "answer": "Same  answer", "source": "a.md"},
            {"id": 2, "question": "q2", "answer": "same answer\n", "source": "b.md"},
            {"id": 3, "question": "q3", "answer": "Other", "source": "c.md"},
        ])
        self.assertEqual(len(corpus), 2)
        self.assertEqual(corpus[0].original_ids, {1, 2})
        self.assertEqual(corpus[0].source, "a.md")  # first source kept
        self.assertEqual([q.target_doc_id for q in queries], ["doc_0", "doc_0", "doc_1"])

    def test_normalize_passage(self):
        self.assertEqual(normalize_passage("  A\t B\n"), "a b")
        self.assertEqual(normalize_passage("\uff21\uff22"), "ab")  # NFKC full-width

    def test_real_golden_set_counts_and_one_gold_per_query(self):
        corpus, queries = load_golden_set()
        self.assertEqual((len(queries), len(corpus)), (158, 142))
        for q in queries:
            gold = [d for d in corpus
                    if is_relevant_hit(SearchResult(doc_id=d.doc_id, score=0, text=d.text, source=d.source,
                                                    original_ids=d.original_ids), q)]
            self.assertEqual(len(gold), 1, q.query_id)


def _res(doc_id, text, source="s.md", ids=()):
    return SearchResult(doc_id=doc_id, score=1.0, text=text, source=source, original_ids=set(ids))


class TestStrictRelevance(unittest.TestCase):
    def setUp(self):
        self.q = GoldenQuery(query_id=7, question="q", target_answer="Raise the limit to 600s",
                             target_source="s.md", target_doc_id="doc_3")

    def test_exact_doc_id(self):
        self.assertTrue(is_relevant_hit(_res("doc_3", "anything"), self.q))

    def test_query_id_membership(self):
        self.assertTrue(is_relevant_hit(_res("doc_9", "x", ids={7}), self.q))

    def test_normalized_exact_text(self):
        self.assertTrue(is_relevant_hit(_res("doc_9", "  raise THE limit  to 600s\n"), self.q))

    def test_same_source_is_not_relevant(self):
        self.assertFalse(is_relevant_hit(_res("doc_9", "unrelated", source="s.md"), self.q))

    def test_substring_not_relevant_in_strict(self):
        longer = _res("doc_9", "Raise the limit to 600s and restart the daemon")
        self.assertFalse(is_relevant_hit(longer, self.q))
        self.assertTrue(is_relevant_hit(longer, self.q, mode="legacy_substring"))

    def test_short_answer_substring_not_relevant(self):
        q = GoldenQuery(query_id=1, question="q", target_answer="5", target_doc_id="doc_1")
        self.assertFalse(is_relevant_hit(_res("doc_2", "Listen on port 5000"), q))
        q2 = GoldenQuery(query_id=1, question="q", target_answer="no", target_doc_id="doc_1")
        self.assertFalse(is_relevant_hit(_res("doc_2", "no entry"), q2))

    def test_unknown_mode(self):
        with self.assertRaises(ValueError):
            is_relevant_hit(_res("d", "t"), self.q, mode="loose")

    def test_ranks_use_strict_rule(self):
        results = [_res("doc_1", "other", source="s.md"), _res("doc_2", "Raise the limit to 600s and more"),
                   _res("doc_3", "Raise the limit to 600s")]
        ev = evaluate_single_query(results, self.q)
        self.assertEqual(ev["first_rank"], 3)
        self.assertFalse(ev["hit_at_1"])
        self.assertTrue(ev["hit_at_5"])
        self.assertAlmostEqual(ev["reciprocal_rank"], 1 / 3)


def _fake_client():
    client = MagicMock(spec=OllamaEmbeddingClient)
    client.get_model_info.return_value = {}
    client.unload_model.return_value = True
    seen = []

    def embed(model, inputs, truncate=True):
        items = [inputs] if isinstance(inputs, str) else list(inputs)
        seen.append((model, items))
        return [[1.0, 0.0] for _ in items], 0.001

    client.embed.side_effect = embed
    return client, seen


class TestHarnessFormatting(unittest.TestCase):
    def setUp(self):
        self.corpus = [CorpusRecord("doc_0", "ans", "a.md", {1})]
        self.queries = [GoldenQuery(1, "ques", "ans", "a.md", "doc_0")]

    def _run(self, formatter, resident=frozenset(), unload_after=True):
        client, seen = _fake_client()
        res = benchmark_single_candidate("m", client, self.corpus, self.queries, set(resident), [],
                                         formatter=formatter, unload_after=unload_after)
        return client, seen, res

    def test_embedded_strings_per_formatter(self):
        expect = {
            "nomic": (["search_document: ans"], ["search_query: ques"]),
            "raw": (["ans"], ["ques"]),
            "qwen3": (["ans"], [QWEN_Q + "ques"]),
            "gemma": (["title: a.md | text: ans"], ["task: search result | query: ques"]),
            "bge-m3": (["ans"], ["ques"]),
        }
        for name, (docs, qs) in expect.items():
            _, seen, res = self._run(name)
            self.assertEqual(seen[0][1], docs, name)
            self.assertEqual(seen[1][1], qs, name)
            self.assertEqual(res["formatter"], name)
            self.assertEqual(res["relevance_mode"], "strict")
            self.assertEqual(res["metrics"]["recall_at_5"], 1.0)

    def test_resident_model_never_unloaded(self):
        client, _, res = self._run("raw", resident={"m"}, unload_after=True)
        client.unload_model.assert_not_called()
        self.assertEqual(res["unload_status"], "kept_loaded_resident_at_start")

    def test_deferred_unload_between_variants(self):
        client, _, res = self._run("raw", unload_after=False)
        client.unload_model.assert_not_called()
        self.assertEqual(res["unload_status"], "unload_deferred")


class TestSequentialLoop(unittest.TestCase):
    def test_order_ablation_resident_preserved_no_pull_delete(self):
        client, seen = _fake_client()
        client.check_health.return_value = {"connected": True, "version": "x"}
        client.get_running_models.return_value = {"resident-llm:1b", "nomic-embed-text", "nomic-embed-text:latest"}
        client.get_available_models.return_value = [{"name": "nomic-embed-text:latest", "size": 1},
                                                    {"name": "bge-m3:latest", "size": 1}]
        client.get_available_model_names.return_value = {
            "nomic-embed-text", "nomic-embed-text:latest", "bge-m3", "bge-m3:latest"}
        with patch("embedding_bench.harness.OllamaEmbeddingClient", return_value=client):
            out = run_embedding_benchmark(max_queries=3)
        c = out["candidates"]
        self.assertEqual(list(c), ["nomic-embed-text", "nomic-embed-text@raw", "qwen3-embedding:0.6b",
                                   "embeddinggemma", "bge-m3"])
        self.assertEqual(c["qwen3-embedding:0.6b"]["status"], "not_pulled")
        self.assertEqual(c["nomic-embed-text@raw"]["ablation_of"], "nomic-embed-text")
        self.assertEqual(c["nomic-embed-text"]["unload_status"], "kept_loaded_resident_at_start")
        self.assertEqual(c["nomic-embed-text@raw"]["unload_status"], "kept_loaded_resident_at_start")
        self.assertEqual(c["bge-m3"]["unload_status"], "unloaded_to_reclaim_ram")
        client.unload_model.assert_called_once_with("bge-m3")  # resident nomic never unloaded
        self.assertEqual(out["relevance_mode"], "strict")
        self.assertEqual(out["total_queries"], 3)
        self.assertFalse(any(hasattr(OllamaEmbeddingClient, n) for n in ("pull", "delete", "remove")))

    def test_ablation_never_wins(self):
        def mk(r5, ab=None):
            return {"status": "completed", "ablation_of": ab, "total_ram_mb": 100.0,
                    "metrics": {"recall_at_5": r5, "mrr": r5, "latency_p50_ms": 10.0}}
        w = select_winner({"a": mk(0.5), "a@raw": mk(0.99, "a")})
        self.assertEqual(w["winner"], "a")


if __name__ == "__main__":
    unittest.main()
