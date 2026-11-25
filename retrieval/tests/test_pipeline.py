"""Tests for the retrieval pipeline."""

import unittest
import tempfile
import threading
import warnings
from pathlib import Path
from unittest.mock import Mock, patch
from retrieval.pipeline import RetrievalPipeline
from storage import MemoryStorage

class TestRetrievalPipeline(unittest.TestCase):
    def setUp(self):
        # Create a mock storage
        self.storage = Mock()
        self.storage.connect.return_value = None
        # Mock search_keyword to return some dummy facts
        self.storage.search_keyword.return_value = [
            {"id": 1, "text": "Test fact 1", "source": "test", "writer": "test", 
             "created_at": "2026-01-01", "valid_from": "2026-01-01", "valid_to": "2026-12-31",
             "status": "active", "superseded_by": None, "confidence": 0.9, "tags": "test"},
            {"id": 2, "text": "Test fact 2", "source": "test", "writer": "test", 
             "created_at": "2026-01-01", "valid_from": "2026-01-01", "valid_to": "2026-12-31",
             "status": "active", "superseded_by": None, "confidence": 0.8, "tags": "test"}
        ]
        # Mock search_vector to return some dummy facts
        self.storage.search_vector.return_value = [
            {"id": 2, "text": "Test fact 2", "source": "test", "writer": "test", 
             "created_at": "2026-01-01", "valid_from": "2026-01-01", "valid_to": "2026-12-31",
             "status": "active", "superseded_by": None, "confidence": 0.8, "tags": "test", "distance": 0.1},
            {"id": 3, "text": "Test fact 3", "source": "test", "writer": "test", 
             "created_at": "2026-01-01", "valid_from": "2026-01-01", "valid_to": "2026-12-31",
             "status": "active", "superseded_by": None, "confidence": 0.7, "tags": "test", "distance": 0.2}
        ]
        
    def test_init(self):
        """Test pipeline initialization."""
        pipeline = RetrievalPipeline(self.storage)
        self.assertEqual(pipeline.storage, self.storage)
        self.assertIsNone(pipeline.embedder)
        self.assertIsNone(pipeline.reranker)
        
    def test_search_keyword_only(self):
        """Test search with no embedder (keyword-only)."""
        pipeline = RetrievalPipeline(self.storage, embedder=None)
        results = pipeline.search("test query", limit=2)
        # Should have called search_keyword
        self.storage.search_keyword.assert_called()
        # Should have results
        self.assertGreaterEqual(len(results), 0)
        # Each result should have score and confidence
        for result in results:
            self.assertIn("score", result)
            self.assertIn("confidence", result)
            
    def test_search_with_embedder(self):
        """Test search with an embedder."""
        embedder = Mock(return_value=[0.1] * 768)  # Mock embedding vector
        pipeline = RetrievalPipeline(self.storage, embedder=embedder)
        results = pipeline.search("test query", limit=2)
        # Should have called both search methods
        self.storage.search_keyword.assert_called()
        self.storage.search_vector.assert_called()
        # Should have results
        self.assertGreaterEqual(len(results), 0)
        # Each result should have score and confidence
        for result in results:
            self.assertIn("score", result)
            self.assertIn("confidence", result)

    def test_shared_sqlite_connection_searches_are_serialized(self):
        """Keyword and vector reads share one SQLite connection and must stay on one thread."""
        with tempfile.TemporaryDirectory(prefix="retrieval-thread-check-") as td:
            store = MemoryStorage(Path(td) / "db.sqlite")
            store.initialize_schema()
            vector = [1.0] + [0.0] * 767
            store.save_fact("A searchable vector fact", vector=vector)
            caller = threading.get_ident()
            seen = []
            keyword = store.search_keyword
            vector_search = store.search_vector

            def tracked_keyword(*args, **kwargs):
                seen.append(threading.get_ident())
                return keyword(*args, **kwargs)

            def tracked_vector(*args, **kwargs):
                seen.append(threading.get_ident())
                return vector_search(*args, **kwargs)

            store.search_keyword = tracked_keyword
            store.search_vector = tracked_vector
            try:
                pipeline = RetrievalPipeline(store, embedder=lambda _q: vector, cache_size=0)
                pipeline.search("searchable fact")
                self.assertEqual(seen, [caller, caller])
            finally:
                store.close()
            
    def test_search_with_filter(self):
        """Test search with a filter function."""
        def filter_func(fact):
            return fact["id"] == 1
        pipeline = RetrievalPipeline(self.storage, embedder=None)
        results = pipeline.search("test query", limit=2, filter_func=filter_func)
        # All results should satisfy the filter
        for result in results:
            self.assertEqual(result["id"], 1)
            
    def test_search_limit(self):
        """Test that limit is respected."""
        pipeline = RetrievalPipeline(self.storage, embedder=None)
        results = pipeline.search("test query", limit=1)
        self.assertLessEqual(len(results), 1)
        
    def test_invalid_query(self):
        """Test that invalid queries raise ValueError."""
        pipeline = RetrievalPipeline(self.storage, embedder=None)
        with self.assertRaises(ValueError):
            pipeline.search("")  # Empty query after stripping
        with self.assertRaises(ValueError):
            pipeline.search("a" * 1001)  # Too long
            
    def test_invalid_limit(self):
        """Test that invalid limits raise ValueError."""
        pipeline = RetrievalPipeline(self.storage, embedder=None)
        with self.assertRaises(ValueError):
            pipeline.search("test", limit=-1)
        with self.assertRaises(ValueError):
            pipeline.search("test", limit="invalid")

    def test_invalid_constructor_bounds(self):
        for kwargs in ({"cache_size": -1}, {"cache_size": 501}, {"rrf_k": 0},
                       {"rrf_k": True}, {"max_candidates_for_rerank": 31},
                       {"low_confidence_threshold": float("nan")},
                       {"low_confidence_threshold": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RetrievalPipeline(self.storage, **kwargs)

    def test_embedder_exception_and_invalid_vector_fall_back(self):
        self.storage.search_keyword.return_value = [{"id": 1, "text": "fallback", "confidence": 0.9}]
        pipeline = RetrievalPipeline(self.storage, embedder=lambda _q: (_ for _ in ()).throw(RuntimeError("offline")), cache_size=0)
        self.assertEqual(pipeline.search("fallback")[0]["score"], 1.0)
        self.storage._validate_vector.side_effect = ValueError("invalid vector")
        pipeline = RetrievalPipeline(self.storage, embedder=lambda _q: [1.0], cache_size=0)
        self.assertEqual(pipeline.search("fallback")[0]["text"], "fallback")

    def test_rrf_ties_are_deterministic(self):
        pipeline = RetrievalPipeline(self.storage)
        keyword = [{"id": 1, "text": "one"}, {"id": 2, "text": "two"}]
        vector = [{"id": 2, "text": "two"}, {"id": 1, "text": "one"}]
        orders = [tuple(row["id"] for row in pipeline._rrf_fusion(keyword, vector)) for _ in range(5)]
        self.assertEqual(orders, [(1, 2)] * 5)

    def test_cache_key_includes_limit_and_invalidation_clears(self):
        self.storage.search_keyword.return_value = [{"id": 1, "text": "one", "confidence": 1.0}]
        pipeline = RetrievalPipeline(self.storage, cache_size=4)
        pipeline.search("same query", limit=1)
        pipeline.search("same query", limit=2)
        self.assertEqual(self.storage.search_keyword.call_count, 2)
        pipeline.invalidate_cache()
        pipeline.search("same query", limit=1)
        self.assertEqual(self.storage.search_keyword.call_count, 3)

    def test_callable_filters_skip_cache(self):
        first = lambda _fact: True
        second = lambda _fact: False
        pipeline = RetrievalPipeline(self.storage)
        self.assertIsNone(pipeline._get_cache_key("q", 1, first))
        self.assertIsNone(pipeline._get_cache_key("q", 1, second))

        class UnhashableFilter:
            __hash__ = None
            def __call__(self, _fact):
                return True

        unhashable = UnhashableFilter()
        pipeline.search("same", filter_func=unhashable)
        pipeline.search("same", filter_func=unhashable)
        self.assertEqual(self.storage.search_keyword.call_count, 2)

    def test_mutable_filter_state_does_not_reuse_stale_cached_results(self):
        state = {"allow": True}

        def filter_func(_fact):
            return state["allow"]

        pipeline = RetrievalPipeline(self.storage, cache_size=4)
        first = pipeline.search("same", limit=2, filter_func=filter_func)
        state["allow"] = False
        second = pipeline.search("same", limit=2, filter_func=filter_func)

        self.assertTrue(first)
        self.assertEqual(second, [])
        self.assertEqual(self.storage.search_keyword.call_count, 2)

    def test_reranker_is_gated_and_capped(self):
        reranker = Mock(side_effect=lambda rows: rows)
        pipeline = RetrievalPipeline(self.storage, embedder=lambda _q: [0.1] * 768,
                                     reranker=reranker, max_candidates_for_rerank=1,
                                     low_confidence_threshold=0.5, cache_size=0)
        self.storage._validate_vector.return_value = [0.1] * 768
        self.storage.search_keyword.return_value = [{"id": 1, "text": "one", "confidence": 0.9}]
        self.storage.search_vector.return_value = [{"id": 1, "text": "one", "confidence": 0.9, "distance": 0.1}]
        pipeline.search("rerank query")
        reranker.assert_not_called()
        self.storage.search_keyword.return_value = [{"id": 1, "text": "one", "confidence": 0.1},
                                                    {"id": 2, "text": "two", "confidence": 0.1}]
        self.storage.search_vector.return_value = [{"id": 2, "text": "two", "confidence": 0.1, "distance": 0.1},
                                                   {"id": 1, "text": "one", "confidence": 0.1, "distance": 0.2}]
        pipeline.search("low confidence")
        self.assertEqual(len(reranker.call_args.args[0]), 1)


class TestSourceContextAssembly(unittest.TestCase):
    """context_mode="source": parent-source aggregation of ranked fragments (Round 3)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStorage(Path(self._tmp.name) / "ctx.db")
        self.store.initialize_schema()
        # ids 1..6: source a.md holds 1,2,3 ; b.md holds 4,5 ; c.md holds 6
        self.ids = {}
        for text, src in [("alpha full sentence about timeout", "a.md"), ("3", "a.md"), ("(15", "a.md"),
                          ("beta unrelated fragment", "b.md"), ("2026-08-30", "b.md"),
                          ("gamma lone fact", "c.md")]:
            self.ids[text] = self.store.save_fact(text, source=src, writer="t", tags="", confidence=0.5)

    def tearDown(self):
        self.store.close()
        self._tmp.cleanup()

    def _pipe(self, ranked_keyword_texts, **kw):
        # keyword search is stubbed so fragment ranking is exact and deterministic
        order = [self.ids[t] for t in ranked_keyword_texts]
        facts = {f["id"]: f for f in (self.store.get_fact(i) for i in order)}
        pipe = RetrievalPipeline(self.store, cache_size=0, **kw)
        pipe.storage = Mock(wraps=self.store)
        pipe.storage.search_keyword = Mock(return_value=[dict(facts[i], vector=None) for i in order])
        return pipe

    def test_default_mode_unchanged_and_validated(self):
        pipe = self._pipe(["beta unrelated fragment", "3", "gamma lone fact"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pipe.search("q", 3)
        self.assertEqual([r["id"] for r in res], [self.ids["beta unrelated fragment"], self.ids["3"], self.ids["gamma lone fact"]])
        for bad in ({"context_mode": "x"}, {"source_score_mode": "mean"}):
            with self.assertRaises(ValueError):
                RetrievalPipeline(self.store, **bad)

    def test_source_mode_pulls_sibling_facts_ahead_of_other_sources(self):
        # fragment "3" of a.md ranks 2nd; the complete a.md sentence was never retrieved
        pipe = self._pipe(["beta unrelated fragment", "3"], context_mode="source")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pipe.search("q", 5)
        # b.md (rank 1) first, then a.md retrieved fragment, then a.md siblings in id order
        self.assertEqual([r["source"] for r in res], ["b.md", "b.md", "a.md", "a.md", "a.md"])
        a_ids = [r["id"] for r in res if r["source"] == "a.md"]
        self.assertEqual(a_ids[0], self.ids["3"])
        self.assertEqual(sorted(a_ids), sorted([self.ids["alpha full sentence about timeout"], self.ids["3"], self.ids["(15"]]))
        self.assertEqual(len({r["id"] for r in res}), len(res))  # no duplicates

    def test_source_mode_order_by_best_fragment_and_ids_preserved(self):
        pipe = self._pipe(["3", "beta unrelated fragment", "(15"], context_mode="source")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pipe.search("q", 6)
        self.assertEqual(res[0]["source"], "a.md")  # a.md owns the best fragment
        self.assertEqual({r["id"] for r in res}, set(range(1, 6)) | set())  # a.md + b.md facts only
        self.assertEqual(res, pipe.search("q", 6))  # deterministic repeat

    def test_unretrieved_siblings_get_zero_score_and_top_score_not_inflated(self):
        pipe = self._pipe(["3"], context_mode="source")
        base = self._pipe(["3"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, ref = pipe.search("q", 5), base.search("q", 5)
        self.assertEqual(max(r["score"] for r in res), max(r["score"] for r in ref))
        self.assertTrue(all(r["score"] == 0.0 for r in res if r["id"] != self.ids["3"]))
        self.assertTrue(all("source_score" in r for r in res))

    def test_downweighted_fragments_are_kept_not_excluded(self):
        pipe = self._pipe(["3", "gamma lone fact"], context_mode="source", downweight_fragments=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pipe.search("q", 6)
        self.assertEqual(res[0]["source"], "c.md")  # the 1-char fragment no longer wins its source
        self.assertIn(self.ids["3"], [r["id"] for r in res])

    def test_filter_applies_to_siblings(self):
        pipe = self._pipe(["3"], context_mode="source")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pipe.search("q", 5, filter_func=lambda f: f["text"] != "(15")
        self.assertNotIn(self.ids["(15"], [r["id"] for r in res])

class TestAnswerabilityScore(unittest.TestCase):
    """Query-level dense answerability_score (mean of top-five cosine similarities)."""

    @staticmethod
    def _fact(i, distance, source="s.md"):
        return {"id": i, "text": f"fact {i} text", "source": source, "writer": "t", "created_at": "2026-01-01",
                "valid_from": "2026-01-01", "valid_to": None, "status": "active", "superseded_by": None,
                "confidence": 0.9, "tags": "", "distance": distance}

    def _pipe(self, distances, keyword=(), **kw):
        storage = Mock()
        storage.search_keyword.return_value = [self._fact(i, None) for i in keyword]
        storage.search_vector.return_value = [self._fact(i, d) for i, d in enumerate(distances, start=1)]
        kw.setdefault("embedder", lambda _q: [0.1] * 768)
        return RetrievalPipeline(storage, **kw), storage

    def test_mean_of_top_five_only(self):
        pipe, _ = self._pipe([0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.5], cache_size=0)
        res = pipe.search("q", 10)
        expected = sum(1 - d for d in (0.1, 0.2, 0.3, 0.4, 0.5)) / 5
        self.assertEqual(len(res), 7)
        for r in res:
            self.assertAlmostEqual(r["answerability_score"], expected, places=12)

    def test_fewer_than_five_uses_available_and_unsorted_input_takes_best(self):
        pipe, _ = self._pipe([0.3, 0.1], cache_size=0)
        self.assertAlmostEqual(pipe.search("q", 5)[0]["answerability_score"], 0.8, places=12)
        pipe, _ = self._pipe([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6], cache_size=0)  # best five: .1 .2 .3 .6 .7
        self.assertAlmostEqual(pipe.search("q", 5)[0]["answerability_score"], 1 - (0.1 + 0.2 + 0.3 + 0.6 + 0.7) / 5, places=12)

    def test_unaffected_by_limit_and_other_fields_unchanged(self):
        pipe, _ = self._pipe([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], cache_size=0)
        one, many = pipe.search("q", 1), pipe.search("q", 6)
        self.assertEqual(one[0]["answerability_score"], many[0]["answerability_score"])
        for r in many:
            self.assertEqual(r["score"], r["rrf_score"])  # RRF stays the score
            self.assertEqual(r["confidence"], 0.9)        # stored confidence untouched
            self.assertNotIn("distance", r)

    def test_same_value_on_source_siblings(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStorage(Path(tmp) / "a.db")
            store.initialize_schema()
            try:
                v = [1.0] + [0.0] * 767
                hit = store.save_fact("hit fragment", source="a.md", writer="t", tags="", confidence=0.5, vector=v)
                sib = store.save_fact("unretrieved sibling", source="a.md", writer="t", tags="", confidence=0.5)
                other = store.save_fact("other source", source="b.md", writer="t", tags="", confidence=0.5,
                                        vector=[0.6, 0.8] + [0.0] * 766)
                pipe = RetrievalPipeline(store, embedder=lambda _q: v, cache_size=0, context_mode="source")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    res = pipe.search("hit fragment", 10)
                self.assertEqual({r["id"] for r in res}, {hit, sib, other})
                expected = (1.0 + 0.6) / 2  # only two dense hits exist (the sibling has no vector)
                for r in res:
                    self.assertAlmostEqual(r["answerability_score"], expected, places=6)
                sibling = next(r for r in res if r["id"] == sib)
                self.assertEqual(sibling["score"], 0.0)  # sibling score semantics unchanged
                self.assertIn("source_score", sibling)
            finally:
                store.close()

    def test_cache_preserves_score_and_skips_recomputation(self):
        pipe, storage = self._pipe([0.1, 0.2, 0.3], cache_size=8)
        first = pipe.search("q", 5)
        second = pipe.search("q", 5)
        self.assertEqual(storage.search_vector.call_count, 1)  # second call served from cache
        self.assertEqual(first, second)
        self.assertAlmostEqual(second[0]["answerability_score"], 0.8, places=12)
        self.assertEqual(pipe.search("q", 2)[0]["answerability_score"], second[0]["answerability_score"])

    def test_keyword_only_omits_feature(self):
        for kw in ({"embedder": None}, {"embedder": lambda _q: (_ for _ in ()).throw(RuntimeError("down"))},
                   {"embedder": lambda _q: [1.0]}):  # last: vector rejected by storage validation
            pipe, storage = self._pipe([0.1], keyword=[1, 2], cache_size=2, **kw)
            storage._validate_vector.side_effect = ValueError("bad dimension")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = pipe.search("q", 5)
                cached = pipe.search("q", 5)
            self.assertEqual(len(res), 2)
            self.assertTrue(all("answerability_score" not in r for r in res + cached))
            storage.search_vector.assert_not_called()

    def test_source_mode_keyword_only_omits_feature(self):
        pipe, _ = self._pipe([0.1], keyword=[1], embedder=None, cache_size=0, context_mode="source")
        pipe.storage._ensure_connection = Mock()
        pipe.storage.conn.execute.return_value.fetchall.return_value = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pipe.search("q", 5)
        self.assertTrue(res and all("answerability_score" not in r for r in res))

    def test_no_dense_hits_omits_feature(self):
        pipe, _ = self._pipe([], keyword=[1], cache_size=0)
        res = pipe.search("q", 5)
        self.assertEqual([r["id"] for r in res], [1])
        self.assertNotIn("answerability_score", res[0])

    def test_invalid_distance_omits_feature_with_warning(self):
        for bad in (float("nan"), float("inf"), -float("inf"), None, "0.1", True, -0.5, 2.5):
            pipe, _ = self._pipe([0.1, 0.2, bad], cache_size=0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                res = pipe.search("q", 5)
            self.assertTrue(res, bad)
            self.assertTrue(all("answerability_score" not in r for r in res), bad)
            self.assertTrue(any(issubclass(w.category, RuntimeWarning) for w in caught), bad)

    def test_float32_rounding_slack_is_clamped_not_rejected(self):
        pipe, _ = self._pipe([-1e-8, 2.0 + 1e-8], cache_size=0)
        score = pipe.search("q", 5)[0]["answerability_score"]
        self.assertEqual(score, 0.0)  # (1.0 + -1.0) / 2 after clamping to [0, 2]


if __name__ == "__main__":
    unittest.main()
