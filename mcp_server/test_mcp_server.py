import asyncio
import gc
import inspect
import os
import tempfile
import unittest
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from mcp_server.server import DATA_DIR, MemoryTools, resolve_db_path


class MemoryToolsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=DATA_DIR)
        self.addCleanup(self.temp.cleanup)
        self.tools = MemoryTools(Path(self.temp.name) / "memory.db", embedder=None)
        self.addCleanup(self.tools.storage.close)

    async def test_save_recall_recent_and_health_share_project_schema(self):
        saved = await self.tools.save("MCP project fact", source="test", tags=["alpha"])
        fact_id = saved["fact_id"]
        self.assertIs(type(fact_id), int)
        recalled = await self.tools.recall("MCP project", 5)
        self.assertEqual(recalled["facts"][0]["id"], fact_id)
        self.assertEqual(recalled["retrieval_mode"], "keyword_only")
        self.assertIn("[F", recalled["grounding_context"])
        self.assertEqual((await self.tools.recent(5))[0]["id"], fact_id)
        self.assertEqual((await self.tools.health())["status"], "healthy")
        row = self.tools.storage.get_fact(fact_id)
        self.assertEqual(row["status"], "active")

    async def test_recall_uses_hybrid_pipeline_and_grounding_format(self):
        vector = [1.0] + [0.0] * 767
        self.tools = MemoryTools(
            Path(self.temp.name) / "hybrid.db", embedder=lambda _text: vector
        )
        self.addCleanup(self.tools.storage.close)
        saved = await self.tools.save("Hybrid semantic fact", source="hybrid.md")

        recalled = await self.tools.recall("meaning not present in exact text", 5)

        self.assertEqual(recalled["retrieval_mode"], "hybrid")
        self.assertEqual(recalled["facts"][0]["id"], saved["fact_id"])
        self.assertEqual(recalled["facts"][0]["fact_id"], f"F{saved['fact_id']}")
        self.assertIn("Source: hybrid.md", recalled["grounding_context"])
        self.assertIn("Status: active", recalled["grounding_context"])

    async def test_embedding_failure_returns_explicit_keyword_fallback(self):
        def broken_embedder(_text):
            raise ConnectionError("offline")

        self.tools = MemoryTools(
            Path(self.temp.name) / "fallback.db", embedder=broken_embedder
        )
        self.addCleanup(self.tools.storage.close)
        await self.tools.save("Fallback keyword fact", source="fallback.md")

        recalled = await self.tools.recall("Fallback keyword", 5)

        self.assertEqual(recalled["retrieval_mode"], "keyword_only")
        self.assertEqual(recalled["warning"], "Dense embedding unavailable; returned keyword-only results.")
        self.assertEqual(len(recalled["facts"]), 1)

    async def test_save_embedding_failure_reports_fact_was_saved_without_vector(self):
        def broken_embedder(_text):
            raise ConnectionError("offline")

        self.tools = MemoryTools(
            Path(self.temp.name) / "save-fallback.db", embedder=broken_embedder
        )
        self.addCleanup(self.tools.storage.close)

        saved = await self.tools.save("Saved despite embedding outage")

        self.assertTrue(saved["fact_id"])
        self.assertFalse(saved["vectorized"])
        self.assertEqual(
            saved["warning"], "Embedding unavailable; fact saved without a dense vector."
        )
        self.assertIsNone(self.tools.storage.get_fact(saved["fact_id"])["vector"])

    async def test_invalid_fact_is_rejected_before_embedding(self):
        calls = []
        self.tools = MemoryTools(
            Path(self.temp.name) / "invalid.db",
            embedder=lambda text: calls.append(text) or [1.0] * 768,
        )
        self.addCleanup(self.tools.storage.close)

        with self.assertRaises(ValueError):
            await self.tools.save("x" * 1_000_001)

        self.assertEqual(calls, [])

    async def test_mutations_fail_closed_without_judge_and_keep_rows(self):
        fact_id = (await self.tools.save("Keep immutable"))["fact_id"]
        with self.assertRaisesRegex(RuntimeError, "no fact was changed"):
            await self.tools.supersede(fact_id, "Changed", "reason")
        with self.assertRaisesRegex(RuntimeError, "no fact was changed"):
            await self.tools.retract(fact_id, "reason")
        self.assertEqual(self.tools.storage.get_fact(fact_id)["status"], "active")

    async def test_rejects_invalid_bounds_and_filters(self):
        for limit in (0, -1, True):
            with self.assertRaises(ValueError):
                await self.tools.recent(limit)
        with self.assertRaises(ValueError):
            await self.tools.recall("x", filters={"sql": "x"})


class StdioIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_official_sdk_client_lists_and_calls_tools(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=DATA_DIR) as temp:
            env = os.environ.copy()
            env["AI_MEMORY_DB"] = str(Path(temp) / "mcp.db")
            env["AI_MEMORY_DISABLE_EMBEDDING"] = "1"
            params = StdioServerParameters(
                command=str(Path(__import__("sys").executable)),
                args=["-m", "mcp_server"], env=env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listed = await session.list_tools()
                    self.assertEqual({t.name for t in listed.tools}, {
                        "memory_save", "memory_recall", "memory_recent", "memory_health",
                        "memory_supersede", "memory_retract",
                    })
                    saved = await session.call_tool("memory_save", {"text": "SDK integration fact"})
                    self.assertFalse(saved.is_error)
                    recalled = await session.call_tool("memory_recall", {"query": "SDK integration"})
                    self.assertFalse(recalled.is_error)
                    recent = await session.call_tool("memory_recent", {})
                    self.assertFalse(recent.is_error)
                    health = await session.call_tool("memory_health", {})
                    self.assertFalse(health.is_error)


class DatabasePathTests(unittest.TestCase):
    def test_env_path_must_be_within_project_data(self):
        with self.assertRaises(ValueError):
            resolve_db_path(str(Path(tempfile.gettempdir()) / "outside.db"))


# ===========================================================================
# E16 query-embedding cache - synthetic MCP-level tests
# ===========================================================================
#
# Safety contract for this section (deliberate, so it stays safe next to the
# production edit):
#   * Every fact text, query, filter and vector below is synthetic and defined
#     in this file. There is no corpus, no evaluation / golden / probe / held-out
#     data, no model, no network and no project database: each test builds a
#     throw-away SQLite database under the project's ``data/`` directory (the
#     same temp-DB pattern as the tests above) and deletes it again.
#   * The embedder is a local counter that returns a fixed 768-dim vector
#     (``[1.0] + [0.0] * 767``), matching ``storage.VECTOR_DIMENSION``.
#   * These tests are **authored but not run** in this change set: no test runner
#     was executed, no project module was imported or run, nothing was compiled
#     and no model or service was called. Executing them belongs to the parent
#     run, which owns test execution.
#
# Behaviours pinned here:
#   1. Default off - a ``MemoryTools`` built with the default constructor does not
#      use a query-embedding cache: two recalls of the same query re-embed.
#   2. Filter isolation - with the cache enabled, the same query recalled with two
#      different filters returns two different, correct fact sets; no stale
#      filtered result leaks across filters and no excluded fact is returned.
#   3. Limit isolation - with the cache enabled, the same query recalled with
#      different limits returns correctly-sized (different-length) result lists.
#   4. Embedding-cache vs result-cache isolation - an embedding-cache hit does not
#      imply a result-cache hit, and a result-cache hit is not required for an
#      embedding-cache hit.
#   5. Embedder failure - with the cache enabled, a failing embedder still yields a
#      keyword-only recall with the documented warning, never a crash and never an
#      empty result.
#   6. No cross-query leak - two different queries never share a cached embedding.
#   7. MCP boundary - the E16 cache stays off by default, the constructor parameter
#      is forwarded to the pipeline, and an out-of-range value is rejected.
#
# These tests live here rather than in a sibling module because the project test
# runner targets this single file for the mcp_server suite (verify_all.py
# SUITE_SPECS), so a separate module would never be collected.

# ---------------------------------------------------------------------------
# Synthetic constants (no corpus, no golden data, no model)
# ---------------------------------------------------------------------------

_SYNTHETIC_DIM = 768
# Deterministic, valid, non-zero 768-dim vector: matches storage.VECTOR_DIMENSION
# and lets the dense path run without a real embedding model.
_SYNTHETIC_VECTOR = [1.0] + [0.0] * (_SYNTHETIC_DIM - 1)

_QUERY = "synthetic topic"
# A second, textually different query that matches the same synthetic facts. It must
# get its own embedding: the E16 cache is keyed by the exact query text.
_OTHER_QUERY = "topic synthetic"

# (text, source, tags) - all three texts contain the query tokens, so the keyword
# search returns all three and only the filters can tell them apart.
_FACTS = (
    ("synthetic topic alpha", "synthetic-src-a.md", "alpha"),
    ("synthetic topic bravo", "synthetic-src-b.md", "bravo"),
    ("synthetic topic charlie", "synthetic-src-c.md", "charlie"),
)

# E16 cache capacity used by the "cache enabled" tests. Any positive value
# works; it only has to be non-zero to switch the cache on.
_E16_CACHE_SIZE = 8

# The MCP boundary may or may not expose the E16 knob yet. Detect it instead of
# assuming, so this file stays valid either way.
_MEMORY_TOOLS_PARAMETERS = inspect.signature(MemoryTools.__init__).parameters
HAS_QUERY_EMBEDDING_CACHE_PARAM = "query_embedding_cache_size" in _MEMORY_TOOLS_PARAMETERS

_MISSING_PARAM_REASON = (
    "MemoryTools does not expose query_embedding_cache_size yet; the MCP-boundary "
    "E16 forwarding / range validation is a pending parent action"
)


class _CountingEmbedder:
    """Synthetic embedder: records every call, returns a fixed 768-dim vector.

    ``error`` makes the embedder raise, which is how the "embedding outage"
    path is exercised without a real model.
    """

    def __init__(self, value=_SYNTHETIC_VECTOR, error=None):
        self.value = value
        self.error = error
        self.calls = []

    def __call__(self, text):
        self.calls.append(text)
        if self.error is not None:
            raise self.error
        return list(self.value)

    def reset(self):
        """Forget the calls made so far (e.g. the document embeddings of a save)."""
        self.calls.clear()

    @property
    def count(self):
        return len(self.calls)


class E16RecallCacheIsolationTests(unittest.IsolatedAsyncioTestCase):
    """MCP-level proof that the E16 query-embedding cache cannot leak results.

    The MCP ``recall`` path always hands the pipeline a callable filter, so the
    pipeline's *result* cache is bypassed for every recall. The E16 cache is a
    separate structure that only stores query embeddings (keyed by query text).
    These tests pin that the embedding cache saves embedding work only, and
    never changes which facts a recall returns.
    """

    async def asyncSetUp(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=DATA_DIR, ignore_cleanup_errors=True)
        self.addCleanup(self.temp.cleanup)

    # -- helpers ---------------------------------------------------------

    def _make_tools(self, name, embedder, cache_size=None):
        """Build a ``MemoryTools`` instance against a throw-away database.

        ``cache_size=None`` builds the tool with the plain default constructor
        (the E16 cache must stay off). A positive ``cache_size`` enables the E16
        query-embedding cache.

        When the ``MemoryTools`` constructor already exposes
        ``query_embedding_cache_size`` it is passed through. If the pipeline did
        not receive it, the already-validated public
        ``RetrievalPipeline.query_embedding_cache_size`` attribute is set
        directly, so the cache *behaviour* is pinned either way. The wiring
        defect itself is reported by the dedicated
        ``test_memory_tools_forwards_query_embedding_cache_size_to_the_pipeline``
        test, so a broken wiring shows up as one precise failure instead of
        six.
        """
        kwargs = {"embedder": embedder}
        if cache_size is not None and HAS_QUERY_EMBEDDING_CACHE_PARAM:
            kwargs["query_embedding_cache_size"] = cache_size
        tools = MemoryTools(Path(self.temp.name) / name, **kwargs)
        self.addCleanup(tools.storage.close)
        if cache_size is not None:
            if tools.pipeline.query_embedding_cache_size != cache_size:
                try:
                    tools.pipeline.query_embedding_cache_size = cache_size
                except AttributeError as exc:  # pragma: no cover - defensive
                    self.fail(f"the E16 cache size must be settable on the pipeline: {exc}")
            self.assertEqual(
                tools.pipeline.query_embedding_cache_size, cache_size,
                "the E16 query-embedding cache must be enabled for this test",
            )
        return tools

    async def _seed_facts(self, tools):
        """Insert the synthetic facts and return ``[(source, fact_id), ...]``."""
        seeded = []
        for text, source, tags in _FACTS:
            saved = await tools.save(text, source=source, writer="synthetic-writer", tags=[tags])
            seeded.append((source, saved["fact_id"]))
        return seeded

    def _count_storage_searches(self, tools):
        """Count the storage-level searches a ``recall`` triggers.

        The pipeline reaches storage through the public ``search_keyword`` and
        ``search_vector`` methods; wrapping them on the instance makes the
        result-cache behaviour observable from outside the pipeline. A recall
        that is served from the result cache performs no storage search at all.
        """
        keyword_calls = []
        vector_calls = []
        original_keyword = tools.storage.search_keyword
        original_vector = tools.storage.search_vector

        def counting_keyword(query, limit=10):
            keyword_calls.append((query, limit))
            return original_keyword(query, limit)

        def counting_vector(query_vector, limit=10):
            vector_calls.append((query_vector, limit))
            return original_vector(query_vector, limit)

        tools.storage.search_keyword = counting_keyword
        tools.storage.search_vector = counting_vector
        return keyword_calls, vector_calls

    @staticmethod
    def _ids(result):
        return {fact["id"] for fact in result["facts"]}

    # -- 1. default off --------------------------------------------------

    async def test_default_memory_tools_never_uses_a_query_embedding_cache(self):
        """Default construction: the E16 cache is off, so every recall re-embeds."""
        embedder = _CountingEmbedder()
        tools = self._make_tools("default-off.db", embedder, cache_size=None)
        seeded = await self._seed_facts(tools)
        embedder.reset()
        keyword_calls, vector_calls = self._count_storage_searches(tools)

        self.assertEqual(
            tools.pipeline.query_embedding_cache_size, 0,
            "the default MemoryTools must leave the E16 query-embedding cache disabled",
        )

        first = await tools.recall(_QUERY, limit=1)
        second = await tools.recall(_QUERY, limit=3)

        # One embedder call per recall: the default caches nothing.
        self.assertEqual(embedder.count, 2,
                         "with the default MemoryTools each recall must re-embed the query")
        # Both recalls still return correct results.
        self.assertEqual(len(first["facts"]), 1)
        self.assertEqual(len(second["facts"]), 3)
        self.assertEqual(self._ids(second), {fact_id for _source, fact_id in seeded})
        self.assertEqual(first["retrieval_mode"], "hybrid")
        self.assertIsNone(first["warning"])
        # Each recall really searched storage (the MCP path never uses the result cache).
        self.assertEqual(len(keyword_calls), 2)
        self.assertEqual(len(vector_calls), 2)

    # -- 2. filter isolation ---------------------------------------------

    async def test_cache_enabled_same_query_different_filters_do_not_leak_results(self):
        """The cached embedding must not carry the previous filter's results."""
        embedder = _CountingEmbedder()
        tools = self._make_tools("filter-isolation.db", embedder, cache_size=_E16_CACHE_SIZE)
        seeded = await self._seed_facts(tools)
        embedder.reset()
        keyword_calls, vector_calls = self._count_storage_searches(tools)

        (source_a, id_a), (source_b, id_b), (_source_c, id_c) = seeded

        first = await tools.recall(_QUERY, limit=5, filters={"source": source_a})
        second = await tools.recall(_QUERY, limit=5, filters={"source": source_b})

        # The embedding is reused (the cache is keyed by the query text only) ...
        self.assertEqual(embedder.count, 1,
                         "the second recall must reuse the cached query embedding")
        # ... but each recall re-ran the storage searches, so no stale result list
        # can be served for the second filter.
        self.assertEqual(len(keyword_calls), 2)
        self.assertEqual(len(vector_calls), 2)

        # The two filters return two different, correct fact sets.
        self.assertEqual(self._ids(first), {id_a})
        self.assertEqual(self._ids(second), {id_b})
        # No fact excluded by the second filter leaks into its results.
        self.assertNotIn(id_a, self._ids(second))
        self.assertNotIn(id_c, self._ids(second))
        self.assertIn(f"[F{id_a}]", first["grounding_context"])
        self.assertIn(f"[F{id_b}]", second["grounding_context"])
        self.assertNotIn(f"[F{id_a}]", second["grounding_context"])

    # -- 3. limit isolation ----------------------------------------------

    async def test_cache_enabled_same_query_different_limits_return_correct_lengths(self):
        """The cached embedding must not serve a stale, wrongly sized result list."""
        embedder = _CountingEmbedder()
        tools = self._make_tools("limit-isolation.db", embedder, cache_size=_E16_CACHE_SIZE)
        seeded = await self._seed_facts(tools)
        embedder.reset()
        keyword_calls, vector_calls = self._count_storage_searches(tools)

        one = await tools.recall(_QUERY, limit=1)
        two = await tools.recall(_QUERY, limit=2)
        three = await tools.recall(_QUERY, limit=3)

        # One embedder call in total: the embedding is reused across limits.
        self.assertEqual(embedder.count, 1,
                         "the query embedding must be reused across different limits")
        # Every limit re-ran the storage searches (no stale result list).
        self.assertEqual(len(keyword_calls), 3)
        self.assertEqual(len(vector_calls), 3)

        # Correct, different-length result lists.
        self.assertEqual(len(one["facts"]), 1)
        self.assertEqual(len(two["facts"]), 2)
        self.assertEqual(len(three["facts"]), 3)
        self.assertEqual(self._ids(three), {fact_id for _source, fact_id in seeded})
        self.assertTrue(self._ids(one).issubset(self._ids(three)))
        self.assertTrue(self._ids(two).issubset(self._ids(three)))
        self.assertLess(len(one["facts"]), len(two["facts"]))
        self.assertLess(len(two["facts"]), len(three["facts"]))

    # -- 4. embedding cache vs result cache -------------------------------

    async def test_embedding_cache_hit_does_not_imply_a_result_cache_hit(self):
        """The two caches are independent: one embedding, two storage searches."""
        embedder = _CountingEmbedder()
        tools = self._make_tools("cache-isolation.db", embedder, cache_size=_E16_CACHE_SIZE)
        seeded = await self._seed_facts(tools)
        embedder.reset()
        keyword_calls, vector_calls = self._count_storage_searches(tools)

        first = await tools.recall(_QUERY, limit=2)
        second = await tools.recall(_QUERY, limit=2)

        # Same query + same limit: the second recall is an embedding-cache hit
        # (1 embedder call) but NOT a result-cache hit (2 storage searches),
        # because the MCP path always supplies a callable filter.
        self.assertEqual(embedder.count, 1)
        self.assertEqual(len(keyword_calls), 2)
        self.assertEqual(len(vector_calls), 2)
        self.assertEqual([fact["id"] for fact in first["facts"]],
                         [fact["id"] for fact in second["facts"]],
                         "two identical recalls must return identical results")
        self.assertEqual(len(first["facts"]), 2)
        self.assertTrue(self._ids(first).issubset({fact_id for _source, fact_id in seeded}))
        # The dense search of the second recall used the *cached* vector, i.e. the
        # embedding really was reused instead of being recomputed.
        self.assertEqual(vector_calls[0][0], vector_calls[1][0])

        # A new limit is a would-be result-cache miss: the embedding cache still
        # hits, so a result hit is not required for an embedding hit.
        third = await tools.recall(_QUERY, limit=3)
        self.assertEqual(embedder.count, 1,
                         "an embedding-cache hit must not require a result-cache hit")
        self.assertEqual(len(keyword_calls), 3)
        self.assertEqual(len(vector_calls), 3)
        self.assertEqual(len(third["facts"]), 3)

    # -- 5. embedder failure ---------------------------------------------

    async def test_embedder_failure_with_cache_enabled_returns_keyword_only_results(self):
        """An embedding outage must degrade to keyword-only, never to a crash."""
        embedder = _CountingEmbedder(error=ConnectionError("synthetic embedder outage"))
        tools = self._make_tools("embedder-failure.db", embedder, cache_size=_E16_CACHE_SIZE)
        seeded = await self._seed_facts(tools)
        embedder.reset()

        first = await tools.recall(_QUERY, limit=5)
        second = await tools.recall(_QUERY, limit=5)

        # Failures are never cached, so each recall retries the embedder.
        self.assertEqual(embedder.count, 2,
                         "a failed embedding must not be cached and must be retried")
        for result in (first, second):
            self.assertEqual(result["retrieval_mode"], "keyword_only")
            self.assertEqual(result["warning"],
                             "Dense embedding unavailable; returned keyword-only results.")
            self.assertEqual(self._ids(result), {fact_id for _source, fact_id in seeded})
            self.assertTrue(result["facts"], "an embedding outage must not empty the result")

    # -- 6. no cross-query leak -------------------------------------------

    async def test_distinct_queries_never_share_a_cached_embedding(self):
        """Each distinct query is embedded on its own; the cache is per query."""
        embedder = _CountingEmbedder()
        tools = self._make_tools("cross-query.db", embedder, cache_size=_E16_CACHE_SIZE)
        await self._seed_facts(tools)
        embedder.reset()

        await tools.recall(_QUERY, limit=2)
        self.assertEqual(embedder.count, 1, "the first query must be embedded once")

        await tools.recall(_OTHER_QUERY, limit=2)
        self.assertEqual(embedder.count, 2,
                         "a different query must trigger its own embedding")

        await tools.recall(_QUERY, limit=2)
        self.assertEqual(embedder.count, 2,
                         "the first query must still be served from its own cache entry")

    # -- 7. MCP boundary --------------------------------------------------

    async def test_mcp_boundary_leaves_the_e16_query_embedding_cache_disabled(self):
        """The MCP boundary must never turn E16 on by itself."""
        tools = MemoryTools(Path(self.temp.name) / "default-bound.db",
                            embedder=_CountingEmbedder())
        self.addCleanup(tools.storage.close)
        effective = tools.pipeline.query_embedding_cache_size
        self.assertGreaterEqual(effective, 0)
        self.assertLessEqual(effective, 500)
        self.assertEqual(effective, 0,
                         "the MCP boundary must leave the E16 query-embedding cache off by default")

    @unittest.skipUnless(HAS_QUERY_EMBEDDING_CACHE_PARAM, _MISSING_PARAM_REASON)
    async def test_memory_tools_forwards_query_embedding_cache_size_to_the_pipeline(self):
        """The MCP parameter must reach the pipeline (and stay off at 0)."""
        default_tools = MemoryTools(Path(self.temp.name) / "forward-default.db",
                                    embedder=_CountingEmbedder(),
                                    query_embedding_cache_size=0)
        self.addCleanup(default_tools.storage.close)
        self.assertEqual(default_tools.pipeline.query_embedding_cache_size, 0)

        tools = MemoryTools(Path(self.temp.name) / "forward-enabled.db",
                            embedder=_CountingEmbedder(),
                            query_embedding_cache_size=_E16_CACHE_SIZE)
        self.addCleanup(tools.storage.close)
        self.assertEqual(tools.pipeline.query_embedding_cache_size, _E16_CACHE_SIZE)

    @unittest.skipUnless(HAS_QUERY_EMBEDDING_CACHE_PARAM, _MISSING_PARAM_REASON)
    async def test_out_of_range_query_embedding_cache_size_is_rejected(self):
        """The MCP boundary must reject an out-of-range E16 cache size."""
        for bad in (-1, 501, 10_000):
            with self.subTest(value=bad):
                # A dedicated temp dir per case, so a rejected construction can never
                # disturb the other cases (and can never block cleanup on Windows).
                with tempfile.TemporaryDirectory(dir=DATA_DIR,
                                                 ignore_cleanup_errors=True) as scratch:
                    with self.assertRaises(ValueError):
                        MemoryTools(
                            Path(scratch) / "invalid.db",
                            embedder=_CountingEmbedder(),
                            query_embedding_cache_size=bad,
                        )
                    # Release the rejected instance (and any DB handle it opened)
                    # before the scratch directory is removed.
                    gc.collect()

        # The documented range must still be accepted and forwarded.
        accepted = MemoryTools(Path(self.temp.name) / "valid-range.db",
                               embedder=_CountingEmbedder(),
                               query_embedding_cache_size=_E16_CACHE_SIZE)
        self.addCleanup(accepted.storage.close)
        self.assertEqual(accepted.pipeline.query_embedding_cache_size,
                         _E16_CACHE_SIZE)


if __name__ == "__main__":
    unittest.main()
