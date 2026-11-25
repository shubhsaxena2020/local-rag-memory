"""Synthetic-only tests for the E16 query-embedding cache in ``retrieval/pipeline.py``.

Design constraints (deliberate, so this file stays safe next to the production edit):

* Every fact, query and embedding vector is synthetic and defined in this file.
  No corpus, no evaluation / golden / probe / held-out data, no model, no network, no database.
* Only the public surface of ``RetrievalPipeline`` is exercised: the constructor,
  ``search()`` and ``invalidate_cache()``. No private pipeline helper (cache lookup/store
  names, cache attributes) is referenced, so the internal implementation stays free to change.
  ``context_mode`` is a public, constructor-documented attribute; the source-mode test
  switches it on one pipeline so that the embedding cache can be observed to be shared
  across the two modes.
* The tests are *authored only* in this change set: executing them (imports, collection, CI)
  belongs to the parent run, which owns test execution.

Behaviours pinned here (preregistered E16 contract):

* ``query_embedding_cache_size`` defaults to 0 (disabled) and is bounded to 0..500.
* The embedding cache is keyed by the query only: it is not keyed by result limit or filter.
* Only a vector that passed ``storage._validate_vector`` is cached. A rejected vector
  (wrong dimension, non-finite element, boolean element) is never cached.
* Embedder failures (exception, ``None``, invalid vector) are never cached and fall back
  exactly as before (keyword-only search).
* The cache is bounded: once capacity is reached, older entries are evicted.
* Capacity is exact and eviction follows least-recently-used order: a cache hit
  refreshes an entry's recency, so a recently used query survives a newer one.
* The embedding cache and the result cache are separate and evict independently:
  evicting one never removes an entry from the other.
* The cache key is the raw query text: case and surrounding whitespace are not
  normalised, so two spellings of the same question are two entries.
* The cached embedding is a copy: mutating the vector handed to storage cannot
  poison a later cache hit.
* An invalid query (blank, or longer than 1000 characters) is rejected before the
  cache is consulted, so it populates nothing.
* A pipeline without an embedder never consults the embedding cache and still
  returns keyword-only results with the documented RuntimeWarning.
* Source-mode aggregation is unaffected by the embedding cache: the embedding is
  shared, the results are still grouped by source.
* ``invalidate_cache()`` clears both the embedding cache and the result cache.
"""

import math
import unittest
import warnings

from retrieval.pipeline import RetrievalPipeline

_SYNTHETIC_DIM = 768
# The embedder returns integer-valued components; storage._validate_vector converts them to
# floats. That makes "the cached copy is the validated vector" observable by type, not just
# by value (1 == 1.0 would hide a raw-output cache).
_SYNTHETIC_RAW_VECTOR = [1] + [0] * (_SYNTHETIC_DIM - 1)
_SYNTHETIC_VALIDATED_VECTOR = [1.0] + [0.0] * (_SYNTHETIC_DIM - 1)
_UNSET = object()


def _synthetic_fact(fact_id, text, distance=None):
    """One synthetic fact dict with the field set the pipeline reads."""
    fact = {
        "id": fact_id,
        "text": text,
        "source": "synthetic.md",
        "writer": "synthetic",
        "created_at": "2026-01-01",
        "valid_from": "2026-01-01",
        "valid_to": None,
        "status": "active",
        "superseded_by": None,
        "confidence": 0.9,
        "tags": "synthetic",
    }
    if distance is not None:
        fact["distance"] = distance
    return fact


_SYNTHETIC_FACTS = (
    _synthetic_fact(1, "synthetic fact alpha", 0.1),
    _synthetic_fact(2, "synthetic fact bravo", 0.2),
    _synthetic_fact(3, "synthetic fact charlie", 0.3),
    _synthetic_fact(4, "synthetic fact delta", 0.4),
)


def _synthetic_fact_in_source(fact_id, text, source, distance=None):
    """One synthetic fact that belongs to a specific (synthetic) source."""
    fact = _synthetic_fact(fact_id, text, distance)
    fact["source"] = source
    return fact


def _keep_first_fact(fact):
    """Synthetic filter predicate: keep only the fact with id 1."""
    return fact["id"] == 1


# Two synthetic sources, so source-mode grouping is observable (a single source would
# make source order and fragment order identical).
_SYNTHETIC_MULTI_SOURCE_FACTS = (
    _synthetic_fact_in_source(1, "synthetic fact alpha", "synthetic-a.md", 0.1),
    _synthetic_fact_in_source(2, "synthetic fact bravo", "synthetic-b.md", 0.2),
    _synthetic_fact_in_source(3, "synthetic fact charlie", "synthetic-a.md", 0.3),
)


class _FakeCursor:
    """Minimal cursor stand-in: hands back pre-built rows."""

    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return list(self._rows)


class _FakeConnection:
    """Minimal connection stand-in for the "all active facts of one source" query.

    Source-mode aggregation asks the storage for every active fact of a source. This
    connection answers with the synthetic facts that belong to the requested source, in
    the column order the pipeline reads them.
    """

    _COLUMNS = ("id", "text", "source", "writer", "created_at", "valid_from",
                "valid_to", "status", "superseded_by", "confidence", "tags")

    def __init__(self, storage):
        self._storage = storage
        self.queries = []

    def execute(self, sql, params=()):
        self.queries.append((sql, params))
        source = params[0] if params else None
        rows = [
            tuple(fact[column] for column in self._COLUMNS)
            for fact in self._storage.facts
            if fact["source"] == source
        ]
        return _FakeCursor(rows)


class _FakeStorage:
    """In-memory stand-in for ``storage.MemoryStorage``: no DB, no corpus, no network.

    Implements only the surface the pipeline touches: ``connect()``,
    ``_ensure_connection()``, ``_validate_vector()``, ``search_keyword()``,
    ``search_vector()`` and a ``conn`` that answers the source-aggregation query.
    """

    def __init__(self, facts=_SYNTHETIC_FACTS, validate_error=None):
        self.facts = [dict(fact) for fact in facts]
        self.validate_error = validate_error
        self.connect_calls = 0
        self.keyword_calls = []
        self.vector_calls = []
        self.vector_objects = []
        self.validate_calls = 0
        self.validated_vectors = []
        self.conn = _FakeConnection(self)

    def connect(self):
        self.connect_calls += 1

    def _ensure_connection(self):
        return None

    def _validate_vector(self, vector, name="vector"):
        """Mirrors the storage contract: returns a fresh float list or raises ValueError."""
        self.validate_calls += 1
        if self.validate_error is not None:
            raise self.validate_error
        if vector is None:
            return None
        if not isinstance(vector, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple of numbers")
        if len(vector) != _SYNTHETIC_DIM:
            raise ValueError(f"{name} must have dimension {_SYNTHETIC_DIM}, got {len(vector)}")
        validated = []
        for index, value in enumerate(vector):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} elements must be numbers; index {index} is not numeric")
            if not math.isfinite(value):
                raise ValueError(f"{name} contains a non-finite value at index {index}")
            validated.append(float(value))
        self.validated_vectors.append(list(validated))
        return validated

    def search_keyword(self, query, limit=10):
        self.keyword_calls.append((query, limit))
        return [{k: v for k, v in fact.items() if k != "distance"} for fact in self.facts[:limit]]

    def search_vector(self, query_vector, limit=10):
        self.vector_calls.append((list(query_vector), limit))
        # The live object, so a test can check whether the pipeline hands storage a
        # list that is aliased to its cache.
        self.vector_objects.append(query_vector)
        return [dict(fact) for fact in self.facts[:limit]]


class _CountingEmbedder:
    """Synthetic embedder that records every call and returns a fixed vector."""

    def __init__(self, value=_SYNTHETIC_RAW_VECTOR, error=None, as_is=False):
        self.value = value
        self.error = error
        self.as_is = as_is
        self.calls = []

    def __call__(self, query):
        self.calls.append(query)
        if self.error is not None:
            raise self.error
        if self.value is None:
            return None
        if self.as_is:
            # Hand back the value untouched (e.g. a tuple) instead of copying it.
            return self.value
        return list(self.value)


class _QueryEmbeddingCacheHarness:
    """Tiny bounded harness: fake storage + counting embedder + pipeline under test.

    ``query_embedding_cache_size`` is only passed to the constructor when it is not
    ``_UNSET``, so the default-constructor path can be exercised as well.
    """

    def __init__(self, query_embedding_cache_size=_UNSET, cache_size=0,
                 embedder_value=_SYNTHETIC_RAW_VECTOR, embedder_error=None, validate_error=None,
                 facts=_SYNTHETIC_FACTS, context_mode="fragments", with_embedder=True,
                 embedder_as_is=False):
        self.storage = _FakeStorage(facts=facts, validate_error=validate_error)
        self.embedder = _CountingEmbedder(value=embedder_value, error=embedder_error,
                                          as_is=embedder_as_is)
        self.warnings = []
        kwargs = {"cache_size": cache_size, "context_mode": context_mode}
        if with_embedder:
            kwargs["embedder"] = self.embedder
        if query_embedding_cache_size is not _UNSET:
            kwargs["query_embedding_cache_size"] = query_embedding_cache_size
        self.pipeline = RetrievalPipeline(self.storage, **kwargs)

    @property
    def embed_calls(self):
        return len(self.embedder.calls)

    @property
    def keyword_calls(self):
        return len(self.storage.keyword_calls)

    @property
    def vector_calls(self):
        return len(self.storage.vector_calls)

    def search(self, query, limit=5, filter_func=None):
        """Call the public search() and capture any warning it emits.

        A ValueError from the pipeline propagates to the caller, but the warnings
        captured up to that point are still recorded.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                results = self.pipeline.search(query, limit=limit, filter_func=filter_func)
            finally:
                self.warnings = caught
        return results

    def ids(self, results):
        return [row["id"] for row in results]


class TestQueryEmbeddingCache(unittest.TestCase):
    """E16 query-embedding cache: bounded, query-keyed, filter-safe, never caches failures."""

    def test_default_is_disabled_and_re_embeds_on_each_result_cache_miss(self):
        harness = _QueryEmbeddingCacheHarness(cache_size=0)
        first = harness.search("synthetic query alpha", limit=1)
        second = harness.search("synthetic query alpha", limit=2)
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)
        self.assertEqual(harness.ids(first), [1])
        self.assertEqual(harness.ids(second), [1, 2])

    def test_default_disabled_re_embeds_even_when_the_result_cache_is_enabled(self):
        # Result cache on, query-embedding cache left at its default (disabled):
        # a result-cache miss (new limit) must re-embed.
        harness = _QueryEmbeddingCacheHarness(cache_size=4)
        harness.search("synthetic query papa", limit=2)
        harness.search("synthetic query papa", limit=3)
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)

    def test_explicit_zero_disables_the_query_embedding_cache(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=0, cache_size=0)
        harness.search("synthetic query bravo", limit=1)
        harness.search("synthetic query bravo", limit=2)
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)

    def test_enabled_cache_reuses_the_vector_across_different_limits(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        harness.search("synthetic query charlie", limit=1)
        harness.search("synthetic query charlie", limit=3)
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)
        first_vector, _first_limit = harness.storage.vector_calls[0]
        second_vector, _second_limit = harness.storage.vector_calls[1]
        self.assertEqual(first_vector, second_vector)
        self.assertEqual(first_vector, list(_SYNTHETIC_VALIDATED_VECTOR))

    def test_query_embedding_cache_is_independent_of_the_result_cache(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=4)
        harness.search("synthetic query delta", limit=2)
        harness.search("synthetic query delta", limit=2)
        # Identical (query, limit): served by the result cache.
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.keyword_calls, 1)
        self.assertEqual(harness.vector_calls, 1)
        harness.search("synthetic query delta", limit=3)
        # New limit: result-cache miss, embedding-cache hit -> no re-embedding.
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)

    def test_vector_is_reused_across_filters_without_reusing_filtered_results(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        only_first = harness.search("synthetic query echo", limit=5,
                                    filter_func=lambda fact: fact["id"] == 1)
        only_second = harness.search("synthetic query echo", limit=5,
                                     filter_func=lambda fact: fact["id"] == 2)
        # The embedding is reused (query-keyed), but each filter re-runs the storage searches.
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)
        # No stale filtered results are reused.
        self.assertEqual(harness.ids(only_first), [1])
        self.assertEqual(harness.ids(only_second), [2])

    def test_bounded_capacity_evicts_older_entries(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=2, cache_size=0)
        harness.search("synthetic query foxtrot", limit=2)
        harness.search("synthetic query golf", limit=2)
        harness.search("synthetic query hotel", limit=2)
        self.assertEqual(harness.embed_calls, 3)
        harness.search("synthetic query foxtrot", limit=2)
        # Evicted -> re-embedded.
        self.assertEqual(harness.embed_calls, 4)
        harness.search("synthetic query hotel", limit=2)
        # Still resident -> served from the cache.
        self.assertEqual(harness.embed_calls, 4)
        self.assertEqual(harness.keyword_calls, 5)
        self.assertEqual(harness.vector_calls, 5)

    def test_invalid_vector_is_not_cached_and_falls_back_to_keyword_only(self):
        harness = _QueryEmbeddingCacheHarness(
            query_embedding_cache_size=4,
            cache_size=0,
            validate_error=ValueError("synthetic invalid vector"),
        )
        first = harness.search("synthetic query india", limit=2)
        second = harness.search("synthetic query india", limit=2)
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.vector_calls, 0)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.ids(first), [1, 2])
        self.assertEqual(harness.ids(second), [1, 2])
        self.assertEqual(first[0]["score"], 1.0)
        self.assertTrue(any(issubclass(item.category, RuntimeWarning) for item in harness.warnings))

    def test_embedder_exception_is_not_cached_and_falls_back_to_keyword_only(self):
        harness = _QueryEmbeddingCacheHarness(
            query_embedding_cache_size=4,
            cache_size=0,
            embedder_error=RuntimeError("synthetic embedder failure"),
        )
        first = harness.search("synthetic query juliet", limit=2)
        second = harness.search("synthetic query juliet", limit=2)
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.vector_calls, 0)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.ids(first), [1, 2])
        self.assertEqual(harness.ids(second), [1, 2])
        self.assertEqual(first[0]["score"], 1.0)
        self.assertTrue(any(issubclass(item.category, RuntimeWarning) for item in harness.warnings))

    def test_embedder_returning_none_is_not_cached_and_falls_back_to_keyword_only(self):
        harness = _QueryEmbeddingCacheHarness(
            query_embedding_cache_size=4,
            cache_size=0,
            embedder_value=None,
        )
        first = harness.search("synthetic query kilo", limit=2)
        second = harness.search("synthetic query kilo", limit=2)
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.vector_calls, 0)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.ids(first), [1, 2])
        self.assertEqual(harness.ids(second), [1, 2])

    def test_invalidate_cache_clears_the_query_embedding_cache(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        harness.search("synthetic query lima", limit=2)
        self.assertEqual(harness.embed_calls, 1)
        harness.pipeline.invalidate_cache()
        harness.search("synthetic query lima", limit=3)
        self.assertEqual(harness.embed_calls, 2)

    def test_invalidate_cache_clears_result_and_embedding_caches_together(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=4)
        harness.search("synthetic query mike", limit=2)
        harness.search("synthetic query mike", limit=2)
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.keyword_calls, 1)
        harness.pipeline.invalidate_cache()
        harness.search("synthetic query mike", limit=2)
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)

    def test_cached_vector_is_the_validated_vector_used_for_the_dense_search(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        harness.search("synthetic query november", limit=2)
        harness.search("synthetic query november", limit=2)
        self.assertGreaterEqual(harness.storage.validate_calls, 1)
        self.assertEqual(len(harness.storage.validated_vectors), harness.storage.validate_calls)
        for vector in harness.storage.validated_vectors:
            self.assertEqual(len(vector), _SYNTHETIC_DIM)
            self.assertTrue(all(isinstance(value, float) for value in vector))
        # The embedder returned integers; the dense search must have used the validated
        # (float) vector, and the cached copy must be that same validated vector.
        self.assertTrue(all(isinstance(value, int) for value in harness.embedder.value))
        for vector, _limit in harness.storage.vector_calls:
            self.assertEqual(len(vector), _SYNTHETIC_DIM)
            self.assertTrue(all(isinstance(value, float) for value in vector))
            self.assertEqual(vector, list(_SYNTHETIC_VALIDATED_VECTOR))
        # Both dense searches used the validated vector, i.e. the cached copy is the
        # post-validation vector, not the raw embedder output.
        self.assertEqual([vector for vector, _ in harness.storage.vector_calls],
                         [list(_SYNTHETIC_VALIDATED_VECTOR),
                          list(_SYNTHETIC_VALIDATED_VECTOR)])

    def test_query_embedding_cache_size_accepts_the_documented_range(self):
        storage = _FakeStorage()
        for ok in (0, 1, 500):
            with self.subTest(value=ok):
                pipeline = RetrievalPipeline(storage, query_embedding_cache_size=ok)
                self.assertIsNotNone(pipeline)

    def test_out_of_range_query_embedding_cache_size_is_rejected_or_clamped(self):
        for bad in (-1, 501, 10_000):
            with self.subTest(value=bad):
                try:
                    harness = _QueryEmbeddingCacheHarness(
                        query_embedding_cache_size=bad, cache_size=0)
                except ValueError:
                    continue  # rejected outright with the repo's standard error: acceptable
                # If the implementation clamps instead of raising, the effective cache must
                # still be bounded, so the oldest entry is evicted rather than retained.
                queries = [f"synthetic bounded query {index}" for index in range(502)]
                for query in queries:
                    harness.search(query, limit=2)
                self.assertEqual(harness.embed_calls, 502)
                harness.search(queries[0], limit=2)
                self.assertEqual(harness.embed_calls, 503)  # oldest entry was evicted

    # --- capacity, eviction order and cache independence -------------------------

    def test_capacity_of_one_keeps_a_single_entry_and_evicts_on_every_miss(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=1, cache_size=0)
        harness.search("synthetic single alpha", limit=2)
        harness.search("synthetic single alpha", limit=3)
        # The single slot holds the entry: the second search is a cache hit.
        self.assertEqual(harness.embed_calls, 1)
        harness.search("synthetic single bravo", limit=2)
        # A new query needs the slot, so the previous entry is evicted.
        self.assertEqual(harness.embed_calls, 2)
        harness.search("synthetic single bravo", limit=3)
        # Bravo is the resident entry -> still a hit.
        self.assertEqual(harness.embed_calls, 2)
        harness.search("synthetic single alpha", limit=2)
        # Alpha was evicted when bravo took the slot -> re-embedded.
        self.assertEqual(harness.embed_calls, 3)
        # The result cache is disabled, so every search did the storage work.
        self.assertEqual(harness.keyword_calls, 5)
        self.assertEqual(harness.vector_calls, 5)

    def test_eviction_follows_least_recently_used_order_not_insertion_order(self):
        # Sequence chosen so LRU and FIFO disagree: "alpha" is touched again before
        # "charlie" is inserted, so LRU evicts "bravo" while FIFO would evict "alpha".
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=2, cache_size=0)
        harness.search("synthetic lru alpha", limit=2)
        harness.search("synthetic lru bravo", limit=2)
        harness.search("synthetic lru alpha", limit=3)
        # The third search hit the cache (no new embedding) and refreshed alpha's recency.
        self.assertEqual(harness.embed_calls, 2)
        harness.search("synthetic lru charlie", limit=2)
        self.assertEqual(harness.embed_calls, 3)
        harness.search("synthetic lru alpha", limit=2)
        # LRU: alpha survived the eviction (it was the most recently used entry).
        self.assertEqual(harness.embed_calls, 3)
        harness.search("synthetic lru bravo", limit=2)
        # LRU: bravo was the least recently used entry, so it was the one evicted.
        self.assertEqual(harness.embed_calls, 4)

    def test_result_cache_eviction_does_not_evict_the_query_embedding(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=3, cache_size=1)
        harness.search("synthetic query alpha", limit=1)
        harness.search("synthetic query bravo", limit=2)
        # The result cache holds one entry, so the first entry was evicted by the second.
        harness.search("synthetic query alpha", limit=1)
        # The result cache missed, but the embedding was still cached -> no re-embedding.
        self.assertEqual(harness.embed_calls, 2)
        # The storage work ran again because only the result entry was evicted.
        self.assertEqual(harness.keyword_calls, 3)
        self.assertEqual(harness.vector_calls, 3)

    def test_query_embedding_eviction_does_not_evict_the_result_cache_entry(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=1, cache_size=3)
        harness.search("synthetic query alpha", limit=1)
        harness.search("synthetic query bravo", limit=2)
        # The embedding cache holds one entry, so alpha's embedding was evicted by bravo.
        harness.search("synthetic query alpha", limit=1)
        # The result entry survived the embedding eviction -> served from the result cache.
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)
        # A different limit misses the result cache and therefore probes the embedding
        # cache again: alpha's embedding really was evicted, so it is embedded anew.
        harness.search("synthetic query alpha", limit=3)
        self.assertEqual(harness.embed_calls, 3)
        self.assertEqual(harness.keyword_calls, 3)
        self.assertEqual(harness.vector_calls, 3)

    def test_invalidate_cache_clears_embeddings_even_when_the_result_cache_was_never_used(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=4)
        harness.search("synthetic query filtered", limit=2, filter_func=_keep_first_fact)
        harness.search("synthetic query filtered", limit=2, filter_func=_keep_first_fact)
        # A filtered search never populates the result cache, but the embedding is cached.
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.keyword_calls, 2)
        harness.pipeline.invalidate_cache()
        harness.search("synthetic query filtered", limit=2, filter_func=_keep_first_fact)
        # The embedding cache was cleared as well, so the query is embedded again.
        self.assertEqual(harness.embed_calls, 2)

    # --- rejected vectors are never cached ---------------------------------------

    def test_vector_with_the_wrong_dimension_is_not_cached_and_falls_back_to_keyword_only(self):
        for wrong_dim in (_SYNTHETIC_DIM - 1, _SYNTHETIC_DIM + 1):
            with self.subTest(dimension=wrong_dim):
                harness = _QueryEmbeddingCacheHarness(
                    query_embedding_cache_size=4,
                    cache_size=0,
                    embedder_value=[1.0] * wrong_dim,
                )
                first = harness.search("synthetic query wrong dim", limit=2)
                second = harness.search("synthetic query wrong dim", limit=2)
                # Rejected -> never cached -> embedded again on the next search.
                self.assertEqual(harness.embed_calls, 2)
                self.assertEqual(harness.vector_calls, 0)
                self.assertEqual(harness.keyword_calls, 2)
                self.assertEqual(harness.ids(first), [1, 2])
                self.assertEqual(harness.ids(second), [1, 2])
                self.assertEqual(first[0]["score"], 1.0)
                self.assertTrue(any(issubclass(item.category, RuntimeWarning)
                                    for item in harness.warnings))

    def test_vector_with_a_non_finite_element_is_not_cached_and_falls_back_to_keyword_only(self):
        for bad_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(value=bad_value):
                broken = [1.0] * _SYNTHETIC_DIM
                broken[3] = bad_value
                harness = _QueryEmbeddingCacheHarness(
                    query_embedding_cache_size=4,
                    cache_size=0,
                    embedder_value=broken,
                )
                first = harness.search("synthetic query non finite", limit=2)
                second = harness.search("synthetic query non finite", limit=2)
                # Rejected -> never cached -> embedded again on the next search.
                self.assertEqual(harness.embed_calls, 2)
                self.assertEqual(harness.vector_calls, 0)
                self.assertEqual(harness.keyword_calls, 2)
                self.assertEqual(harness.ids(first), [1, 2])
                self.assertEqual(harness.ids(second), [1, 2])
                self.assertTrue(any(issubclass(item.category, RuntimeWarning)
                                    for item in harness.warnings))

    def test_vector_with_a_boolean_element_is_not_cached_and_falls_back_to_keyword_only(self):
        broken = [1.0] * _SYNTHETIC_DIM
        broken[5] = True
        harness = _QueryEmbeddingCacheHarness(
            query_embedding_cache_size=4,
            cache_size=0,
            embedder_value=broken,
        )
        first = harness.search("synthetic query bool element", limit=2)
        second = harness.search("synthetic query bool element", limit=2)
        # Rejected -> never cached -> embedded again on the next search.
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.vector_calls, 0)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.ids(first), [1, 2])
        self.assertEqual(harness.ids(second), [1, 2])
        self.assertEqual(first[0]["score"], 1.0)
        self.assertTrue(any(issubclass(item.category, RuntimeWarning)
                            for item in harness.warnings))

    def test_embedder_returning_a_tuple_is_accepted_and_cached_as_a_float_list(self):
        harness = _QueryEmbeddingCacheHarness(
            query_embedding_cache_size=4,
            cache_size=0,
            embedder_value=tuple(_SYNTHETIC_RAW_VECTOR),
            embedder_as_is=True,
        )
        harness.search("synthetic query tuple", limit=2)
        harness.search("synthetic query tuple", limit=3)
        # A tuple is an acceptable vector shape: it was validated and cached.
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.vector_calls, 2)
        for vector, _limit in harness.storage.vector_calls:
            self.assertIsInstance(vector, list)
            self.assertEqual(len(vector), _SYNTHETIC_DIM)
            self.assertEqual(vector, list(_SYNTHETIC_VALIDATED_VECTOR))

    # --- the cache key is the raw query text -------------------------------------

    def test_cache_key_is_the_raw_query_text_without_case_or_whitespace_normalisation(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        harness.search("synthetic query x", limit=2)
        harness.search(" synthetic query x", limit=2)
        harness.search("SYNTHETIC QUERY X", limit=2)
        # Three different raw keys -> three separate cache entries, three embeddings.
        self.assertEqual(harness.embed_calls, 3)
        self.assertEqual(harness.embedder.calls,
                         ["synthetic query x", " synthetic query x", "SYNTHETIC QUERY X"])
        # Repeating an already cached key is a hit; the other spellings stay separate.
        harness.search("synthetic query x", limit=3)
        self.assertEqual(harness.embed_calls, 3)
        harness.search(" synthetic query x", limit=3)
        self.assertEqual(harness.embed_calls, 3)
        harness.search("SYNTHETIC QUERY X", limit=3)
        self.assertEqual(harness.embed_calls, 3)

    # --- invalid queries populate nothing ----------------------------------------

    def test_blank_query_raises_value_error_and_populates_no_cache_entry(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        for blank in ("", "   ", "\t\n "):
            with self.subTest(query=blank):
                with self.assertRaises(ValueError):
                    harness.search(blank, limit=2)
        # Nothing was embedded, nothing was searched, so nothing could be cached.
        self.assertEqual(harness.embed_calls, 0)
        self.assertEqual(harness.keyword_calls, 0)
        self.assertEqual(harness.vector_calls, 0)
        self.assertEqual(harness.storage.validate_calls, 0)

    def test_query_longer_than_the_maximum_length_raises_value_error(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        too_long = "synthetic " + ("x" * 1000)
        self.assertGreater(len(too_long), 1000)
        with self.assertRaises(ValueError):
            harness.search(too_long, limit=2)
        # The rejected query never reached the embedder or the storage.
        self.assertEqual(harness.embed_calls, 0)
        self.assertEqual(harness.keyword_calls, 0)
        self.assertEqual(harness.vector_calls, 0)
        # A query of exactly the maximum length is accepted and cached normally.
        exact = "synthetic " + ("x" * (1000 - len("synthetic ")))
        self.assertEqual(len(exact), 1000)
        harness.search(exact, limit=2)
        harness.search(exact, limit=3)
        self.assertEqual(harness.embed_calls, 1)

    # --- the cached vector is a copy ---------------------------------------------

    def test_cached_vector_is_not_aliased_to_the_vector_handed_to_storage(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        harness.search("synthetic query alias", limit=2)
        # Poison the vector the pipeline handed to storage on the first search.
        first_vector = harness.storage.vector_objects[0]
        first_vector[0] = -99.0
        harness.search("synthetic query alias", limit=2)
        # The cache hit is unaffected: the cached entry is a copy, not the handed-out list.
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(len(harness.storage.vector_objects), 2)
        self.assertEqual(harness.storage.vector_objects[1], list(_SYNTHETIC_VALIDATED_VECTOR))
        # Poisoning the second handed-out vector must not poison the cache either.
        harness.storage.vector_objects[1][0] = -99.0
        harness.search("synthetic query alias", limit=2)
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.storage.vector_objects[2], list(_SYNTHETIC_VALIDATED_VECTOR))

    # --- a pipeline without an embedder ------------------------------------------

    def test_pipeline_without_an_embedder_never_embeds_and_still_warns(self):
        harness = _QueryEmbeddingCacheHarness(
            query_embedding_cache_size=4,
            cache_size=0,
            with_embedder=False,
        )
        results = harness.search("synthetic query no embedder", limit=2)
        self.assertEqual(harness.embed_calls, 0)
        # No dense search happened: the embedding cache had nothing to give.
        self.assertEqual(harness.vector_calls, 0)
        self.assertEqual(harness.keyword_calls, 1)
        self.assertEqual(harness.ids(results), [1, 2])
        self.assertEqual(results[0]["score"], 1.0)
        self.assertTrue(any(issubclass(item.category, RuntimeWarning)
                            for item in harness.warnings))
        # A second search behaves identically: nothing was cached, nothing changed.
        harness.search("synthetic query no embedder", limit=2)
        self.assertEqual(harness.embed_calls, 0)
        self.assertEqual(harness.vector_calls, 0)
        self.assertEqual(harness.keyword_calls, 2)

    # --- source mode shares the embedding ----------------------------------------

    def test_source_mode_shares_the_embedding_with_fragment_mode_and_still_groups_by_source(self):
        harness = _QueryEmbeddingCacheHarness(
            query_embedding_cache_size=4,
            cache_size=0,
            facts=_SYNTHETIC_MULTI_SOURCE_FACTS,
        )
        fragments = harness.search("synthetic query sources", limit=3)
        self.assertEqual(harness.embed_calls, 1)
        # Source mode is selected on the pipeline's public context_mode attribute.
        harness.pipeline.context_mode = "source"
        grouped = harness.search("synthetic query sources", limit=3)
        # The embedding is reused across the two modes (same raw query key).
        self.assertEqual(harness.embed_calls, 1)
        self.assertEqual(harness.vector_calls, 2)
        # Fragment mode returns the fused fragment order.
        self.assertEqual(harness.ids(fragments), [1, 2, 3])
        # Source mode returns the same facts grouped source-by-source, best source first.
        self.assertEqual(harness.ids(grouped), [1, 3, 2])
        sources = [row["source"] for row in grouped]
        self.assertEqual(sources, ["synthetic-a.md", "synthetic-a.md", "synthetic-b.md"])

    # --- the cache belongs to one pipeline ----------------------------------------

    def test_two_pipelines_keep_separate_query_embedding_caches(self):
        first = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        second = _QueryEmbeddingCacheHarness(query_embedding_cache_size=4, cache_size=0)
        first.search("synthetic query shared text", limit=2)
        second.search("synthetic query shared text", limit=2)
        # The same query text in a different pipeline is a different cache: both embed.
        self.assertEqual(first.embed_calls, 1)
        self.assertEqual(second.embed_calls, 1)
        # And each pipeline still serves its own cached entry afterwards.
        first.search("synthetic query shared text", limit=3)
        second.search("synthetic query shared text", limit=3)
        self.assertEqual(first.embed_calls, 1)
        self.assertEqual(second.embed_calls, 1)

    def test_invalidate_cache_is_harmless_when_the_embedding_cache_is_disabled(self):
        harness = _QueryEmbeddingCacheHarness(query_embedding_cache_size=0, cache_size=4)
        harness.search("synthetic query disabled", limit=2)
        harness.pipeline.invalidate_cache()
        harness.search("synthetic query disabled", limit=2)
        # The cache was disabled throughout, so every search embeds and nothing is reused.
        self.assertEqual(harness.embed_calls, 2)
        self.assertEqual(harness.keyword_calls, 2)
        self.assertEqual(harness.vector_calls, 2)


if __name__ == "__main__":
    unittest.main()
