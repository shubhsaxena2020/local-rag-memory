"""Retrieval pipeline for AI Memory System.

Implements parallel BM25 keyword + dense retrieval with RRF fusion,
optional reranking, query caching, and filters.

Two independent caches exist:

* the *result* cache (``cache_size``) stores finished result lists, keyed by
  (query, limit, filter) and skipped entirely when a filter is supplied;
* the *query embedding* cache (``query_embedding_cache_size``) stores validated
  query embeddings, keyed by the query text only, and is disabled by default.
"""

import math
import warnings
from typing import List, Dict, Any, Optional, Callable, Tuple

from storage import MemoryStorage

# Constants
DEFAULT_SEARCH_LIMIT = 10
MAX_SEARCH_LIMIT = 1000
RRF_K = 60
MAX_CANDIDATES_FOR_RERANK = 30
DEFAULT_CACHE_SIZE = 128
DEFAULT_QUERY_EMBEDDING_CACHE_SIZE = 0  # E16: off by default; opt in once measured
MAX_QUERY_EMBEDDING_CACHE_SIZE = 500
LOW_CONFIDENCE_THRESHOLD = 0.5  # Threshold for considering reranking
CONTEXT_MODES = ("fragments", "source")
SOURCE_SCORE_MODES = ("max", "top2", "sum")
MIN_INFORMATIVE_ALNUM = 4  # fragments with fewer alphanumerics ("3", "(15") cannot identify a source
ANSWERABILITY_TOP_N = 5  # query-level answerability = mean cosine similarity of the N best dense hits
_DISTANCE_EPS = 1e-6  # float32 rounding slack around the valid cosine-distance range [0, 2]

class RetrievalPipeline:
    def __init__(
        self,
        storage: MemoryStorage,
        embedder: Optional[Callable[[str], Optional[List[float]]]] = None,
        reranker: Optional[Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]]] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        rrf_k: int = RRF_K,
        max_candidates_for_rerank: int = MAX_CANDIDATES_FOR_RERANK,
        low_confidence_threshold: float = LOW_CONFIDENCE_THRESHOLD,
        context_mode: str = "fragments",
        source_score_mode: str = "max",
        downweight_fragments: bool = False,
        query_embedding_cache_size: int = DEFAULT_QUERY_EMBEDDING_CACHE_SIZE,
    ):
        """
        Initialize the retrieval pipeline.

        :param storage: An instance of MemoryStorage.
        :param embedder: A callable that takes a string and returns a list of floats (embedding) or None.
        :param reranker: An optional callable that takes a list of results and returns a reranked list.
        :param cache_size: Size of the query cache (0 to disable).
        :param rrf_k: The RRF constant.
        :param max_candidates_for_rerank: Maximum number of candidates to pass to the reranker.
        :param low_confidence_threshold: If the top result's confidence is below this, consider reranking.
        :param context_mode: "fragments" (default, unchanged ranking) or "source". In "source" mode the
            ranked fragments vote for their source; results are ordered source-by-source (best source
            first) and each chosen source's remaining active facts are appended after its retrieved ones,
            so a fragment-sized answer is not buried among fragments of other sources.
        :param source_score_mode: how fragment scores combine into a source score ("max", "top2", "sum").
        :param downweight_fragments: if True, near-empty fragments (< 4 alphanumerics) contribute nothing
            to the source score; they are still kept in the context (never excluded).
        :param query_embedding_cache_size: Size of the query-embedding cache (0 to disable, max 500).
            Disabled by default. When enabled, validated query embeddings are reused across
            searches; the embedding cache is keyed by the query text only (never by limit or
            filter), so it can only save embedding work, never results.
        """
        if type(cache_size) is not int or not 0 <= cache_size <= 500:
            raise ValueError("cache_size must be an integer from 0 to 500")
        if (type(query_embedding_cache_size) is not int
                or not 0 <= query_embedding_cache_size <= MAX_QUERY_EMBEDDING_CACHE_SIZE):
            raise ValueError(
                f"query_embedding_cache_size must be an integer from 0 to {MAX_QUERY_EMBEDDING_CACHE_SIZE}")
        if type(rrf_k) is not int or rrf_k < 1:
            raise ValueError("rrf_k must be a positive integer")
        if type(max_candidates_for_rerank) is not int or not 1 <= max_candidates_for_rerank <= 30:
            raise ValueError("max_candidates_for_rerank must be an integer from 1 to 30")
        if (isinstance(low_confidence_threshold, bool)
                or not isinstance(low_confidence_threshold, (int, float))
                or not math.isfinite(low_confidence_threshold)
                or not 0 <= low_confidence_threshold <= 1):
            raise ValueError("low_confidence_threshold must be finite and between 0 and 1")
        if context_mode not in CONTEXT_MODES:
            raise ValueError(f"context_mode must be one of {CONTEXT_MODES}")
        if source_score_mode not in SOURCE_SCORE_MODES:
            raise ValueError(f"source_score_mode must be one of {SOURCE_SCORE_MODES}")
        self.context_mode = context_mode
        self.source_score_mode = source_score_mode
        self.downweight_fragments = bool(downweight_fragments)
        self.storage = storage
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.max_candidates_for_rerank = max_candidates_for_rerank
        self.low_confidence_threshold = low_confidence_threshold
        self.cache_size = cache_size
        self._query_cache = {}  # Simple dict cache; we'll implement LRU if needed but size is small
        self._cache_keys = []   # To track insertion order for LRU when cache_size is exceeded
        # Query-embedding cache (E16): separate from the result cache above, keyed by query text
        # only. Disabled unless query_embedding_cache_size > 0.
        self.query_embedding_cache_size = query_embedding_cache_size
        self._query_embedding_cache: Dict[str, List[float]] = {}
        self._query_embedding_cache_keys: List[str] = []
        self.rerank_gate_activations = 0  # times the low-confidence gate opened (cache misses only)
        self.rerank_fallbacks = 0         # gate opened but reranker output was unusable -> fused order kept

        # Ensure storage is connected
        self.storage.connect()

    def _validate_query(self, query: str) -> str:
        """Validate query string length and type."""
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if not query.strip():
            raise ValueError("query must not be blank")
        if len(query) > 1000:  # MAX_QUERY_LENGTH from storage.py
            raise ValueError(f"query length ({len(query)}) exceeds maximum allowed (1000)")
        return query

    def _apply_filters(self, results: List[Dict[str, Any]], filter_func: Optional[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
        """Apply a filter function to a list of results."""
        if filter_func is None:
            return results
        return [fact for fact in results if filter_func(fact)]

    def _search_keyword(self, query: str, limit: int, filter_func: Optional[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
        """Perform keyword search and apply filters."""
        validated_query = self._validate_query(query)
        raw_results = self.storage.search_keyword(validated_query, limit=limit)
        return self._apply_filters(raw_results, filter_func)

    def _search_vector(self, query_vector: List[float], limit: int, filter_func: Optional[Callable[[Dict[str, Any]], bool]]) -> List[Dict[str, Any]]:
        """Perform vector search and apply filters."""
        raw_results = self.storage.search_vector(query_vector, limit=limit)
        return self._apply_filters(raw_results, filter_func)

    def _rrf_fusion(
        self,
        keyword_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Perform Reciprocal Rank Fusion on keyword and vector results."""
        keyword_ranks = {fact["id"]: rank for rank, fact in enumerate(keyword_results, start=1)}
        vector_ranks = {fact["id"]: rank for rank, fact in enumerate(vector_results, start=1)}
        all_ids = set(keyword_ranks.keys()) | set(vector_ranks.keys())
        scored_facts = []
        for fact_id in all_ids:
            # Get the fact dict from either list
            fact_dict = None
            for fact in keyword_results:
                if fact["id"] == fact_id:
                    fact_dict = fact.copy()
                    break
            if fact_dict is None:
                for fact in vector_results:
                    if fact["id"] == fact_id:
                        fact_dict = fact.copy()
                        break
            if fact_dict is None:
                continue
            kw_rank = keyword_ranks.get(fact_id, len(keyword_results) + 1)
            vec_rank = vector_ranks.get(fact_id, len(vector_results) + 1)
            rrf_score = (1.0 / (self.rrf_k + kw_rank)) + (1.0 / (self.rrf_k + vec_rank))
            fact_dict["rrf_score"] = rrf_score
            # Remove distance and vector if present (from vector search)
            fact_dict.pop("distance", None)
            fact_dict.pop("vector", None)
            scored_facts.append(fact_dict)
        # Sort by score, then ID so ties are stable across processes.
        scored_facts.sort(key=lambda x: (-x["rrf_score"], x["id"]))
        return scored_facts

    def _answerability_score(self, vector_results: List[Dict[str, Any]]) -> Optional[float]:
        """Query-level dense answerability: mean cosine similarity (1 - distance) of the top five dense hits.

        Computed from the vector results exactly as retrieved (after ``filter_func``, before fusion,
        reranking, source aggregation or slicing). Fewer than five hits -> mean over those available.
        Returns None (feature omitted, never a guessed number) when there are no dense hits, or when any
        of the hits considered carries a missing / non-numeric / bool / non-finite distance or one outside
        the cosine-distance range [0, 2] (beyond float32 slack); the latter also emits a RuntimeWarning.
        Storage guarantees finite distances in [0, 2] (scan clamps cosine; the index path rejects None),
        so this is a defensive guard. No threshold is applied here: calibration belongs to the consumer.
        """
        distances = []
        for fact in vector_results:
            d = fact.get("distance")
            if (isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d)
                    or not -_DISTANCE_EPS <= d <= 2.0 + _DISTANCE_EPS):
                warnings.warn("Invalid dense distance in vector results; omitting answerability_score.", RuntimeWarning)
                return None
            distances.append(float(d))
        if not distances:
            return None
        top = sorted(distances)[:ANSWERABILITY_TOP_N]
        return math.fsum(1.0 - min(max(d, 0.0), 2.0) for d in top) / len(top)

    def _source_facts(self, source: str, filter_func) -> List[Dict[str, Any]]:
        """All active facts of one source in ascending id order (deterministic)."""
        self.storage._ensure_connection()
        rows = self.storage.conn.execute(
            "SELECT id, text, source, writer, created_at, valid_from, valid_to, status, "
            "superseded_by, confidence, tags FROM facts WHERE source = ? AND status = 'active' "
            "ORDER BY id", (source,),
        ).fetchall()
        keys = ("id", "text", "source", "writer", "created_at", "valid_from", "valid_to",
                "status", "superseded_by", "confidence", "tags")
        facts = [dict(zip(keys, r), vector=None) for r in rows]
        return self._apply_filters(facts, filter_func)

    def _aggregate_by_source(self, ranked: List[Dict[str, Any]], limit: int, filter_func) -> List[Dict[str, Any]]:
        """Reorder fragment-ranked results source-by-source (parent-document context assembly).

        ``ranked`` is the fused (or reranked) fragment order; each item has ``score``/``rrf_score``.
        Source score = max / top-2 sum / sum of its fragments' scores. Sources are ordered by
        (-source_score, best fragment rank, source name); inside a source, retrieved fragments keep their
        ranked order and unretrieved active facts follow in id order. Fact ids/sources are preserved and
        each fact appears once. ``score`` stays the fragment's own RRF value (0.0 for unretrieved
        siblings) so downstream abstention gates see no inflated score; ``source_score`` is extra.
        """
        by_source: Dict[Any, List[Tuple[int, Dict[str, Any]]]] = {}
        for rank, fact in enumerate(ranked):
            by_source.setdefault(fact.get("source"), []).append((rank, fact))

        def frag_score(fact: Dict[str, Any]) -> float:
            value = fact.get("rrf_score", fact.get("score", 0.0))
            if self.downweight_fragments and sum(ch.isalnum() for ch in str(fact.get("text", ""))) < MIN_INFORMATIVE_ALNUM:
                return 0.0
            return value

        scored = []
        for source, members in by_source.items():
            values = sorted((frag_score(f) for _, f in members), reverse=True)
            if self.source_score_mode == "max":
                agg = values[0]
            elif self.source_score_mode == "top2":
                agg = sum(values[:2])
            else:
                agg = sum(values)
            scored.append((-agg, members[0][0], str(source), source, agg))
        scored.sort(key=lambda t: t[:3])

        out: List[Dict[str, Any]] = []
        seen = set()
        for _neg, _rank, _name, source, agg in scored:
            for _, fact in by_source[source]:
                if fact["id"] not in seen:
                    seen.add(fact["id"])
                    item = fact.copy()
                    item["source_score"] = agg
                    out.append(item)
            if source is not None:
                for fact in self._source_facts(source, filter_func):
                    if fact["id"] not in seen:
                        seen.add(fact["id"])
                        fact["rrf_score"] = 0.0
                        fact["score"] = 0.0
                        fact["source_score"] = agg
                        out.append(fact)
            if len(out) >= limit:
                break
        return out[:limit]

    def _get_cache_key(self, query: str, limit: int, filter_func: Optional[Callable[[Dict[str, Any]], bool]]) -> Optional[Tuple]:
        """Generate a cache key unless a callable filter could have mutable captured state."""
        if filter_func is not None:
            return None
        return (query, limit, filter_func)

    def _get_from_cache(self, key: Tuple) -> Optional[List[Dict[str, Any]]]:
        """Retrieve results from cache if present and not expired."""
        if key in self._query_cache:
            return self._query_cache[key]
        return None

    def _store_in_cache(self, key: Tuple, value: List[Dict[str, Any]]) -> None:
        """Store results in cache, enforcing LRU if cache size exceeded."""
        if self.cache_size <= 0:
            return
        if key in self._query_cache:
            # Move to end (most recently used)
            self._cache_keys.remove(key)
        elif len(self._cache_keys) >= self.cache_size:
            # Remove least recently used
            oldest_key = self._cache_keys.pop(0)
            del self._query_cache[oldest_key]
        self._query_cache[key] = value
        self._cache_keys.append(key)

    def _get_query_embedding_from_cache(self, query: str) -> Optional[List[float]]:
        """Return a cached, already validated query embedding, or None.

        The embedding cache is keyed by the query text only. It never holds results, so a
        hit can never return stale filtered results; a miss simply re-embeds.
        """
        if self.query_embedding_cache_size <= 0:
            return None
        if query in self._query_embedding_cache:
            # Move to end (most recently used)
            self._query_embedding_cache_keys.remove(query)
            self._query_embedding_cache_keys.append(query)
            return list(self._query_embedding_cache[query])
        return None

    def _store_query_embedding_in_cache(self, query: str, vector: List[float]) -> None:
        """Store a validated query embedding, enforcing LRU if the cache size is exceeded."""
        if self.query_embedding_cache_size <= 0:
            return
        if query in self._query_embedding_cache:
            # Move to end (most recently used)
            self._query_embedding_cache_keys.remove(query)
        elif len(self._query_embedding_cache_keys) >= self.query_embedding_cache_size:
            # Remove least recently used
            oldest_key = self._query_embedding_cache_keys.pop(0)
            del self._query_embedding_cache[oldest_key]
        self._query_embedding_cache[query] = list(vector)
        self._query_embedding_cache_keys.append(query)

    def _embed_query_vector(self, query: str) -> Optional[List[float]]:
        """Embed the query and return a validated vector, or None on failure.

        Uses the (opt-in) query-embedding cache, which is keyed by the query text only and
        therefore can only save embedding work - never results. Failures are never cached:
        an embedder exception, a None result or a vector rejected by
        ``storage._validate_vector`` all return None, so the caller falls back to keyword
        search exactly as before. The cache is not lock-protected, matching the result cache.
        """
        cached_vector = self._get_query_embedding_from_cache(query)
        if cached_vector is not None:
            return cached_vector
        try:
            embedded_vector = self.embedder(query)
            if embedded_vector is None:
                return None
            query_vector = self.storage._validate_vector(embedded_vector, name="query_vector")
        except Exception:
            warnings.warn("Embedding failed or returned an invalid vector; using keyword search.", RuntimeWarning)
            return None
        if isinstance(query_vector, (list, tuple)):
            # Only a vector that passed validation is cached; failures are not.
            self._store_query_embedding_in_cache(query, query_vector)
        return query_vector

    def invalidate_cache(self) -> None:
        """Invalidate the entire query cache (result cache and query-embedding cache)."""
        self._query_cache.clear()
        self._cache_keys.clear()
        self._query_embedding_cache.clear()
        self._query_embedding_cache_keys.clear()

    def _apply_reranker(self, query: str, candidates: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        """Run the reranker on capped candidates; return None to keep the fused (RRF) order.

        Query-aware rerankers (``wants_query = True``) receive ``query=``; legacy callables get
        only the candidates. Any exception or a result that is not a permutation of the input
        ids is treated as a failure: warn, count it, and fall back to the fused order.
        """
        try:
            if getattr(self.reranker, "wants_query", False) is True:
                result = self.reranker(candidates, query=query)
            else:
                result = self.reranker(candidates)
            if (not isinstance(result, list) or len(result) != len(candidates)
                    or not all(isinstance(r, dict) and "id" in r for r in result)
                    or sorted(r["id"] for r in result) != sorted(c["id"] for c in candidates)):
                raise ValueError("reranker must return a permutation of its input candidates")
            return result
        except Exception:
            self.rerank_fallbacks += 1
            warnings.warn("Reranker failed or returned invalid output; using fused order.", RuntimeWarning)
            return None

    def search(
        self,
        query: str,
        limit: int = DEFAULT_SEARCH_LIMIT,
        filter_func: Optional[Callable[[Dict[str, Any]], bool]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform retrieval and return top-limit results.

        :param query: The search query string.
        :param limit: The number of results to return (default 10, max 1000).
        :param filter_func: An optional function that takes a fact dict and returns True if the fact should be included.
        :return: A list of fact dicts, each containing at least: id, text, source, writer, created_at, valid_from, valid_to,
                 status, superseded_by, confidence, tags, and a score field (rrf_score or reranker score).
        """
        # Validate limit
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError(f"limit must be an integer, got {type(limit).__name__}")
        if limit <= 0:
            raise ValueError("limit must be positive")
        limit = min(limit, MAX_SEARCH_LIMIT)

        # Validate query
        query = self._validate_query(query)

        # Check cache
        cache_key = self._get_cache_key(query, limit, filter_func)
        cached_results = self._get_from_cache(cache_key) if cache_key is not None else None
        if cached_results is not None:
            return cached_results[:limit]

        # Determine candidate limit for initial retrieval (we want enough for reranking and top-limit)
        candidate_limit = max(limit * 2, self.max_candidates_for_rerank)
        candidate_limit = min(candidate_limit, MAX_SEARCH_LIMIT)

        query_vector = None
        if self.embedder is not None:
            query_vector = self._embed_query_vector(query)
        if query_vector is None:
            if self.embedder is None:
                warnings.warn("Embedding function unavailable; using keyword search.", RuntimeWarning)
            keyword_results = self._search_keyword(query, candidate_limit, filter_func)
            results = []
            for rank, fact in enumerate(keyword_results[:candidate_limit], start=1):
                fact_copy = fact.copy()
                fact_copy["score"] = 1.0 / rank
                results.append(fact_copy)
            if self.context_mode == "source":
                results = self._aggregate_by_source(results, limit, filter_func)
            else:
                results = results[:limit]
            if cache_key is not None:
                self._store_in_cache(cache_key, results)
            return results

        # Storage exposes one SQLite connection; concurrent operations on that connection
        # can corrupt FTS5's virtual-table query state. Run these reads on the caller thread.
        keyword_results = self._search_keyword(query, candidate_limit, filter_func)
        vector_results = self._search_vector(query_vector, candidate_limit, filter_func)

        # Dense answerability is a property of the query's actual dense top-five, so it is taken here,
        # before fusion / reranking / source aggregation / slicing can change which facts are returned.
        answerability = self._answerability_score(vector_results)

        # Fuse with RRF
        fused_results = self._rrf_fusion(keyword_results, vector_results)

        # Determine if we should rerank
        reranked = False
        if self.reranker is not None and len(fused_results) > 0:
            # We need to decide on a confidence metric for the top result.
            # The fused results do not have a confidence field from the fact? They do because we copied the fact.
            # But note: in _rrf_fusion we copied the fact, so the confidence field is present.
            top_confidence = fused_results[0].get("confidence", 0.0)
            if top_confidence < self.low_confidence_threshold:
                # Take top N candidates for reranking
                candidates = fused_results[:self.max_candidates_for_rerank]
                self.rerank_gate_activations += 1
                reranked_candidates = self._apply_reranker(query, candidates)
                if reranked_candidates is not None:
                    # Replace the top candidates with the reranked ones, keeping the rest
                    fused_results = reranked_candidates + fused_results[self.max_candidates_for_rerank:]
                    reranked = True

        # Take top-limit results
        if self.context_mode == "source":
            top_results = self._aggregate_by_source(fused_results, limit, filter_func)
        else:
            top_results = fused_results[:limit]

        # Prepare output: ensure each result has a score and confidence.
        # For fused results, we have rrf_score. For reranked results, we don't have a score from the reranker.
        # We'll use the rrf_score as the score for non-reranked results, and for reranked results we don't have a score.
        # We can set the score to the rrf_score for all, and then if reranked, we don't change the score? 
        # But the reranker might change the order, so the rrf_score is no longer reflective.
        # Alternatively, we can compute a new score based on the rerank position? 
        # The requirement is to output score/confidence. We'll output:
        #   score: the rrf_score (if available) or 0.0 if not available (should not happen)
        #   confidence: the fact's confidence
        # We'll also note in the REPORT that the score is the RRF score (or 0.0 for keyword-only) and confidence is the fact's confidence.
        output_results = []
        for fact in top_results:
            fact_copy = fact.copy()
            # Ensure we have a score field (rrf_score) from fusion; if not, set to 0.0
            score = fact_copy.get("rrf_score", 0.0)
            fact_copy["score"] = score
            # Same query-level value on every returned fact (incl. source siblings); omitted if unavailable.
            if answerability is not None:
                fact_copy["answerability_score"] = answerability
            # Confidence is already present from the fact
            output_results.append(fact_copy)

        # Store in cache
        if cache_key is not None:
            self._store_in_cache(cache_key, output_results)

        return output_results

# Example usage and tests would be in the tests directory.
