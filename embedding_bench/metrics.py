"""
Evaluation metrics calculation for retrieval recall, MRR, and query latencies.

Computes exact Recall@1, Recall@5, Mean Reciprocal Rank (MRR),
and percentile-based latency statistics (p50, p95).
"""

import math
import statistics
from typing import Any, Dict, List, Optional

from .dataset import GoldenQuery, normalize_passage
from .index import SearchResult


def calculate_percentile(values: List[float], p: float) -> float:
    """Calculate the p-th percentile of a list of floats using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    if p <= 0.0:
        return float(min(values))
    if p >= 100.0:
        return float(max(values))

    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (p / 100.0)
    floor_k = math.floor(k)
    ceil_k = math.ceil(k)

    if floor_k == ceil_k:
        return float(sorted_vals[int(k)])

    d0 = sorted_vals[floor_k] * (ceil_k - k)
    d1 = sorted_vals[ceil_k] * (k - floor_k)
    return float(d0 + d1)


def is_relevant_hit(retrieved: SearchResult, query: GoldenQuery, mode: str = "strict") -> bool:
    """Check if a retrieved passage is a gold answer for the query.

    mode="strict" (default; used for all corrected benchmark numbers):
      1. retrieved.doc_id == query.target_doc_id, or
      2. query.query_id is in the passage's original_ids (the passage is the
         deduplicated answer of this query), or
      3. the passage text equals the target answer after normalize_passage()
         (NFKC, whitespace-collapsed, casefolded).
      Same-source passages and substring containment are NOT relevant.

    mode="legacy_substring": the previous rule, which additionally accepted any
      >=6-char substring containment either way. Kept only so the old/new
      difference can be reproduced; never used by the harness.
    """
    if mode not in ("strict", "legacy_substring"):
        raise ValueError(f"unknown relevance mode: {mode}")

    if query.target_doc_id and retrieved.doc_id == query.target_doc_id:
        return True
    if query.query_id in retrieved.original_ids:
        return True

    ans_clean = normalize_passage(query.target_answer)
    ret_clean = normalize_passage(retrieved.text)
    if ans_clean and ret_clean:
        if ans_clean == ret_clean:
            return True
        if mode == "legacy_substring":
            if len(ans_clean) >= 6 and ans_clean in ret_clean:
                return True
            if len(ret_clean) >= 6 and ret_clean in ans_clean:
                return True
    return False


def is_relevant_hit_appendix_a(retrieved: SearchResult, query: GoldenQuery) -> bool:
    """Check relevance per PLAN-24H.md Appendix A rule.

    A passage is relevant if and only if:
      1. Exact gold passage/fact ID match when explicit (retrieved.doc_id == query.target_doc_id), OR
      2. Full normalized answer phrase containment subject to length/entity/number rule:
         - Normalize both query.target_answer and retrieved.text via normalize_passage()
         - The normalized answer must be fully contained in the normalized retrieved text
         - AND (answer length >= 12 OR answer is a complete entity/number)
      3. query.query_id in retrieved.original_ids (explicit fact ID match)

    Prohibitions:
      - No source-only matches (must be answer containment, not just same source)
      - No unrelated short substrings (answer must be meaningful length or entity/number)
      - No use of query ID as fact ID unless it's an explicit original_ids match

    Returns True if relevant per Appendix A, False otherwise.
    """
    # Condition 1: Explicit gold passage/fact ID match
    if query.target_doc_id and retrieved.doc_id == query.target_doc_id:
        return True

    # Condition 3: Explicit fact ID match via original_ids
    if query.query_id in retrieved.original_ids:
        return True

    # Condition 2: Full normalized answer phrase containment with length/entity/number rule
    ans_clean = normalize_passage(query.target_answer)
    ret_clean = normalize_passage(retrieved.text)
    
    if ans_clean and ret_clean:
        # Check if normalized answer is fully contained in normalized retrieved text
        if ans_clean in ret_clean:
            # Apply length/entity/number rule: answer must be >=12 chars OR be a complete entity/number
            # For simplicity, we'll treat answers >=12 chars as sufficient per the rule
            # In a more sophisticated implementation, we'd detect entities/numbers
            if len(ans_clean) >= 12:
                return True
            # TODO: Add proper entity/number detection for shorter answers
            # For now, we'll be conservative and require >=12 chars
    
    return False

def evaluate_single_query(retrieved_results: List[SearchResult], query: GoldenQuery, mode: str = "strict") -> Dict[str, Any]:
    """Evaluate retrieval results for a single query."""
    hit_at_1 = False
    hit_at_5 = False
    reciprocal_rank = 0.0
    first_rank: Optional[int] = None

    for rank_idx, result in enumerate(retrieved_results):
        rank = rank_idx + 1
        if is_relevant_hit(result, query, mode=mode):
            if first_rank is None:
                first_rank = rank
                reciprocal_rank = 1.0 / rank
            if rank == 1:
                hit_at_1 = True
            if rank <= 5:
                hit_at_5 = True
            break

    return {
        "query_id": query.query_id,
        "hit_at_1": hit_at_1,
        "hit_at_5": hit_at_5,
        "reciprocal_rank": reciprocal_rank,
        "first_rank": first_rank,
    }


def compute_retrieval_metrics(
    query_evaluations: List[Dict[str, Any]],
    latencies_ms: List[float],
) -> Dict[str, Any]:
    """Compute aggregate retrieval metrics across all evaluated queries."""
    total = len(query_evaluations)
    if total == 0:
        return {
            "total_queries": 0,
            "recall_at_1": 0.0,
            "recall_at_5": 0.0,
            "mrr": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_mean_ms": 0.0,
            "latency_min_ms": 0.0,
            "latency_max_ms": 0.0,
        }

    hits_1 = sum(1 for e in query_evaluations if e.get("hit_at_1"))
    hits_5 = sum(1 for e in query_evaluations if e.get("hit_at_5"))
    mrr_sum = sum(float(e.get("reciprocal_rank", 0.0)) for e in query_evaluations)

    recall_at_1 = hits_1 / total
    recall_at_5 = hits_5 / total
    mrr = mrr_sum / total

    p50 = calculate_percentile(latencies_ms, 50.0) if latencies_ms else 0.0
    p95 = calculate_percentile(latencies_ms, 95.0) if latencies_ms else 0.0
    mean_lat = statistics.mean(latencies_ms) if latencies_ms else 0.0
    min_lat = min(latencies_ms) if latencies_ms else 0.0
    max_lat = max(latencies_ms) if latencies_ms else 0.0

    return {
        "total_queries": total,
        "hits_at_1": hits_1,
        "hits_at_5": hits_5,
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "latency_p50_ms": round(p50, 2),
        "latency_p95_ms": round(p95, 2),
        "latency_mean_ms": round(mean_lat, 2),
        "latency_min_ms": round(min_lat, 2),
        "latency_max_ms": round(max_lat, 2),
    }
