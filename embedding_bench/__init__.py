"""
Embedding Benchmark Package (W2) for AI Memory System.
"""

from .client import (
    DEFAULT_OLLAMA_URL,
    OllamaEmbeddingClient,
    validate_embedding,
    validate_loopback_url,
)
from .dataset import (
    CorpusRecord,
    GoldenQuery,
    load_golden_set,
    resolve_golden_set_path,
)
from .harness import (
    CANDIDATE_MODELS,
    estimate_model_memory,
    run_embedding_benchmark,
    select_winner,
)
from .formatting import FORMATTERS, VARIANTS, format_document, format_query, resolve_variants
from .index import CosineIndex, SearchResult
from .metrics import (
    calculate_percentile,
    compute_retrieval_metrics,
    evaluate_single_query,
    is_relevant_hit,
)

__all__ = [
    "DEFAULT_OLLAMA_URL",
    "CANDIDATE_MODELS",
    "OllamaEmbeddingClient",
    "validate_embedding",
    "validate_loopback_url",
    "CosineIndex",
    "SearchResult",
    "CorpusRecord",
    "GoldenQuery",
    "load_golden_set",
    "resolve_golden_set_path",
    "compute_retrieval_metrics",
    "evaluate_single_query",
    "is_relevant_hit",
    "calculate_percentile",
    "estimate_model_memory",
    "select_winner",
    "run_embedding_benchmark",
    "FORMATTERS",
    "VARIANTS",
    "format_document",
    "format_query",
    "resolve_variants",
]
