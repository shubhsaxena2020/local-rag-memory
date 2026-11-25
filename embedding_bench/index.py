"""
Exact cosine retrieval index for embedding benchmark evaluation.

Supports vector normalization, dimension validation, finite vector checks,
exact cosine ranking (via NumPy when available, with pure Python fallback),
and memory footprint estimation.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .client import validate_embedding

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore
    NUMPY_AVAILABLE = False


@dataclass
class SearchResult:
    """Retrieved search result from CosineIndex."""
    doc_id: Any
    score: float
    text: str = ""
    source: str = ""
    original_ids: Set[Any] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CosineIndex:
    """Exact in-memory cosine similarity retrieval index."""

    def __init__(self, dimension: Optional[int] = None):
        self.dimension: Optional[int] = dimension
        self._doc_ids: List[Any] = []
        self._texts: List[str] = []
        self._sources: List[str] = []
        self._original_ids: List[Set[Any]] = []
        self._metadatas: List[Dict[str, Any]] = []
        self._vectors: List[List[float]] = []
        self._np_matrix: Optional[Any] = None
        self._dirty_np: bool = False

    def __len__(self) -> int:
        return len(self._doc_ids)

    def _normalize(self, vec: List[float]) -> List[float]:
        """Compute L2 normalized unit vector. Returns zero vector if norm is 0."""
        norm_sq = sum(x * x for x in vec)
        if norm_sq <= 0.0:
            return [0.0] * len(vec)
        norm = math.sqrt(norm_sq)
        return [x / norm for x in vec]

    def add(
        self,
        doc_id: Any,
        vector: List[float],
        text: str = "",
        source: str = "",
        original_ids: Optional[Set[Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a single document vector and its associated metadata to the index.

        Validates dimension consistency and finite values.
        """
        dim = validate_embedding(vector, expected_dim=self.dimension)
        if self.dimension is None:
            self.dimension = dim

        norm_vec = self._normalize(vector)

        self._doc_ids.append(doc_id)
        self._vectors.append(norm_vec)
        self._texts.append(text or "")
        self._sources.append(source or "")
        self._original_ids.append(set(original_ids) if original_ids else set())
        self._metadatas.append(metadata or {})
        self._dirty_np = True

    def add_batch(
        self,
        doc_ids: List[Any],
        vectors: List[List[float]],
        texts: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        original_ids_list: Optional[List[Set[Any]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Batch-insert multiple documents and vectors."""
        n = len(doc_ids)
        if len(vectors) != n:
            raise ValueError(f"Length mismatch: {n} doc_ids vs {len(vectors)} vectors")

        for idx in range(n):
            doc_id = doc_ids[idx]
            vec = vectors[idx]
            text = texts[idx] if texts and idx < len(texts) else ""
            source = sources[idx] if sources and idx < len(sources) else ""
            orig_ids = original_ids_list[idx] if original_ids_list and idx < len(original_ids_list) else None
            meta = metadatas[idx] if metadatas and idx < len(metadatas) else None
            self.add(doc_id, vec, text=text, source=source, original_ids=orig_ids, metadata=meta)

    def _sync_np_matrix(self) -> None:
        """Sync internal NumPy 2D array if NumPy is available."""
        if NUMPY_AVAILABLE and self._dirty_np:
            if self._vectors:
                self._np_matrix = np.array(self._vectors, dtype=np.float32)
            else:
                self._np_matrix = np.empty((0, self.dimension or 0), dtype=np.float32)
            self._dirty_np = False

    def search(self, query_vector: List[float], top_k: int = 5) -> List[SearchResult]:
        """Search the index for the top_k most similar items by cosine similarity.

        Returns a list of SearchResult sorted in descending order of similarity score.
        """
        if len(self) == 0:
            return []

        validate_embedding(query_vector, expected_dim=self.dimension)
        norm_query = self._normalize(query_vector)

        if NUMPY_AVAILABLE:
            self._sync_np_matrix()
            q_arr = np.array(norm_query, dtype=np.float32)
            # Dot product against all rows
            scores = np.dot(self._np_matrix, q_arr)
            # Get top indices
            k = min(top_k, len(scores))
            # argpartition for efficiency, then sort top k
            if k < len(scores):
                top_indices = np.argpartition(-scores, k)[:k]
                top_indices = top_indices[np.argsort(-scores[top_indices])]
            else:
                top_indices = np.argsort(-scores)

            results: List[SearchResult] = []
            for idx in top_indices:
                idx_int = int(idx)
                results.append(
                    SearchResult(
                        doc_id=self._doc_ids[idx_int],
                        score=float(scores[idx_int]),
                        text=self._texts[idx_int],
                        source=self._sources[idx_int],
                        original_ids=self._original_ids[idx_int],
                        metadata=self._metadatas[idx_int],
                    )
                )
            return results

        # Pure Python fallback
        scores_list: List[float] = []
        for v in self._vectors:
            # Dot product of unit vectors is cosine similarity
            s = sum(a * b for a, b in zip(norm_query, v))
            scores_list.append(s)

        # Sort indices by score descending
        sorted_indices = sorted(range(len(scores_list)), key=lambda i: scores_list[i], reverse=True)
        top_indices_py = sorted_indices[:top_k]

        return [
            SearchResult(
                doc_id=self._doc_ids[idx],
                score=scores_list[idx],
                text=self._texts[idx],
                source=self._sources[idx],
                original_ids=self._original_ids[idx],
                metadata=self._metadatas[idx],
            )
            for idx in top_indices_py
        ]

    def estimate_memory(self) -> Dict[str, Any]:
        """Estimate the in-memory footprint of indexed vectors and text metadata."""
        n = len(self._doc_ids)
        d = self.dimension or 0

        # Vector memory: 4 bytes per float32, 8 bytes per float64
        vector_bytes_float32 = n * d * 4
        vector_bytes_float64 = n * d * 8

        # Metadata memory estimation (character counts in utf-8 + structure overhead)
        text_bytes = sum(len(t.encode("utf-8")) for t in self._texts)
        source_bytes = sum(len(s.encode("utf-8")) for s in self._sources)
        metadata_bytes = text_bytes + source_bytes + (n * 128)  # 128 bytes dict/object overhead per entry

        total_bytes_float32 = vector_bytes_float32 + metadata_bytes
        total_mb_float32 = total_bytes_float32 / (1024 * 1024)

        return {
            "num_vectors": n,
            "dimension": d,
            "vector_bytes_float32": vector_bytes_float32,
            "vector_bytes_float64": vector_bytes_float64,
            "vector_ram_mb": round(vector_bytes_float32 / (1024 * 1024), 4),
            "metadata_bytes": metadata_bytes,
            "metadata_ram_mb": round(metadata_bytes / (1024 * 1024), 4),
            "total_ram_mb": round(total_mb_float32, 4),
            "backend": "numpy" if NUMPY_AVAILABLE else "pure_python",
        }
