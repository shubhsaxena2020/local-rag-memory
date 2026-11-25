"""
AI Memory System - SQLite Storage Layer Package.
Provides MemoryStorage with WAL mode, FTS5 keyword search, and vector/hybrid search.
"""

from .storage import (
    MemoryStorage,
    SQLITECVEC_AVAILABLE,
    NUMPY_AVAILABLE,
    VECTOR_DIMENSION,
    MAX_TEXT_LENGTH,
    MAX_SOURCE_LENGTH,
    MAX_WRITER_LENGTH,
    MAX_TAGS_LENGTH,
    MAX_QUERY_LENGTH,
    MAX_SEARCH_LIMIT,
)

__all__ = [
    "MemoryStorage",
    "SQLITECVEC_AVAILABLE",
    "NUMPY_AVAILABLE",
    "VECTOR_DIMENSION",
    "MAX_TEXT_LENGTH",
    "MAX_SOURCE_LENGTH",
    "MAX_WRITER_LENGTH",
    "MAX_TAGS_LENGTH",
    "MAX_QUERY_LENGTH",
    "MAX_SEARCH_LIMIT",
]
