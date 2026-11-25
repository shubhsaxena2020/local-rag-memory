"""
SQLite storage layer for AI Memory System.
Provides a DB connection, schema initialization, and basic CRUD operations for facts.
Designed for in-process use with WAL mode, FTS5, and optional vector search (sqlite-vec or numpy fallback).
"""

import sqlite3
import json
import struct
import sys
import math
import re
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

# Constants for input validation and boundaries
MAX_TEXT_LENGTH = 10000
MAX_TAGS_LENGTH = 1000
MAX_SOURCE_LENGTH = 500
MAX_WRITER_LENGTH = 200
VECTOR_DIMENSION = 768  # Default dimension for embedding models; adjust if needed
MAX_QUERY_LENGTH = 1000
MAX_SEARCH_LIMIT = 1000
DEFAULT_SEARCH_LIMIT = 10

# sqlite-vec derived index. facts.vector remains the source of truth; this table is a
# rebuildable cosine KNN index keyed by facts.id. Names/SQL below are constants only.
VEC_TABLE = "facts_vec"
VEC_TABLE_DDL = (
    f"CREATE VIRTUAL TABLE IF NOT EXISTS {VEC_TABLE} USING vec0("
    f"embedding float[{VECTOR_DIMENSION}] distance_metric=cosine)"
)
VEC_MAX_K = 4096  # sqlite-vec 0.1.9 hard limit for KNN k
VEC_SYNC_BATCH = 500

# Initialize availability flags on every import path
SQLITECVEC_AVAILABLE = False
NUMPY_AVAILABLE = False

try:
    import sqlite_vec
    SQLITECVEC_AVAILABLE = True
except ImportError:
    SQLITECVEC_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

class MemoryStorage:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn: Optional[sqlite3.Connection] = None
        self.vec_available = SQLITECVEC_AVAILABLE
        self.numpy_available = NUMPY_AVAILABLE
        self._vec_table_ok = False  # facts_vec exists with the expected definition
        self._vec_marker: Optional[Tuple[int, int]] = None  # (total_changes, data_version) when index verified in sync
        self._vec_unindexed = 0  # facts with a vector the index cannot rank (zero-norm / wrong size); >0 forces scan

    def connect(self) -> None:
        """Open a connection to the SQLite database with WAL mode and foreign keys."""
        if self.conn is not None:
            return
        self.conn = sqlite3.connect(
            self.db_path,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False
        )
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self._vec_table_ok = False
        self._vec_marker = None
        self._vec_unindexed = 0
        # Load sqlite-vec extension if available
        if self.vec_available:
            try:
                self.conn.enable_load_extension(True)
                sqlite_vec.load(self.conn)
                self.conn.execute("SELECT vec_version()").fetchone()  # prove the extension is functional
            except Exception as e:
                # Failed extension loading must clearly select numpy/pure-Python fallback
                self.vec_available = False
            finally:
                try:
                    self.conn.enable_load_extension(False)
                except Exception:
                    pass

    def close(self) -> None:
        """Close the database connection and release file locks."""
        if self.conn is not None:
            try:
                self.conn.close()
            finally:
                self.conn = None
                self._vec_table_ok = False
                self._vec_marker = None

    @property
    def vector_backend(self) -> str:
        """Return the active vector search backend: 'sqlite-vec', 'numpy' or 'pure-python'."""
        if self.vec_available and self._vec_table_ok:
            return "sqlite-vec"
        if self.numpy_available:
            return "numpy"
        return "pure-python"

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _ensure_connection(self) -> None:
        if self.conn is None:
            self.connect()

    def _validate_fact_input(
        self,
        text: str,
        source: Optional[str] = None,
        writer: Optional[str] = None,
        tags: Optional[str] = None,
        confidence: float = 1.0
    ) -> None:
        """Validate input sizes and types for a fact."""
        if not isinstance(text, str):
            raise ValueError(f"text must be a string, got {type(text).__name__}")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"text length ({len(text)}) exceeds maximum allowed ({MAX_TEXT_LENGTH})")
        if source is not None:
            if not isinstance(source, str):
                raise ValueError(f"source must be a string, got {type(source).__name__}")
            if len(source) > MAX_SOURCE_LENGTH:
                raise ValueError(f"source length ({len(source)}) exceeds maximum allowed ({MAX_SOURCE_LENGTH})")
        if writer is not None:
            if not isinstance(writer, str):
                raise ValueError(f"writer must be a string, got {type(writer).__name__}")
            if len(writer) > MAX_WRITER_LENGTH:
                raise ValueError(f"writer length ({len(writer)}) exceeds maximum allowed ({MAX_WRITER_LENGTH})")
        if tags is not None:
            if not isinstance(tags, str):
                raise ValueError(f"tags must be a string, got {type(tags).__name__}")
            if len(tags) > MAX_TAGS_LENGTH:
                raise ValueError(f"tags length ({len(tags)}) exceeds maximum allowed ({MAX_TAGS_LENGTH})")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"confidence must be a float between 0.0 and 1.0, got {type(confidence).__name__}")
        if not math.isfinite(confidence) or not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be a finite float between 0.0 and 1.0, got {confidence}")

    def _validate_vector(self, vector: Any, name: str = "vector") -> Optional[List[float]]:
        """Validate vector dimension and finite float values."""
        if vector is None:
            return None
        if not isinstance(vector, (list, tuple)):
            raise ValueError(f"{name} must be a list or tuple of numbers, got {type(vector).__name__}")
        if len(vector) != VECTOR_DIMENSION:
            raise ValueError(f"{name} must have dimension {VECTOR_DIMENSION}, got {len(vector)}")
        validated = []
        for i, val in enumerate(vector):
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ValueError(f"{name} elements must be numbers; index {i} has type {type(val).__name__}")
            if not math.isfinite(val):
                raise ValueError(f"{name} contains non-finite value at index {i}: {val}")
            validated.append(float(val))
        return validated

    def _validate_limit(self, limit: Any) -> int:
        """Validate and cap search limit."""
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError(f"limit must be an integer, got {type(limit).__name__}")
        if limit <= 0:
            return 0
        return min(limit, MAX_SEARCH_LIMIT)

    def _validate_keyword_query(self, query: Any) -> str:
        """Validate keyword query type and length."""
        if not isinstance(query, str):
            raise ValueError(f"query must be a string, got {type(query).__name__}")
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError(f"query length ({len(query)}) exceeds maximum allowed ({MAX_QUERY_LENGTH})")
        return query

    def _serialize_vector(self, vector: List[float]) -> bytes:
        """Serialize a list of floats to bytes (little-endian float32)."""
        validated = self._validate_vector(vector)
        if validated is None:
            return b""
        return struct.pack(f"{len(validated)}f", *validated)

    def _deserialize_vector(self, blob: bytes) -> List[float]:
        """Deserialize bytes to a list of floats."""
        if not blob:
            return []
        if len(blob) % 4 != 0:
            raise ValueError("Invalid vector blob length: not a multiple of 4 bytes")
        count = len(blob) // 4
        return list(struct.unpack(f"{count}f", blob))

    def _vector_distance(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine distance (1 - cosine similarity) between two vectors."""
        if self.numpy_available:
            try:
                import numpy as np
                v1 = np.asarray(vec1, dtype=np.float32)
                v2 = np.asarray(vec2, dtype=np.float32)
                dot = float(np.dot(v1, v2))
                norm1 = float(np.linalg.norm(v1))
                norm2 = float(np.linalg.norm(v2))
                if norm1 == 0.0 or norm2 == 0.0:
                    return 1.0
                cosine_sim = dot / (norm1 * norm2)
                cosine_sim = max(-1.0, min(1.0, cosine_sim))
                return 1.0 - cosine_sim
            except Exception:
                pass

        # Pure Python fallback
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        if norm1 == 0.0 or norm2 == 0.0:
            return 1.0
        cosine_sim = dot / (norm1 * norm2)
        cosine_sim = max(-1.0, min(1.0, cosine_sim))
        return 1.0 - cosine_sim

    # ------------------------------------------------------------------
    # sqlite-vec derived index helpers. facts.vector is always the source of truth.
    # ------------------------------------------------------------------
    _VEC_BLOB_LEN = VECTOR_DIMENSION * 4
    _VEC_VALID_SQL = (
        "vector IS NOT NULL AND length(vector) = ? AND vector != zeroblob(?)"
    )

    def _vec_state(self) -> Tuple[int, int]:
        assert self.conn is not None
        return (self.conn.total_changes, self.conn.execute("PRAGMA data_version").fetchone()[0])

    def _vec_in_sync(self) -> bool:
        if self._vec_marker is None:
            return False
        try:
            return self._vec_marker == self._vec_state()
        except Exception:
            return False

    def _ensure_vec_table(self) -> bool:
        """Create (or recreate on definition mismatch) the derived vec0 table. Never raises."""
        if not self.vec_available or self.conn is None:
            return False
        if self._vec_table_ok:
            return True
        try:
            row = self.conn.execute(
                "SELECT sql FROM sqlite_master WHERE name = ?", (VEC_TABLE,)
            ).fetchone()
            if row is not None:
                sql = (row[0] or "").lower()
                if f"float[{VECTOR_DIMENSION}]" not in sql or "cosine" not in sql:
                    self.conn.execute(f"DROP TABLE {VEC_TABLE}")  # derived data only; rebuilt on demand
                    row = None
            if row is None:
                self.conn.execute(VEC_TABLE_DDL)
            self.conn.commit()
            self._vec_table_ok = True
            self._vec_marker = None
            return True
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            self._vec_table_ok = False
            return False

    def _vec_reconcile(self) -> bool:
        """Make facts_vec match facts.vector (insert missing, delete stale) in one transaction.

        Returns True when the index is usable for KNN. Rows the index cannot rank
        (zero-norm vectors, wrong blob size) are not indexed and counted in _vec_unindexed.
        """
        if not self._ensure_vec_table():
            return False
        assert self.conn is not None
        try:
            n = self._VEC_BLOB_LEN
            want = {r[0] for r in self.conn.execute(
                f"SELECT id FROM facts WHERE {self._VEC_VALID_SQL}", (n, n))}
            have = {r[0] for r in self.conn.execute(f"SELECT rowid FROM {VEC_TABLE}")}
            unindexed = self.conn.execute(
                "SELECT COUNT(*) FROM facts WHERE vector IS NOT NULL "
                f"AND NOT ({self._VEC_VALID_SQL})", (n, n)).fetchone()[0]
            stale = sorted(have - want)
            missing = sorted(want - have)
            if stale or missing:
                self.conn.execute("SAVEPOINT vec_reconcile")
                try:
                    for i in range(0, len(stale), VEC_SYNC_BATCH):
                        chunk = stale[i:i + VEC_SYNC_BATCH]
                        self.conn.execute(
                            f"DELETE FROM {VEC_TABLE} WHERE rowid IN ({','.join('?' * len(chunk))})", chunk)
                    for i in range(0, len(missing), VEC_SYNC_BATCH):
                        chunk = missing[i:i + VEC_SYNC_BATCH]
                        rows = self.conn.execute(
                            f"SELECT id, vector FROM facts WHERE id IN ({','.join('?' * len(chunk))})", chunk
                        ).fetchall()
                        self.conn.executemany(
                            f"INSERT INTO {VEC_TABLE}(rowid, embedding) VALUES (?, ?)", rows)
                except Exception:
                    self.conn.execute("ROLLBACK TO SAVEPOINT vec_reconcile")
                    raise
                finally:
                    self.conn.execute("RELEASE SAVEPOINT vec_reconcile")
            self.conn.commit()
            self._vec_unindexed = unindexed
            self._vec_marker = self._vec_state()
            return True
        except Exception:
            try:
                self.conn.rollback()
            except Exception:
                pass
            self._vec_marker = None
            return False

    def _search_vector_indexed(
        self, query_vector: List[float], limit: int
    ) -> Optional[List[Dict[str, Any]]]:
        """KNN via sqlite-vec. Returns None whenever the caller must use the scan fallback."""
        if not (self.vec_available and self.conn is not None):
            return None
        if limit > VEC_MAX_K or not any(query_vector):
            return None  # zero query: scan defines distance 1.0 for everything
        try:
            if not self._vec_in_sync() and not self._vec_reconcile():
                return None
            if self._vec_unindexed:
                return None  # some vectors are unrankable by the index; scan keeps exact semantics
            hits = self.conn.execute(
                f"SELECT rowid, distance FROM {VEC_TABLE} WHERE embedding MATCH ? AND k = ?",
                (struct.pack(f"{VECTOR_DIMENSION}f", *query_vector), limit),
            ).fetchall()
            if any(d is None for _, d in hits):
                return None
            ids = [h[0] for h in hits]
            if not ids:
                return []
            dist = {h[0]: float(h[1]) for h in hits}
            rows = self.conn.execute(
                "SELECT id, text, source, writer, created_at, valid_from, valid_to, "
                "status, superseded_by, confidence, tags, vector FROM facts "
                f"WHERE id IN ({','.join('?' * len(ids))})", ids
            ).fetchall()
            if len(rows) != len(ids):
                self._vec_marker = None
                return None
            results = [{
                "id": r[0], "text": r[1], "source": r[2], "writer": r[3],
                "created_at": r[4], "valid_from": r[5], "valid_to": r[6],
                "status": r[7], "superseded_by": r[8], "confidence": r[9],
                "tags": r[10], "vector": self._deserialize_vector(r[11]),
                "distance": dist[r[0]],
            } for r in rows]
            results.sort(key=lambda x: (x["distance"], x["id"]))
            return results
        except Exception:
            self._vec_marker = None
            return None

    def initialize_schema(self) -> None:
        """Create tables and run migrations if needed. Repeated initialization is safe."""
        self._ensure_connection()
        assert self.conn is not None
        migration_path = Path(__file__).parent / "migrations" / "V1__init.sql"
        with open(migration_path, "r", encoding="utf-8") as f:
            sql = f.read()
        self.conn.executescript(sql)
        self.conn.commit()
        # Derived vector index; failure here must never affect the facts/FTS schema.
        self._ensure_vec_table()

    def health_check(self) -> bool:
        """Check if the database is accessible and schema is initialized.

        Returns False if the required schema is absent, and True only when
        the facts, facts_fts, and schema_migrations tables exist and can be queried.
        """
        try:
            self._ensure_connection()
            if self.conn is None:
                return False
            # Verify basic SQL execution
            self.conn.execute("SELECT 1;").fetchone()

            # Check that required tables exist in sqlite_master
            cursor = self.conn.execute(
                """
                SELECT name FROM sqlite_master
                WHERE name IN ('facts', 'facts_fts', 'schema_migrations');
                """
            )
            found = {row[0] for row in cursor.fetchall()}
            required = {"facts", "facts_fts", "schema_migrations"}
            if not required.issubset(found):
                return False

            # Confirm tables are actually queryable
            self.conn.execute("SELECT 1 FROM facts LIMIT 0;").fetchall()
            self.conn.execute("SELECT 1 FROM facts_fts LIMIT 0;").fetchall()
            self.conn.execute("SELECT 1 FROM schema_migrations LIMIT 0;").fetchall()
            return True
        except Exception:
            return False

    def save_fact(
        self,
        text: str,
        source: Optional[str] = None,
        writer: Optional[str] = None,
        tags: Optional[str] = None,
        confidence: float = 1.0,
        vector: Optional[List[float]] = None
    ) -> int:
        """Save a new fact to the database. Returns the row ID of the inserted fact."""
        self._validate_fact_input(text, source, writer, tags, confidence)
        validated_vector = self._validate_vector(vector, name="vector") if vector is not None else None

        self._ensure_connection()
        assert self.conn is not None

        vector_blob = self._serialize_vector(validated_vector) if validated_vector is not None else None

        table_ok = self._ensure_vec_table()
        was_synced = table_ok and self._vec_in_sync()
        # Zero-norm vectors are not indexed (cosine undefined in vec0); search falls back to the scan for them.
        indexable = table_ok and vector_blob is not None and any(validated_vector)
        use_index = indexable

        cursor = self.conn.execute(
            """
            INSERT INTO facts (text, source, writer, tags, confidence, vector)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (text, source, writer, tags, confidence, vector_blob)
        )
        fact_id = cursor.lastrowid
        mirrored = False
        if use_index:
            # Same transaction as the fact insert. A savepoint isolates index failures so the
            # fact itself is never lost; the next search reconciles any missing index row.
            try:
                self.conn.execute("SAVEPOINT vec_sync")
                try:
                    self.conn.execute(f"DELETE FROM {VEC_TABLE} WHERE rowid = ?", (fact_id,))
                    self.conn.execute(
                        f"INSERT INTO {VEC_TABLE}(rowid, embedding) VALUES (?, ?)",
                        (fact_id, vector_blob)
                    )
                    mirrored = True
                except Exception:
                    self.conn.execute("ROLLBACK TO SAVEPOINT vec_sync")
                finally:
                    self.conn.execute("RELEASE SAVEPOINT vec_sync")
            except Exception:
                mirrored = False
        self.conn.commit()
        if vector_blob is not None and not indexable:
            self._vec_marker = None  # unindexed vector: recount on next indexed search
        elif use_index and not mirrored:
            self._vec_marker = None  # force verification/repair on next indexed search
        elif was_synced:
            self._vec_marker = self._vec_state()
        return fact_id

    def get_fact(self, fact_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a fact by its ID."""
        if isinstance(fact_id, bool) or not isinstance(fact_id, int):
            raise ValueError(f"fact_id must be an integer, got {type(fact_id).__name__}")
        self._ensure_connection()
        assert self.conn is not None
        cursor = self.conn.execute(
            """
            SELECT id, text, source, writer, created_at, valid_from, valid_to,
                   status, superseded_by, confidence, tags, vector
            FROM facts
            WHERE id = ?
            """,
            (fact_id,)
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "text": row[1],
            "source": row[2],
            "writer": row[3],
            "created_at": row[4],
            "valid_from": row[5],
            "valid_to": row[6],
            "status": row[7],
            "superseded_by": row[8],
            "confidence": row[9],
            "tags": row[10],
            "vector": self._deserialize_vector(row[11]) if row[11] is not None else None
        }





    def search_keyword(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> List[Dict[str, Any]]:
        """Search facts using FTS5 keyword search.

        Preserves parameterized SQL and handles malformed FTS syntax predictably.
        """
        query = self._validate_keyword_query(query)
        capped_limit = self._validate_limit(limit)
        if capped_limit == 0 or not query.strip():
            return []

        self._ensure_connection()
        assert self.conn is not None

        sql = """
            SELECT f.id, f.text, f.source, f.writer, f.created_at, f.valid_from, f.valid_to,
                   f.status, f.superseded_by, f.confidence, f.tags
            FROM facts f
            JOIN facts_fts ft ON f.id = ft.rowid
            WHERE facts_fts MATCH ?
            ORDER BY bm25(facts_fts)
            LIMIT ?
        """

        try:
            cursor = self.conn.execute(sql, (query, capped_limit))
            rows = cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            if not isinstance(exc, sqlite3.OperationalError) and "fts5:" not in str(exc).lower():
                raise
            # Handle malformed FTS syntax predictably (e.g. unclosed quotes, bare operators, punctuation)
            # Extract word tokens and wrap in double quotes for safe phrase matching
            tokens = re.findall(r"[A-Za-z0-9_]+", query)
            if not tokens:
                return []
            safe_query = " ".join(f'"{t}"' for t in tokens)
            try:
                cursor = self.conn.execute(sql, (safe_query, capped_limit))
                rows = cursor.fetchall()
            except sqlite3.DatabaseError as exc:
                if not isinstance(exc, sqlite3.OperationalError) and "fts5:" not in str(exc).lower():
                    raise
                return []

        results = []
        for row in rows:
            results.append({
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "writer": row[3],
                "created_at": row[4],
                "valid_from": row[5],
                "valid_to": row[6],
                "status": row[7],
                "superseded_by": row[8],
                "confidence": row[9],
                "tags": row[10],
                "vector": None
            })
        return results

    def search_vector(
        self,
        query_vector: List[float],
        limit: int = DEFAULT_SEARCH_LIMIT
    ) -> List[Dict[str, Any]]:
        """Search facts using vector similarity (cosine distance)."""
        validated_vector = self._validate_vector(query_vector, name="query_vector")
        if validated_vector is None:
            raise ValueError("query_vector cannot be None")
        capped_limit = self._validate_limit(limit)
        if capped_limit == 0:
            return []

        self._ensure_connection()
        assert self.conn is not None

        indexed = self._search_vector_indexed(validated_vector, capped_limit)
        if indexed is not None:
            return indexed

        # Fallback: NumPy / pure-Python cosine scan over facts.vector.
        cursor = self.conn.execute(
            """
            SELECT id, text, source, writer, created_at, valid_from, valid_to,
                   status, superseded_by, confidence, tags, vector
            FROM facts
            WHERE vector IS NOT NULL
            ORDER BY id
            """
        )
        results = []
        for row in cursor.fetchall():
            vector_blob = row[11]
            if vector_blob is None:
                continue
            vector = self._deserialize_vector(vector_blob)
            distance = self._vector_distance(validated_vector, vector)
            results.append({
                "id": row[0],
                "text": row[1],
                "source": row[2],
                "writer": row[3],
                "created_at": row[4],
                "valid_from": row[5],
                "valid_to": row[6],
                "status": row[7],
                "superseded_by": row[8],
                "confidence": row[9],
                "tags": row[10],
                "vector": vector,
                "distance": distance
            })

        results.sort(key=lambda x: (x["distance"], x["id"]))
        return results[:capped_limit]

    def hybrid_search(
        self,
        keyword_query: str,
        query_vector: Optional[List[float]] = None,
        limit: int = DEFAULT_SEARCH_LIMIT,
        keyword_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining keyword and vector search using Reciprocal Rank Fusion (RRF)."""
        capped_limit = self._validate_limit(limit)
        if capped_limit == 0:
            return []

        if query_vector is None:
            return self.search_keyword(keyword_query, limit=capped_limit)

        validated_vector = self._validate_vector(query_vector, name="query_vector")

        keyword_results = self.search_keyword(keyword_query, limit=capped_limit * 2)
        vector_results = self.search_vector(validated_vector, limit=capped_limit * 2)

        keyword_ranks = {fact["id"]: rank for rank, fact in enumerate(keyword_results, start=1)}
        vector_ranks = {fact["id"]: rank for rank, fact in enumerate(vector_results, start=1)}

        all_ids = set(keyword_ranks.keys()) | set(vector_ranks.keys())
        scored_facts = []
        k = 60
        for fact_id in all_ids:
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
            rrf_score = (1.0 / (k + kw_rank)) + (1.0 / (k + vec_rank))
            fact_dict["rrf_score"] = rrf_score
            fact_dict.pop("distance", None)
            fact_dict.pop("vector", None)
            scored_facts.append(fact_dict)

        scored_facts.sort(key=lambda x: (-x["rrf_score"], x["id"]))
        return scored_facts[:capped_limit]

    # Additional methods for fact lifecycle (supersede, retract, etc.) can be added here.
    # For the storage layer, we focus on the core CRUD and search.
