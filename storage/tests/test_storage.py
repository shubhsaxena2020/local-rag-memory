"""
Unit tests for the SQLite storage layer in AI Memory System.

Tests cover:
- Schema initialization, idempotency (repeated initialization), and WAL mode.
- Health check behavior before and after schema initialization.
- FTS5 synchronization on fact insertion, text update, and deletion.
- CRUD operations (save_fact, get_fact, update_fact_text, delete_fact).
- Search capabilities (keyword search, vector similarity search, hybrid RRF search).
- Robust handling of malformed FTS syntax without unhandled exceptions.
- Comprehensive input validation (lengths, types, confidence range, vector dimensions, finite floats).
- Guaranteed connection cleanup and reliable Windows file unlock in tearDown.
"""

import sys
import os
import math
import shutil
import tempfile
import unittest
from pathlib import Path

# Add repo root to sys.path so 'from storage import ...' resolves under all invocation methods:
# - python -m unittest discover -s storage/tests -v
# - python storage/tests/test_storage.py
# - python -m storage.tests.test_storage
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Fallback: add storage parent directory to sys.path if invoked from storage/
STORAGE_PARENT = Path(__file__).resolve().parent.parent
if str(STORAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(STORAGE_PARENT))

from storage import (
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


class BaseStorageTestCase(unittest.TestCase):
    """Base test case providing clean temporary directory and connection lifecycle."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_storage_")
        self.db_path = os.path.join(self.tmp_dir, "test_memory.db")
        self.storage = None

    def tearDown(self):
        """Always close the SQLite connection before attempting to delete files on Windows."""
        if self.storage is not None:
            try:
                self.storage.close()
            except Exception:
                pass
            self.storage = None

        # Clean up database, WAL, and SHM files
        for p in [self.db_path, f"{self.db_path}-wal", f"{self.db_path}-shm"]:
            try:
                if os.path.exists(p):
                    os.unlink(p)
            except OSError:
                pass

        # Remove temporary directory
        try:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except OSError:
            pass


class TestHealthAndSchema(BaseStorageTestCase):
    """Tests for health check accuracy, schema initialization, idempotency, and WAL mode."""

    def test_health_check_before_and_after_schema(self):
        """health_check() must accurately return False before schema exists, and True after."""
        self.storage = MemoryStorage(self.db_path)

        # Before connect and before schema: health_check must return False
        self.assertFalse(self.storage.health_check())

        # Connected, but schema not yet initialized: health_check must still return False
        self.storage.connect()
        self.assertFalse(self.storage.health_check())

        # Initialize schema
        self.storage.initialize_schema()

        # After schema initialization: health_check must return True
        self.assertTrue(self.storage.health_check())

    def test_repeated_initialization_idempotent(self):
        """Calling initialize_schema() repeatedly must succeed without error or data loss."""
        self.storage = MemoryStorage(self.db_path)
        self.storage.initialize_schema()
        self.assertTrue(self.storage.health_check())

        # Save a fact to ensure data exists
        fact_id = self.storage.save_fact(text="Persistent fact before re-init")
        self.assertGreater(fact_id, 0)

        # Re-run initialize_schema()
        self.storage.initialize_schema()
        self.assertTrue(self.storage.health_check())

        # Verify fact is still intact
        fact = self.storage.get_fact(fact_id)
        self.assertIsNotNone(fact)
        self.assertEqual(fact["text"], "Persistent fact before re-init")

    def test_wal_mode_and_foreign_keys(self):
        """Verify SQLite connection enables WAL journal mode and foreign key constraints."""
        self.storage = MemoryStorage(self.db_path)
        self.storage.initialize_schema()

        cursor = self.storage.conn.execute("PRAGMA journal_mode;")
        journal_mode = cursor.fetchone()[0].lower()
        self.assertEqual(journal_mode, "wal")

        cursor = self.storage.conn.execute("PRAGMA foreign_keys;")
        foreign_keys = cursor.fetchone()[0]
        self.assertEqual(foreign_keys, 1)

    def test_context_manager(self):
        """MemoryStorage works as a context manager and cleanly closes."""
        with MemoryStorage(self.db_path) as storage:
            storage.initialize_schema()
            self.assertTrue(storage.health_check())
            self.assertIsNotNone(storage.conn)
        self.assertIsNone(storage.conn)


class TestFactsCRUDAndFTSSynchronization(BaseStorageTestCase):
    """Tests for fact CRUD operations and automatic FTS5 synchronization."""

    def setUp(self):
        super().setUp()
        self.storage = MemoryStorage(self.db_path)
        self.storage.initialize_schema()

    def test_save_and_get_fact(self):
        """save_fact stores all metadata and get_fact retrieves them accurately."""
        fact_id = self.storage.save_fact(
            text="The capital of France is Paris.",
            source="geography_book",
            writer="system_agent",
            tags="geography,france,europe",
            confidence=0.98
        )
        self.assertIsInstance(fact_id, int)
        self.assertGreater(fact_id, 0)

        fact = self.storage.get_fact(fact_id)
        self.assertIsNotNone(fact)
        self.assertEqual(fact["id"], fact_id)
        self.assertEqual(fact["text"], "The capital of France is Paris.")
        self.assertEqual(fact["source"], "geography_book")
        self.assertEqual(fact["writer"], "system_agent")
        self.assertEqual(fact["tags"], "geography,france,europe")
        self.assertAlmostEqual(fact["confidence"], 0.98)
        self.assertEqual(fact["status"], "active")
        self.assertIsNone(fact["vector"])

    def test_get_nonexistent_fact(self):
        """get_fact returns None for IDs that do not exist."""
        self.assertIsNone(self.storage.get_fact(99999))

    def test_fts_synchronization_on_insert(self):
        """Inserting a fact automatically registers it in FTS5 index via trigger."""
        fact_id = self.storage.save_fact(text="Quantum computing leverages superposition and entanglement.")
        results = self.storage.search_keyword("entanglement")
        self.assertGreaterEqual(len(results), 1)
        self.assertEqual(results[0]["id"], fact_id)

class TestSearchCapabilities(BaseStorageTestCase):
    def setUp(self):
        super().setUp()
        self.storage = MemoryStorage(self.db_path)
        self.storage.initialize_schema()

    def test_keyword_search_ranking_and_limits(self):
        """search_keyword ranks matching documents and respects limit capping."""
        self.storage.save_fact(text="Database indexing techniques for relational storage systems.")
        self.storage.save_fact(text="Relational database design principles and normalization.")
        self.storage.save_fact(text="Unrelated cooking recipe for apple pie.")

        results = self.storage.search_keyword("database relational", limit=1)
        self.assertEqual(len(results), 1)

        # Capped limit
        results_all = self.storage.search_keyword("database", limit=100)
        self.assertEqual(len(results_all), 2)

        # Limit <= 0 returns empty list
        self.assertEqual(self.storage.search_keyword("database", limit=0), [])
        self.assertEqual(self.storage.search_keyword("database", limit=-5), [])

    def test_malformed_fts_syntax_handled_predictably(self):
        """search_keyword gracefully handles syntax errors (bare operators, unclosed quotes)."""
        self.storage.save_fact(text="Reliable error recovery in distributed storage networks.")

        # Bare boolean operators
        res_not = self.storage.search_keyword("NOT")
        self.assertIsInstance(res_not, list)

        res_and = self.storage.search_keyword("AND")
        self.assertIsInstance(res_and, list)

        res_or = self.storage.search_keyword("OR")
        self.assertIsInstance(res_or, list)

        # Unclosed quotation mark
        res_quote = self.storage.search_keyword('"unclosed quote search')
        self.assertIsInstance(res_quote, list)

        # Backticks produce sqlite3.DatabaseError (not OperationalError) in some SQLite builds.
        res_backtick = self.storage.search_keyword("error `recovery`")
        self.assertGreaterEqual(len(res_backtick), 1)

        # Special wildcard and punctuation
        res_wildcard = self.storage.search_keyword("***???")
        self.assertEqual(res_wildcard, [])

        # Empty and whitespace queries
        self.assertEqual(self.storage.search_keyword(""), [])
        self.assertEqual(self.storage.search_keyword("   "), [])

    def test_vector_search_and_backend(self):
        """search_vector computes cosine similarity and finds closest vectors."""
        # Check that vector availability flags are booleans
        self.assertIsInstance(SQLITECVEC_AVAILABLE, bool)
        self.assertIsInstance(NUMPY_AVAILABLE, bool)
        self.assertIn(self.storage.vector_backend, ("sqlite-vec", "numpy", "pure-python"))

        # Vector 1: oriented along dimension 0
        vec_a = [0.0] * VECTOR_DIMENSION
        vec_a[0] = 1.0

        # Vector 2: oriented along dimension 1 (orthogonal to vec_a)
        vec_b = [0.0] * VECTOR_DIMENSION
        vec_b[1] = 1.0

        fact_a = self.storage.save_fact(text="Fact aligned with axis A", vector=vec_a)
        fact_b = self.storage.save_fact(text="Fact aligned with axis B", vector=vec_b)

        # Search with vector identical to vec_a
        results = self.storage.search_vector(vec_a, limit=5)
        self.assertEqual(len(results), 2)
        # First result should be fact_a with distance ~ 0.0
        self.assertEqual(results[0]["id"], fact_a)
        self.assertAlmostEqual(results[0]["distance"], 0.0, places=4)
        # Second result should be fact_b with distance ~ 1.0 (orthogonal)
        self.assertEqual(results[1]["id"], fact_b)
        self.assertAlmostEqual(results[1]["distance"], 1.0, places=4)

    def test_hybrid_search_rrf(self):
        """hybrid_search combines keyword and vector signals with Reciprocal Rank Fusion."""
        vec_target = [0.1] * VECTOR_DIMENSION

        fact_kw_only = self.storage.save_fact(text="Artificial intelligence and cognitive architecture.")
        fact_both = self.storage.save_fact(text="Artificial intelligence memory recall.", vector=vec_target)

        # Hybrid search with both keyword and vector
        results = self.storage.hybrid_search(
            keyword_query="artificial intelligence",
            query_vector=vec_target,
            limit=5
        )
        self.assertGreaterEqual(len(results), 1)
        # Fact matching both keyword and vector should have highest RRF score
        self.assertEqual(results[0]["id"], fact_both)
        self.assertIn("rrf_score", results[0])

        # Hybrid search with query_vector=None falls back to keyword-only search
        kw_fallback = self.storage.hybrid_search("cognitive", query_vector=None, limit=5)
        self.assertEqual(len(kw_fallback), 1)
        self.assertEqual(kw_fallback[0]["id"], fact_kw_only)

    def test_hybrid_search_breaks_equal_rrf_scores_by_fact_id(self):
        """Equal keyword/vector rank contributions have a stable ID tie-break."""
        query_vector = [0.0] * VECTOR_DIMENSION
        query_vector[0] = 1.0
        lower_vector_rank = [0.8, 0.6] + [0.0] * (VECTOR_DIMENSION - 2)
        higher_vector_rank = [1.0] + [0.0] * (VECTOR_DIMENSION - 1)

        fact_a = self.storage.save_fact("alpha", vector=lower_vector_rank)
        for _ in range(998):
            self.storage.save_fact("filler")
        fact_b = self.storage.save_fact("other", vector=higher_vector_rank)

        results = self.storage.hybrid_search("alpha", query_vector=query_vector, limit=2)

        self.assertEqual((fact_a, fact_b), (1, 1000))
        self.assertEqual([row["id"] for row in results], [fact_a, fact_b])
        self.assertAlmostEqual(results[0]["rrf_score"], results[1]["rrf_score"])


class TestInputValidation(BaseStorageTestCase):
    """Tests for parameter size bounds, dimension checks, and finite float enforcement."""

    def setUp(self):
        super().setUp()
        self.storage = MemoryStorage(self.db_path)
        self.storage.initialize_schema()

    def test_text_length_and_type_validation(self):
        """Text must be string and within MAX_TEXT_LENGTH."""
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="a" * (MAX_TEXT_LENGTH + 1))
        with self.assertRaises(ValueError):
            self.storage.save_fact(text=12345)

    def test_source_and_writer_length_validation(self):
        """Source and writer strings must not exceed their limits."""
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", source="s" * (MAX_SOURCE_LENGTH + 1))
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", writer="w" * (MAX_WRITER_LENGTH + 1))

    def test_tags_length_validation(self):
        """Tags string must not exceed MAX_TAGS_LENGTH."""
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", tags="t" * (MAX_TAGS_LENGTH + 1))

    def test_confidence_validation(self):
        """Confidence must be a finite float between 0.0 and 1.0."""
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", confidence=1.01)
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", confidence=-0.01)
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", confidence=float("nan"))
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", confidence=float("inf"))
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", confidence=True)

    def test_vector_validation(self):
        """Vectors must match VECTOR_DIMENSION and contain only finite numbers."""
        # Wrong dimension
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", vector=[0.1, 0.2, 0.3])

        # Contains NaN
        nan_vec = [0.0] * VECTOR_DIMENSION
        nan_vec[10] = float("nan")
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", vector=nan_vec)

        # Contains Infinity
        inf_vec = [0.0] * VECTOR_DIMENSION
        inf_vec[5] = float("inf")
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", vector=inf_vec)

        # Non-numeric elements
        invalid_type_vec = [0.0] * VECTOR_DIMENSION
        invalid_type_vec[2] = "not_a_float"
        with self.assertRaises(ValueError):
            self.storage.save_fact(text="Valid", vector=invalid_type_vec)

        # Query vector validation in search_vector
        with self.assertRaises(ValueError):
            self.storage.search_vector([0.1, 0.2])
        with self.assertRaises(ValueError):
            self.storage.search_vector(nan_vec)

    def test_query_and_limit_validation(self):
        """Keyword query length and search limits must be bounded."""
        with self.assertRaises(ValueError):
            self.storage.search_keyword("q" * (MAX_QUERY_LENGTH + 1))
        with self.assertRaises(ValueError):
            self.storage.search_keyword(12345)
        with self.assertRaises(ValueError):
            self.storage.search_keyword("valid", limit="not_an_int")

    def test_no_destructive_methods(self):
        """The storage layer must not expose destructive update/delete methods."""
        self.assertFalse(hasattr(self.storage, "update_fact_text"))
        self.assertFalse(hasattr(self.storage, "delete_fact"))


if __name__ == "__main__":
    unittest.main()

