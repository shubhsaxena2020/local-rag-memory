"""
Chaos, concurrency, crash recovery, and scale tests for AI Memory System.

Tests cover:
1. Concurrent threaded saves across multiple workers and subsequent count/integrity verification.
2. Subprocess transaction crash recovery: child starts transaction, inserts synthetic fact,
   signals parent; parent terminates only that child process; verifies uncommitted row is absent,
   committed rows survive, and PRAGMA integrity_check is ok.
3. WAL checkpointing (TRUNCATE) and valid SQLite online backup and restore.
4. Scale benchmark: 50,000 synthetic facts bulk insert + FTS5 keyword latency measurement
   (marked @unittest.skipUnless(os.getenv('RUN_SLOW_SCALE') == '1', ...)).
5. Retrieval Ollama-down fallback: tests graceful keyword-only fallback when embedding service fails,
   with explicit pending designation (no fake) for non-existent built-in Ollama adapter modules.

All operations use temporary directories and databases, closing all connections and cleaning up.
No external or system directories are touched.
"""

import os
import sys
import time
import shutil
import sqlite3
import tempfile
import threading
import subprocess
import statistics
import unittest
from pathlib import Path
from typing import List, Optional

# Ensure repository root is on sys.path for direct script, module, or discover execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage.storage import MemoryStorage


class BaseChaosTestCase(unittest.TestCase):
    """Base test case managing temporary database lifecycle and safe cleanup on Windows."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="test_chaos_")
        self.db_path = Path(self.tmp_dir) / "chaos_test.db"
        self._storages: List[MemoryStorage] = []
        self._procs: List[subprocess.Popen] = []

    def track_storage(self, storage: MemoryStorage) -> MemoryStorage:
        """Register a MemoryStorage instance for guaranteed teardown closure."""
        self._storages.append(storage)
        return storage

    def track_proc(self, proc: subprocess.Popen) -> subprocess.Popen:
        """Register a subprocess for guaranteed termination in teardown."""
        self._procs.append(proc)
        return proc

    def tearDown(self):
        """Cleanly terminate child processes, close SQLite connections, and remove temp directory."""
        # 1. Terminate any tracked subprocesses still running
        for proc in self._procs:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except Exception:
                    try:
                        proc.kill()
                        proc.wait(timeout=2.0)
                    except Exception:
                        pass
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()
        self._procs.clear()

        # 2. Close all opened MemoryStorage connections to unlock files on Windows
        for s in self._storages:
            try:
                s.close()
            except Exception:
                pass
        self._storages.clear()

        # 3. Clean up database, WAL, and SHM files in temp dir
        temp_path = Path(self.tmp_dir)
        if temp_path.exists():
            for p in temp_path.glob("*"):
                try:
                    if p.is_file():
                        p.unlink()
                except OSError:
                    pass

        # 4. Remove temp directory
        try:
            if os.path.exists(self.tmp_dir):
                shutil.rmtree(self.tmp_dir, ignore_errors=True)
        except OSError:
            pass


class TestConcurrentSaves(BaseChaosTestCase):
    """Concurrency chaos tests: multiple threads saving facts simultaneously."""

    def test_concurrent_threaded_saves_and_integrity(self):
        """Execute concurrent threaded saves (8 threads x 25 facts = 200 total)

        and verify final fact count, FTS5 sync, and PRAGMA integrity_check.
        """
        # Initialize schema with primary connection
        init_storage = MemoryStorage(self.db_path)
        init_storage.initialize_schema()
        init_storage.close()

        num_threads = 8
        facts_per_thread = 25
        expected_total_facts = num_threads * facts_per_thread
        thread_errors: List[tuple] = []
        threads: List[threading.Thread] = []

        def worker(thread_idx: int):
            try:
                # Each thread operates its own MemoryStorage instance
                storage = MemoryStorage(self.db_path)
                storage.connect()
                try:
                    # Configure busy timeout for smooth contention handling under concurrency
                    if storage.conn is not None:
                        storage.conn.execute("PRAGMA busy_timeout = 30000;")

                    for fact_idx in range(facts_per_thread):
                        fact_text = (
                            f"Concurrent worker thread {thread_idx} fact {fact_idx}: "
                            f"data integrity probe under concurrent WAL write load."
                        )
                        # Save fact with retry backoff for extreme thread scheduling contention
                        saved = False
                        for attempt in range(10):
                            try:
                                storage.save_fact(
                                    text=fact_text,
                                    source=f"thread-{thread_idx}",
                                    writer="concurrency-chaos-worker",
                                    tags=f"concurrency,worker_{thread_idx}",
                                    confidence=0.90
                                )
                                saved = True
                                break
                            except sqlite3.OperationalError as op_err:
                                if "locked" in str(op_err).lower() or "busy" in str(op_err).lower():
                                    time.sleep(0.02 * (attempt + 1))
                                else:
                                    raise
                        if not saved:
                            raise RuntimeError(
                                f"Thread {thread_idx} failed to save fact {fact_idx} after retries."
                            )
                finally:
                    storage.close()
            except Exception as exc:
                thread_errors.append((thread_idx, exc))

        for t_idx in range(num_threads):
            t = threading.Thread(target=worker, args=(t_idx,), name=f"ChaosWorker-{t_idx}")
            threads.append(t)
            t.start()

        for t in threads:
            t.join(timeout=30.0)
            self.assertFalse(t.is_alive(), f"Thread {t.name} did not finish within timeout.")

        # Assert no worker encountered errors
        self.assertEqual(thread_errors, [], f"Thread errors occurred during concurrent saves: {thread_errors}")

        # Open database with parent connection and verify state
        storage = MemoryStorage(self.db_path)
        storage.connect()
        self.track_storage(storage)

        # 1. Total facts count
        cursor = storage.conn.execute("SELECT COUNT(*) FROM facts;")
        total_facts = cursor.fetchone()[0]
        self.assertEqual(total_facts, expected_total_facts, "Total inserted facts mismatch.")

        # 2. Total FTS5 rows count (trigger synchronization)
        cursor = storage.conn.execute("SELECT COUNT(*) FROM facts_fts;")
        total_fts = cursor.fetchone()[0]
        self.assertEqual(total_fts, expected_total_facts, "FTS5 index count mismatch with facts count.")

        # 3. Database integrity check
        cursor = storage.conn.execute("PRAGMA integrity_check;")
        integrity_rows = cursor.fetchall()
        self.assertEqual(integrity_rows, [("ok",)], "PRAGMA integrity_check failed after concurrent saves.")

        # 4. Storage health check
        self.assertTrue(storage.health_check(), "Storage health_check returned False.")

        # 5. Searchability verification
        search_results = storage.search_keyword("concurrent", limit=20)
        self.assertGreater(len(search_results), 0, "Keyword search failed to return concurrent facts.")


class TestCrashRecovery(BaseChaosTestCase):
    """Crash recovery chaos test: abrupt process termination during uncommitted transaction."""

    def test_uncommitted_subprocess_crash_recovery(self):
        """Subprocess begins transaction, inserts synthetic fact, signals parent and waits;

        parent terminates only that child process, then verifies uncommitted row is absent,
        pre-existing committed facts are intact, and PRAGMA integrity_check=ok.
        """
        # Setup initial database and insert 1 committed baseline fact
        storage = MemoryStorage(self.db_path)
        storage.initialize_schema()
        storage.save_fact(
            text="Committed baseline fact surviving crash recovery test.",
            source="parent_baseline",
            writer="parent",
            tags="baseline,committed",
            confidence=1.0
        )
        storage.close()

        # Child subprocess script: begins transaction, inserts uncommitted synthetic fact,
        # signals READY, then blocks waiting to be terminated.
        child_script = """
import sqlite3
import sys
import time

db_path = sys.argv[1]
conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
conn.execute("PRAGMA journal_mode=WAL;")
conn.execute("BEGIN IMMEDIATE;")
conn.execute(
    "INSERT INTO facts (text, source, writer, tags, confidence) VALUES (?, ?, ?, ?, ?);",
    ("SYNTHETIC_UNCOMMITTED_CRASH_FACT_ID_987654321", "subprocess_crash", "child_worker", "chaos,crash", 0.5)
)
# Intentionally DO NOT COMMIT
sys.stdout.write("READY\\n")
sys.stdout.flush()

try:
    while True:
        time.sleep(0.5)
except Exception:
    pass
"""

        proc = subprocess.Popen(
            [sys.executable, "-c", child_script, str(self.db_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        self.track_proc(proc)

        # Wait for child to reach uncommitted state and signal READY
        ready_line = proc.stdout.readline()
        self.assertEqual(ready_line.strip(), "READY", f"Child subprocess failed to signal READY: {ready_line}")

        # Terminate ONLY that child process (simulating sudden crash / kill -9)
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)

        # Brief pause to allow OS file handles to release
        time.sleep(0.1)

        # Connect to database with parent storage
        storage = MemoryStorage(self.db_path)
        storage.connect()
        self.track_storage(storage)

        # 1. Verify uncommitted row is ABSENT
        cursor = storage.conn.execute(
            "SELECT COUNT(*) FROM facts WHERE text = 'SYNTHETIC_UNCOMMITTED_CRASH_FACT_ID_987654321';"
        )
        uncommitted_count = cursor.fetchone()[0]
        self.assertEqual(uncommitted_count, 0, "Uncommitted transaction row was unexpectedly found in facts table.")

        # 2. Verify pre-existing committed fact is INTACT
        cursor = storage.conn.execute("SELECT COUNT(*) FROM facts;")
        total_facts = cursor.fetchone()[0]
        self.assertEqual(total_facts, 1, "Expected exactly 1 committed baseline fact.")

        # 3. Verify FTS table matches facts table
        cursor = storage.conn.execute("SELECT COUNT(*) FROM facts_fts;")
        total_fts = cursor.fetchone()[0]
        self.assertEqual(total_fts, 1, "FTS5 table count mismatch after crash recovery.")

        # 4. Verify PRAGMA integrity_check = ok
        cursor = storage.conn.execute("PRAGMA integrity_check;")
        integrity = cursor.fetchall()
        self.assertEqual(integrity, [("ok",)], "PRAGMA integrity_check failed after crash recovery.")

        # 5. Verify health_check
        self.assertTrue(storage.health_check(), "Storage health_check returned False after crash recovery.")


class TestWALCheckpointAndBackup(BaseChaosTestCase):
    """Durability tests: WAL checkpointing and SQLite online backup and restore."""

    def test_wal_checkpoint(self):
        """Verify WAL checkpoint (TRUNCATE) successfully flushes write-ahead log to DB."""
        storage = MemoryStorage(self.db_path)
        storage.initialize_schema()
        self.track_storage(storage)

        # Insert 50 facts to populate WAL log
        for i in range(50):
            storage.save_fact(
                text=f"WAL durability test fact {i}: validating log flush and checkpointing.",
                source="wal_test",
                writer="checkpoint_tester",
                tags="wal,durability",
                confidence=0.95
            )

        wal_path = Path(f"{self.db_path}-wal")
        self.assertTrue(wal_path.exists(), "WAL file should exist in WAL mode after writes.")
        self.assertGreater(wal_path.stat().st_size, 0, "WAL file should contain written frames.")

        # Execute PRAGMA wal_checkpoint(TRUNCATE)
        cursor = storage.conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        busy, log_frames, checkpointed_frames = cursor.fetchone()
        self.assertEqual(busy, 0, f"WAL checkpoint returned busy={busy} (checkpoint blocked).")

        # Verify database integrity and facts integrity
        cursor = storage.conn.execute("PRAGMA integrity_check;")
        self.assertEqual(cursor.fetchall(), [("ok",)], "PRAGMA integrity_check failed after WAL checkpoint.")

        cursor = storage.conn.execute("SELECT COUNT(*) FROM facts;")
        self.assertEqual(cursor.fetchone()[0], 50, "Fact count mismatch after WAL checkpoint.")

        # Verify search works after checkpoint
        results = storage.search_keyword("checkpointing", limit=10)
        self.assertGreater(len(results), 0, "Keyword search failed after WAL checkpoint.")

    def test_sqlite_backup_and_restore(self):
        """Verify valid SQLite backup using online backup API and subsequent restore."""
        storage = MemoryStorage(self.db_path)
        storage.initialize_schema()
        self.track_storage(storage)

        for i in range(50):
            storage.save_fact(
                text=f"Backup restore test fact {i}: validating SQLite online backup API.",
                source="backup_test",
                writer="backup_tester",
                tags="backup,restore",
                confidence=0.9
            )

        backup_path = Path(self.tmp_dir) / "backup_facts.db"

        # 1. Perform online backup from live storage connection
        backup_conn = sqlite3.connect(str(backup_path))
        try:
            storage.conn.backup(backup_conn, pages=100)
        finally:
            backup_conn.close()

        self.assertTrue(backup_path.exists(), "Backup database file was not created.")
        self.assertGreater(backup_path.stat().st_size, 0, "Backup database file is empty.")

        # 2. Verify backup database independently
        backup_storage = MemoryStorage(backup_path)
        backup_storage.connect()
        self.track_storage(backup_storage)

        self.assertTrue(backup_storage.health_check(), "Backup database failed health_check.")
        cursor = backup_storage.conn.execute("PRAGMA integrity_check;")
        self.assertEqual(cursor.fetchall(), [("ok",)], "Backup database failed PRAGMA integrity_check.")

        cursor = backup_storage.conn.execute("SELECT COUNT(*) FROM facts;")
        self.assertEqual(cursor.fetchone()[0], 50, "Backup database row count mismatch.")

        backup_search = backup_storage.search_keyword("validating", limit=10)
        self.assertGreater(len(backup_search), 0, "Keyword search failed on backup database.")

        # 3. Restore from backup into a third clean database file
        restore_path = Path(self.tmp_dir) / "restored_facts.db"
        src_conn = sqlite3.connect(str(backup_path))
        dst_conn = sqlite3.connect(str(restore_path))
        try:
            src_conn.backup(dst_conn)
        finally:
            src_conn.close()
            dst_conn.close()

        self.assertTrue(restore_path.exists(), "Restored database file was not created.")
        restored_storage = MemoryStorage(restore_path)
        restored_storage.connect()
        self.track_storage(restored_storage)

        self.assertTrue(restored_storage.health_check(), "Restored database failed health_check.")
        cursor = restored_storage.conn.execute("PRAGMA integrity_check;")
        self.assertEqual(cursor.fetchall(), [("ok",)], "Restored database failed PRAGMA integrity_check.")
        cursor = restored_storage.conn.execute("SELECT COUNT(*) FROM facts;")
        self.assertEqual(cursor.fetchone()[0], 50, "Restored database row count mismatch.")

        fact_item = restored_storage.get_fact(1)
        self.assertIsNotNone(fact_item, "Failed to retrieve fact 1 from restored database.")
        self.assertIn("Backup restore test fact 0", fact_item["text"])


class TestScaleBenchmark(BaseChaosTestCase):
    """Large-corpus scale test: 50,000 synthetic facts bulk insert + FTS5 keyword latency."""

    @unittest.skipUnless(
        os.getenv("RUN_SLOW_SCALE") == "1",
        "Scale benchmark skipped by default; set RUN_SLOW_SCALE=1 to execute."
    )
    def test_fifty_thousand_facts_bulk_insert_and_keyword_latency(self):
        """Bulk insert 50,000 synthetic facts into temporary DB and measure keyword latency."""
        storage = MemoryStorage(self.db_path)
        storage.initialize_schema()
        self.track_storage(storage)

        target_count = 50000
        batch_size = 5000

        # Bulk insert in batched transactions so FTS5 sync triggers execute cleanly and rapidly
        t_start = time.perf_counter()
        for batch_start in range(0, target_count, batch_size):
            batch_data = []
            for i in range(batch_start, batch_start + batch_size):
                text = (
                    f"Synthetic telemetry fact {i}: system node cluster alpha-{i % 50} "
                    f"reporting memory consumption {i * 17 % 1024} MB and status nominal."
                )
                source = f"cluster-node-{i % 50}"
                writer = "scale_generator"
                tags = f"telemetry,node_{i % 50},batch_{batch_start // batch_size}"
                confidence = 0.95
                batch_data.append((text, source, writer, tags, confidence))

            with storage.conn:
                storage.conn.executemany(
                    "INSERT INTO facts (text, source, writer, tags, confidence) VALUES (?, ?, ?, ?, ?);",
                    batch_data
                )
        insert_duration = time.perf_counter() - t_start

        # Verify fact count and FTS synchronization
        cursor = storage.conn.execute("SELECT COUNT(*) FROM facts;")
        fact_count = cursor.fetchone()[0]
        self.assertEqual(fact_count, target_count, "Bulk insert fact count mismatch.")

        cursor = storage.conn.execute("SELECT COUNT(*) FROM facts_fts;")
        fts_count = cursor.fetchone()[0]
        self.assertEqual(fts_count, target_count, "Bulk insert FTS5 index count mismatch.")

        # Integrity check on 50,000 row database
        cursor = storage.conn.execute("PRAGMA integrity_check;")
        self.assertEqual(cursor.fetchall(), [("ok",)], "PRAGMA integrity_check failed on 50k row database.")
        self.assertTrue(storage.health_check(), "health_check failed on 50k row database.")

        # Keyword search latency test
        queries = [
            "telemetry",
            "cluster alpha-5",
            "consumption",
            "nominal",
            "cluster alpha-25",
            "reporting memory",
            "system node",
            "alpha-42 nominal",
            "telemetry consumption",
            "synthetic telemetry fact 12345",
            "telemetry cluster",
            "node cluster",
            "nominal consumption",
            "cluster alpha-10",
            "synthetic 45000",
            "cluster alpha-30 reporting",
            "memory consumption nominal",
            "system node cluster",
            "telemetry nominal",
            "cluster alpha-0"
        ]

        latencies_ms = []
        # Warm-up query
        storage.search_keyword("telemetry", limit=10)

        for q in queries:
            t_q = time.perf_counter()
            results = storage.search_keyword(q, limit=10)
            latency_ms = (time.perf_counter() - t_q) * 1000.0
            latencies_ms.append(latency_ms)
            self.assertGreater(len(results), 0, f"Expected non-empty results for query: {q}")

        latencies_ms.sort()
        p50_latency = statistics.median(latencies_ms)
        p95_latency = latencies_ms[int(len(latencies_ms) * 0.95)]

        # Enforce search latency bounds on 50k dataset (FTS5 typically sub-50ms)
        self.assertLess(p50_latency, 200.0, f"p50 latency {p50_latency:.2f}ms exceeded 200ms threshold.")
        self.assertLess(p95_latency, 500.0, f"p95 latency {p95_latency:.2f}ms exceeded 500ms threshold.")

        sys.stderr.write(
            f"\n[SCALE BENCHMARK] 50,000 facts inserted in {insert_duration:.2f}s "
            f"({target_count / insert_duration:.0f} facts/sec). "
            f"Keyword search latency: p50={p50_latency:.2f}ms, p95={p95_latency:.2f}ms.\n"
        )


class TestRetrievalOllamaFallback(BaseChaosTestCase):
    """Retrieval service fallback when embedding generation or Ollama is unavailable."""

    def test_retrieval_pipeline_embedder_down_fallback(self):
        """When embedder returns None (simulating Ollama connection failure),

        RetrievalPipeline must fall back to keyword-only search without unhandled crash.
        """
        try:
            from retrieval.pipeline import RetrievalPipeline
        except ImportError:
            self.skipTest("retrieval package not available in environment.")

        storage = MemoryStorage(self.db_path)
        storage.initialize_schema()
        self.track_storage(storage)

        # Save facts with keyword searchable content
        storage.save_fact(
            text="Quantum computing utilizes quantum superposition and entanglement principles.",
            source="quantum_paper",
            writer="physicist",
            tags="quantum,physics",
            confidence=0.95
        )
        storage.save_fact(
            text="Classical computer architecture relies on deterministic binary semiconductor gates.",
            source="cs_textbook",
            writer="engineer",
            tags="classical,architecture",
            confidence=0.85
        )

        # Simulate an embedder that catches Ollama failure (ConnectionRefused/down) and returns None
        def ollama_down_embedder(query: str):
            # Simulating Ollama server being down or unreachable
            return None

        pipeline = RetrievalPipeline(storage=storage, embedder=ollama_down_embedder)

        # Execute search - must gracefully fall back to keyword search
        results = pipeline.search("quantum computing", limit=5)
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0, "Fallback keyword search returned no results.")
        self.assertIn("quantum", results[0]["text"].lower())
        self.assertIn("score", results[0], "Result missing score field in fallback mode.")
        self.assertIn("confidence", results[0], "Result missing confidence field in fallback mode.")

    def test_direct_ollama_retrieval_module(self):
        """Check for dedicated built-in Ollama retrieval adapter package.

        If no dedicated package/module exists in the repo (retrieval exposes
        RetrievalPipeline with caller-injected embedder callable), mark
        explicit pending with no fake.
        """
        try:
            import importlib
            importlib.import_module("retrieval.ollama")
        except ImportError:
            self.skipTest(
                "Explicit pending: retrieval package exposes RetrievalPipeline with injected embedder callable; "
                "no built-in retrieval.ollama module exists in the codebase (no fake)"
            )


if __name__ == "__main__":
    unittest.main()
