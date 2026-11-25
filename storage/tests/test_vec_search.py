"""Focused tests for the sqlite-vec search path in storage.storage.MemoryStorage.

Run: .\\.venv\\Scripts\\python.exe -m unittest discover -s storage/tests -v
"""
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from storage import MemoryStorage, SQLITECVEC_AVAILABLE, VECTOR_DIMENSION  # noqa: E402

DIM = VECTOR_DIMENSION


def rand_vec(rng):
    return [rng.uniform(-1.0, 1.0) for _ in range(DIM)]


def _open(db_path, force_fallback=False):
    s = MemoryStorage(db_path)
    s.connect()
    if force_fallback:
        s.vec_available = False  # same state a failed extension load produces
    s.initialize_schema()
    return s


@unittest.skipUnless(SQLITECVEC_AVAILABLE, "sqlite-vec not installed")
class TestSqliteVecSearch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_vec_")
        self.db = os.path.join(self.tmp, "m.db")
        self.opened = []
        self.rng = random.Random(1234)

    def tearDown(self):
        for s in self.opened:
            try:
                s.close()
            except Exception:
                pass
        shutil.rmtree(self.tmp, ignore_errors=True)

    def open(self, force_fallback=False):
        s = _open(self.db, force_fallback)
        self.opened.append(s)
        return s

    def assert_equivalent(self, a, b):
        self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
        for ra, rb in zip(a, b):
            self.assertAlmostEqual(ra["distance"], rb["distance"], places=4)
            self.assertEqual(ra["text"], rb["text"])

    def test_vec_path_matches_forced_fallback(self):
        vec = self.open()
        self.assertEqual(vec.vector_backend, "sqlite-vec")
        for i in range(60):
            vec.save_fact(f"fact {i}", vector=rand_vec(self.rng))
        q = rand_vec(self.rng)

        # Prove the indexed path really answers (not a silent scan).
        indexed = vec._search_vector_indexed(q, 7)
        self.assertIsNotNone(indexed)
        via_vec = vec.search_vector(q, limit=7)

        fb = self.open(force_fallback=True)
        self.assertNotEqual(fb.vector_backend, "sqlite-vec")
        self.assertIsNone(fb._search_vector_indexed(q, 7))
        via_scan = fb.search_vector(q, limit=7)

        self.assertEqual(len(via_vec), 7)
        self.assert_equivalent(via_vec, via_scan)
        self.assert_equivalent(indexed, via_scan)

    def test_existing_rows_without_index_are_rebuilt(self):
        fb = self.open(force_fallback=True)  # rows written with no index at all
        for i in range(25):
            fb.save_fact(f"old {i}", vector=rand_vec(self.rng))
        fb.save_fact("no vector")
        fb.close()
        # Simulate a DB created before facts_vec existed.
        raw = sqlite3.connect(self.db)
        raw.execute("DROP TABLE IF EXISTS facts_vec")
        raw.commit()
        raw.close()

        vec = self.open()
        q = rand_vec(self.rng)
        got = vec.search_vector(q, limit=5)
        self.assertIsNotNone(vec._search_vector_indexed(q, 5))
        n_idx = vec.conn.execute("SELECT COUNT(*) FROM facts_vec").fetchone()[0]
        self.assertEqual(n_idx, 25)
        self.assert_equivalent(got, self.open(force_fallback=True).search_vector(q, limit=5))

    def test_stale_and_external_changes_reconciled(self):
        vec = self.open()
        ids = [vec.save_fact(f"f{i}", vector=rand_vec(self.rng)) for i in range(20)]
        q = rand_vec(self.rng)
        vec.search_vector(q, limit=3)  # index verified in sync

        # Another connection deletes a fact and adds a vectorless-index row behind our back.
        raw = sqlite3.connect(self.db)
        raw.execute("DELETE FROM facts WHERE id = ?", (ids[0],))
        raw.commit()
        raw.close()
        top = vec.search_vector(rand_vec(self.rng), limit=20)
        self.assertNotIn(ids[0], [r["id"] for r in top])
        self.assertEqual(len(top), 19)
        self.assertEqual(vec.conn.execute("SELECT COUNT(*) FROM facts_vec").fetchone()[0], 19)

    def test_unindexable_vectors_fall_back_but_agree(self):
        vec = self.open()
        vec.save_fact("zero", vector=[0.0] * DIM)
        for i in range(10):
            vec.save_fact(f"r{i}", vector=rand_vec(self.rng))
        q = rand_vec(self.rng)
        self.assertIsNone(vec._search_vector_indexed(q, 5))  # zero-norm row forces exact scan
        self.assert_equivalent(vec.search_vector(q, limit=5),
                               self.open(force_fallback=True).search_vector(q, limit=5))
        # Zero query vector also uses scan and agrees.
        z = [0.0] * DIM
        self.assert_equivalent(vec.search_vector(z, limit=5),
                               self.open(force_fallback=True).search_vector(z, limit=5))

    def test_save_fact_mirrors_in_same_transaction_and_index_survives_reopen(self):
        vec = self.open()
        fid = vec.save_fact("a", vector=rand_vec(self.rng))
        self.assertEqual(vec.conn.execute("SELECT rowid FROM facts_vec").fetchall(), [(fid,)])
        # Invalid vector must not leave index debris.
        with self.assertRaises(ValueError):
            vec.save_fact("bad", vector=[1.0] * (DIM - 1))
        self.assertEqual(vec.conn.execute("SELECT COUNT(*) FROM facts_vec").fetchone()[0], 1)
        vec.close()
        again = self.open()
        q = rand_vec(self.rng)
        self.assertEqual([r["id"] for r in again.search_vector(q, limit=3)], [fid])

    def test_limit_beyond_vec_k_cap_uses_fallback(self):
        vec = self.open()
        vec.save_fact("a", vector=rand_vec(self.rng))
        self.assertIsNone(vec._search_vector_indexed(rand_vec(self.rng), 5000))
        self.assertEqual(len(vec.search_vector(rand_vec(self.rng), limit=5000)), 1)

    def test_extension_load_failure_selects_fallback(self):
        mod = sys.modules[MemoryStorage.__module__]  # works whether imported as storage or storage.storage
        real = mod.sqlite_vec.load
        mod.sqlite_vec.load = lambda conn: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            s = self.open()
        finally:
            mod.sqlite_vec.load = real
        self.assertFalse(s.vec_available)
        self.assertNotEqual(s.vector_backend, "sqlite-vec")
        a = rand_vec(self.rng)
        fid = s.save_fact("x", vector=a)
        self.assertEqual(s.search_vector(a, limit=1)[0]["id"], fid)


if __name__ == "__main__":
    unittest.main()
