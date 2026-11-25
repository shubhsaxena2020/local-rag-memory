# Chaos, Concurrency, and Scale Test Report

Package `chaos/` implements comprehensive chaos, concurrency, crash recovery, WAL durability, scale benchmarking, and retrieval fallback tests for the AI Memory System (`storage.MemoryStorage`).

## Test Suite Overview

All tests are unittest-discoverable in `chaos/tests/test_chaos.py` and run against isolated temporary databases in temporary directories with guaranteed connection closure and cleanup in `tearDown`. No operations are performed outside test temp dirs.

1. **Concurrent Threaded Writes (`TestConcurrentSaves`)**:
   - Spawns 8 concurrent worker threads, each saving 25 distinct facts (200 total) through independent `MemoryStorage` instances in WAL mode.
   - Configures `PRAGMA busy_timeout = 30000;` and operational retry backoff to handle high thread contention smoothly.
   - Verifies 100% thread completion with zero errors, total fact count (`200`), FTS5 index synchronization count (`200`), `PRAGMA integrity_check = ok`, `health_check() = True`, and keyword searchability.

2. **Subprocess Crash Recovery (`TestCrashRecovery`)**:
   - Parent establishes baseline committed database state.
   - Isolated child subprocess connects in WAL mode, begins an `IMMEDIATE` transaction, inserts an uncommitted synthetic fact (`SYNTHETIC_UNCOMMITTED_CRASH_FACT_ID_987654321`), and signals `READY` via stdout.
   - Parent terminates only that child PID (simulating sudden crash / ungraceful termination).
   - Parent reconnects, verifies SQLite WAL recovery rolls back the uncommitted transaction: uncommitted row is absent (`count = 0`), baseline committed rows survive (`count = 1`), FTS5 index is clean, `PRAGMA integrity_check = ok`, and `health_check() = True`.

3. **WAL Checkpointing and Backup/Restore (`TestWALCheckpointAndBackup`)**:
   - `test_wal_checkpoint`: Populates WAL file with writes, executes `PRAGMA wal_checkpoint(TRUNCATE);`, verifies non-busy return (`busy = 0`), integrity check `ok`, and fact retrieval.
   - `test_sqlite_backup_and_restore`: Uses SQLite's live online backup API (`conn.backup()`) to create a valid backup database file, confirms backup health and integrity, then restores into a fresh database verifying full round-trip data fidelity.

4. **50,000 Fact Scale Benchmark (`TestScaleBenchmark`)**:
   - Decorated with `@unittest.skipUnless(os.getenv('RUN_SLOW_SCALE') == '1', ...)`.
   - Bulk-inserts 50,000 synthetic facts in batched transactions with active FTS5 triggers.
   - Validates 50,000 row counts across `facts` and `facts_fts`, integrity check `ok`, and measures keyword search latency across 20 distinct queries, enforcing p50 < 200ms and p95 < 500ms bounds.

5. **Retrieval Ollama-Down Fallback (`TestRetrievalOllamaFallback`)**:
   - `test_retrieval_pipeline_embedder_down_fallback`: Exercises `RetrievalPipeline` when an injected embedder fails / returns `None` (simulating Ollama service down). Confirms graceful fallback to keyword search with valid scores and confidence without unhandled crash.
   - `test_direct_ollama_retrieval_module`: Probes for a dedicated built-in `retrieval.ollama` adapter module. Because `retrieval` provides an embedder injection interface rather than a built-in Ollama client, this test marks an explicit pending skip (`"Explicit pending: ... (no fake)"`).

## Exact Root Test Commands

### 1. Fast Suite (Concurrency, Crash Recovery, WAL Checkpoint, Backup, Retrieval Fallback)

From project root (`C:\Users\shubh\ai-memory-system`):

**PowerShell / Windows:**
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s chaos/tests -v
```

**Bash / POSIX:**
```bash
python -m unittest discover -s chaos/tests -v
```

### 2. Full Suite (Including 50,000 Synthetic Fact Scale Benchmark)

From project root (`C:\Users\shubh\ai-memory-system`):

**PowerShell / Windows:**
```powershell
$env:RUN_SLOW_SCALE="1"; .\.venv\Scripts\python.exe -m unittest discover -s chaos/tests -v
```

**Bash / POSIX:**
```bash
RUN_SLOW_SCALE=1 python -m unittest discover -s chaos/tests -v
```

## Constraints Adherence
- Only `chaos/` was built; no edits outside `chaos/`.
- No test execution was initiated (per instruction: "Do not run tests yourself").
- Destructive operations strictly restricted to isolated test temporary directories.

## Orchestrator verification update

The project venv fast suite passed 7 tests with 2 intentional skips (the opt-in scale benchmark and the explicitly pending built-in Ollama adapter probe). The initial run revealed a test query typo (`concurrency` while fixtures contained `concurrent`) and leaked child streams; both test-only issues were corrected and the suite then passed. Evidence: `evidence/chaos-fast-tests.txt`. The 50,000-fact benchmark remains to be executed with `RUN_SLOW_SCALE=1`.

The opt-in 50,000-fact benchmark passed in the project venv: 50,000 facts inserted in 1.15 s (43,344 facts/s); FTS5 query latency p50 44.60 ms and p95 55.81 ms; integrity and row counts passed. Evidence: `evidence/chaos-scale-50k.txt`.
