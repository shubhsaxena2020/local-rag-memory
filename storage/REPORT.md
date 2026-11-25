# Storage package report

## Implementation

- One SQLite database, WAL mode, foreign keys, FTS5 keyword search, immutable public fact rows, and validated vector blobs.
- Vector search uses the optional sqlite-vec 0.1.9 cosine KNN index when it loads and is synchronized. `facts.vector` remains authoritative; the derived index is rebuilt/reconciled as needed. If extension loading, indexing, or query fails, storage falls back to the NumPy scan and then pure Python.
- No public `delete_fact` or `update_fact_text` methods exist. Lifecycle changes must update status metadata while retaining fact rows.
- Inputs are bounded for fact text, source/writer/tags, query length, result limits, vector dimension, and finite confidence/vector components.

## Verification

From the project root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q storage
.\.venv\Scripts\python.exe -m unittest discover -s storage/tests -v
```

Round 2 orchestrator result: compile succeeded; 25 tests passed, including seven vec-index/fallback tests. Evidence: `../evidence/r2-storage-vec-tests.txt`.

## Limitations

- First indexed search on an existing database reconciles the full ID set; any unindexable vector forces the scan to preserve semantics.
- No automated migration from the two baseline records is implemented here; cutover migration remains a separate step.
- Round 2 concurrency, crash recovery, WAL checkpoint/backup, and the 50k-fact scale case passed in the current chaos run; see `../evidence/r2-final-chaos-tests.txt` and `../evidence/r2-final-chaos-scale-50k-tests.txt`.

## Orchestrator vector-backend benchmark

The earlier direct synthetic probe (10,000 x 768 vectors, 10 queries) measured NumPy p50 860.456 ms / p95 909.842 ms and sqlite-vec p50 46.168 ms / p95 54.593 ms, with top-1 agreement 10/10. Round 2 now wires the indexed path into storage and tests it against forced fallback on 60 vectors. The dedicated 10,000-row probe compared raw backends; it is not a production workload result. Evidence: `../evidence/vector-backend-benchmark.txt` and `../evidence/r2-storage-vec-tests.txt`.

The end-to-end retrieval run exercised a backtick-containing query that raised `sqlite3.DatabaseError` rather than `OperationalError`; the prior FTS5 fallback missed it. The orchestrator catches recognized FTS syntax errors, re-tokenizes safely, and re-raises other database failures. Regression test is included in the passing Round 2 storage suite.
