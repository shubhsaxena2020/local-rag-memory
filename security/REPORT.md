# Independent review and disposition

The W12 worker’s final report was checked against the actual current code. Four of its five findings were false positives or had already been fixed; the remaining vector-scan observation is valid and documented below.

## Confirmed remaining limitation

- **Vector-search scaling** — [storage/storage.py](/C:/Users/shubh/ai-memory-system/storage/storage.py:400) scans vector rows and computes cosine distances in Python. On 10,000 synthetic 768-dimensional vectors, NumPy scan p50/p95 was 860.456/909.842 ms versus sqlite-vec 46.168/54.593 ms; top-1 agreement was 10/10. This is a scaling/performance risk, not a data-integrity defect. sqlite-vec is not yet integrated into the shared storage lifecycle. Measurements: `evidence/vector-backend-benchmark.txt`.

## Worker findings that were rejected

- `lifecycle/manager.py` `_get` calls `MemoryStorage.get_fact`, which uses `_ensure_connection`; the manager already owns an open connection, so the same connection is reused. No nested connection or inconsistent transaction was observed.
- The FTS fallback tokenizes only alphanumeric/underscore tokens, so its wrapped tokens cannot contain embedded quotes. A backtick query did expose an uncaught `sqlite3.DatabaseError`; the orchestrator fixed that syntax-error path and added a regression test. Other database errors are re-raised.
- MCP tag filtering uses exact membership in a comma-split list; it does not use substring matching.
- The lifecycle post-insert fact read supplies the complete new-fact payload to the optional conflict judge. It is an in-transaction lookup and may cost a small query, but no incorrect mutation was found.

## Fixes made during review

- Retrieval cache keys now retain hashable filter objects to prevent object-ID reuse collisions; unhashable filters bypass caching.
- Grounding rejects boolean confidence values instead of treating `True` as numeric confidence 1.0.
- MCP supersede/retract remain fail-closed because no validated local judge adapter is configured.
- Evaluation abstain precision now computes TP/(TP+FP), rather than classification accuracy; the updated distinction test passes. End-to-end precision remains null because no calibrated abstainer made predictions.

## Coverage and cutover limits

All package suites, the 50k chaos case, MCP client subprocess smoke, embedding-model benchmark, hybrid retrieval benchmark, and vector-backend benchmark were run and recorded in `evidence/`. The abstain set is hand-built and has no calibrated abstention callback; unsupported-claim rate has no independent answerer/judge callback. The read-only baseline adapter has no configured local credentials, so there is no measured baseline comparison. These gaps leave comparative quality gates unmet; no client configuration was changed.

This is a bounded static review, not a full adversarial audit of every dependency or deployment environment.
