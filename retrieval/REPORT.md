# Retrieval pipeline verification

`RetrievalPipeline` combines keyword and dense results with reciprocal rank fusion, optional gated reranking, and a bounded in-memory query cache.

## Verified behavior

- Imports `MemoryStorage` through the package API; no machine-specific path injection.
- Rejects blank or overlong queries, nonpositive/bool limits, and invalid constructor bounds.
- RRF ties sort deterministically by fact ID.
- Cache keys include query, limit, and filter identity; `invalidate_cache()` clears cached rows. Callers must invalidate after writes.
- Embedder exceptions, `None`, and invalid vectors emit a warning and fall back to keyword search.
- Reranking runs only below the configured confidence threshold and receives no more than the configured (maximum 30) candidates.
- Keyword-only result `score` is a reciprocal-rank proxy; `confidence` remains the stored fact confidence.

Round 2 verification: `python -m compileall -q retrieval` and `python -m unittest discover -s retrieval/tests -v`; 14 tests passed, including the shared-connection serialization regression. Evidence: `../evidence/r2-retrieval-tests.txt`, `../evidence/r2-retrieval-race-regression.txt`, and `../evidence/r2-retrieval-race-stress.txt`.

## Limitations

- Reranking depends on the caller-provided callback; no reranker model is configured here.
- Cache invalidation after writes is explicit because `MemoryStorage` has no write-notification hook.
- Keyword and vector reads run sequentially on the shared SQLite connection. Round 2 reproduced FTS parser failures when both reads ran concurrently on that connection (40 failures in 1,650 hybrid searches); the serial regression stress completed 1,650 searches with zero failures.
- This package does not enforce loopback itself; use the supplied loopback-only Ollama embedding client.

## Orchestrator end-to-end benchmark

Round 1's headline R@5 0.8667 used the flawed source-match predicate in `eval._is_relevant`; it is not comparable and is superseded. On the corrected strict predicate and same 165 questions / 145 passages, `embeddinggemma` with the documented `gemma` query/document templates scored dense R@5 0.4970 and MRR 0.2939. The new FTS+dense pipeline scored R@1 0.2667, R@5 0.5576, MRR 0.3685, warm p50 162.16 ms / p95 400.20 ms. The old loopback service scored R@5 0.4606 and MRR 0.2700 on the same golden set and strict answer rule. Quality gates pass; p95 is 0.20 ms above the 400 ms target. Unsupported-claim rate and abstention metrics are still pending the real Sonnet + independent-judge run. Evidence: `../evidence/r2-retrieval-e2e.json`, `../evidence/r2-retrieval-e2e.txt`, `../evidence/baseline-round2.txt`, and `../evidence/retrieval_e2e_round2.py`.
