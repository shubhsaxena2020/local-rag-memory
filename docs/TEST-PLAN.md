# TEST PLAN

Principle: nothing is "done" without a saved command and its real output under evidence/. Pass counts alone are not enough; read the output.

## 1. Automated suites (re-run before every phase close)
Packages: eval, judge, storage, lifecycle, retrieval, grounding, graph, embedding_bench, reranker, chaos, and mcp_server. Current saved functional baseline: 314 run, 312 pass, 2 optional chaos skips, 0 failures/errors (2026-10-20 23:25 IST; `evidence/r3-functional-run-20261020-2326.md`). The full Round 3 requirements and evidence status are tracked in `docs/TEST-MATRIX.md`; this baseline does not close the broader matrix.

## 2. Retrieval quality (golden set)
Use the current frozen split manifest and pre-registration for membership and counts; earlier Round 2 counts are historical only. One strict relevance rule for old and new: exact target passage, or substantive full-answer containment; no same-file hits, no short substrings. Metrics: recall@1/5, MRR, abstain precision/recall, latency p50/p95, peak RAM. Old FTS5 service is scored on a separate isolated benchmark tenant.

## 3. Hallucination measurement
Round 2's 40-probe setup is historical. For Round 3, use the current frozen dev/held-out manifest; run isolated conditions (a) without memory, (b) with retrieved facts and the grounding rule, and (c) on the current no-answer probes. Keep each condition in its own folder, use a judge from a different model family, and have Claude Code hand-read the preregistered sample. Current execution status is in `docs/TEST-MATRIX.md`.

## 4. Fact lifecycle
Supersede/retract keep the audit trail; superseded facts never come back as current; conflicting writes from two CLIs resolve deterministically; contradiction set of 60+ labelled pairs (negation, number/unit change, same entity different attribute, paraphrase = not a contradiction); judge accuracy reported.

## 5. MCP protocol
Real MCP client session over stdio: initialize, list tools, call each tool, bad input, oversized input, concurrent calls. Old tool names still work.

## 6. Concurrency, crash, scale
Several processes writing at once (WAL), kill -9 mid-write, corrupt-file recovery, Ollama down (keyword-only with warning), sqlite-vec unavailable (NumPy path), 50,000 facts, then a longer soak under the watchdog.

## 7. Cutover tests
Per CLI: config backup exists, CLI still starts, save and recall work through the real server. Cross-CLI: saved in A, recalled in B. Rollback script restores every file byte-for-byte (hash check).

## 8. Security review
Fresh reviewer: SQL injection, path traversal, input caps, resource caps, secrets never logged, no network egress except loopback Ollama.

## 9. Acceptance
All PRD gates pass on held-out data, no open blocker from the review, rollback proven.
