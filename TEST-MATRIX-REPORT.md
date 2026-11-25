# Functional Test Matrix Report

**Run:** 2026-10-20 03:04 IST
**Command:** `python verify_all.py`
**Result:** PASS
**Run evidence:** `evidence/1792445690-24316-5812/`

The root runner validates the watchdog and physical free memory, holds the shared owner-token lock across the batch, executes each suite serially, saves stdout/stderr, and fails on any nonzero exit or zero-test discovery. The final run was performed with watchdog RUN, temperature 64.9 C, and about 18.06 GiB free. User Ollama inventory included `qwen2.5:1.5b-instruct` on GPU; these are functional tests, not timing measurements.

| Package | Tests | Passed | Skipped | Failed | Status |
|---|---:|---:|---:|---:|---|
| eval | 14 | 14 | 0 | 0 | PASS |
| judge | 46 | 46 | 0 | 0 | PASS |
| storage | 26 | 26 | 0 | 0 | PASS |
| lifecycle | 5 | 5 | 0 | 0 | PASS |
| retrieval | 31 | 31 | 0 | 0 | PASS |
| grounding | 60 | 60 | 0 | 0 | PASS |
| graph | 6 | 6 | 0 | 0 | PASS |
| embedding_bench | 51 | 51 | 0 | 0 | PASS |
| reranker | 18 | 18 | 0 | 0 | PASS |
| chaos | 7 | 5 | 2 | 0 | PASS |
| mcp_server | 9 | 9 | 0 | 0 | PASS |
| **Total** | **273** | **271** | **2** | **0** | **PASS** |

The two chaos tests are explicitly skipped by their test definitions. The MCP tests emitted a keyword-search fallback warning because an embedding function was unavailable in that test environment; the suite still passed. This report covers functional package suites only. Still not run: full MCP protocol edge cases/fuzz, multi-process concurrency soak, 200/1000 crash drills, corruption/recovery, dependency-failure matrix, 1k-100k scale, migration, clock/time edge cases, full security/red-team suite, mutation testing, six-hour soak, and gated latency/RAM benchmarks.

Additional focused verification outside the root matrix: E7 harness 19/19; E10 harness 33/33 after Codex added a CLI fail-closed regression; E7 independent metric recomputation matched all six metric cohorts and their 10,000-bootstrap intervals exactly. E10 split verification yielded exactly 79 golden +20 probes. No sealed or heldout scoring was performed. No cutover is authorized by these results.
