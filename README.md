# AI Memory System

> **This is a sanitized public snapshot.** Personal data and credentials were removed: run logs and evidence (`evidence/`), the personal AI-fleet ops tooling (`ops/`, `work/`), status/handoff docs, and the golden Q/A set (it was generated from private notes). Docs that cite `evidence/...` files refer to artefacts that are not included. To run the eval/embedding benchmarks, generate your own `eval/golden_set.jsonl` from your own notes with `eval/generate_sets_v2.py` (set `MEM_DIR`).

Local-first memory prototype for Windows. SQLite WAL/FTS5 stores append-only facts; lifecycle metadata preserves superseded and retracted facts. The retrieval package combines FTS5 and dense cosine results with RRF. The MCP package exposes six tools over stdio.

## Current deployment status

The new MCP server is verified locally but is not registered with any CLI. The legacy service remains available. Round 2 measured better Recall@5 and MRR than the old service, but the heldout abstention precision is 0.2985 (gate: 0.90), live warm p95 is 3.195 seconds (gate: 400 ms), and sampled benchmark plus Ollama runner RAM is 10.23 GB (gate: 2 GB). Do not cut over. No client configuration was edited. The MCP recall handler currently uses FTS5; it is not yet wired to the hybrid retrieval or grounding packages. Supersede/retract fail closed until a validated local mutation judge is configured.

Commands and operational limits are in [RUNBOOK.md](RUNBOOK.md).
