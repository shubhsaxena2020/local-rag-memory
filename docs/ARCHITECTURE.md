# ARCHITECTURE

## Overview
```
 CLIs (Claude Code, Codex, OpenCode, Goose, Hermes, Antigravity, Kali Hermes)
        | MCP over stdio  (memory_save / memory_recall / memory_recent / memory_supersede / memory_health)
        v
 +----------------------- ai-memory MCP server (one Python process per CLI session) -----------------------+
 |  grounding layer: attach source + confidence + status; "unverified" flag when weak                       |
 |  retrieval: BM25 (FTS5) || dense (vector)  -> Reciprocal Rank Fusion -> gated cross-encoder rerank      |
 |  fact lifecycle: dedupe -> conflict check -> supersede/retract (audit log, never delete)                |
 |  graphify bridge: fact <-> graph node ids; unified query                                                |
 +--------------------------------------------------------------------------------------------------------+
        | in-process                                   | HTTP 127.0.0.1:11434
        v                                              v
 one SQLite file (WAL): facts, FTS5, vectors     Ollama: ONE embedding model + ONE small judge model
 (sqlite-vec, NumPy fallback)
```
Everything is local and in-process except the Ollama calls on loopback.

## Components and decisions (with evidence so far)
- Storage: single SQLite file, WAL mode, so several CLIs can read and write at once. 50,000 facts load in 1.15 s; FTS p50 45 ms, p95 56 ms (measured).
- Vector search: sqlite-vec is 19x faster than NumPy at 10,000 x 768 (46 ms vs 860 ms, same top-1 on 10/10 queries). Round 2 wires it into storage with automatic NumPy fallback.
- Embedding model: chosen by strict recall@5 then MRR, with latency and RAM as tie-breakers. Candidates: nomic-embed-text, qwen3-embedding:0.6b, embeddinggemma, bge-m3. Round 2 selector currently favours embeddinggemma; final choice is confirmed in FINAL-REPORT.md.
- Retrieval: BM25 and dense run in parallel, fused with RRF. A cross-encoder reranker is used only when first-pass confidence is low, and only if it measurably helps.
- Fact lifecycle: every fact has id, text, source, writer, created_at, valid_from/valid_to, status (active | superseded | retracted), superseded_by, confidence, tags, optional graphify node ids. Conflicts are found deterministically first (same entity key, near-duplicate by vector and FTS); a small local LLM judge is called only when a fact is being changed or removed. Rationale: a 2026 study found an LLM at mutation time recovered intent-aware deletion best (78-85%) at about 2.3 s per change, with recall paths unchanged.
- graphify: structure layer for code and documents (AST, no LLM for code). Facts store node ids. Quirk: `graphify update` writes manifest.json relative to the current directory, so always cd into the target first. Never mutate the vault graphs.
- Grounding rule for agents: answer from retrieved facts, cite fact ids, say "unverified" when nothing is found or confidence is low.
- MCP server: stdio, backward-compatible tool names, bounded inputs, no secrets in logs.

## Rejected options (and why)
Qdrant/Neo4j/Graphiti (server or heavy dependency, Kuzu deprecated), Cognee (demo-grade extractor, RAM unknown), Letta (full runtime), multi-tenant-rag as a service (dead SaaS; only retrieval patterns reused), Obsidian as a context source (not used by the user), any VPS or cloud store.

## Failure behaviour
Ollama down: keyword-only search with an explicit warning in the response. sqlite-vec fails to load: NumPy fallback. Database corruption: documented recovery from WAL plus nightly copy. Memory pressure: see OPS-SAFETY-AND-QUOTA.md.

## Known open items
Judge-model winner, reranker keep/drop, final embedder, strict re-measurement of end-to-end recall (round-1 end-to-end numbers used a lenient relevance rule and are invalid), grounded-answer metrics.
