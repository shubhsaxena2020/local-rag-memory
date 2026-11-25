# PRD - Local AI Memory System

Owner: Shubh. Status: draft v1, written 2026-10-19. Companion docs: ARCHITECTURE.md, ROADMAP-24H.md, TEST-PLAN.md.

## 1. Problem
Shubh runs many AI CLIs on one Windows laptop (Claude Code, Codex, OpenCode, Goose, Hermes, Antigravity, Kali-WSL Hermes). Each keeps its own memory, or none. Facts decided in one tool are unknown to the others, old facts are never retired, and the AI answers from guesswork. The previous shared memory ran on a VPS that is now deleted; the local fallback is keyword-only and holds 2 records.

## 2. Goal
One local, automatic, fast memory shared by every CLI so answers are grounded in real, sourced, current facts, and the AI says "unverified" instead of inventing. Honest framing: 0% hallucination is not achievable by any system. The measurable targets are in section 6.

## 3. Users
Shubh (single user) and the AI agents acting for them. No multi-tenant, no sharing, no UI required.

## 4. Hard constraints
- 100% local. No VPS, no cloud service in the loop. Everything on this laptop (Core Ultra 5 225H, 48 GB RAM, Arc iGPU, Windows 11).
- No Obsidian in the loop (vault may be read as test text only). No multi-tenant-rag system (code patterns may be borrowed).
- Uses Ollama and graphify. Smallest sufficient Ollama models; delete all others at the end.
- Writes are automatic (no approval step). Nothing is deleted from memory; changed facts are superseded.
- Must not overheat or exhaust RAM while unattended (see OPS-SAFETY-AND-QUOTA.md).

## 5. Functional requirements
FR1 Save a fact with source, writer, time. FR2 Hybrid recall (keyword + vector, fused) returning facts with source, confidence and status. FR3 Supersede/retract with full audit trail; recall never returns a superseded fact as current. FR4 Abstain signal when retrieval is weak ("unverified"). FR5 One MCP stdio server, backward compatible with tool names memory_save / memory_recall / memory_recent. FR6 graphify bridge: facts link to graph nodes; one query can use both. FR7 Registration in every CLI with tested rollback. FR8 Health tool and local log.

## 6. Success metrics (measured on a golden set built from real data; same strict scoring for old and new)
| Metric | Old baseline (measured) | Target |
|---|---|---|
| Recall@5 | 0.46 | clearly higher than baseline |
| MRR | 0.27 | clearly higher than baseline |
| Unsupported-claim rate with memory vs without | not measured | lower with memory |
| Abstain precision on no-answer probes (held-out) | not measured | >= 0.90 |
| Warm p95 retrieval latency | 28 ms (keyword only) | < 400 ms |
| Whole-system RAM, excluding the answerer | ~0 (FTS5 only) | < 2 GB |
| Test suite | - | all green incl. concurrency, crash, 50k-fact scale |

## 7. Non-goals
Zero hallucination guarantees, cloud sync, multi-user access control, a note-taking UI, replacing Claude Code's own flat memory files.

## 8. Release gates
Cut over only if every metric target above passes on held-out data, all tests pass, rollback is proven, and an independent adversarial review has no open blockers.
