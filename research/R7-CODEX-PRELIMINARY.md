# R7 — Temporal facts and forgetting: preliminary source synthesis

Status: Codex orchestrator synthesis during watchdog pause. This is not the Hermes R4/R7 package and must be compared with that worker's report before G1. No local tests or data runs were performed.

## Scope and method

- Focus: updating, superseding, deleting, and retrieving facts that change over time.
- Primary sources opened: Zep paper, Graphiti repository, LongMemEval paper, LifelongAgentBench paper, MemBench (ACL 2025), and MemoryBench (ICML 2026 proceedings).
- Vendor performance claims are not adopted as measurements for this project.
- Main design choice: keep event history/provenance while making the current-state query deterministic and explicit.

## Evidence from benchmarks and systems

- LongMemEval defines five long-term memory abilities: information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention. It provides 500 questions and scalable histories, including standard S and M contexts (paper reports about 115k and 1.5M tokens). This is a benchmark coverage statement, not a guarantee that its labels or evaluator fit this project's corpus.
- LongMemEval's authors test updates and temporal questions explicitly. A project-specific suite should preserve earlier facts and ask both “what is current?” and “what was true at time t?” so a single current-state metric cannot hide historical corruption.
- LifelongAgentBench examines sequential, interdependent tasks across database, OS, and knowledge-graph environments. Its authors identify retention, transfer, and prior experience reuse as important dimensions. It is broader than personal fact memory and not a drop-in score for this system.
- ACL 2025 MemBench evaluates factual and reflective memory across participation and observation settings and includes knowledge updating and temporal reasoning. Its setup underscores that observing user messages differs from interactive dialogue in which the assistant also contributes messages.
- ICML 2026 MemoryBench focuses on learning from accumulated user feedback in service time across domains, languages, and tasks. This supports adding update sequences and explicit user corrections to evaluation, beyond static QA.
- Zep's paper describes a temporal knowledge graph with episodes, semantic entities, and time-scoped relationships. The paper reports benchmark results, but those are system-specific and not accepted as our expected performance.
- Graphiti's official repository describes facts with validity windows and lineage back to source episodes. Treat this as an architectural reference; it does not prove that our SQLite implementation needs a graph database.
- The inspected sources consistently separate a current fact from the evidence/history from which it was derived. This separation is useful even if the implementation remains relational.

## Recommended fact record and semantics

For each assertion, retain: fact ID; subject/predicate/object or normalized value; scope/tenant; valid_from and valid_to (when known); recorded_at; source episode/document ID; source span; author/source type; confidence/review status; supersedes/superseded_by links; deletion/tombstone status.

Use valid time for when the assertion is true in the world or user's stated timeline. Use transaction/recorded time for when this system learned or recorded it. Do not silently substitute ingestion time for the event time.

Represent unknown temporal bounds as unknown. Do not synthesize an exact date from “recently,” “last summer,” or a relative phrase without the conversation timestamp and timezone needed to resolve it.

A “current” retrieval filters to active facts and clearly labels unknown or conflicting validity. A historical question may retrieve superseded facts, but it must return their time bounds and provenance.

A new assertion that conflicts with an old fact creates a candidate supersession edge; it does not physically overwrite the old record. Source evidence and validity interval remain available for audit and as-of queries.

## Supersession and conflict policy

1. A direct, explicit user correction scoped to the same subject and attribute takes precedence over the user's earlier assertion for the current state.
2. Preserve the prior value as historical with an end boundary only when the correction indicates a real-world change or correction of the value. Ask for clarification via the product when it is ambiguous; until then mark conflict instead of inventing a timeline.
3. Different subjects, projects, tenants, or scopes must never supersede each other just because text is similar.
4. If two active assertions conflict and neither has explicit correction semantics, surface the conflict. Do not use “last writer wins” as universal authority.
5. Within an explicitly append-only source such as a dated log, order by source event time when well-formed; retain source and recorded time to detect ingestion reordering.
6. Near duplicates are candidates for review or linked provenance, not automatic merges. A duplicate merge must retain all source spans and support complete rollback.
7. Corrections to one attribute must not erase neighboring attributes. Update at the smallest fact granularity that the evidence supports.
8. A deletion request has separate intent from a fact update. Honor an explicit forget/delete action across current facts, indexes, caches, graph projections, exports, and backups according to a documented retention policy; avoid reintroducing the value from stale derived state.
9. A historical correction and a privacy deletion are different operations: the former can preserve auditable history when authorized; the latter may require erasure/tombstoning of content and all searchable copies. Do not equate an old fact being superseded with permission to forget it.
10. Failed embedding or index updates must remain visible as an operational warning and must not imply that an unindexed fact is retrievable.

These are recommendations for a user-owned, local memory application. They are not claims that a specific privacy law mandates a particular retention implementation.

## Required temporal and forgetting tests

- Same attribute changes twice: old and intermediate values are returned for their respective as-of times; current query returns only the latest supported value.
- Explicit correction (“I meant X, not Y”): correction is current; the original is marked corrected, not treated as an independent simultaneous value.
- Real-world change (“I moved from A to B in June”): preserve A before June, B on/after June; ask/flag if dates conflict.
- Conflicting unqualified statements from two sources: both remain visible as unresolved; current query does not choose based solely on write order.
- Late-arriving event: insert an older valid-time event after a newer one; current state is derived by valid time and precedence, not insertion order.
- Partial update: change one field of a multi-field profile; other supported facts remain unchanged.
- Timezone and daylight-saving boundary: do not cross a date boundary incorrectly; preserve source timezone or mark time unknown.
- Relative dates with no resolvable anchor: mark time unknown and test that no invented date is returned.
- Repeated identical ingest: idempotent result; no duplicate active rows or changed provenance.
- Near-duplicate but contradictory text: do not merge or discard without conflict evidence.
- Delete / forget: assert the fact is unavailable in current and historical search, FTS, vector index, cache, graph projection, backup/export as policy specifies; test after restart and rebuild.
- Delete followed by stale backup restore: deletion tombstone wins or restore is rejected; deleted content does not reappear.
- Cross-scope isolation: an update or delete in tenant/project A has no effect on tenant/project B.
- Invalid temporal intervals: reject or mark invalid where valid_to precedes valid_from; preserve diagnostic provenance.
- Transaction rollback: a mid-update fault leaves either the old consistent state or the complete new state, never a half-superseded record.
- Clock skew: recorded_at remains monotonic or is separately sourced; ingestion timestamps do not reorder valid-time truth.
- Search index rebuild: current/historical outputs match database semantics before and after full rebuild.
- Conflict query: both candidate facts are returned with concise source/time evidence; no unsupported answer is generated.
- Stale-fact leak regression: current-state queries must have zero responses containing only a superseded value unless explicitly asked for history.

## ForgetEval-style project slice

Create a private, synthetic 40-sequence suite, separate from golden/held-out data. Each sequence contains 5–12 fact events with explicit source IDs and a timestamped truth table. Include 10 update sequences, 6 explicit corrections, 6 unresolved conflicts, 6 deletion requests, 4 late-arriving facts, 4 near-duplicates, and 4 cross-scope pairs. Include current, as-of, conflict, and deleted-value probes for each applicable sequence.

Have an independent author provide inputs and gold event semantics. The scorer should calculate current-value accuracy, as-of accuracy, stale-fact leakage, conflict preservation/false resolution, deletion persistence, cross-scope violations, index/database consistency, and update transaction atomicity.

Hash and freeze this slice before any system run. Do not reuse the SEALED author or inspect the sealed set during development. If including this suite as a scored experiment, preregister it in EXPERIMENTS.md first and keep its files separate from all tuned dev artifacts.

## Adoption decision

- ADOPT: preserve source provenance and distinguish event-valid time from system-recorded time in the data model, if schema migration and compatibility tests prove safe.
- ADOPT: model corrections, real-world updates, conflicts, and explicit deletion as separate operations.
- TEST: temporal filtering and supersession behavior on an independent synthetic slice before any migration or CLI integration.
- TEST: duplicate candidate thresholds; a match score must never silently authorize merge/deletion.
- REJECT: global last-write-wins; embedding similarity as a temporal/conflict authority; physically overwriting old facts on ordinary updates; “delete” implemented only as current-state filtering while content remains searchable elsewhere.

## Ready experiment spec (not run)

Hypothesis: explicit temporal intervals, supersession provenance, and separate deletion intent will reduce stale-fact leakage to zero and improve current/as-of accuracy against the present behavior without creating cross-scope deletions.

Change: add a disposable dev database with explicit interval/provenance fields and guarded update/delete transactions. Baseline is the current implementation at the exact same commit and fixture events.

Metrics: current-value accuracy, as-of accuracy, stale leakage (must be 0), conflict retention, deletion persistence, duplicate rate, cross-scope violations (must be 0), crash consistency, and query/write overhead.

Split: synthetic dev sequences for debugging; a separately authored, hashed temporal test slice for one final evaluation. No sealed questions or labels are used here.

Cost: API-only for schema review; local tests only after watchdog RUN and free RAM >12 GB. Use disposable copies only. No old service or personal vault is modified.

Decision rule: retain the change only if all integrity/privacy invariants pass, current and historical accuracy improve or remain statistically non-inferior under paired bootstrap, and no material latency/RAM gate regresses. Any stale leak or cross-scope effect is an automatic reject.

## Sources

- Wu et al., LongMemEval, arXiv: https://arxiv.org/abs/2410.10813
- Zheng et al., LifelongAgentBench, arXiv: https://arxiv.org/abs/2505.11942
- Tan et al., MemBench, ACL Anthology: https://aclanthology.org/2025.findings-acl.989/
- Ai et al., MemoryBench, PMLR ICML 2026: https://proceedings.mlr.press/v306/ai26e.html
- Rasmussen et al., Zep: A Temporal Knowledge Graph Architecture for Agent Memory, arXiv: https://arxiv.org/abs/2501.13956
- Graphiti official repository: https://github.com/getzep/graphiti

## Verification status

This 100+ line synthesis cites opened primary sources. It is not the worker output. All paper-reported scores remain paper-specific, with no claim that they are reproduced. It has no benchmark result and does not satisfy R7 acceptance until compared with the paused Hermes R4/R7 worker report. No sealed content, local tests, or model calls were used.
