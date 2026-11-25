# Round 3 Test Matrix

**Status:** In progress; fresh functional-suite snapshot 2026-10-20 23:26 IST (314 run, 312 pass, 2 optional chaos skips), with E16's focused cache/retrieval/MCP verification recorded in `evidence/r3-e16-integration-review1-codex-review-20261020-2325.md`. E8 correction4's isolated synthetic suite passed all 96 tests; the E8 experiment remains a Gap because grammar/policy registration, approved dev-only input, production index/integration, quality, and performance evidence are absent. E9 correction2 has been rejected for scoring/execution after a source-only independent review; no E9 test or data was run. S5's independent review preserves Partial status and corrects the stale BOM-probe note. The O2/O3 source-only second pass is complete and leaves R9 Partial. Other row statuses retain their individually cited evidence dates. This is the authoritative distinction between the functional package-suite run and the full Round 3 acceptance matrix. A green unit suite does not close any row marked Partial or Gap.

## Verified functional baseline

The prior 15:06 IST run remains historical. A fresh project-venv `verify_all.py` run at 23:25 IST passed all 11 functional package suites: 314 run, 312 passed, 2 optional chaos skips, 0 failures/errors. Codex independently recounted all 11 saved suite outputs and recomputed their SHA-256 values. Evidence index: `evidence/r3-functional-run-20261020-2326.md`; raw outputs: `evidence/1792518952-36072-4139/`; full command capture: `evidence/r3-verifyall-functional-20261020-232552.txt`. This closes only the functional package-suite check; all broader Round 3 matrix and release gates remain open.

**One-command matrix gap:** current `verify_all.py` runs the 11 functional package suites listed below; it does not automate the full 14-item Round 3 matrix. The latest draft is rejected: tracked-file cleanliness does not pin untracked row scripts, and the candidate runs row/baseline processes with repository-root access without enforcing sandbox confinement. Correction1 is a design/blocker document only; it records that no verified host boundary exists and requires an operator/admin decision. It remains unrun and is not an accepted runner; do not describe the full matrix as runnable through one command yet.

| Suite | Run | Pass | Skip | Evidence |
|---|---:|---:|---:|---|
| eval | 14 | 14 | 0 | `evidence/1792518952-36072-4139/eval_output.txt` |
| judge | 46 | 46 | 0 | `evidence/1792518952-36072-4139/judge_output.txt` |
| storage | 26 | 26 | 0 | `evidence/1792518952-36072-4139/storage_output.txt` |
| lifecycle | 5 | 5 | 0 | `evidence/1792518952-36072-4139/lifecycle_output.txt` |
| retrieval | 63 | 63 | 0 | `evidence/1792518952-36072-4139/retrieval_output.txt` |
| grounding | 60 | 60 | 0 | `evidence/1792518952-36072-4139/grounding_output.txt` |
| graph | 6 | 6 | 0 | `evidence/1792518952-36072-4139/graph_output.txt` |
| embedding_bench | 51 | 51 | 0 | `evidence/1792518952-36072-4139/embedding_bench_output.txt` |
| reranker | 18 | 18 | 0 | `evidence/1792518952-36072-4139/reranker_output.txt` |
| chaos | 7 | 5 | 2 | `evidence/1792518952-36072-4139/chaos_output.txt` |
| mcp_server | 18 | 18 | 0 | `evidence/1792518952-36072-4139/mcp_server_output.txt` |
| **Total** | **314** | **312** | **2** | `evidence/r3-functional-run-20261020-2326.md` |

Independent previous review of the functional runner and its earlier saved run: `evidence/r3-verifyall-20261020-0628-codex-review.md`. The current saved run directory above was separately recounted. No evaluation or held-out rows were read by that recount.

## Plan section 5.2 requirements

Status definitions: **Pass** = the stated scope is evidenced; **Partial** = narrower unit/source coverage exists but the planned stress/protocol scope is not met; **Gap** = no qualifying execution evidence.

| # | Requirement | Status | Evidence and remaining work |
|---:|---|---|---|
| 1 | Unit and Hypothesis/property tests for lifecycle, RRF, escaping, dimensions | Partial | Functional suites pass. No comprehensive Hypothesis/property campaign is evidenced; add properties and preserve reproducible output. |
| 2 | MCP argument fuzzing, malformed/oversized inputs and no crash/hang | Gap | No automated fuzz corpus/run recorded. |
| 3 | Real MCP client over stdio: initialize, tools, cancellation, malformed/partial frames, concurrency, restart, legacy names | Partial | The 2026-10-20 23:25 functional snapshot passes 18 `mcp_server` tests (`evidence/r3-functional-run-20261020-2326.md`). Full real-client conformance sequence and restart/cancellation cases are not evidenced. |
| 4 | 10-process, 5-minute mixed concurrency including slow Ollama | Partial | The chaos package has 5 passing tests and 2 optional skips; this does not establish the required duration/process count or slow-Ollama run. |
| 5 | 200 randomized writer crashes, integrity/WAL checks and power-loss simulation | Partial | Unit crash-recovery coverage exists in the passing package suite; no 200-run randomized or power-loss campaign is evidenced. |
| 6 | Truncated DB/WAL, missing extension/schema mismatch and backup recovery | Partial | Backup/WAL unit coverage exists; the required corruption matrix and recovery report remain unverified. |
| 7 | Ollama down/hung/invalid vectors, missing model, sqlite-vec absent, disk full/read-only DB | Partial | Some fallback unit coverage passes; the complete failure injection table is not evidenced. |
| 8 | 1k/10k/50k/100k scale curves, RAM and DB size, 500k extrapolation | Gap | Two optional chaos cases were skipped. No current scale curve is accepted. |
| 9 | Schema up/down, two-record bit-exact migration and idempotence | Gap | No current migration run is evidenced. |
| 10 | Same timestamps, backwards clock, DST/timezone and far-future dates | Gap | No current time-behavior matrix is evidenced. |
| 11 | Emoji, RTL, Devanagari, mixed scripts, combining marks and long tokens | Partial | Some focused Unicode tests/probes exist, but the full planned matrix and saved run are not established. |
| 12 | R9 security cases: injection, poisoned facts, provenance, secret refusal, scope and logs | Partial | S2 remediation/rerun remains open. A non-executing synthetic Appendix C test plan is being drafted; no attack has been run and no security gate is closed. |
| 13 | Regression gate against accepted metrics and confidence intervals | Gap | No automated gate comparing all frozen metrics against an accepted baseline is evidenced. |
| 14 | Mutation testing for storage/lifecycle/retrieval/abstention | Gap | No mutation run or surviving-mutant dispositions are recorded. |

## Other required verification work

| Workstream | Status | Current evidence / next gate |
|---|---|---|
| S1 CLI integration inventory | Partial | `research/INTEGRATION.md` exists; verify every CLI path/startup and grounding placement before cutover. |
| S2 static analysis/security/dependencies | Partial | PR #24 now has a lifecycle reconnect fix and save-after-reconnect regression; its MCP test module passes 10/10. Full scan scope, findings disposition, dependency/license check, and broader suite remain open. |
| S3 golden-set label audit | Partial | Dev-only corrections are recorded. Full label-noise rate is not established; never use held-out rows for tuning. |
| S4 profiling/FD/SQLite query plans | Gap | Correction3 is accepted as an isolated source candidate only: it checks HEAT_HOLD/HOLD_LAUNCHES before work, preserves watchdog/temp/CPU/RAM gates, and uses the shared owner-token lock. Synthetic tests and profiler are unrun; Windows lock behavior and a non-atomic owner-check/unlink window remain. No profile or measurement exists. Evidence: `evidence/r3-s4-profile-guard-correction3-codex-review-20261020-1126.md`. |
| S5 Windows paths/locking/sleep/clock | Partial | Independent review confirms saved synthetic checks for spaces, Unicode, a 456-character raw-sqlite path, and four-thread/single-process WAL writes. The `LongPathsEnabled=1` assertion lacks a registry artifact; the extended-prefix check reuses a short path; long-path `MemoryStorage` and cross-process locks are untested. Current storage.py/test_storage.py hashes differ from the harness pins, though their mtimes postdate the recorded run and historical equality is not independently confirmed. The 13:01 isolated BOM parser demo exists (UTF-8 rejects; utf-8-sig accepts); this is not a loader audit, so earlier “pending” wording was stale. UNC, Defender, sleep/resume, clock changes, other interpreters/builds, and harness reproducibility remain unverified. Evidence: `evidence/r3-s5-windows-independent-review1-codex-review-20261020-1617.md`. |
| S6 soak resource scan | Gap | No accepted low-intensity or burst soak series. Correction2 is not accepted: stale-lock/path races and Windows assumptions remain. Correction3 failed before a deliverable because Claude returned HTTP 429; the immediate probe was limited and correction3-retry2 produced source/test candidates but no report/hash manifest; Codex rejected it as incomplete and no further correction will be dispatched under the two-correction cap. No collector has been parsed or run; lock, path, and PowerShell/.NET behavior remain unverified. S6 remains a Gap. |
| Latency/RAM baseline and final measurement | Gap | Must be one-at-a-time benchmark windows with current backend/model/worker metadata and all applicable safety gates. |
| Hallucination study and independent scorer | Gap | No complete isolated conditions, judge agreement, CI report, or required hand-readings. |
| Red team and independent adversarial code review | Partial | Independent Codex and Sonnet source-only reviews confirm O2/O3 and enumerate 12 Appendix C/R9 findings, including untrusted fact formatting, source-sibling confidence, provenance, lifecycle policy, filtering, FTS semantics, resource caps, prompt delimiting, path/error handling, and time/cache/cleanup concerns. The O2/O3 candidate?s 13 synthetic regressions passed, but the second-pass review found total-prompt size/escaping, fail-open sibling confidence, and filter-after-200 gaps. Default patch check fails on CRLF bases (ignore-whitespace check passes). The candidate is unmerged; no exploit/fuzz campaign or runtime validation has run. R9 remains Partial. Evidence: evidence/r3-o2-o3-second-pass1-codex-review-20261020-1533.md. |
| Rollback/cutover and cross-CLI smoke | Gap | No cutover. Keep production configs and old service unchanged until every release gate passes. |
| Held-out and sealed final scoring | Blocked on approval | Held-out final scoring waits for a frozen pipeline; sealed scoring requires Shubh's approval and is exactly once. No sealed content was accessed in this matrix audit. |

## Quality experiment status relevant to release

E5 has only a partial development diagnostic with no statistically supported winner; the run is quarantined for selection because the E5 loader deserializes the full mixed golden JSONL before filtering to dev. E7 correction4 completed its paired dev quality comparison with the staged ONNX model through DirectML with CPU fallback and canonical PLAN-24H Appendix A relevance. The measured deltas did not demonstrate the preregistered positive quality lift, so reranker adoption is rejected; this is not a significant-degradation claim. The fresh 11:52 functional rerun also passed all 18 reranker unit tests, including DirectML preference and CPU-only fallback coverage. No latency/RAM benchmark was run; the timing gates remain open. E10's threshold-only dev diagnostic was infeasible; the standalone JSON transport adapter passed 71 offline tests and its self-check, but no valid prompt A/B score exists. Scorer correction4 remains an isolated source candidate: its pinned ordered manifest binds each ID to its answerable/probe label and synthetic swap-refusal coverage is authored; tests are unrun and producer/approval ownership remain unresolved. The integrated live-dev4-retry2 runner was independently rejected: its sequential mixed-file reader cannot guarantee held-out bytes stay unread; it deserializes the full split manifest, does not verify dev/split manifest pins, rebuilds contexts from the whole unpinned corpus, and uses check-then-write output creation. Its report discloses prohibited AST/symtable parsing and recommends forbidden whole-file hashes. No runner smoke or score is accepted, and the four correction attempts are exhausted. Evidence: evidence/r3-e10-live-dev4-retry2-codex-review-20261020-1430.md. Future scoring requires a separately approved dev-only input and an approved runner. E2 retry3 was rejected by Codex review for unverified dev/split pins, sequential mixed-file reads that cannot guarantee held-out bytes remain unread, unconstrained source paths, check-then-write outputs and missing runtime guards. Final source-only correction4 was rejected at 18:14 for missing README/REPORT outputs and failure to enforce the registered relevance rule; no E2 metric is accepted. See `evidence/r3-e2-harness4-final-review1-codex-review-20261020-1814.md`. See evidence/r3-e2-harness3-codex-review-20261020-1409.md and evidence/r3-e2-harness4-launch-verified-20261020-1411.md. E1 correction2 is rejected for strict-relevance metric nonconformance; correction3 adds safer input handling but still diverges from canonical `grounding.relevance.answer_in_text()` semantics (NFKC/entity/number/boundary rules). Correction4 is active, synthetic-only, and tests remain unrun. No approved dev artifact or E1 result exists. E3 and E4 are blocked/exhausted after four correction rounds and have no valid candidate pools. E4 correction4 is independently rejected: the submitted source still deserializes mixed golden/probe rows before filtering, contains a visibly misplaced lock-release block, does not implement the reported candidate-pool-only path, and writes output without overwrite protection; see `evidence/r3-e4-correction4-codex-review-20261020-1124.md`. The four-round ceiling is reached, with no E4 score or gate pass. E6 has only a feasibility scan and no benchmark. E8 correction2 remains an isolated proposal and is rejected for integration: its synthetic suite ran 64 tests with 16 failures (48 passed); three entity-key cases contradict the candidate grammar, punctuation expectations conflict, and 12 pipeline tests fail because the synthetic storage double does not implement the SQL connection contract used by the candidate. A synthetic smoke reproduced a false whole-token match from `café-123abc`; fact-token extraction lacks whole-token boundaries and builds its map with a full-table scan rather than the registered trigram/prefix lookup. Grammar, false-exact policy, approved dev subset, and a bounded indexed lookup remain prerequisites. Correction3 is active as source/synthetic-test authoring only and its tests remain unrun. No E8 metric or gate pass is accepted. Evidence: evidence/r3-e8-identifier-correction3-dispatch-20261020-1523.md; evidence/r3-e8-correction2-codex-review-20261020-1517.md, evidence/r3-e8-correction2-unittest-20261020-1517.txt, and evidence/r3-e8-token-boundary-smoke-20261020-1522.txt. E9 correction1 now explicitly labels its implemented control as keyword-only fallback and blocks production claims; unresolved preregistration, a real control, and an approved dev-only input remain, with tests unrun and no result. E11-E18 are pending or stretch work. The authoritative arm-level ledger is `EXPERIMENTS.md`.

## Release decision

**No-go for cutover.** The functional package baseline passes, but the required fuzz/protocol/stress/scale/migration/security/mutation/latency/soak, quality, held-out, independent-review, rollback and approval gates are incomplete. `FINAL-REPORT.md` and `ops/PROJECT_DONE` remain pending until the full plan is finished and the final gate decision is evidence-based.

## Reproduction

The saved-output count audit is `python evidence/r3-test-matrix-recompute.py`. A fresh functional rerun must use the policy-aware `verify_all.py` safety checks, acquire the common local-job lock, and wait for the plan's watchdog/temperature/free-memory gate; do not infer permission to run from this report.




## 2026-10-20 11:03 Codex review checkpoint

S2 reconnect review2 and scan-triage2 both identify the stale `FactLifecycleManager` connection after `recent()` reconnect; PR #24 branch commit `709c8cd` now resolves the connection lazily and tests a second save after reconnect. The MCP test module passes 10/10; PR #24 remains draft while full scan coverage and broader gates remain open. Scan-triage2's report hash and nine input hashes match, but it is not accepted as prompt-compliant because it discloses AST parsing of source despite the no-parse instruction. The E5 safe loader remains unexecuted with no approved artifact; its caller-supplied approval/hash and path replacement checks need an authoritative trust boundary. S3 audit1 disclosed reading mixed held-out count summaries despite its dev-only scope; those claims are quarantined and do not establish a dev noise rate. Evidence: evidence/r3-s2-reconnect-review2-codex-review-20261020-1102.md, evidence/r3-s2-scan-triage2-codex-review-20261020-1625.md, evidence/r3-s2-pr24-reconnect-fix-codex-review-20261020-1628.md, evidence/r3-e5-loader-correction1-codex-review-20261020-1102.md, evidence/r3-s3-dev-repair-audit1-codex-review-20261020-1102.md.


## 2026-10-20 11:52 Codex functional rerun

Fresh policy-aware run of all 11 functional package suites passed: 273 run, 271 passed, 2 optional chaos skips, 0 failures/errors. Codex independently recounted every suite output and verified each file hash. Evidence: evidence/r3-functional-run-20261020-1152.md; raw outputs: work/r3-testmatrix-verifyall1/evidence/1792477323-9996-8962/. This closes only the functional suite; all broader matrix gaps and release gates remain open.


## 2026-10-20 11:54 E7 DirectML status correction

The DirectML runtime/provider integration was already present. Independently checked E7 correction4's report, source/result hashes, and saved metric recomputation: 99 dev queries, 198 paired sanitized rows; no positive reranker lift, so retain the baseline. Fresh 11:52 reranker functional tests passed 18/18. Evidence: evidence/r3-e7-quality-dev-results-v2.json, evidence/r3-e7-independent-recompute.txt, and evidence/r3-functional-run-20261020-1152.md. The CPU<35 latency/RAM gate remains unmeasured.


## 2026-10-20 14:52 Documentary coverage crosswalk review

Codex reviewed the isolated documentary crosswalk against the current plan, test plan, matrix, and verification log. The 37 plan units map to 24 distinct acceptance-table rows; the remaining table row is the D8-only Latency/RAM gate. A0 is matrix prose, and C8/C9 have no matrix row. Status recount remains 20 Gap, 14 Partial, 1 Blocked on approval, 2 with no row, and 0 Pass. The worker crosswalk's “30 of 25 rows” sentence and byte-identical-row claim are incorrect; the matrix changed after its pinned snapshot, including the red-team row, though the statuses remain as counted. Evidence: evidence/r3-test-matrix-crosswalk3-codex-review-20261020-1452.md. No gate status changed.


## 2026-10-20 15:04 O2/O3 candidate2 synthetic regressions

The isolated O2/O3 correction2 candidate's test file passed 13/13 synthetic tests using the preinstalled system Python 3.13.14/pytest 9.1.1; plugin autoload was disabled because the project venv lacks pytest. Tests cover fact-line escaping, strict total output cap, source-sibling marking/bounds, and MCP confidence mapping with in-memory storage. This does not exercise the full search path, live attacks, or the shared production code; O2/O3/R9 remain Partial. Evidence: evidence/r3-r9-o2-o3-fix2-codex-review-20261020-1504.md.


## 2026-10-20 15:06 Fresh functional package suite

Fresh policy-aware `verify_all.py` run passed 273 tests, 271 passed, 2 optional chaos skips, 0 failures/errors. Codex independently checked every suite's exit, unittest counts, and output hash. The runner acquired/released its owner-token lock and rechecked watchdog/RAM before each package. Only functional suites ran; no matrix stress, quality, latency, security, soak, held-out, or sealed gate is closed. Evidence: evidence/r3-functional-run-20261020-1506.md and raw outputs under evidence/1792488981-48500-4935/.



## R3-RAM-1
Correction1 is rejected for measurement/adoption after independent final-source review. Blocking defects remain in process-snapshot failure handling, the pure Ollama parser, measured-window sample validity, authored tests, and unverified live PDH temperature. No RAM metric or PRD gate credit exists; do not execute this candidate. Evidence: `evidence/r3-r3-ram-harness1-correction1-codex-review-20261021-0703.md`.


## 2026-10-21 07:05 IST - R3-RAM correction2
The single remaining correction round is active in isolated folder `work/r3-r3-ram-harness1-correction2` under issue #27. The protected launcher started one full-profile MiniMax session (PID27340); review waits for root exit and five minutes of `run.log` idleness. No tests or measurements have run. Dispatch guards/evidence: `evidence/r3-r3-ram-harness1-correction2-dispatch-20261021-0705.md`.
