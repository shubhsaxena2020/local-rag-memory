# Research synthesis - Round 3 restart (G1 draft; gate not passed)

**Status:** Working synthesis dated 2026-10-19. Do not treat this as G1 acceptance. R2 worker attempt4 was rejected and replaced by a Codex source-reviewed report; final R3/R10 repair and S5 correction are active. S3 produced only a negative source-availability audit, not a truth/label audit. Worker-exhausted R1/R4/R7 were replaced by explicitly Codex-authored, source-grounded reports. No project experiment or benchmark is claimed here.

## Evidence status by lane

| Lane | Current status | What the evidence supports | Open limit |
|---|---|---|---|
| R1 | Codex replacement in research/R1.md after four worker attempts | Architectural patterns and benchmark caveats across 14 systems/papers | Literature scores vary in reader, judge, setup, release; no portable leaderboard |
| R2 | Final worker attempt4 rejected; Codex source-reviewed replacement in research/R2.md | Correct benchmark/framework distinctions, corrected source join and license caveats; no invented thresholds | No dataset downloaded/run; source rights must be checked per corpus; no independent worker report accepted |
| R3 | Retry3 rejected; final retry4 active | Candidate facts remain TEST-only; all local performance and compatibility UNKNOWN | Retry3 invented splits/thresholds and contained insufficiently traceable claims; Codex review in evidence/r3-r3-r10-retry3-codex-review.md. No local measurement |
| R4 | Codex replacement in research/R4.md after four worker attempts | Claim-level evaluation, calibrated judge use, selective abstention, conflict-aware answers are testable | No local effect size; generic published results do not transfer to this answerer/corpus |
| R5 | Worker report reviewed and accepted in research/R5.md | Exact flat search is a plausible baseline at expected corpus size; sqlite-vec offers an in-file option | Windows/225H timings and compatibility remain unmeasured; estimates are not observed facts |
| R6 | Worker report reviewed and accepted in research/R6.md | Keep lexical+dense; test fusion; route identifiers to exact match; keep query rewriting gated | Effects depend on dataset; no project ablation or absolute local cost |
| R7 | Codex replacement in research/R7.md after four worker attempts | Event history, valid time, provenance, explicit supersession/retraction, conflict handling are correctness invariants | Schema behavior and query correctness remain untested locally |
| R8 | Accepted in research/R8.md as docs research | Official CLI/MCP documentation supports a read-only integration map and candidate smoke plan | No server registration or cross-CLI save/recall was tested |
| R9 | Accepted in research/R9.md as threat model/test specification | Threat cases for poisoning, injection, provenance spoofing, scopes, secrets | No adversarial code test or security signoff has run |
| R10 | Retry3 rejected; final retry4 active | Official Intel/Microsoft docs support candidate paths only | Retry3 omitted target SKU and included unsupported gates; no local compatibility, speed, parity, driver, or stability result |
| S1 | Corrected report independently checked/promoted as research/INTEGRATION.md | Local CLI availability/version/path inventory plus official doc map | No MCP config or grounding instructions were changed; no functional integration proof |
| S2 | Focused pip and loopback remediations verified; broad triage open | pip 26.2.1 strict pip-audit clean; 51/51 embedding tests plus B018 pass; MCP tests 9/9; loopback URL/proxy/redirect boundary covered | Broad Ruff/Pyflakes/MyPy/Bandit/Vulture findings and B404/B603 subprocess review remain open; no clean S2 gate claimed |
| S3 | Final attempt4 accepted only as a negative source-availability audit | 165-row schema verified; worker reports 34 missing source names and one ambiguous name; zero source-backed answers | No label truth/noise/leakage rate. Worker report over-discloses question/source metadata; cite only minimized Codex review `evidence/r3-s3-retry3-codex-review.md`; S3 remains blocked |
| S4/S6 | Not complete | Plan specifies profiles and resource soak | No profiling/soak evidence in this restart yet |
| S5 | Initial worker report rejected; correction attempt2 active | Source review flags unsupported SQLite/Defender, console, line-ending and runtime conclusions; Unicode fallback is a real narrow risk | Review in evidence/r3-s5-codex-review.md; await independent corrected report and review |

## Cross-lane conclusions

1. **Keep a small, inspectable baseline.** At expected facts scale, first measure flat cosine scan and the current FTS+vector path. Add ANN only if measured at realistic volume and latency fails. sqlite-vec is a candidate backend, not adopted based on external timings.
2. **Use lexical and dense signals together, then test fusion.** R6 supports RRF as a baseline but reports that tuned alternatives can win. E4 should compare RRF (k=10/30/60/100) with a one-parameter normalized weighted challenger on the same queries. Preserve an exact/prefix/trigram route for identifiers (E8).
3. **Filter lifecycle/scope before semantic ranking.** Exclude retracted/superseded/out-of-scope items from current-state retrieval. Preserve history for dated questions. Conflicts that cannot be resolved from source authority and valid time must be surfaced or abstained, never silently picked by score.
4. **Do not choose an embedder from aggregate leaderboard averages.** Candidate model-card scores and languages are not local retrieval quality. Use a fixed tokenizer/prefix/pooling and local corrected DEV/HOLDOUT set; report per-query category, paired CIs, warm/cold latency, and RAM. R3/R10 reports do not yet meet this bar.
5. **Treat generated answers as a separate quality gate.** Retrieval recall alone does not measure unsupported claims. Use one fixed answerer across the full comparison, record its exact route/model on each row, and annotate claim-level support vs source truth. Report answerable false refusal separately from no-answer precision/recall.
6. **Automated judges need calibration.** ARES/RAGAS/FActScore/RAGTruth and other papers motivate decomposed metrics; they do not provide project ground truth. Hand-label an overlap, report two-family agreement, and discard a judge whose agreement fails the plan's threshold.
7. **Write-time complexity has a cost.** Graphs, reflections, query rewriting, extra model calls, and memory-manager policies should be gated experiments. For this small local corpus, exact keys, SQLite/FTS, dense search, status/provenance filters, and optional rerank are the default candidateâ€”not a conclusion that richer methods never help.
8. **No external score clears a PRD gate.** Every floor must be measured on corrected held-out data, then sealed exactly once after configuration freeze. No sealed data has been opened.

## Experiment backlog rank for G1 review

This is a proposed order by expected information per cost. Before any run, confirm the matching entry and decision rule in EXPERIMENTS.md, freeze the split, and capture the model/runtime details.

| Order | Plan experiment | Why first | Adopt only if |
|---:|---|---|---|
| 1 | E8 exact-match fast path | Likely cheap and isolates identifiers that embeddings miss | Held-out identifier recall@1 meets >=0.95 and no cross-entity/path errors |
| 2 | E4 fusion | A direct retrieval-quality comparison with little model cost | Paired CIs support lift or same quality with lower latency |
| 3 | E9 current/superseded/recency handling | Correctness/security invariant, not just ranking preference | Zero stale-fact leakage in deterministic tests and no false supersession increase |
| 4 | E5 embedder/settings | Large quality/latency impact, but more model/RAM cost | Strict retrieval metrics improve on the same corpus and latency/RAM remain inside gates |
| 5 | E1 chunking / E3 query handling | Changes evidence units and query normalization; moderate cost | Paired held-out gain with no identifier/temporal regression |
| 6 | E7 reranking | Extra latency; only justified if top-result quality materially rises | Significant measured gain per added millisecond within p95 budget |
| 7 | E10 abstention calibration and E11 verifier | Hallucination/over-refusal critical but requires reliable answer labels and fixed answerer | No-answer and answerable rates both meet their preregistered bounds |
| 8 | E6 vector backend and E16 caching | Optimize only after a correct pipeline exists and corpus scale is measured | Realistic size/latency or RAM exceeds budget; measured improvement is stable |
| 9 | E2 contextual retrieval, E12 daemon, E13 graph, E14 extraction, E15 judge, E17 public replication, E18 language mix | Higher cost or dependent on unresolved data/model access; keep targeted | Run only when research and data gates justify the cost |

## Proposed untested baseline

- Append-only SQLite facts with provenance, source IDs, timestamps, active/superseded/retracted status, and explicit supersession links.
- FTS5/BM25 plus dense retrieval and RRF, exact-match routing for identifiers, strict scope/status filters before ranking, deterministic tie breaks, and optional confidence-gated reranking.
- Evidence-first answer output with source IDs and an abstain/conflict state. No automatic mutation based only on an uncalibrated model judge.
- Candidate local embedder and answerer remain unselected until E5 and the fixed-answerer quality study pass on valid splits.

## G1 gate remains blocked

Do not declare G1 passed until R2 and R3/R10 correction packages finish and pass Codex review, the missing R1/R4/R7 worker packages are represented by their clearly labeled Codex replacements, experiment registrations are checked against these findings, and S2/S3 defects have an explicit triage disposition. R1/R4/R7 worker attempts are exhausted; Codex fallback reports fill research coverage but are not independent worker verification. S3's source-availability failure is a plan blocker: do not compute â€œlabel noiseâ€, correct labels, or report a new quality metric without an authorized source-backed audit. G1 must record this failure honestly if the source boundary cannot be resolved.






