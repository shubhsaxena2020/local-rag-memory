# Round 3 failure analysis (interim)

**Status:** Interim as of 2026-10-20 15:30 IST. This records only verified failures, negative results, and open test-matrix gates. It is not a system error-rate estimate or final release analysis: quality and held-out runs are incomplete, and the sealed set remains unopened. No cutover is authorized.

## Measured functional-suite outcome

The latest full functional package rerun executed 273 tests: 271 passed, 2 optional chaos cases were skipped, and 0 failed or errored. This establishes only the 11 package suites executed by `verify_all.py`; it does not close the larger Round 3 property, protocol, stress, scale, migration, security, latency, mutation, or soak requirements. Evidence: [fresh functional run](evidence/r3-functional-run-20261020-1506.md) and raw outputs under `evidence/1792488981-48500-4935/`.

## Verified experiment failures and negative results

| Area | Verified observation | Meaning and remaining limit | Evidence |
|---|---|---|---|
| E7 reranker quality | On 99 dev answerable queries, always-on reranking point estimates were below control for R@1, R@5, and MRR. The paired 10,000-bootstrap intervals include zero; no positive lift was demonstrated. | Reject reranker adoption; this is not evidence of statistically significant degradation. Latency/RAM and a calibrated gated arm remain unmeasured. | `evidence/r3-e7-quality-dev-results-v2.json`; `evidence/r3-e7-independent-scorer-review.md` |
| E8 candidate tests | Correction2 synthetic suite ran 64 tests: 48 passed and 16 failed. Three entity-key cases use strings the candidate grammar rejects; punctuation expectations conflict; 12 pipeline cases fail because the fake storage omits the SQL connection contract. | Reject for integration. These test results do not measure retrieval quality. A separate invented-text smoke demonstrated `123abc` extracted from inside `café-123abc`, violating the whole-token claim. The candidate also full-scans the facts table instead of using the registered trigram/prefix lookup. | `evidence/r3-e8-correction2-codex-review-20261020-1517.md`; `evidence/r3-e8-correction2-unittest-20261020-1517.txt`; `evidence/r3-e8-token-boundary-smoke-20261020-1522.txt` |
| E1 pre-overlay candidate tests | Before the Codex test-only overlay, the authored suite had 2 assertion failures and 2 errors among 312 selected cases; the overlay fixed three test defects and passed 311/311 on a source-identical production copy. | This validates only that isolated source candidate under synthetic fixtures. It is not an E1 metric or approved-data result. The worker disclosed a prohibited AST parse; four correction rounds are exhausted. | `evidence/r3-e1-failed-preoverlay-runs-index-20261020-1358.md`; `evidence/r3-e1-codex-testfix1-review-20261020-1255.md` |

## Rejected or blocked experiment execution

- E1, E2, E4, E5, and E10 do not have accepted quality scores. The submitted harnesses/loaders were rejected or quarantined where they parsed mixed golden/probe rows before dev filtering, failed to verify pins, or could not establish a separately approved dev-only input boundary. Do not infer results from those packages.
- E3 and E4 have no valid candidate result; their correction allowances are exhausted. E8 has no approved identifier subset, registered grammar, false-exact definition, or bounded index provider. E9 has no approved temporal dev input or complete preregistration.
- E10 has no valid prompt A/B score. Its threshold-only diagnostic was infeasible, and the integrated runner was rejected for split/input/manifest isolation defects. No abstention floor is established.
- O2/O3 has source-level findings and an isolated candidate; 13 synthetic tests passed, but the candidate is unmerged and no exploit/fuzz campaign or security gate has passed. The completed source-only second pass confirms that the full request remains unbounded, sibling handling is fail-open in other paths, and the per-source row cap precedes caller filtering. Default patch application also fails against CRLF working-tree files; no patch was applied. Evidence: `evidence/r3-o2-o3-second-pass1-codex-review-20261020-1533.md`.

## Remaining system-level unknowns

The acceptance matrix remains No-go. It records no Pass rows; significant gaps include MCP-client fuzz/concurrency/restart, randomized writer-crash and corruption campaigns, scale curves, schema migration/restore, time/Unicode matrices, regression scoring, mutation testing, approved quality evaluation, latency/RAM under a quiet window, and soak. The test-matrix one-command runner is not accepted because it lacks an enforceable Windows isolation boundary. The source-only code-review dispositions are consolidated in `CODE-REVIEW-REPORT.md`; see `docs/TEST-MATRIX.md` for row-level evidence and prerequisites.

Do not claim zero hallucination, perfect accuracy, readiness, or a cutover result from this interim ledger. Update this document only when a new result has exact raw evidence and an independent review; write the final per-gate analysis after the full plan's gates have been evaluated.
