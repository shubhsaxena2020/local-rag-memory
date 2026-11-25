# Round 3 code-review report (interim)

**Status:** Interim as of 2026-10-20 15:35 IST. The reviews below are source-only. No exploit, fuzz campaign, full Appendix C suite, mutation run, service test, or cutover review has passed. “Confirmed” means confirmed by source inspection, not reproduced at runtime. The current release decision remains No-go.

## Scope and evidence

- An independent source-only Appendix C/R9 review recorded 12 findings and hash-checked its ten scoped files. Codex spot-checked the main O2/O3, provenance, lifecycle, FTS/filtering, resource, prompt, error, and path observations. Evidence: `evidence/r3-redteam-code-review1-codex-review-20261020-1444.md`; worker inventory: `work/r3-redteam-code-review1/REPORT.md`.
- An isolated O2/O3 correction2 candidate passed 13 synthetic tests, but remained unmerged. A separate Sonnet second pass was reviewed by Codex; all nine cited hashes match. The second pass confirmed that only the facts block is capped, source-sibling handling remains fail-open in unreviewed consumers, and the SQL row cap precedes the caller filter. Default patch check fails on CRLF bases; `--ignore-whitespace` check succeeds, but no patch was applied. Evidence: `evidence/r3-r9-o2-o3-fix2-codex-review-20261020-1504.md` and `evidence/r3-o2-o3-second-pass1-codex-review-20261020-1533.md`.
- No database, user corpus, evaluation, held-out, aggregate, or sealed content was read during these reviews. The sealed set remains unopened.

## Finding dispositions

| ID | Source-level finding | Disposition | Evidence / remaining work |
|---|---|---|---|
| F1 | Stored fact text/source can forge trusted-looking grounding lines through raw newline, delimiter, id, and confidence content. | Confirmed in baseline; **partly addressed only in an isolated candidate**. | Candidate escaping and per-facts-block cap are not merged; complete answer/judge prompts remain unbounded and contain other raw untrusted fields. Needs registered format amendment, integrated tests, and separately approved synthetic attack validation. |
| F2 | Same-source siblings can inherit query answerability/confidence and occupy retrieval slots despite not matching the query. | Confirmed; **partly addressed only in an isolated candidate**. | Candidate marks appended siblings and removes their confidence in MCP output, but absence of the marker is fail-open for other consumers; matched fragments still carry a query-level value. The 200-row limit is applied before the Python filter and has no global source-scan budget. |
| F3 | Writer/source labels are caller-supplied and do not authenticate a caller or enforce per-agent access scope. | Open design/policy finding. | Requires a defined single-user versus multi-agent trust model and server-assigned origin distinct from caller labels. |
| F4 | Conflicting/poisoned facts can persist; MCP supersede/retract are disabled and no-judge creation remains active. | Open policy finding; transaction rollback defect claim rejected. | Codex confirmed create commits without a judge by design and rolls back exceptions. Need explicit conflict, retirement, and secret-handling policy with synthetic tests. |
| F5 | Keyword/vector top-k truncation occurs before status/writer/source/tag filtering, allowing crowd-out. | Confirmed at source level; runtime magnitude unknown. | Requires a bounded over-fetch/filter strategy and synthetic temporary-DB coverage. |
| F6 | Bound user input is still raw FTS5 syntax; ASCII fallback can silently drop non-ASCII text after parser errors. | Confirmed at source level; malformed-query runtime behavior untested. | Requires an explicit query grammar, Unicode-aware fallback/error semantics, and temporary-DB operator/Unicode tests. This is not SQL parameter injection. |
| F7 | Resource controls are incomplete, including unbounded sibling reads, arbitrary fact growth, full vector scans, unbounded Ollama response reads, and prompt output size. | Confirmed at source level; resource impact not measured. | Candidate’s per-source `LIMIT 200` is not a complete fix: it is before caller filtering, lacks a global bound, and is unmerged. Agree resource budgets and run isolated stress tests under the guarded program. |
| F8 | Entity keys are unnormalized, unscoped free-text exact matches. | Confirmed design risk; practical impact depends on an inactive/no-judge path. | Define entity and namespace semantics; add synthetic Unicode/case/whitespace tests. |
| F9 | Answer and judge prompts have weak delimiters for free text, including answer/reference fields. | Confirmed at source level; no prompt attack attempted. | Requires a registered prompt change, bounded escaping/delimiters, synthetic builder tests, and separate judge-integrity verification. |
| F10 | Error surfaces can include input/path details; MCP failures lack a clear operator audit trail. | Partly confirmed; SDK rendering unverified. | Generic client errors and server-side audit/origin policy need review against the MCP SDK; no claim about its behavior is made. |
| F11 | MCP DB path confinement exists, but Windows reparse-point/ADS/TOCTOU behavior and data-folder ACLs are unverified. | Partial source mitigation; Windows boundary unverified. | Requires a host-level path/ACL review and tests in a disposable temp directory. |
| F12 | Clock formats, lone surrogates/NUL, cache invalidation for direct pipeline callers, connection concurrency/cleanup, and cutover rollback remain open. | Mixed: source behavior observed; runtime and cutover safety unverified. | No full module review or mutation/soak/cross-CLI validation has run. There is no accepted cutover/rollback script review. |

## Positive observations (limited to reviewed source)

- SQL values in the scoped code are parameter-bound; FTS5 syntax semantics remain a separate issue.
- Loopback Ollama URL checks and lifecycle transaction rollback paths were present in reviewed code.
- MCP refuses direct supersede/retract operations, and recall filters inactive facts; these observations do not close lifecycle or crowd-out concerns.

## Gate status and next verification

Appendix C/R9 remains **Partial**. O2/O3 are **partly addressed only in an unmerged candidate**. The second-pass source review is complete, but remediation review, integrated regression tests, bounded adversarial/fuzz testing, runtime behavior, resource measurement, mutation testing, and cutover rollback verification remain open. The test matrix still has no passing release rows; sealed scoring still requires Shubh’s explicit approval after the other gates.

This report is not a security certification. Do not rely on it to claim exploit resistance, zero hallucinations, perfect accuracy, readiness, or permission to cut over.
