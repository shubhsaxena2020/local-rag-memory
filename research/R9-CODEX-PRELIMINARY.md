# R9 — Shared memory security: preliminary source synthesis

Status: Codex source synthesis for comparison with the assigned Hermes R8/R9/S1 report. This is not the worker deliverable or a completed security review. No active testing or project-specific SAST was run; the watchdog remains PAUSED.

## Scope and assumptions

The project is a local memory service exposed through agent-facing tools. Its data may be personalized or sensitive. A malicious or compromised caller, note, tool result, local process, or dependency may try to write deceptive data, retrieve another scope's facts, change system behavior, or persist an instruction for later turns.

Local-only deployment reduces remote exposure but does not make stored content trusted. Retrieved fact text, provenance strings, tool outputs, imported markdown, and source metadata can all be attacker-controlled.

This threat model concerns the ai-memory-system checkout only. It does not authorize scanning or changing the user's other services, personal vault, global CLI configuration, or external accounts.

## Source-backed threat findings

- OWASP's current RAG Security Cheat Sheet treats ingestion, storage, retrieval, generation, output validation, and agent integration as one attack surface. It explicitly identifies corpus poisoning, hidden/invisible text, access control, chunk isolation, index integrity, caching, and fail-closed design.
- OWASP's MCP Security Cheat Sheet says all tool responses must be handled as untrusted data; filtering or instruction-pattern detection cannot prove text is safe. Authorization and input/output validation belong in trusted application code.
- OWASP recommends least privilege per MCP tool, pinned/inspected schemas, sandboxing, human approval for destructive actions, dependency/source review, and auditable invocation logs.
- NIST's Generative AI Profile includes indirect prompt injection from data likely to be retrieved as a risk and discusses data poisoning as manipulation of model/system behavior through compromised data.
- AgentPoison is a peer-reviewed red-team study of poisoning agent memory/RAG. It reports a paper-specific 63% end-to-end attack success rate with poisoning ratio below 0.1% on its studied agents. This demonstrates risk plausibility; it is not a success-rate prediction for this codebase.
- Provenance identifies where a fact came from; provenance alone does not establish that the source is truthful or that the fact should override another source.

## Threat model by asset and boundary

| Asset | Boundary / threat | Required property |
|---|---|---|
| Fact text and source spans | Malicious instructions embedded in user/imported notes | Treated as data, never as privileged instructions |
| Tenant / project scope | Caller supplies or tampers with scope filter | Scope derived from authenticated caller; no cross-scope read/write |
| Vector index and FTS | Partial update, stale index, modified DB, wrong vector row | DB/index consistency and integrity are checked; fail closed on mismatch |
| Provenance / time | Spoofed author/source/valid-time metadata | Provenance fields are stored separately from content and cannot grant trust by themselves |
| Secrets | API keys, tokens, credentials accidentally saved or returned | Detect/redact at ingestion and suppress in output/logs |
| Deletion intent | Stale vector/cache/backup resurrects forgotten facts | Delete semantics cover all searchable projections and documented backups |
| MCP tool schema | Changed tool descriptions or command without review | Pin/hash schema and require user approval for install/change |
| Local server process | Malicious caller, path manipulation, unbounded input | Validate input size/type/path; loopback/stdio boundary; no arbitrary command execution |
| Retrieval-to-answer boundary | Stored prompt injection changes assistant behavior | Mark retrieved text untrusted and preserve source IDs / separation |

## Controls to verify in project source

1. Derive user/tenant identity from trusted connection context, not a caller-chosen filter string.
2. Enforce authorization in database operations and every retrieval channel (FTS, vector, cache, graph) before returning text.
3. Use SQL parameters for user data; escape FTS query syntax and cap token, query, vector, and result sizes.
4. Treat embeddings and their metadata as derived, untrusted indexes; reconcile them to source fact IDs and active/scope state.
5. Perform writes, supersession, and index updates transactionally or use a durable reconciliation marker that refuses unsafe reads.
6. Keep fact content distinct from trusted instructions; quote/source-bound retrieved data and never pass it to tools as executable instructions.
7. Validate mutation payloads in code; deletion and destructive operations require explicit caller authorization and clear semantics.
8. Redact secrets before storage and return; logs contain event IDs and outcome, not full secrets or private fact bodies.
9. Limit local model inference to loopback, bounded timeouts, request-size caps, and no arbitrary remote URL fetching.
10. Pin dependencies and inspect build/install provenance; do not silently load unverified extensions.
11. Protect DB files and backups with least-privilege ACLs; ensure rollback and backup verification do not restore intentionally deleted content.
12. Restrict MCP tool schemas and validate any tool arguments again at the server boundary.

## Required non-destructive tests for section 6

- Stored-note prompt injection: save a harmless fact with embedded “ignore instructions” text; recall must quote it as untrusted evidence, not follow its instruction.
- Hidden-content variants: zero-width/control characters, HTML comments, markdown links, encoded text, and multilingual imperative phrasing must not gain authority.
- Poisoning influence: one low-trust fact contradicts a direct user correction; output should preserve conflict or follow explicit correction semantics with provenance, not trust semantic similarity alone.
- Tenant isolation: caller A attempts query, delete, and filter tampering against caller B; assert zero disclosure and zero mutation.
- Provenance spoofing: content claims to be “official/system verified”; forged metadata cannot elevate it without trusted ingestion evidence.
- Secret canary: insert a synthetic token-shaped value in a test DB; ensure retrieval policies, tool output, errors, and logs redact or suppress it as designed.
- SQL/FTS injection: operators, quotes, parentheses, wildcard patterns, and malformed unicode produce bounded, safe behavior with no widened query or SQL execution.
- Path handling: traversal, UNC, ADS, and symlink-like paths remain inside approved project root when file import/export exists.
- Tool output injection: malicious tool-return text cannot alter a subsequent sensitive tool call; use a sandboxed stub and observe exact validated arguments.
- Mutation authorization: save/update/delete operations check caller scope independently of model-generated instructions.
- Index tampering/staleness: mismatch between DB and index is detected and causes bounded recovery or fail-closed retrieval; never leak deleted/superseded records.
- Resource exhaustion: oversized payload, large vector, huge query, and repeated concurrency remain within explicit limits; no unbounded allocation or hangs.
- Cache poisoning: same normalized query with different authorized scopes cannot reuse cross-scope results; filtered search cache is disabled or includes immutable scope/filter identity.
- Crash and restore: kill during transaction/rebuild; reopen and verify no partial mutation, integrity-check result, and delete-tombstone semantics.
- Secret-safe error: path/SQL/model error responses do not expose tokens, full private records, environment values, or machine-specific secrets.
- Schema/config changes: compare server/tool schema hash to approved value and refuse silent unexpected changes.

Every test should use dummy facts, isolated disposable databases, and instrumented tools. No live credential, external tenant, unrelated MCP server, or real user secret belongs in the test corpus.

## Adoption decisions

- ADOPT: mark retrieved content and tool returns as untrusted data; enforce scope at the trusted storage boundary; keep source/provenance IDs; validate every mutation in application code.
- ADOPT: use least privilege, bounded input/output sizes, secret-safe logs, explicit deletion rules, and scope-aware caches as release invariants.
- TEST: poisoned instruction strings, cross-scope attack attempts, DB/index inconsistency, cache isolation, and deletion resurrection using only dummy data.
- TEST: exact tool schema/command approval and configuration hashes later, only after every release gate passes.
- REJECT: keyword-only injection filtering as a security guarantee; provenance as proof of truth; caller-provided scope as authorization; global read/write access because the server runs on localhost.

## Evidence gap / red-team plan

No source-level SAST or dependency scan has been run in Round 3. The security-reviewer skill's recommended Semgrep, Bandit, Gitleaks, and Trivy commands are unavailable in both global PATH and the project `.venv`. Under PAUSED watchdog, no packages were installed and no scan launched. This document therefore does not certify the project secure.

After watchdog RUN and resource gates: first install/use an approved scanner only if the ordinary project environment supports it without weakening controls; capture tool versions and raw findings under `evidence/`. Then manual review should cover auth, parameterized SQL and FTS construction, input caps, path safety, secrets, timeouts, subprocess use, and dependency pins. Each finding needs a real source location and a disposition.

Red-team cases should be run against a disposable copy and sandboxed answerer/tool stub. Record the exact input, retrieved records, exact output/attempted tool call, and expected policy. Do not test against live CLI configs, the user's personal vault, other services, or unrelated sessions.

## Sources

- OWASP RAG Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/RAG_Security_Cheat_Sheet.html
- OWASP LLM Prompt Injection Prevention Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/LLM_Prompt_Injection_Prevention_Cheat_Sheet.html
- OWASP MCP Security Cheat Sheet: https://cheatsheetseries.owasp.org/cheatsheets/MCP_Security_Cheat_Sheet.html
- NIST AI 600-1, Generative AI Profile: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.600-1.pdf
- MITRE ATLAS: https://atlas.mitre.org/
- Chen et al., AgentPoison (arXiv): https://arxiv.org/abs/2407.12784

## Verification status

This is a 100+ line preliminary synthesis from opened public sources. Worker R9/S1 report and project SAST are still pending. It makes no claim that controls have been fully audited or that the product has passed security testing. No local test, dependency scan, config access, or system modification occurred.
Primary near-term priority: prevent cross-scope disclosure, ensure retrieved injections cannot authorize tools, and preserve deletion integrity.
