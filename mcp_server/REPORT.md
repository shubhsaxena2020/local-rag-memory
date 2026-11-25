# MCP package verification

## Implemented

- Six local stdio tools: `memory_save`, `memory_recall`, `memory_recent`, `memory_health`, `memory_supersede`, and `memory_retract`.
- Save and search use the shared `storage.MemoryStorage` schema and append-only `lifecycle.FactLifecycleManager`, with integer fact IDs.
- `AI_MEMORY_DB` must resolve under this projectâ€™s `data/` directory. Default: `data/memory.db`.
- No HTTP listener is started.
- Supersede and retract fail closed because no validated local mutation judge is configured. They do not alter facts.

## Previous implementation verification (Round 2)

The earlier FTS-only implementation passed `python -m compileall -q mcp_server` and `python -m unittest discover -s mcp_server -p 'test*.py' -v` in the project virtual environment. That historical run covered five tests, including a real SDK `ClientSession` subprocess smoke test. Output: `evidence/mcp-tests-final.txt`. It does not verify the Round 3 hybrid-retrieval edits below.

An earlier worker version did not use the shared storage schema, attempted to delete facts during lifecycle operations, and did not run under the installed SDK. The orchestrator replaced those handlers and tests before accepting this package.

## Limitations

- Before Round 3, recall used FTS keyword search only. Round 3 now wires hybrid retrieval; that implementation is awaiting tests and live verification.
- Mutation tools remain unavailable until an independently verified local judge is configured.
- This package test uses a temporary database beneath project `data/`; it does not touch the legacy service database or client configuration.

## Round 3 integration status (verification pending)

The Round 3 implementation now wires `memory_recall` to `RetrievalPipeline` with active-fact and caller filters, and returns structured facts plus a grounding-ready context string. `memory_save` attempts the same EmbeddingGemma document format used by the benchmark; query embeddings use the matching query format and request a 30-minute Ollama keep-alive. Embedding failures preserve keyword search and are reported in the tool result. MCP tool names remain unchanged. Regression tests cover hybrid retrieval, grounding context, explicit keyword fallback, and the keep-alive request.

These edits are not yet verified. The laptop watchdog paused project processes after a 95.9 C reading; test execution is deferred until it resumes and resource limits permit. Do not treat this section as a passed gate until the MCP and embedding suites, full `verify_all`, and real stdio checks pass.

