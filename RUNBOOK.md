# Runbook

## Current state

The project is a local prototype. The new MCP server is not registered with any CLI, the old `rag-memory-mcp` service/configuration remains untouched, and no migration was performed. The MCP smoke suite left `data/memory.db` as a local empty test database (zero facts); it contains no migrated baseline records. Round 2 measured better retrieval Recall@5/MRR than the old service, but failed heldout abstention precision (0.2985 vs 0.90), live warm p95 latency (3.195 s vs <400 ms), and sampled benchmark plus Ollama runner RAM (10.23 GB vs <2 GB). Do not register or point clients at this server; see `FINAL-REPORT.md` for the complete gate evidence.

## Environment

Use PowerShell from `C:\Users\shubh\ai-memory-system`:

```powershell
\.venv\Scripts\Activate.ps1
```

Python dependencies are installed in the project virtual environment. Ollama and graphify are local dependencies. Selected benchmark candidates are `embeddinggemma` (embedder) and `qwen3:1.7b` (judge fallback); all installed/pulled models remain in place because cleanup was not authorized.

## Run checks

```powershell
$python = '.\.venv\Scripts\python.exe'
& $python -m compileall -q eval judge storage lifecycle retrieval grounding graph embedding_bench mcp_server chaos
& $python -m unittest discover -s eval\tests -v
& $python -m unittest discover -s judge\tests -v
& $python -m unittest discover -s storage\tests -v
& $python -m unittest discover -s lifecycle\tests -v
& $python -m unittest discover -s retrieval\tests -v
& $python -m unittest discover -s grounding\tests -v
& $python -m unittest discover -s graph\tests -v
& $python -m unittest discover -s embedding_bench\tests -v
& $python -m unittest discover -s mcp_server -v
& $python -m unittest discover -s chaos\tests -v
$env:RUN_SLOW_SCALE = '1'
& $python -m unittest discover -s chaos\tests -p 'test_chaos.py' -v
Remove-Item Env:\RUN_SLOW_SCALE
```

The current Round 2 package outputs are in `evidence/r3-full-*-tests.txt`; `evidence/r3-full-suite-summary.json` records 261 test executions, three intentional skips, and no failures. The scale run is recorded with the suite evidence.

## Run the MCP server manually

The server uses stdio and defaults to `data/memory.db`. `AI_MEMORY_DB` can select a database only under the project `data` directory. A client must launch the server as a subprocess using the MCP stdio transport; it does not open an HTTP listener. Current tool names are `memory_save`, `memory_recall`, `memory_recent`, `memory_health`, `memory_supersede`, and `memory_retract`.

The integration suite starts a real local MCP client/server pair. Supersede and retract fail closed until a validated local mutation judge is configured. `memory_recall` uses the shared hybrid retrieval pipeline and returns facts plus a grounding-ready context, retrieval mode, and a warning when dense embeddings fall back to keyword search. The Round 3 wiring and its integration tests remain pending verification; do not treat the integration gate as passed until the current suite and live stdio check pass.

## Cutover and rollback

No client configuration was edited, so there is no backup set or rollback script to run. Keep the existing service and registrations as they are. If a later release passes the comparative-quality, grounding, memory, and CLI smoke gates, cutover must first back up each configuration file, create and exercise a rollback script, migrate the two existing records, then prove each registered CLI can save and recall a fact through the new server. Never edit Claude Code's configuration as part of this project; its coordinator registers itself.
