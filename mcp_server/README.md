# AI Memory System MCP server

This package provides six tools over local MCP stdio: `memory_save`, `memory_recall`, `memory_recent`, `memory_health`, `memory_supersede`, and `memory_retract`.

## Run

From the project root, using its virtual environment:

```powershell
.\.venv\Scripts\python.exe -m mcp_server
```

The server uses the shared SQLite schema in `data/memory.db`. `AI_MEMORY_DB` may select another database only when its resolved path stays under this project’s `data/` directory. The server has no network listener.

Save, recall, recent, and health use the shared storage and lifecycle packages. Save and recall use the local `embeddinggemma` model through loopback Ollama when available. Recall uses the shared hybrid BM25+dense RRF pipeline and returns `facts`, `retrieval_mode`, `warning`, and a grounding-ready `grounding_context`; if embedding fails it reports keyword-only fallback. The embed call asks Ollama to keep the model resident for 30 minutes. Set `AI_MEMORY_DISABLE_EMBEDDING=1` for tests that must not contact Ollama. Supersede and retract return an error without changing facts because a validated local judge is not configured. This is fail-closed behavior.

## Verification

Run `\.venv\Scripts\python.exe -m unittest discover -s mcp_server -p 'test*.py' -v`. The tests include an official MCP SDK client subprocess check. See [REPORT.md](REPORT.md) and `evidence/mcp-tests-final.txt` for current results.
