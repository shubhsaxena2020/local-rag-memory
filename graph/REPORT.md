# Graphify bridge report

`GraphifyBridge` updates a graph using subprocess `cwd` instead of changing process CWD, queries graphify with plain argv strings, and falls back to NetworkX with a warning when the CLI is unavailable. Fact-to-node association tokenizes graph IDs, labels, and source paths and matches whole tokens; it applies graph size and result bounds. The scratch project under `graph/scratch-project/` is separate from the user's vault.

## Verification

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe -m compileall -q graph
.\.venv\Scripts\python.exe -m unittest discover -s graph/tests -v
Push-Location graph\scratch-project; graphify update .; Pop-Location
graphify query 'What does the add function do?' --graph 'C:\Users\shubh\ai-memory-system\graph\scratch-project\graphify-out\graph.json'
```

Orchestrator verified compilation and 6 discoverable unittest cases, regenerated/checked the scratch graph, and observed a query matching `add()` and its source node. Evidence is in `../evidence/graph-compile.txt`, `../evidence/graph-tests-final.txt`, `../evidence/graphify-update.txt`, and `../evidence/graph-query-final.txt`.

## Limitations

- The NetworkX fallback performs simple label matching, not semantic retrieval.
- The bridge is not yet integrated into the MCP retrieval path or persisted on fact rows.
