# Fact lifecycle report

`FactLifecycleManager` works against the existing `MemoryStorage` facts schema and creates only two side tables: `fact_entity_keys` and `fact_audit`. It never creates a competing `facts` table and never deletes or edits fact text.

`create_fact` inserts compatible facts as active. For same-key or duplicate-hook candidates, the optional mutation judge receives the explicit action `supersede`, old/new fact records, and context. Only approved proposals mark the old row superseded and set `superseded_by` / `valid_to`; rows are retained. `retract` invokes the judge with action `retract` and context and changes status only after approval. Insert, status changes, and audit records use SQLite transactions. The judge is injectable; a caller must provide a local implementation for automatic mutation decisions.

## Verification

```powershell
.\.venv\Scripts\python.exe -m compileall -q lifecycle
.\.venv\Scripts\python.exe -m unittest discover -s lifecycle/tests -v
```

Orchestrator result: compile succeeded; 5 tests passed. Evidence: `../evidence/lifecycle-tests-final.txt`.

## Limitations

- The judge and near-duplicate detector are injected callbacks; this module does not call Ollama itself.
- SQLite permits one writer at a time. Cross-process concurrency and crash recovery need W10 verification.
