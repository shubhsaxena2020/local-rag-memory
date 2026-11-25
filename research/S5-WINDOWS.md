# S5 Windows-Specific Scan (Static Review)

**Status:** Static source/test review plus one guarded temporary-database runtime probe completed; broader Windows runtime cases remain unverified. No product code or machine settings changed. This does not close S5.

## Scope and evidence

Read `storage/storage.py`, `mcp_server/server.py`, `embedding_bench/dataset.py`, `graph/graphify_bridge.py`, `mcp_server/test_mcp_server.py`, and `storage/tests/test_storage.py`; searched the core test trees for Windows path, Unicode, encoding, locking, clock, and sleep coverage. Source inspection was followed by `evidence/r3-s5-windows-runtime-probe.py`, run with the common owner-token lock and a temporary DB under project `data/`. It passed a Unicode-and-spaces DB path, exact multilingual fact/source round-trip (including Japanese, Devanagari, emoji, accented combining text, Hebrew, Chinese and Cyrillic), Devanagari FTS recall, and rejection of an outside DB path. No user DB was read or modified. No Defender, scheduled-task, hibernate, clock, UNC, or long-path experiment was run. Output: `evidence/r3-s5-windows-runtime-probe.txt`.

## Findings

1. **MCP database confinement is explicit.** `mcp_server/server.py:22-30` resolves `AI_MEMORY_DB`, then requires the resolved location to be relative to this project's `data/`. A direct test covers an outside temp path (`mcp_server/test_mcp_server.py:142-145`). The tests do not cover drive-case differences, `..`, junction/symlink escapes, UNC paths, or data-directory names containing spaces/Unicode. The production default is repository-relative, which avoids dependence on the caller's current directory.
2. **Storage itself accepts any `Path`.** `storage/storage.py:53-56` creates parent directories and passes the supplied path to SQLite. MCP callers are confined upstream, but direct library callers are not. This appears intentional for tests and portable library use; do not treat it as a server escape without an entry path that bypasses `resolve_db_path`.
3. **SQLite locking and Windows cleanup have partial coverage only.** `MemoryStorage.connect()` enables WAL (`storage/storage.py:64-73`). The chaos tests use temporary DBs, multiple connections, cleanup and WAL recovery; `storage/tests/test_storage.py` cleanup catches `OSError` when deleting DB/WAL/SHM files. There is no evidence here of Defender-held-file contention, antivirus scanning behavior, or a multi-process soak under Windows AV. The chaos thread test's explicit busy timeout is per test connection, not proof of the production default under prolonged contention.
4. **Unicode handling has useful implementation support but insufficient Windows-specific tests.** Dataset normalization uses NFKC and casefold (`embedding_bench/dataset.py:17-21`), JSONL is read as `utf-8-sig` (`:119`), and SQLite FTS uses the `unicode61` tokenizer (`storage/migrations/V1__init.sql`). The search found no targeted emoji, Devanagari, RTL, combining-mark, or Unicode-path tests in the examined suites.
5. **Subprocess argument/path handling is mostly structured.** Graphify uses argument lists, a `cwd` parameter and timeouts (`graph/graphify_bridge.py:60-65,108-119,206-211`), so spaces do not rely on manual shell quoting. However, `text=True` uses the platform's default text encoding at subprocess boundaries (`:64` and query call); non-ASCII output decoding under varied Windows console/code-page settings was not verified. Graph JSON files are explicitly read as UTF-8 (`:273`).
6. **Line endings vary.** Byte inspection found `storage/storage.py` mostly CRLF (684 CRLF, 2 bare LF), `embedding_bench/dataset.py` CRLF-only, `mcp_server/server.py` and `mcp_server/test_mcp_server.py` LF-only, and `graph/graphify_bridge.py` mixed (475 CRLF, 28 bare LF). This is not shown to break execution, but editors/patch tooling should preserve existing endings.
7. **Machine lifecycle cases are open.** No automated or controlled evidence was found for scheduled-task startup, Defender DB scans, sleep/hibernate resume, timezone/DST or backward clock changes. These remain S5 evidence gaps, not assumed passes.

## Remaining S5 work

- Extend isolated path tests to `..`, junction/symlink escapes and long paths only if Windows long-path policy/filesystem support them; record environment limits. UNC remains untested because MCP deliberately confines DB paths to project `data/`.
- Exercise a two-process write/read case and Windows cleanup while the DB is open/closed. Any Defender-specific reproduction must stay observational; do not change Defender exclusions or service settings.
- Leave scheduled-task, sleep/hibernate and clock-jump tests pending until an explicitly controlled test window is available. Record current policy and exact setup before any system-level action.

No claims are made about external CLI configs, personal data, sealed evaluation data, or user-owned services.
