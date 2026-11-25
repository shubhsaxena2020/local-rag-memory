# INTEGRATION - read-only inventory of local agent CLIs (S1)

Page verification date for the official docs used to derive paths: 2026-10-19 (same fetches as R8.md; direct URLs listed in section 6).
Inventory date: 2026-10-19, Windows 11 host plus the kali-linux WSL distribution (targeted correction for Kali Goose: 2026-10-19).

## 0. Method and safety statement

- Only two kinds of local probes were run: (1) command lookup on PATH plus `<exe> --version`, (2) path-existence tests (Test-Path in PowerShell, `[ -e ]` in the Kali shell) on paths that the OFFICIAL docs name.
- For the targeted correction in this worker, only read-only `command -v goose` and `goose --version` were rerun inside kali-linux.
- No user config, auth file, environment file, credential store, session database or memory file was opened, read, listed by content or parsed. Directories were tested for existence only (never enumerated).
- No config was edited, no MCP server was registered, no CLI was started interactively, no model call was made.
- Redaction applied: the home, local-app-data and roaming-app-data directories appear as %USERPROFILE%, %LOCALAPPDATA%, %APPDATA%; account name, machine name, IPs, URLs from local files, tokens and keys are not present in this document because none were read. In Kali Linux, `~` denotes `/home/shubh`.
- Precision caveat: existence of a path says nothing about its content, validity or whether the CLI is currently using it.
- "Plan-listed" CLIs come from the project's PRD/MISSION/PLAN documents (project docs, not product evidence): Codex, OpenCode, Goose, Hermes, Antigravity (agy), Kali-WSL Hermes; Claude Code is the seventh CLI named in the PRD and is inventoried as the "other plan-listed CLI". Gemini CLI is not plan-listed; it was checked on PATH only to avoid confusing it with Antigravity.

## 1. Executable inventory (LOCAL)

| CLI | Executable looked up | Resolved location (redacted) | Version output | Status |
|-----|---------------------|------------------------------|----------------|--------|
| Codex | codex | %USERPROFILE%\.npm-global\codex.ps1 (shim; codex.cmd alongside) | codex-cli 0.160.0 | reachable |
| OpenCode | opencode | %USERPROFILE%\.npm-global\opencode.ps1 (shim) | 1.18.34 | reachable |
| Goose | goose | %USERPROFILE%\bin\goose.cmd | 1.53.0 | reachable |
| Hermes (Windows) | hermes | %LOCALAPPDATA%\hermes\hermes-agent\venv\Scripts\hermes.exe | Hermes Agent v0.21.5+6750 (2026.9.24), git install | reachable |
| Antigravity CLI | agy | %LOCALAPPDATA%\agy\bin\agy.exe | 1.3.6 | reachable. `antigravity` is NOT on PATH. That agy is the Antigravity CLI is taken from the project plan; the docs page https://antigravity.google/docs/mcp names an "Antigravity CLI" but does not state the executable name on the page read: executable-to-product mapping UNVERIFIED from docs |
| Kali-WSL Hermes | hermes inside distro kali-linux | ~/.local/bin/hermes and ~/.hermes/hermes-agent/venv/bin/hermes present (not on non-login PATH, `command -v hermes` failed) | Hermes Agent v0.21.5+6528 (2026.9.24) | reachable by explicit path |
| Claude Code (other plan-listed) | claude | %USERPROFILE%\.npm-global\claude.ps1 (shim) | 2.1.288 (Claude Code) | reachable |
| Gemini CLI (not plan-listed) | gemini | not found | n/a | unreachable (not needed) |
| Kali-WSL Goose | goose inside distro kali-linux | ~/.local/bin/goose (/home/shubh/.local/bin/goose; native WSL binary) | 1.49.0 | reachable (`command -v goose` resolves) |

WSL distributions present (names only): Ubuntu, docker-desktop, kali-linux. Only kali-linux was entered, and only to run version and existence checks.
Additional Kali LOCAL facts: codex (codex-cli 0.160.0) and opencode (1.18.34) are installed there. Goose is available at `~/.local/bin/goose` (`/home/shubh/.local/bin/goose`, resolved via `command -v goose`), reporting version `1.49.0` via `goose --version`. Evidence indicates this is a native WSL Linux binary rather than Windows interop: the binary resides within the native Linux filesystem hierarchy (`~/.local/bin/goose` rather than a `/mnt/c/...` Windows interop mount path), and the reported version (1.49.0) differs from the Windows host install (%USERPROFILE%\bin\goose.cmd reporting 1.53.0). These are separate installs from the Windows ones; their configs are in the Linux home.

## 2. Config-path existence (LOCAL existence; path and schema from DOCS only)

Legend: path column is what the cited official page names. "exists" = path present on this machine. Schema column is from the docs, never from the local file.

| CLI | Path (redacted) | Doc source | Exists | Doc-derived schema |
|-----|-----------------|------------|--------|--------------------|
| Codex | %USERPROFILE%\.codex\config.toml | https://developers.openai.com/codex/mcp | True | TOML, [mcp_servers.<name>] |
| Codex | %USERPROFILE%\.codex\AGENTS.md | https://developers.openai.com/codex/guides/agents-md | True | Markdown, no schema |
| Codex | %USERPROFILE%\.codex\AGENTS.override.md | same | False | Markdown; if present it takes precedence over AGENTS.md at that level |
| OpenCode | %USERPROFILE%\.config\opencode\opencode.json | https://opencode.ai/docs/config/ | True | JSON, $schema https://opencode.ai/config.json, key mcp |
| OpenCode | %USERPROFILE%\.config\opencode\opencode.jsonc | same | False | JSONC variant |
| OpenCode | %USERPROFILE%\.config\opencode\AGENTS.md | https://opencode.ai/docs/rules/ | True | Markdown |
| OpenCode | %ProgramData%\opencode (managed dir) | https://opencode.ai/docs/config/ | False | optional managed config dir (Windows) |
| Goose | %APPDATA%\Block\goose\config\config.yaml | https://raw.githubusercontent.com/block/goose/main/documentation/docs/guides/config-files.md | True | YAML, key extensions |
| Goose | %USERPROFILE%\.config\goose\.goosehints | https://goose-docs.ai/docs/guides/context-engineering/using-goosehints/ | False | Markdown/plain text; page gives this path for global hints |
| Goose | %APPDATA%\Block\goose\config\.goosehints | not named by docs; probed as a plausible Windows analogue | False | UNVERIFIED location (analogy only) |
| Hermes | %USERPROFILE%\.hermes\config.yaml | https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp and /docs/user-guide/configuration | True | YAML, key mcp_servers |
| Hermes | %LOCALAPPDATA%\hermes\config.yaml | project MISSION.md names this path (project doc, not product doc); docs mention HERMES_HOME as the home override | True | YAML |
| Hermes | %LOCALAPPDATA%\hermes\SOUL.md | https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files (SOUL.md lives in HERMES_HOME) | True | Markdown |
| Hermes | %LOCALAPPDATA%\hermes\memories (directory) | https://hermes-agent.nousresearch.com/docs/user-guide/features/memory (MEMORY.md, USER.md under memories/) | True | Markdown files |
| Antigravity | %USERPROFILE%\.gemini\config\mcp_config.json | https://antigravity.google/docs/mcp | True | JSON, key mcpServers, remote key serverUrl |
| Antigravity | %USERPROFILE%\.gemini\AGENTS.md | https://antigravity.google/docs/rules | True | Markdown, no frontmatter |
| Antigravity | %USERPROFILE%\.gemini\GEMINI.md | same | True | Markdown |
| Antigravity | %USERPROFILE%\.gemini\config\AGENTS.md | same | True | Markdown |
| Antigravity | %USERPROFILE%\.gemini\config\GEMINI.md | same | False | Markdown |
| Antigravity | %USERPROFILE%\.gemini\config\rules (directory) | same | False | *.md with YAML frontmatter, trigger key required |
| Claude Code | %USERPROFILE%\.claude.json | https://code.claude.com/docs/en/mcp | True | JSON, key mcpServers (user/local scopes) |
| Claude Code | %USERPROFILE%\.claude\CLAUDE.md | https://code.claude.com/docs/en/memory | True | Markdown |
| Kali Hermes | ~/.hermes/config.yaml, ~/.hermes/SOUL.md, ~/.hermes/memories (Linux home) | same Hermes docs | True, True, True | YAML / Markdown |
| Kali Codex | ~/.codex/config.toml ; ~/.codex/AGENTS.md | Codex docs | True ; False | TOML ; Markdown |
| Kali OpenCode | ~/.config/opencode/opencode.json ; ~/.config/opencode/AGENTS.md | OpenCode docs | True ; False | JSON ; Markdown |

## 3. Findings and unknowns

1. FACT (LOCAL): all six plan-listed targets are reachable; Kali Hermes requires an explicit path because it is not on the non-login PATH.
1b. FACT (LOCAL): Goose is installed and available in Kali WSL at `~/.local/bin/goose` (`/home/shubh/.local/bin/goose`), reporting version `1.49.0` via `goose --version` (resolved via `command -v goose`). Evidence distinguishes this native WSL Linux binary from Windows interop: the executable is located on the Linux root filesystem (`~/.local/bin/goose`, not under `/mnt/c/`), and the version is `1.49.0`, whereas the Windows host installation is at `%USERPROFILE%\bin\goose.cmd` running version `1.53.0`.
2. FACT (LOCAL): Hermes on Windows has config.yaml at TWO existing locations (%USERPROFILE%\.hermes and %LOCALAPPDATA%\hermes). Which one is active is UNKNOWN; determining it would require reading config or environment, which is prohibited here. The Hermes docs say the default home is ~/.hermes and that HERMES_HOME overrides it (https://hermes-agent.nousresearch.com/docs/user-guide/configuration). Any future registration must first identify the active home by a non-secret, official command (see planned check P-04).
3. FACT (LOCAL): OpenCode global config file is opencode.json (JSON), not a different name; no opencode.jsonc locally.
4. FACT (LOCAL): Goose global hints file does not exist at the doc-named path; Windows-specific hints location is not stated on the page read: UNKNOWN.
5. FACT (LOCAL): Antigravity has both ~/.gemini/AGENTS.md and ~/.gemini/GEMINI.md present. The rules doc lists several global filenames but does not say, on the page read, which wins when both exist: UNKNOWN. Precedence must be tested before placing a grounding block.
6. FACT (LOCAL): Codex has no AGENTS.override.md locally, so ~/.codex/AGENTS.md is the effective global file per the Codex docs.
7. UNKNOWN: whether the installed versions match the behaviour of the current docs; docs were read as latest.
8. UNKNOWN: whether Windows and Kali instances of Codex/OpenCode/Goose/Hermes share state; the paths are separate (Linux home vs Windows profile) per the existence results; no sharing mechanism was inspected.
9. UNKNOWN: contents of every file above (deliberately not inspected). No statement here describes what any config contains, including whether an MCP server for memory is already registered.
10. UNVERIFIED: the agy executable corresponds to docs product "Antigravity CLI" (section 1).

## 4. Per-CLI placement targets (DOC-derived, nothing applied)

| CLI | MCP registration target (doc) | Grounding block target (doc) | Reload step (doc) |
|-----|------------------------------|------------------------------|-------------------|
| Codex | ~/.codex/config.toml [mcp_servers.x] | ~/.codex/AGENTS.md | "Codex rebuilds the instruction chain on every run"; MCP: restart extension/new session (IDE doc) |
| OpenCode | ~/.config/opencode/opencode.json mcp.x (type local, command array) | ~/.config/opencode/AGENTS.md | restart the CLI (UNKNOWN whether hot reload exists) |
| Goose | config.yaml extensions.x (type stdio) | ~/.config/goose/.goosehints global per docs | docs: direct edits usually need restart for existing sessions |
| Hermes | active home config.yaml mcp_servers.x | SOUL.md in active HERMES_HOME | /reload-mcp or new session; memory/SOUL read at session start |
| Antigravity | ~/.gemini/config/mcp_config.json mcpServers.x | ~/.gemini/AGENTS.md or GEMINI.md | /mcp overlay can reload (CLI) |
| Kali Hermes | ~/.hermes/config.yaml inside Linux | ~/.hermes/SOUL.md | as Hermes; can a Linux process reach a Windows stdio server: UNKNOWN |
| Claude Code | not to be edited by workers (project rule: Claude Code registers itself) | ~/.claude/CLAUDE.md | n/a |

## 5. Planned checks (NOT EXECUTED in this package). Label: R = read-only, M = would mutate state.

| ID | Label | Check | Pass condition |
|----|-------|-------|----------------|
| P-01 | R | Re-run `<exe> --version` for all seven and compare with section 1 | identical strings |
| P-02 | R | Run each CLI's official non-secret listing command, only where docs name one: `codex mcp list` (Codex docs), `claude mcp list` (Claude Code docs), `/mcp` inside Antigravity CLI (docs), `goose info -v` (Goose docs, prints settings so output must be redacted before saving) | command exits 0; output stored only after redaction pass |
| P-03 | R | Validate the OpenCode schema URL is reachable and that the local opencode version accepts a candidate snippet using a scratch copy, never the real file | schema validation passes on scratch |
| P-04 | R | Identify active Hermes home using the official docs command or documented HERMES_HOME semantics without printing values (e.g. report only "home = default \| override") | one of two labels recorded |
| P-05 | R | Build candidate registration snippets for each CLI in a scratch folder using the key names in R8.md section 3; validate syntax only (TOML/JSON/YAML parsers on the scratch files) | all parse |
| P-06 | R | Start the memory MCP server alone with a scripted MCP client over stdio (no CLI involved), run initialize, tools/list, one recall against a temp DB | valid JSON-RPC responses, stdout clean |
| P-07 | R | Record Windows ACL display of the memory data directory with `icacls <dir>` (display only) | no broad write grants |
| P-08 | M | Back up a CLI config to <file>.bak-aimem-<timestamp> (project rule) | backup hash equals original hash |
| P-09 | M | Add the memory MCP entry to one CLI config using that CLI's documented command or schema (for example `codex mcp add ...` writes config.toml) | CLI lists the server; rollback script restores the backup |
| P-10 | M | Install the grounding block into the documented global instruction file | file diff shows only the block |
| P-11 | M | Start each CLI and perform a real save then recall through it | tool response with fact id |
| P-12 | M | Cross-CLI test: save in CLI A, recall in CLI B | B returns A's fact id and writer label |
| P-13 | M | Per-CLI echo test: ask the CLI to quote the first line of its grounding block | exact first line returned (proves the file is loaded) |
| P-14 | M | Antigravity precedence test: place distinct canary lines in AGENTS.md vs GEMINI.md and observe which is loaded | result recorded (resolves unknown 5) |
| P-15 | R | Kali: from inside kali-linux run only `--version` and existence checks; do NOT start a server bound to a Windows port until a loopback/WSL networking doc is read | no new listeners |
| P-16 | M | Rollback drill: restore each backup and rerun P-01/P-02 | CLI starts with original config |

Everything labelled M requires explicit user approval at execution time. None of P-01..P-16 was run for this package; the inventory in sections 1-2 used only version and existence probes described in section 0.

## 6. Official sources opened for path/schema derivation (verification date 2026-10-19)

- https://developers.openai.com/codex/mcp
- https://developers.openai.com/codex/guides/agents-md
- https://developers.openai.com/codex/config-basic
- https://opencode.ai/docs/config/
- https://opencode.ai/docs/mcp-servers/
- https://opencode.ai/docs/rules/
- https://opencode.ai/config.json
- https://raw.githubusercontent.com/block/goose/main/documentation/docs/guides/config-files.md
- https://goose-docs.ai/docs/guides/context-engineering/using-goosehints/
- https://goose-docs.ai/docs/getting-started/using-extensions/
- https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp
- https://hermes-agent.nousresearch.com/docs/user-guide/configuration
- https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files
- https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
- https://antigravity.google/docs/mcp
- https://antigravity.google/docs/rules
- https://code.claude.com/docs/en/mcp
- https://code.claude.com/docs/en/memory
- https://code.claude.com/docs/en/settings

## 7. Checklist against the S1 requirements

| Requirement | Done |
|-------------|------|
| Codex, OpenCode, Goose, Hermes, Antigravity, Kali WSL Hermes, other plan-listed CLI | yes (Claude Code; Kali Goose verified) |
| executable names and versions | yes, section 1 |
| config-path existence only | yes, section 2 |
| no config/auth/env/credential file opened | yes |
| no hostnames, IPs, URLs from local configs, usernames, tokens, headers, keys, values | none read, none printed |
| path/schema from official docs, missing binaries marked unreachable | yes (gemini unreachable, antigravity executable name not on PATH) |
| test section lists safe planned checks, mutating ones labelled, not executed | yes, section 5 |
