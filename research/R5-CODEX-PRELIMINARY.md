# R5 — Vector retrieval at laptop scale: preliminary source synthesis

Status: Codex research notes for later comparison with the assigned Hermes R5/R6 report. This file does not replace research/R5.md and is not a measured project result. Primary sources were opened through official project/PyPI pages on 2026-10-19.

## Decision framing

The memory corpus is currently small enough that exact/flat search is a serious baseline. ANN should be adopted only when measured end-to-end recall/latency/RAM at the expected corpus size justifies its added index lifecycle and consistency burden. At laptop scale, a simple index is often the right default.

The answer depends on vector dimension, scalar type, number of facts, update/delete rate, filters, and whether index RAM is counted with the MCP process. Always compare under identical vectors, normalization, candidate filters, CPU thread limits, and query set.

## Candidate summary

| Option | Shape | Expected fit | Main risk to check |
|---|---|---|---|
| SQLite-vec flat KNN | SQL extension and vector virtual table | Small corpora; minimal new service; one transaction domain | Flat scan scaling and extension loading on deployed Windows SQLite |
| USearch | Native HNSW/filtered vector index | Moderate/large corpora; compact native package | Index snapshot durability, filter/delete behavior, score semantics |
| hnswlib | Native HNSW library | Mature approximate search when build/install is feasible | Windows/Python wheel/build support and app-level persistence glue |
| FAISS CPU | Broad vector index toolkit | Larger research workloads and many index variants | Windows packaging/build complexity; likely excess for a laptop memory corpus |
| NMSLIB HNSW | Broader non-metric search library | Alternative if its current wheels and APIs fit deployment | Verify package/version support and maintenance for target Python |

This table is a selection frame, not a benchmark ranking.

## Findings from official sources

- sqlite-vec describes itself as a small SQLite extension and documents float32, int8, and binary vectors in `vec0` virtual tables.
- sqlite-vec states it is pre-v1 and may introduce breaking changes. Pin an exact release and rerun storage integrity tests before upgrading.
- The project README lists Windows among supported platforms, and its install guide lists a Python PyPI package. This is vendor documentation; verify the wheel/load path on the actual supported Python and SQLite build.
- USearch's PyPI release page currently lists a CPython 3.13 Windows x86-64 wheel. The retrieved release is 2.26.5, uploaded 2026-10-18. This is package-availability evidence, not a quality result.
- The official hnswlib repository gives pip/source build instructions and supports updates, persistence, filtering callback, and multi-threaded search. The inspected README does not itself establish a current Windows CPython 3.13 wheel; inspect PyPI files and test clean installation before choosing it.
- Current FAISS install documentation describes CPU package availability on Windows x86-64; its current project build config advertises Python 3.13 and includes a Windows wheel build target. This suggests feasibility but still requires checking a released matching wheel and the actual clean-install path.
- NMSLIB's official repository currently says it provides Windows binaries for Python 3.8–3.13. It is worth checking as a comparator if deployment packaging proves reliable.
- PLAID and ColBERT-style late-interaction retrieval retain multiple vectors per token/document representation; these approaches bring substantial indexing and memory costs relative to one vector per fact. They are not a first choice for a small, local fact memory.

## Capacity and resource arithmetic

For N float32 vectors of dimension d, raw vector storage is `N * d * 4` bytes, before SQLite or ANN graph overhead. For example, at d=768, the raw matrix alone is approximately 30 MiB at 10k vectors, 293 MiB at 100k, and 2.86 GiB at 1M. These are arithmetic estimates, not measured RSS.

int8 storage is approximately one quarter of float32 bytes; binary representations can be smaller still but use different distance semantics and may harm retrieval quality. Quantization needs a paired relevance evaluation and must not be conflated with ANN index compression.

HNSW graph memory adds per-vector links, labels, allocator/index overhead, and often a resident copy. Measure process working set, not only the serialized index file. Include rebuild peak because save/restore or compaction can temporarily hold two copies.

A flat vector scan can be competitive at 10k when d and CPU are modest, especially if exact filtering sharply reduces candidates. The crossover is empirical: report QPS/latency and recall for every N rather than assume ANN wins at a particular size.

At 100k and 1M, compare at least exact flat, SQLite-vec, and a supported HNSW implementation. Record write amplification and stale-index repair after updates/deletes.

## Proposed benchmark design

Use the same frozen vector corpus at 10k/100k/1M sizes. Generate vectors once and store their checksum. Include uniform random and representative corpus distributions; random-only ANN tests can overstate quality on actual data.

Queries include exact identifiers, short facts, paraphrases, multilingual English/Hindi if present in the real corpus, filtered scope searches, and updates/deletes. The gold relevance definition remains PLAN-24H Appendix A.

For each backend, report warm and cold start, p50/p95/p99, recall@1/5, MRR, process RSS/working set, index bytes, write time, delete/update visibility, crash recovery, and clean Windows Python 3.13 installation status.

Use at least 20 warm-up calls and 500 timed searches as the plan requires. Keep workers/model activity out of the latency window; log watchdog status during each measurement.

## Adopt / test / reject

- ADOPT: retain exact/flat retrieval as the transparent baseline and preserve the database as source of truth.
- TEST: SQLite-vec against existing flat scan for small collections; USearch as a Windows/Python 3.13 ANN candidate where the matching release wheel exists; hnswlib only after a clean wheel/build and persistence test.
- TEST: int8 quantization only as a separate paired ablation; do not combine its effect with ANN in the first comparison.
- REJECT for initial laptop release: DiskANN-style infrastructure or late-interaction ColBERT/PLAID until a measured corpus scale demonstrates need. The deployment complexity and multi-vector storage are not justified by current evidence.
- REJECT: selecting from vendor speed claims or theoretical complexity without measurements on this machine.

## Experiment spec (not run)

Hypothesis: at the observed corpus size, exact/SQLite-vec reaches the warm p95 target while preserving strict recall; at larger N, a supported HNSW alternative may improve p95 under a fixed recall loss bound.

Change: backend only; use one checksum-identified vector set and identical filters / tie-breaks. Keep each backend's data in isolated directories.

Metrics: strict R@1/R@5/MRR and identifier R@1; warm/cold p50/p95/p99; RSS steady/max; build/rebuild/update/delete cost; crash recoverability; clean Windows install outcome.

Split: tune ANN parameters on dev; evaluate one registered held-out set. The SEALED set is never used for tuning and is scored only once at final.

Decision rule: choose an ANN index only if recall remains within the registered non-inferiority margin, it improves the target latency at expected N, and RAM, durability, filtering, and package gates pass. Otherwise keep exact search.

## Sources

- sqlite-vec README: https://github.com/asg017/sqlite-vec/blob/main/README.md
- sqlite-vec installation guide: https://github.com/asg017/sqlite-vec/blob/main/site/getting-started/installation.md
- USearch current package files: https://pypi.org/project/usearch/
- hnswlib official repository: https://github.com/nmslib/hnswlib
- FAISS install guide: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
- FAISS current build configuration: https://github.com/facebookresearch/faiss/blob/main/pyproject.toml
- NMSLIB official repository: https://github.com/nmslib/nmslib
- PLAID paper: https://arxiv.org/abs/2205.09707

## Review status

All numeric storage values above are direct arithmetic from dimension and scalar width. No vector benchmark, package installation, local test, or model call ran. This preliminary note is pending independent Hermes R5 output and local verification after watchdog RUN.
