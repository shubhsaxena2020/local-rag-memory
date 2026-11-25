# Embedding Model Benchmark Package (W2) - Report

## Round 2 corrected benchmark and selection

The corrected benchmark used 165 golden questions and 145 deduplicated passages with strict target-passage relevance. All model-specific query/document formats were applied. The quality-selected embedder is **EmbeddingGemma** because it is the only candidate that exceeded the old service on both strict Recall@5 and MRR in the end-to-end hybrid pipeline; the harness composite score alone is not the selection rule.

| Variant | R@1 | R@5 | MRR | p50 ms | p95 ms | estimated RAM MB |
|---|---:|---:|---:|---:|---:|---:|
| nomic-embed-text v1.5, prefixed | .1939 | .2182 | .2040 | 125.78 | 325.01 | 262 |
| nomic-embed-text v1.5, raw ablation | .2000 | .2727 | .2267 | 180.05 | 365.64 | 262 |
| qwen3-embedding:0.6b | .1939 | .2667 | .2193 | 493.86 | 980.52 | 610 |
| embeddinggemma | .1879 | .4970 | .2939 | 172.32 | 340.31 | 594 |
| bge-m3 | .1879 | .2788 | .2206 | 222.68 | 519.79 | 1105 |

The old loopback service scored strict R@5 .4606 and MRR .2700. EmbeddingGemma's standalone dense R@5 .4970 and MRR .2939 clear that reference; the fused end-to-end pipeline scored R@5 .5576 and MRR .3685. Prefixed Nomic underperformed its raw ablation here; this is benchmark evidence for this corpus/runtime and does not establish a general formatting rule. EmbeddingGemma p95 and RAM are within the local target estimates; whole-system memory must still be judged separately.

The strict passage identity, per-model formatters, ablation exclusion, and tested model unload policy are described below. Corrected results and raw timings are saved in `../evidence/r2-embedding-all.txt`; the final quality-first winner calculation is `../evidence/r2-embedding-quality-selection.json`; pipeline comparison is in `../evidence/r2-retrieval-e2e.txt`. Models were not deleted.

---
# Historical round-1 report below (superseded where it conflicts with the above)

# Embedding Model Benchmark Package (W2) - Worker Report

## 1. Executive Summary

The embedding-model benchmark package has been built in `C:\Users\shubh\ai-memory-system\embedding_bench\`.
It implements a concise, robust, and non-destructive benchmark harness designed to evaluate candidate embedding models against the repository's real golden set (`eval/golden_set.jsonl`) using local Ollama.

### Candidate Models
1. `nomic-embed-text` (v1.5, 768 dimensions, ~137M parameters, ~274 MB RAM footprint, 8192 context window)
2. `qwen3-embedding:0.6b` (0.6B dense retriever, 1024 dimensions, ~650 MB RAM footprint)
3. `embeddinggemma` (Gemma embedding architecture, 768 dimensions, ~300M parameters, ~400 MB RAM footprint)

---

## 2. Package Architecture and Components

The package is structured into clean, modular, single-responsibility files:

- **`embedding_bench/client.py`**:
  - `validate_loopback_url(url)`: Strictly validates that the Ollama URL uses `http` and points only to local loopback hosts (`127.0.0.1`, `localhost`, `::1`). Rejects all remote hosts and non-HTTP schemes.
  - `validate_embedding(vec, expected_dim)`: Validates that returned vectors are non-empty lists of finite numbers (detects and rejects `NaN`, `+Inf`, `-Inf`, non-numerics, and dimension mismatches).
  - `OllamaEmbeddingClient`: Loopback HTTP client implementing health checks, model catalog inspection (`/api/tags`), active resident inspection (`/api/ps`), metadata querying (`/api/show`), vector embedding via `/api/embed`, and safe memory release via `keep_alive: 0` without deleting models.

- **`embedding_bench/index.py`**:
  - `CosineIndex`: Exact in-memory cosine similarity retrieval index.
  - Automatically $L_2$-normalizes document and query vectors: unit-vector dot products yield exact cosine similarities.
  - Dual execution engine: Uses NumPy matrix operations when available; falls back to pure Python dot product when NumPy is absent.
  - `estimate_memory()`: Computes exact float32 (4-byte) and float64 (8-byte) vector footprints plus metadata overhead in bytes and MB.
  - `SearchResult`: Structured dataclass carrying document ID, similarity score, text, source file, and original query ID mappings.

- **`embedding_bench/dataset.py`**:
  - `resolve_golden_set_path(path)`: Robust repo-root path resolver that dynamically discovers `eval/golden_set.jsonl` from repo root, package directory, current working directory, or `GOLDEN_SET_PATH` environment variable.
  - `load_golden_set(filepath, deduplicate_answers)`: Validates line-by-line JSONL schema (non-empty `question` and `answer`/`text`), constructs unique canonical `CorpusRecord` documents, and generates `GoldenQuery` objects referencing target document IDs.

- **`embedding_bench/metrics.py`**:
  - `calculate_percentile(values, p)`: Exact linear interpolation for p50 and p95 latency percentiles.
  - `is_relevant_hit(retrieved, query)`: Multi-strategy ground-truth verification matching target document ID, original query ID set, exact answer match, and substantive substring match.
  - `compute_retrieval_metrics(evaluations, latencies)`: Computes exact Recall@1, Recall@5, Mean Reciprocal Rank (MRR), and latency statistics (p50, p95, mean, min, max in milliseconds).

- **`embedding_bench/harness.py`**:
  - Benchmark coordinator orchestrating model inspection, sequential execution, memory protection, and winner selection.
  - `estimate_model_memory(model, client, available_models)`: Extracts model size from Ollama `/api/tags` and `/api/show` with fallback to known architecture specifications.
  - `select_winner(candidate_results)`: Multi-criteria decision engine evaluating Recall, Latency, and RAM footprint.
  - `run_embedding_benchmark(...)`: Core benchmark runner.
  - `main()`: Safe CLI entry point. Dormant by default (inspects dataset and Ollama without running embeddings); `--run` flag required to start the benchmark.

- **`embedding_bench/tests/test_embedding_bench.py`**:
  - Discoverable test suite with comprehensive unit and end-to-end mocked tests (zero network calls).

---

## 3. Strict Safety and Operational Guarantees

1. **100% Local Loopback Enforcement**:
   - The harness rejects any Ollama URL whose hostname is not `127.0.0.1`, `localhost`, or `::1`.
   - No external API or cloud endpoint can be contacted.

2. **No Model Downloads in this Worker Round**:
   - The harness queries `/api/tags` to check if a candidate model is already pulled.
   - If a candidate is absent, it is marked as `status: "not_pulled"` with an explanatory message.
   - The harness never calls `/api/pull` or executes `ollama pull`.

3. **No Model Deletions**:
   - The harness does not invoke `DELETE /api/delete` or `ollama rm`. All existing models remain untouched.

4. **Resident Model Protection at Start**:
   - At benchmark start, `/api/ps` is queried to record all models currently loaded in RAM/VRAM (`resident_at_start`).
   - Any model resident at start is never unloaded.
   - When sequentially evaluating candidates, non-resident candidates are cleanly unloaded after their evaluation via Ollama's `keep_alive: 0` API to reclaim RAM.

5. **Strictly Sequential Execution**:
   - Candidates are evaluated strictly one at a time (never in parallel) to keep RAM footprint low.

6. **Repo-Root Robust Path Resolution**:
   - The golden set is located dynamically relative to the package anchor and repo root, ensuring seamless execution regardless of the caller's current working directory.

7. **Dormant by Default**:
   - Running `python -m embedding_bench.harness` without `--run` only displays configuration, dataset statistics, and model availability; it does not trigger embeddings or benchmark calls.

---

## 4. Winner Selection Methodology

The harness selects a single winner based on a multi-criteria decision formula balancing retrieval accuracy, query latency, and memory footprint:

$$\text{Composite Score} = 0.50 \times \text{Quality} + 0.25 \times \text{Latency Score} + 0.25 \times \text{RAM Score}$$

Where:
- $\text{Quality} = 0.35 \times \text{Recall@5} + 0.15 \times \text{MRR}$ (higher is better)
- $\text{Latency Score} = \frac{\max(\text{p50}) - \text{p50}}{\max(\text{p50}) - \min(\text{p50}) + \epsilon}$ (lower latency yields higher score)
- $\text{RAM Score} = \frac{\max(\text{RAM}) - \text{RAM}}{\max(\text{RAM}) - \min(\text{RAM}) + \epsilon}$ (lower RAM yields higher score, where $\text{RAM} = \text{Model RAM} + \text{Vector RAM}$)

If only one candidate is pulled, that candidate is selected if its metrics pass validity checks. If no candidate is pulled, the winner is reported as `None` with status `not_pulled`.

---

## 5. Environmental Prerequisites

1. **Python Environment**:
   - Python 3.10+ (tested for Python 3.13 / 3.14).
   - Standard library (`json`, `urllib`, `math`, `statistics`, `dataclasses`, `pathlib`).
   - Optional: `numpy` (if installed, accelerates cosine retrieval; pure Python fallback is automatically used if absent).

2. **Local Ollama Daemon**:
   - Ollama service running on loopback: `http://127.0.0.1:11434`.
   - Accessible `/api/embed`, `/api/tags`, `/api/ps`, and `/api/show` endpoints.

3. **Candidate Models in Ollama** (to be pulled by orchestrator/user):
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen3-embedding:0.6b
   ollama pull embeddinggemma
   ```

4. **Golden Set Dataset**:
   - `eval/golden_set.jsonl` located in repo root (already present, 165 validated question/answer/source triples).

---

## 6. Exact Commands

### Command A: Safe Inspection / Status Check (Dormant, No Benchmark Run)
Inspects golden set statistics, checks loopback Ollama health, and reports candidate model availability without running any model or network requests:
```bash
python -m embedding_bench.harness
```
*(Can be executed from repository root or from any directory)*

### Command B: Run Discoverable Unit Tests (Mocked Local API, No Network Calls)
Executes all unit tests with mocked Ollama endpoints and synthetic datasets:
```bash
python -m unittest discover -s embedding_bench/tests -v
```

### Command C: Run Full Benchmark on Real Golden Set
When candidate models are pulled and ready to evaluate:
```bash
python -m embedding_bench.harness --run
```

### Command D: Run Benchmark with Custom Options (Optional)
Evaluate a specific subset of candidate models or limit query count:
```bash
python -m embedding_bench.harness --run --models nomic-embed-text,embeddinggemma --max-queries 50
```

---

## 7. Package File Inventory

| File | Purpose |
|------|---------|
| `embedding_bench/__init__.py` | Package initialization and public API exports |
| `embedding_bench/client.py` | Ollama loopback client, validation, model lifecycle control |
| `embedding_bench/index.py` | Exact cosine retrieval index and memory estimation |
| `embedding_bench/dataset.py` | Repo-root robust golden set loader and schema validator |
| `embedding_bench/metrics.py` | Recall@1, Recall@5, MRR, and p50/p95 latency computation |
| `embedding_bench/harness.py` | Sequential runner, model memory estimator, winner selector, safe CLI |
| `embedding_bench/tests/__init__.py` | Test package marker |
| `embedding_bench/tests/test_embedding_bench.py` | Discoverable unit tests with mocked Ollama API |
| `embedding_bench/REPORT.md` | Comprehensive documentation and execution guide |

## Orchestrator verification (2026-10-19)
- Project venv: `sqlite-vec==0.1.9`, `networkx==3.7`; Python 3.13.14, SQLite 3.50.4.
- `python -m compileall -q embedding_bench` passed; `python -m unittest discover -s embedding_bench/tests -v` passed 21 mocked/offline tests.
- Dormant CLI confirmed Ollama is loopback, `qwen2.5:1.5b-instruct` was resident before benchmark, and all three candidate embedding models were `not_pulled`. No embedding benchmark or model pull has been run yet. Preserve that pre-existing resident model.
- Use `python -m embedding_bench` for CLI help/dormant status; `python -m embedding_bench.harness --run` runs the explicit benchmark after candidate models are pulled.
- Evidence: `../evidence/embedding-compile.txt`, `../evidence/embedding-tests.txt`, `../evidence/embedding-dormant.txt`, and `../evidence/embedding-help.txt`.

## Orchestrator model benchmark

The real local Ollama run completed all 165 golden queries over 145 deduplicated answer passages. qwen2.5:1.5b-instruct was resident at start and remained resident; each of the three benchmark candidates was unloaded after its run, without deletion.

| Model | Recall@1 | Recall@5 | MRR | p50 | p95 | Estimated model + vector RAM |
|---|---:|---:|---:|---:|---:|---:|
| nomic-embed-text | 0.2727 | 0.3879 | 0.3186 | 137.04 ms | 333.28 ms | 262.05 MB |
| qwen3-embedding:0.6b | 0.2303 | 0.3636 | 0.2743 | 301.05 ms | 686.36 ms | 610.13 MB |
| embeddinggemma | 0.2182 | 0.3758 | 0.2783 | 109.84 ms | 249.57 ms | 593.52 MB |

The harness composite score selected `nomic-embed-text` (0.6480). This is a small, source-specific corpus result; recall@5 is under 0.39, so it is not sufficient by itself for a cutover. No model was installed into project configuration. Full output: `../evidence/embedding-benchmark-all.txt`.
