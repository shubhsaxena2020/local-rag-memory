# Round 2 orchestrator verification

`reranker/` is integrated into the project tree and uses the pinned real ONNX graph at `models/reranker/model_quint8_avx2.onnx` with `tokenizer.json`. The graph SHA256 is `c80a8b34256ea453093d612e3ac48d3d965a0c0a48c7906709af8b8e28461bf9`. Project venv dependencies: onnxruntime 1.31.0, tokenizers 0.23.3, NumPy 2.5.4. All 16 reranker tests and all 14 retrieval tests passed; compileall passed. Evidence: `../evidence/r2-reranker-tests.txt`, `../evidence/r2-retrieval-tests.txt`.

Real gated benchmark: 165 golden questions, 145 passages, `embeddinggemma` with the `gemma` formatter, strict relevance, top 5, 30-candidate cap, fact confidence 0.0 (gate open for every question), 256-token max length. The real cross-encoder had zero fallbacks.

| Arm | R@1 | R@5 | MRR | p50 total | p95 total | gate activations |
|---|---:|---:|---:|---:|---:|---:|
| No reranker | 0.2667 | 0.5576 | 0.3685 | 162.92 ms | 360.44 ms | 0 |
| Cross-encoder | 0.2848 | 0.4000 | 0.3271 | 1,236.10 ms | 2,727.91 ms | 165 |

The reranker raises R@1 by 0.0182, but lowers R@5 by 0.1576 and MRR by 0.0414, while adding 2,367.47 ms to p95. Drop it from cutover. The all-open gate measures its reranking effect on every query; a confidence-closed gate would yield zero activations and no quality change. Evidence: `../evidence/r2-reranker-benchmark.json` and `../evidence/r2-reranker-benchmark.txt`.

# W6 Reranker worker report (Round 3)

Local cross-encoder reranker wired to `retrieval.pipeline.RetrievalPipeline`, plus a no-reranker vs
real-reranker benchmark runner. No retrieval/storage/eval/corpus/config files were modified.

## Files (all under C:\Users\shubh\ai-memory-system)
- work/W6-reranker/reranker/onnx_reranker.py  - `CrossEncoderReranker`, `OnnxPairScorer`, `load_default_reranker`, `RerankerUnavailable`
- work/W6-reranker/reranker/benchmark.py      - benchmark runner (`python -m reranker.benchmark`)
- work/W6-reranker/reranker/tests/test_onnx_reranker.py (+ tests/__init__.py) - 16 tests, fake scorer/tokenizer/session only
- work/W6-reranker/reranker/__init__.py, requirements.txt (onnxruntime>=1.17, tokenizers>=0.15, numpy>=1.24)
- evidence/w6-reranker-tests.txt, evidence/w6-reranker-real-onnx-smoke.txt

## Design
- `wants_query = True`, so the pipeline calls `reranker(candidates, query=query)`.
- Pipeline keeps ownership of the gate (top fused confidence < threshold), the cap (<=30) and fallback.
  The reranker also caps at `max_candidates` (<=30) and returns the uncapped tail untouched, so output is always a permutation.
- Scoring: (query, passage) pairs batched (16) through ONNX; tokenizer.json truncation 256 tokens, dynamic padding.
  Handles 1-logit and 2-class heads. Sort is stable: score desc, ties keep fused RRF order. Adds `rerank_score`; inputs not mutated.
- Non-finite/wrong-count scores, missing query or scorer exceptions raise -> pipeline counts `rerank_fallbacks` and keeps fused order.
- Missing model/tokenizer/dependency raises `RerankerUnavailable`; the benchmark exits 2 rather than faking the real arm.

## Commands and results
- `cd work/W6-reranker && python -m unittest reranker.tests.test_onnx_reranker -v` -> Ran 16 tests, OK
- `python -m compileall -q reranker` -> COMPILE_OK
- `python -m unittest discover -s retrieval/tests -t .` (project root) -> 13 tests OK (existing pipeline untouched)
- Real ONNX smoke (4 hand-written passages, query "capital of France"): model loads in 3.4 s, inputs
  input_ids/attention_mask/token_type_ids, output `logits`; Paris passage ranked first (9.33) vs unrelated (-11.1);
  4-passage rerank 21 ms; repeat run deterministic. See evidence/w6-reranker-real-onnx-smoke.txt.
- `python -m reranker.benchmark --model-dir C:/nonexistent` -> error, exit 2 (no fake substitution).
- Wiring-only smoke of `compare()` with the REAL ONNX model but a toy hash-bag embedder, 60 passages, 10 queries:
  gate activations 0 vs 10, no fallbacks, p50 30 ms vs 734 ms. NOT benchmark evidence (toy embedder, tiny sample, not saved).

## Real benchmark (NOT yet run - needs live embedder; prohibited this round)
    cd C:\Users\shubh\ai-memory-system\work\W6-reranker
    python -m reranker.benchmark --out ..\..\evidence\w6-reranker-benchmark.json
Builds one SQLite store from eval/golden_set.jsonl (nomic-embed-text via Ollama, 768-d), runs both arms on the identical queries
with cache off and a discarded warm-up query; reports R@1, R@5, MRR (strict eval harness rule), total per-search p50/p95 latency
(embedding+search+fusion+gate+rerank), gate activations and fallbacks. `--fact-confidence` (default 0.0 = gate always open) sets the stored
confidence that drives the gate; 1.0 gives production-like closed gate (0 activations).

## Model requirements
- models/reranker/model_quint8_avx2.onnx (23.2 MB, cross-encoder/ms-marco-MiniLM-L6-v2, Apache-2.0 per orchestrator) and tokenizer.json (present, not re-verified by me beyond load/inference).
- Python 3.14 env: onnxruntime 1.27.0, numpy 2.4.6 already present; tokenizers 0.23.3 installed by me via pip (declared in requirements.txt).

## Limitations
- No real golden-set quality/latency numbers yet; the smoke latency suggests ~0.7 s p50 for 30 candidates at 256 tokens on this CPU,
  so the gate matters. Real numbers must come from the benchmark run.
- Golden facts have no genuine confidence signal, so the gate's behaviour on real data depends on `--fact-confidence`.
- ms-marco model is English, passage-search trained, max 256 tokens used (longer passages truncated); scores are logits, not probabilities.
- Package lives in work/W6-reranker/ (not integrated into the main tree); uses sys.path discovery of the project root.
- Pipeline reports `score` = rrf_score even after reranking; `rerank_score` is on the (copied) candidate dicts but the pipeline's output copies keep it only because it copies the dict (not changed here).
