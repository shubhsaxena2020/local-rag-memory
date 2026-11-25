"""Benchmark: no-reranker vs. real local cross-encoder on IDENTICAL golden queries.

Same corpus, same SQLite store, same embedder, same queries, cache disabled; only the pipeline's
``reranker`` differs. Metrics come from ``eval.eval_harness.evaluate_retrieval`` (strict relevance
rule): R@1, R@5, MRR, and per-search *total* latency p50/p95 (query embedding + keyword/vector
search + fusion + gate + rerank). Also reports gate activations / fallbacks from the pipeline.

Gate note: the pipeline opens its gate when the TOP fused fact's ``confidence`` is below
``low_confidence_threshold``. Golden facts carry no real confidence signal, so ``--fact-confidence``
(default 0.0 -> gate always open, i.e. measures the reranker's ceiling) controls what is stored.
Use ``--fact-confidence 1.0`` to see production-like behaviour with the gate closed (0 activations).

Requires a live embedder (Ollama ``nomic-embed-text`` by default, 768-d to match storage) and the
real model files in ``--model-dir`` (default ``<root>/models/reranker``). It refuses to run
the "real" arm if the model files are missing; it never substitutes a fake scorer.

    python -m reranker.benchmark --limit 0 --out evidence/w6-reranker-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
PKG_PARENT = HERE.parent
ROOT = next((p for p in HERE.parents if (p / "retrieval" / "pipeline.py").is_file()), None)
for _p in (ROOT, PKG_PARENT):
    if _p is not None and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from eval.eval_harness import evaluate_retrieval, load_jsonl  # noqa: E402
from embedding_bench.formatting import format_document, format_query  # noqa: E402
from retrieval.pipeline import RetrievalPipeline  # noqa: E402
from storage import MemoryStorage  # noqa: E402

from reranker.onnx_reranker import CrossEncoderReranker, RerankerUnavailable, load_default_reranker  # noqa: E402

ARMS = ("no_reranker", "cross_encoder")


def run_arm(store: MemoryStorage, embedder: Callable[[str], Any], reranker: Optional[Callable],
            golden: List[Dict[str, Any]], *, top_k: int, threshold: float, warmup: bool = True) -> Dict[str, Any]:
    pipe = RetrievalPipeline(store, embedder=embedder, reranker=reranker, cache_size=0,
                             low_confidence_threshold=threshold)
    if warmup and golden:  # first ONNX/embedder call is slow; exclude it, then reset counters
        pipe.search(golden[0]["question"], top_k)
        pipe.rerank_gate_activations = 0
        pipe.rerank_fallbacks = 0
    metrics = evaluate_retrieval(golden, lambda q, k: pipe.search(q, k), top_k=top_k)
    return {
        "recall_at_1": metrics["recall_at_1"], "recall_at_5": metrics["recall_at_5"], "mrr": metrics["mrr"],
        "total_latency_p50_ms": metrics["latency_p50_ms"], "total_latency_p95_ms": metrics["latency_p95_ms"],
        "gate_activations": pipe.rerank_gate_activations, "rerank_fallbacks": pipe.rerank_fallbacks,
        "queries": metrics["total_questions"],
    }


def compare(store: MemoryStorage, embedder: Callable[[str], Any], reranker: Callable,
            golden: List[Dict[str, Any]], *, top_k: int = 5, threshold: float = 0.5) -> Dict[str, Any]:
    """Run both arms on the identical ``golden`` list and return a result dict."""
    base = run_arm(store, embedder, None, golden, top_k=top_k, threshold=threshold)
    real = run_arm(store, embedder, reranker, golden, top_k=top_k, threshold=threshold)
    delta = {k: (real[k] - base[k]) for k in ("recall_at_1", "recall_at_5", "mrr",
                                                 "total_latency_p50_ms", "total_latency_p95_ms")
             if real[k] is not None and base[k] is not None}
    return {"no_reranker": base, "cross_encoder": real, "delta_cross_encoder_minus_none": delta}


def build_store(path: Path, corpus: list, embed_batch: Callable[[List[str]], List[List[float]]],
                confidence: float, formatter: str) -> MemoryStorage:
    store = MemoryStorage(path)
    store.initialize_schema()
    for start in range(0, len(corpus), 16):
        batch = corpus[start:start + 16]
        texts = [format_document(formatter, d.text, d.source) for d in batch]
        for doc, vec in zip(batch, embed_batch(texts)):
            store.save_fact(doc.text, source=doc.source, writer="benchmark", tags="",
                            confidence=confidence, vector=vec)
    return store


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--golden", default="eval/golden_set.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="first N golden queries (0 = all)")
    ap.add_argument("--model-dir", default=str((ROOT or Path(".")) / "models" / "reranker"))
    ap.add_argument("--embed-model", default="nomic-embed-text")
    ap.add_argument("--embed-formatter", choices=("raw", "nomic", "qwen3", "gemma", "bge-m3"), default=None)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fact-confidence", type=float, default=0.0)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    try:
        reranker = load_default_reranker(args.model_dir, max_length=args.max_length)
    except RerankerUnavailable as exc:
        print(f"ERROR: real reranker unavailable ({exc}); not substituting a fake.", file=sys.stderr)
        return 2

    from embedding_bench.client import OllamaEmbeddingClient
    from embedding_bench.dataset import load_golden_set

    client = OllamaEmbeddingClient(timeout=60)
    resident_start = client.get_running_models()
    corpus, _ = load_golden_set(args.golden)
    golden = load_jsonl(args.golden)
    if args.limit:
        golden = golden[:args.limit]
    model = args.embed_model
    formatter = args.embed_formatter or {
        "nomic-embed-text": "nomic", "qwen3-embedding:0.6b": "qwen3",
        "embeddinggemma": "gemma", "bge-m3": "bge-m3",
    }.get(model, "raw")
    try:
        with tempfile.TemporaryDirectory(prefix="w6-rerank-") as td:
            store = build_store(Path(td) / "eval.db", corpus,
                                lambda texts: client.embed(model, texts)[0], args.fact_confidence, formatter)
            try:
                result = compare(store, lambda q: client.embed(model, format_query(formatter, q))[0][0], reranker, golden,
                                 top_k=args.top_k, threshold=args.threshold)
            finally:
                store.close()
    finally:
        if model not in resident_start and f"{model}:latest" not in resident_start:
            client.unload_model(model)
    result["config"] = {"embed_model": model, "embed_formatter": formatter, "reranker_model_dir": str(args.model_dir),
                        "corpus_passages": len(corpus), "fact_confidence": args.fact_confidence,
                        "gate_threshold": args.threshold, "top_k": args.top_k, "max_length": args.max_length}
    text = json.dumps(result, indent=2, sort_keys=True)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
