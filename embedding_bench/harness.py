"""
Embedding Model Benchmark Harness for AI Memory System.

Coordinates sequential benchmarking of candidate embedding models against the golden set:
- nomic-embed-text (task prefixes) plus a raw-text ablation (nomic-embed-text@raw)
- qwen3-embedding:0.6b (instruction on queries only)
- embeddinggemma (task/title prompts)
- bge-m3 (raw text)

Corrected relevance: strict exact target passage only (see metrics.is_relevant_hit).
Input formats per model live in formatting.py with primary-doc links.

Guarantees:
- 100% local loopback communication.
- No model downloads in this round (absent candidates reported as not_pulled).
- No model deletions.
- Never unloads models that were resident at start.
- Strictly sequential candidate execution to keep RAM footprint low.
- Validates vector dimensions and finite values.
- Builds an exact cosine retrieval index.
- Computes Recall@1, Recall@5, MRR, p50/p95 latency, and estimates model/vector memory.
- Determines one winner based on recall, latency, and RAM trade-offs.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional, Set

from .client import DEFAULT_OLLAMA_URL, OllamaEmbeddingClient
from .formatting import Variant, format_document, format_query, resolve_variants
from .dataset import CorpusRecord, GoldenQuery, load_golden_set, resolve_golden_set_path
from .index import CosineIndex
from .metrics import compute_retrieval_metrics, evaluate_single_query

CANDIDATE_MODELS = [
    "nomic-embed-text",
    "qwen3-embedding:0.6b",
    "embeddinggemma",
    "bge-m3",
]

# Variant keys run by default (adds the Nomic raw-vs-prefixed ablation).
DEFAULT_VARIANT_KEYS = [
    "nomic-embed-text",
    "nomic-embed-text@raw",
    "qwen3-embedding:0.6b",
    "embeddinggemma",
    "bge-m3",
]

KNOWN_MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "nomic-embed-text": {
        "parameters": "137M",
        "default_dim": 768,
        "typical_ram_mb": 274.0,
        "description": "Nomic Embed Text v1.5 (8192 context, Matryoshka-capable)",
    },
    "qwen3-embedding:0.6b": {
        "parameters": "596M",
        "default_dim": 1024,
        "typical_ram_mb": 650.0,
        "description": "Qwen 3 Embedding 0.6B small dense retriever",
    },
    "bge-m3": {
        "parameters": "567M",
        "default_dim": 1024,
        "typical_ram_mb": 1200.0,
        "description": "BAAI BGE-M3 dense retriever (XLM-RoBERTa, 8192 context)",
    },
    "embeddinggemma": {
        "parameters": "300M",
        "default_dim": 768,
        "typical_ram_mb": 400.0,
        "description": "Gemma Embedding architecture (Google)",
    },
}


def estimate_model_memory(
    model_name: str,
    client: OllamaEmbeddingClient,
    available_models: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Estimate model RAM footprint from local Ollama metadata or known specs."""
    # Check Ollama tags for reported file/model size
    reported_size_bytes = 0
    for m in available_models:
        name = m.get("name", "")
        if name == model_name or name.startswith(f"{model_name}:") or model_name.startswith(f"{name}:"):
            reported_size_bytes = m.get("size", 0)
            break

    # Also inspect /api/show if possible
    info = client.get_model_info(model_name)
    details = info.get("details", {})
    param_size = details.get("parameter_size", "")
    quant = details.get("quantization_level", "")

    fallback = KNOWN_MODEL_SPECS.get(model_name, {})
    if not param_size and "parameters" in fallback:
        param_size = fallback["parameters"]

    if reported_size_bytes > 0:
        ram_mb = round(reported_size_bytes / (1024 * 1024), 2)
    else:
        ram_mb = fallback.get("typical_ram_mb", 350.0)

    return {
        "model_name": model_name,
        "parameter_size": param_size or "unknown",
        "quantization": quant or "unknown",
        "size_bytes": reported_size_bytes,
        "model_ram_mb": ram_mb,
    }


def select_winner(candidate_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Select by strict retrieval quality, then use latency and RAM as tie-breakers.

    Recall@5 is primary and MRR is secondary. Latency and RAM only break quality
    ties, so a fast but materially less relevant model cannot win. Ablation
    variants (ablation_of set) are reported but never win.
    """
    completed = {
        name: res
        for name, res in candidate_results.items()
        if res.get("status") == "completed" and res.get("metrics") and not res.get("ablation_of")
    }

    if not completed:
        return {
            "winner": None,
            "composite_score": None,
            "rankings": [],
            "reason": "No candidate models were completed (candidates are not_pulled or encountered errors).",
        }

    if len(completed) == 1:
        winner_name = next(iter(completed.keys()))
        res = completed[winner_name]
        return {
            "winner": winner_name,
            "composite_score": 1.0,
            "rankings": [{"model": winner_name, "composite_score": 1.0, "metrics": res["metrics"]}],
            "reason": f"Only candidate '{winner_name}' completed the benchmark.",
        }

    # Multiple candidates completed: calculate normalized metrics
    models = list(completed.keys())
    recalls_5 = [completed[m]["metrics"]["recall_at_5"] for m in models]
    mrrs = [completed[m]["metrics"]["mrr"] for m in models]
    latencies = [completed[m]["metrics"]["latency_p50_ms"] for m in models]
    rams = [completed[m]["total_ram_mb"] for m in models]

    min_lat, max_lat = min(latencies), max(latencies)
    min_ram, max_ram = min(rams), max(rams)

    rankings: List[Dict[str, Any]] = []
    for idx, model in enumerate(models):
        rec5 = recalls_5[idx]
        mrr = mrrs[idx]
        lat = latencies[idx]
        ram = rams[idx]

        # Normalized latency (lower is better -> higher score)
        if max_lat > min_lat:
            norm_lat_score = (max_lat - lat) / (max_lat - min_lat)
        else:
            norm_lat_score = 1.0

        # Normalized RAM (lower is better -> higher score)
        if max_ram > min_ram:
            norm_ram_score = (max_ram - ram) / (max_ram - min_ram)
        else:
            norm_ram_score = 1.0

        quality_score = 0.35 * rec5 + 0.15 * mrr
        composite = quality_score + 0.25 * norm_lat_score + 0.25 * norm_ram_score

        rankings.append({
            "model": model,
            "composite_score": round(composite, 4),
            "quality_score": round(quality_score, 4),
            "latency_score": round(norm_lat_score, 4),
            "ram_score": round(norm_ram_score, 4),
            "recall_at_5": rec5,
            "mrr": mrr,
            "latency_p50_ms": lat,
            "total_ram_mb": ram,
        })

    # The quality gates prioritize strict retrieval; efficiency only resolves ties.
    rankings.sort(key=lambda r: (r["recall_at_5"], r["mrr"], -r["latency_p50_ms"], -r["total_ram_mb"]), reverse=True)
    winner = rankings[0]

    reason = (
        f"Model '{winner['model']}' ranked first by strict Recall@5, then MRR "
        f"(Recall@5: {winner['recall_at_5']}, MRR: {winner['mrr']}); "
        f"p50 latency: {winner['latency_p50_ms']}ms, total RAM: {winner['total_ram_mb']:.1f}MB."
    )

    return {
        "winner": winner["model"],
        "composite_score": winner["composite_score"],
        "rankings": rankings,
        "reason": reason,
    }


def benchmark_single_candidate(
    model: str,
    client: OllamaEmbeddingClient,
    corpus_records: List[CorpusRecord],
    queries: List[GoldenQuery],
    resident_at_start: Set[str],
    available_models: List[Dict[str, Any]],
    batch_size: int = 16,
    formatter: str = "raw",
    unload_after: bool = True,
) -> Dict[str, Any]:
    """Benchmark a single candidate model sequentially and manage memory safely.

    Documents and queries are rewritten by `formatter` (formatting.FORMATTERS) before
    embedding; ranking and strict relevance still use the original passage ids.
    unload_after=False lets the caller defer unloading (e.g. between ablation variants
    of one model). Models resident at start are never unloaded either way.
    """
    # 1. Embed corpus records in batches (documents formatted per model docs)
    corpus_texts = [format_document(formatter, doc.text, doc.source) for doc in corpus_records]
    corpus_vectors: List[List[float]] = []

    index = CosineIndex()
    t_start_index = time.perf_counter()

    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i : i + batch_size]
        vecs, _ = client.embed(model, batch)
        corpus_vectors.extend(vecs)

    # 2. Build index
    for idx, doc in enumerate(corpus_records):
        index.add(
            doc_id=doc.doc_id,
            vector=corpus_vectors[idx],
            text=doc.text,
            source=doc.source,
            original_ids=doc.original_ids,
        )
    index_build_sec = time.perf_counter() - t_start_index

    # 3. Evaluate queries sequentially
    query_evaluations: List[Dict[str, Any]] = []
    latencies_ms: List[float] = []

    for query in queries:
        # Time the question embedding
        t0 = time.perf_counter()
        q_vecs, _ = client.embed(model, format_query(formatter, query.question))
        t_embed = time.perf_counter() - t0
        q_vec = q_vecs[0]

        # Time the cosine retrieval
        t1 = time.perf_counter()
        results = index.search(q_vec, top_k=5)
        t_search = time.perf_counter() - t1

        total_query_lat_ms = (t_embed + t_search) * 1000.0
        latencies_ms.append(total_query_lat_ms)

        eval_res = evaluate_single_query(results, query)
        eval_res["latency_ms"] = total_query_lat_ms
        query_evaluations.append(eval_res)

    # 4. Compute metrics
    metrics = compute_retrieval_metrics(query_evaluations, latencies_ms)
    metrics["index_build_seconds"] = round(index_build_sec, 3)

    # 5. Memory estimation
    vector_mem = index.estimate_memory()
    model_mem = estimate_model_memory(model, client, available_models)
    total_ram_mb = round(model_mem["model_ram_mb"] + vector_mem["total_ram_mb"], 2)

    # 6. Strictly sequential RAM management:
    # Never unload a model that was resident at start!
    # Unload candidate ONLY if it was NOT resident at start.
    unloaded = False
    if model in resident_at_start or f"{model}:latest" in resident_at_start:
        unload_status = "kept_loaded_resident_at_start"
    elif not unload_after:
        unload_status = "unload_deferred"
    else:
        unloaded = client.unload_model(model)
        unload_status = "unloaded_to_reclaim_ram" if unloaded else "unload_failed"

    return {
        "model": model,
        "formatter": formatter,
        "relevance_mode": "strict",
        "status": "completed",
        "metrics": metrics,
        "vector_memory": vector_mem,
        "model_memory": model_mem,
        "total_ram_mb": total_ram_mb,
        "dimension": index.dimension,
        "unloaded": unloaded,
        "unload_status": unload_status,
    }


def run_embedding_benchmark(
    golden_set_path: Optional[str] = None,
    candidates: Optional[List[str]] = None,
    base_url: str = DEFAULT_OLLAMA_URL,
    max_queries: Optional[int] = None,
    batch_size: int = 16,
) -> Dict[str, Any]:
    """Execute the embedding benchmark on local Ollama sequentially.

    Checks:
    - Loopback Ollama availability.
    - Captures resident models at start (to protect them from unload).
    - Checks candidate availability in /api/tags without pulling.
    - Evaluates candidates sequentially.
    - Picks one winner by strict Recall@5 and MRR, using latency/RAM only as tie-breakers.
    """
    client = OllamaEmbeddingClient(base_url=base_url)

    # Health check
    health = client.check_health()
    if not health.get("connected"):
        raise ConnectionError(
            f"Ollama is not accessible at loopback URL {base_url}. Error: {health.get('error')}"
        )

    # Inspect resident models at start to enforce preservation
    resident_at_start = client.get_running_models()

    # Inspect available models
    available_models = client.get_available_models()
    available_names = client.get_available_model_names()

    # Load golden set
    resolved_path = resolve_golden_set_path(golden_set_path)
    corpus_records, queries = load_golden_set(resolved_path)

    if max_queries and max_queries > 0:
        queries = queries[:max_queries]

    # candidates=None -> every variant (4 models + Nomic raw ablation); tokens are variant
    # keys (e.g. "nomic-embed-text@raw") or plain Ollama model names (raw formatter).
    variants = resolve_variants(candidates)
    candidate_results: Dict[str, Dict[str, Any]] = {}

    # Sequential execution loop (one variant at a time; never parallel)
    for pos, variant in enumerate(variants):
        candidate = variant.model
        is_pulled = (
            candidate in available_names
            or f"{candidate}:latest" in available_names
            or any(m.get("name", "").startswith(f"{candidate}:") for m in available_models)
        )

        if not is_pulled:
            candidate_results[variant.key] = {
                "model": candidate,
                "variant": variant.key,
                "formatter": variant.formatter,
                "ablation_of": variant.ablation_of,
                "status": "not_pulled",
                "reason": (
                    f"Model '{candidate}' is not present in local Ollama tags. "
                    "Skipping download in this round per non-destructive worker constraint."
                ),
                "metrics": None,
                "total_ram_mb": None,
            }
            continue

        # Keep the model loaded while later variants of the same model still need it.
        more_for_model = any(v.model == candidate for v in variants[pos + 1:])

        try:
            res = benchmark_single_candidate(
                model=candidate,
                client=client,
                corpus_records=corpus_records,
                queries=queries,
                resident_at_start=resident_at_start,
                available_models=available_models,
                batch_size=batch_size,
                formatter=variant.formatter,
                unload_after=not more_for_model,
            )
            res["variant"] = variant.key
            res["ablation_of"] = variant.ablation_of
            candidate_results[variant.key] = res
        except Exception as e:
            candidate_results[variant.key] = {
                "model": candidate,
                "variant": variant.key,
                "formatter": variant.formatter,
                "ablation_of": variant.ablation_of,
                "status": "error",
                "error": str(e),
                "metrics": None,
                "total_ram_mb": None,
            }

    # Select winner
    winner_info = select_winner(candidate_results)

    return {
        "timestamp": time.time(),
        "ollama_url": base_url,
        "ollama_version": health.get("version", "unknown"),
        "resident_at_start": list(resident_at_start),
        "golden_set_path": str(resolved_path),
        "relevance_mode": "strict",
        "corpus_documents": len(corpus_records),
        "total_queries": len(queries),
        "candidates": candidate_results,
        "winner_summary": winner_info,
    }


def main():
    """Safe dormant CLI entry point.

    Default invocation inspects dataset and Ollama status without running the benchmark.
    Pass --run to execute the benchmark explicitly.
    """
    parser = argparse.ArgumentParser(description="Embedding Model Benchmark Harness (W2)")
    parser.add_argument("--run", action="store_true", help="Execute the real benchmark")
    parser.add_argument("--golden-set", type=str, default=None, help="Custom path to golden_set.jsonl")
    parser.add_argument("--base-url", type=str, default=DEFAULT_OLLAMA_URL, help="Ollama loopback URL")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated variant keys/model names (default: all incl. nomic-embed-text@raw)")
    parser.add_argument("--max-queries", type=int, default=None, help="Limit number of queries")
    args = parser.parse_args()

    print("=== Embedding Benchmark Harness (AI Memory System W2) ===")
    try:
        resolved_path = resolve_golden_set_path(args.golden_set)
        corpus, queries = load_golden_set(resolved_path)
        print(f"Golden Set Path: {resolved_path}")
        print(f"Corpus Passages: {len(corpus)}")
        print(f"Total Queries:   {len(queries)}")
    except Exception as e:
        print(f"Dataset resolution note: {e}")

    client = OllamaEmbeddingClient(base_url=args.base_url)
    health = client.check_health()
    print(f"Ollama Loopback: {args.base_url} (connected: {health.get('connected')})")

    if health.get("connected"):
        running = client.get_running_models()
        available = client.get_available_model_names()
        print(f"Resident Models at Start: {sorted(running) if running else 'None'}")
        print(f"Locally Available Models: {sorted(available) if available else 'None'}")

        print("\nCandidate Status:")
        candidates = [v.model for v in resolve_variants([m for m in args.models.split(",")] if args.models else None)]
        for c in dict.fromkeys(candidates):
            status = "available" if (c in available or f"{c}:latest" in available) else "not_pulled"
            print(f"  - {c}: {status}")

    if not args.run:
        print("\n[DORMANT MODE] Benchmark was not executed.")
        print("To run the benchmark when models are pulled, use:")
        print("  python -m embedding_bench.harness --run")
        return

    print("\nExecuting benchmark sequentially...")
    results = run_embedding_benchmark(
        golden_set_path=args.golden_set,
        candidates=[m.strip() for m in args.models.split(",")] if args.models else None,
        base_url=args.base_url,
        max_queries=args.max_queries,
    )

    print("\n=== Benchmark Results ===")
    for model, res in results["candidates"].items():
        print(f"\nModel: {model} [Status: {res.get('status')}]")
        if res.get("status") == "completed":
            m = res["metrics"]
            print(f"  Recall@1: {m['recall_at_1']}")
            print(f"  Recall@5: {m['recall_at_5']}")
            print(f"  MRR:      {m['mrr']}")
            print(f"  p50 Lat:  {m['latency_p50_ms']} ms")
            print(f"  p95 Lat:  {m['latency_p95_ms']} ms")
            print(f"  RAM Est:  {res['total_ram_mb']} MB (Model: {res['model_memory']['model_ram_mb']} MB, Vec: {res['vector_memory']['total_ram_mb']} MB)")
            print(f"  Unload:   {res['unload_status']}")
        elif res.get("status") == "not_pulled":
            print(f"  Reason:   {res.get('reason')}")

    print("\nWinner Decision:")
    print(f"  Winner: {results['winner_summary'].get('winner')}")
    print(f"  Reason: {results['winner_summary'].get('reason')}")


if __name__ == "__main__":
    main()
