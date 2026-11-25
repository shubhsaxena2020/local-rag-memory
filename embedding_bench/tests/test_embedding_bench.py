"""
Unit tests for the embedding_bench package.

All network calls are strictly mocked with zero network traffic.
Discoverable via:
    python -m unittest discover -s embedding_bench/tests -v
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repository root is on sys.path for direct or discovered execution
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from embedding_bench.client import (
    _NoRedirectHandler,
    OllamaEmbeddingClient,
    validate_embedding,
    validate_loopback_url,
)
from embedding_bench.dataset import (
    CorpusRecord,
    GoldenQuery,
    load_golden_set,
    resolve_golden_set_path,
)
from embedding_bench.harness import (
    CANDIDATE_MODELS,
    benchmark_single_candidate,
    estimate_model_memory,
    run_embedding_benchmark,
    select_winner,
)
from embedding_bench.index import CosineIndex, SearchResult
from embedding_bench.metrics import (
    calculate_percentile,
    compute_retrieval_metrics,
    evaluate_single_query,
    is_relevant_hit,
)


class TestClientLoopbackAndValidation(unittest.TestCase):
    """Test local loopback enforcement and vector dimension/finite validation."""

    def test_loopback_url_allowed_hosts(self):
        valid_urls = [
            "http://127.0.0.1:11434",
            "http://localhost:11434",
            "http://127.0.0.1",
            "http://localhost:8080",
        ]
        for url in valid_urls:
            try:
                validate_loopback_url(url)
            except Exception as e:
                self.fail(f"validate_loopback_url unexpectedly raised on valid URL '{url}': {e}")

    def test_loopback_url_rejects_remote_and_non_http(self):
        invalid_urls = [
            "https://127.0.0.1:11434",
            "http://192.168.1.100:11434",
            "http://evil.com:11434",
            "http://127.0.0.1.attacker.com:11434",
            "ftp://localhost:11434",
            "http://user@localhost:11434",
            "http://localhost:11434/?token=x",
            "http://localhost:11434/#fragment",
            "http://localhost:notaport",
            "http://localhost:0",
            "",
            "   ",
        ]
        for url in invalid_urls:
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_loopback_url(url)

    def test_loopback_redirects_are_not_followed(self):
        handler = _NoRedirectHandler()
        self.assertIsNone(
            handler.redirect_request(
                None, None, 302, "Found", None, "http://127.0.0.1:11434/redirect"
            )
        )

    def test_validate_embedding_valid(self):
        vec = [0.1, -0.25, 0.0, 1.5]
        dim = validate_embedding(vec)
        self.assertEqual(dim, 4)
        dim2 = validate_embedding(vec, expected_dim=4)
        self.assertEqual(dim2, 4)

    def test_validate_embedding_rejects_empty(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            validate_embedding([])

    def test_validate_embedding_rejects_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            validate_embedding([0.1, 0.2], expected_dim=3)

    def test_validate_embedding_rejects_non_numeric(self):
        with self.assertRaisesRegex(ValueError, "not numeric"):
            validate_embedding([0.1, "bad", 0.3])

    def test_validate_embedding_rejects_nan(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_embedding([0.1, float("nan"), 0.3])

    def test_embed_passes_keep_alive_to_ollama(self):
        client = OllamaEmbeddingClient()
        with patch.object(client, "_request", return_value={"embeddings": [[0.1]]}) as request:
            vectors, _ = client.embed("embeddinggemma", "text", keep_alive="30m")
        self.assertEqual(vectors, [[0.1]])
        request.assert_called_once_with(
            "/api/embed",
            payload={"model": "embeddinggemma", "input": "text", "truncate": True, "keep_alive": "30m"},
            method="POST",
        )

    def test_embed_rejects_invalid_keep_alive_type(self):
        with self.assertRaisesRegex(ValueError, "keep_alive"):
            OllamaEmbeddingClient().embed("embeddinggemma", "text", keep_alive=True)

    def test_validate_embedding_rejects_infinities(self):
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_embedding([0.1, float("inf"), 0.3])
        with self.assertRaisesRegex(ValueError, "not finite"):
            validate_embedding([0.1, float("-inf"), 0.3])


class TestCosineIndex(unittest.TestCase):
    """Test exact cosine retrieval mathematics, ranking, and memory estimation."""

    def test_exact_cosine_similarity_math(self):
        index = CosineIndex(dimension=2)
        # doc1 is along x axis
        index.add(doc_id="doc1", vector=[2.0, 0.0], text="item 1")
        # doc2 is along y axis (orthogonal, cosine = 0.0)
        index.add(doc_id="doc2", vector=[0.0, 3.0], text="item 2")
        # doc3 is along negative x axis (opposite, cosine = -1.0)
        index.add(doc_id="doc3", vector=[-5.0, 0.0], text="item 3")
        # doc4 is at 45 degrees [1, 1] (cosine = 1/sqrt(2) ~ 0.7071)
        index.add(doc_id="doc4", vector=[1.0, 1.0], text="item 4")

        # Query along x axis: [1.0, 0.0]
        results = index.search([1.0, 0.0], top_k=4)

        self.assertEqual(len(results), 4)
        # Rank 1: doc1 (score ~ 1.0)
        self.assertEqual(results[0].doc_id, "doc1")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)

        # Rank 2: doc4 (score ~ 0.7071)
        self.assertEqual(results[1].doc_id, "doc4")
        self.assertAlmostEqual(results[1].score, 1.0 / math.sqrt(2), places=4)

        # Rank 3: doc2 (score ~ 0.0)
        self.assertEqual(results[2].doc_id, "doc2")
        self.assertAlmostEqual(results[2].score, 0.0, places=4)

        # Rank 4: doc3 (score ~ -1.0)
        self.assertEqual(results[3].doc_id, "doc3")
        self.assertAlmostEqual(results[3].score, -1.0, places=4)

    def test_zero_vector_handling(self):
        index = CosineIndex(dimension=2)
        index.add(doc_id="zero", vector=[0.0, 0.0], text="zero vector")
        index.add(doc_id="unit", vector=[1.0, 0.0], text="unit vector")

        results = index.search([1.0, 0.0], top_k=2)
        self.assertEqual(results[0].doc_id, "unit")
        self.assertAlmostEqual(results[0].score, 1.0, places=4)
        self.assertEqual(results[1].doc_id, "zero")
        self.assertAlmostEqual(results[1].score, 0.0, places=4)

    def test_memory_estimation(self):
        index = CosineIndex(dimension=768)
        for i in range(100):
            vec = [0.01 * (j % 10) for j in range(768)]
            index.add(doc_id=f"doc_{i}", vector=vec, text="sample document text", source="doc.md")

        mem = index.estimate_memory()
        self.assertEqual(mem["num_vectors"], 100)
        self.assertEqual(mem["dimension"], 768)
        # 100 * 768 * 4 = 307200 bytes
        self.assertEqual(mem["vector_bytes_float32"], 307200)
        # 100 * 768 * 8 = 614400 bytes
        self.assertEqual(mem["vector_bytes_float64"], 614400)
        self.assertGreater(mem["vector_ram_mb"], 0.28)
        self.assertLess(mem["vector_ram_mb"], 0.31)


class TestDatasetAndPathResolution(unittest.TestCase):
    """Test golden set schema validation and path resolution."""

    def test_resolve_real_golden_set_path(self):
        # Must resolve the real golden set under eval/
        resolved = resolve_golden_set_path()
        self.assertTrue(resolved.is_file(), f"Path did not resolve to a file: {resolved}")
        self.assertTrue(resolved.name.endswith(".jsonl"))

    def test_load_golden_set_valid_tempfile(self):
        records = [
            {"id": 1, "question": "What is A?", "answer": "Alpha is first", "source": "alpha.md"},
            {"id": 2, "question": "What is B?", "answer": "Beta is second", "source": "beta.md"},
            {"id": 3, "question": "How about A again?", "answer": "Alpha is first", "source": "alpha.md"},
        ]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as tf:
            for r in records:
                tf.write(json.dumps(r) + "\n")
            temp_path = Path(tf.name)

        try:
            corpus, queries = load_golden_set(temp_path, deduplicate_answers=True)
            # Deduplication should yield 2 unique corpus passages from 3 lines
            self.assertEqual(len(corpus), 2)
            self.assertEqual(len(queries), 3)

            # Query 1 and 3 should point to the same doc_id
            self.assertEqual(queries[0].target_doc_id, queries[2].target_doc_id)
            self.assertNotEqual(queries[0].target_doc_id, queries[1].target_doc_id)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_golden_set_rejects_missing_question(self):
        records = [{"id": 1, "answer": "only answer"}]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as tf:
            for r in records:
                tf.write(json.dumps(r) + "\n")
            temp_path = Path(tf.name)

        try:
            with self.assertRaisesRegex(ValueError, "missing or empty 'question'"):
                load_golden_set(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_load_golden_set_rejects_missing_answer(self):
        records = [{"id": 1, "question": "only question"}]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", encoding="utf-8", delete=False) as tf:
            for r in records:
                tf.write(json.dumps(r) + "\n")
            temp_path = Path(tf.name)

        try:
            with self.assertRaisesRegex(ValueError, "missing or empty 'answer'"):
                load_golden_set(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)


class TestMetricsCalculation(unittest.TestCase):
    """Test retrieval recall, MRR, and percentile latency math."""

    def test_percentile_linear_interpolation(self):
        data = [10.0, 20.0, 30.0, 40.0, 50.0]
        self.assertEqual(calculate_percentile(data, 0.0), 10.0)
        self.assertEqual(calculate_percentile(data, 50.0), 30.0)
        self.assertEqual(calculate_percentile(data, 100.0), 50.0)
        # p95 on 5 elements: k = 4 * 0.95 = 3.8 -> 40.0 * 0.2 + 50.0 * 0.8 = 48.0
        self.assertAlmostEqual(calculate_percentile(data, 95.0), 48.0)

    def test_compute_retrieval_metrics(self):
        # 4 queries:
        # q1: rank 1 (hit@1, hit@5, RR=1.0)
        # q2: rank 3 (hit@5, RR=1/3)
        # q3: rank 5 (hit@5, RR=1/5)
        # q4: not found (miss, RR=0.0)
        query_evals = [
            {"hit_at_1": True, "hit_at_5": True, "reciprocal_rank": 1.0},
            {"hit_at_1": False, "hit_at_5": True, "reciprocal_rank": 1.0 / 3.0},
            {"hit_at_1": False, "hit_at_5": True, "reciprocal_rank": 1.0 / 5.0},
            {"hit_at_1": False, "hit_at_5": False, "reciprocal_rank": 0.0},
        ]
        latencies = [15.0, 25.0, 35.0, 45.0]

        metrics = compute_retrieval_metrics(query_evals, latencies)
        self.assertEqual(metrics["total_queries"], 4)
        self.assertEqual(metrics["hits_at_1"], 1)
        self.assertEqual(metrics["hits_at_5"], 3)
        self.assertEqual(metrics["recall_at_1"], 0.25)
        self.assertEqual(metrics["recall_at_5"], 0.75)
        expected_mrr = (1.0 + 1.0 / 3.0 + 1.0 / 5.0 + 0.0) / 4.0
        self.assertAlmostEqual(metrics["mrr"], round(expected_mrr, 4))
        self.assertEqual(metrics["latency_p50_ms"], 30.0)


class TestCandidateHandlingAndSafeguards(unittest.TestCase):
    """Test candidate reporting, resident model preservation, and sequential execution."""

    @patch.object(OllamaEmbeddingClient, "check_health")
    @patch.object(OllamaEmbeddingClient, "get_available_models")
    @patch.object(OllamaEmbeddingClient, "get_available_model_names")
    @patch.object(OllamaEmbeddingClient, "get_running_models")
    def test_candidate_not_pulled_reported_without_pull(
        self,
        mock_running,
        mock_available_names,
        mock_available_models,
        mock_health,
    ):
        mock_health.return_value = {"connected": True, "version": "0.3.12"}
        mock_running.return_value = {"qwen2.5:1.5b-instruct"}
        # None of the candidates are present in tags
        mock_available_models.return_value = [{"name": "qwen2.5:1.5b-instruct", "size": 986000000}]
        mock_available_names.return_value = {"qwen2.5:1.5b-instruct"}

        results = run_embedding_benchmark(
            candidates=["nomic-embed-text", "qwen3-embedding:0.6b"],
            max_queries=2,
        )

        candidates = results["candidates"]
        self.assertEqual(candidates["nomic-embed-text"]["status"], "not_pulled")
        self.assertEqual(candidates["qwen3-embedding:0.6b"]["status"], "not_pulled")
        self.assertIn("not present in local Ollama", candidates["nomic-embed-text"]["reason"])
        self.assertIsNone(results["winner_summary"]["winner"])

    def test_resident_model_at_start_is_preserved(self):
        client = MagicMock(spec=OllamaEmbeddingClient)
        # Mock embed returning 2D vector for 1 document and 1 query
        client.embed.side_effect = [
            ([[1.0, 0.0]], 0.01),  # corpus batch
            ([[1.0, 0.0]], 0.005),  # query
        ]
        client.get_model_info.return_value = {"details": {"parameter_size": "137M"}}
        client.unload_model.return_value = True

        corpus = [CorpusRecord(doc_id="d1", text="text1", source="s1", original_ids={1})]
        queries = [GoldenQuery(query_id=1, question="q1", target_answer="text1", target_doc_id="d1")]

        # Case 1: Model WAS resident at start
        resident_start = {"nomic-embed-text"}
        res1 = benchmark_single_candidate(
            model="nomic-embed-text",
            client=client,
            corpus_records=corpus,
            queries=queries,
            resident_at_start=resident_start,
            available_models=[{"name": "nomic-embed-text", "size": 274000000}],
        )
        self.assertEqual(res1["unload_status"], "kept_loaded_resident_at_start")
        client.unload_model.assert_not_called()

        # Case 2: Model was NOT resident at start -> should be unloaded
        client.embed.side_effect = [
            ([[1.0, 0.0]], 0.01),
            ([[1.0, 0.0]], 0.005),
        ]
        resident_start_empty = set()
        res2 = benchmark_single_candidate(
            model="nomic-embed-text",
            client=client,
            corpus_records=corpus,
            queries=queries,
            resident_at_start=resident_start_empty,
            available_models=[{"name": "nomic-embed-text", "size": 274000000}],
        )
        self.assertEqual(res2["unload_status"], "unloaded_to_reclaim_ram")
        client.unload_model.assert_called_once_with("nomic-embed-text")

    def test_select_winner_scoring(self):
        # Candidate 1: High recall, low latency, low RAM
        # Candidate 2: Low recall, high latency, high RAM
        cand_results = {
            "nomic-embed-text": {
                "status": "completed",
                "metrics": {
                    "recall_at_1": 0.80,
                    "recall_at_5": 0.95,
                    "mrr": 0.88,
                    "latency_p50_ms": 15.0,
                },
                "total_ram_mb": 275.0,
            },
            "qwen3-embedding:0.6b": {
                "status": "completed",
                "metrics": {
                    "recall_at_1": 0.60,
                    "recall_at_5": 0.70,
                    "mrr": 0.65,
                    "latency_p50_ms": 45.0,
                },
                "total_ram_mb": 650.0,
            },
        }

        winner = select_winner(cand_results)
        self.assertEqual(winner["winner"], "nomic-embed-text")
        self.assertGreater(winner["composite_score"], 0.7)
        self.assertEqual(len(winner["rankings"]), 2)
        self.assertEqual(winner["rankings"][0]["model"], "nomic-embed-text")

    def test_select_winner_prioritizes_quality_over_efficiency(self):
        candidates = {
            "embeddinggemma": {
                "status": "completed",
                "metrics": {"recall_at_5": 0.50, "mrr": 0.40, "latency_p50_ms": 500.0},
                "total_ram_mb": 1200.0,
            },
            "fast-low-recall": {
                "status": "completed",
                "metrics": {"recall_at_5": 0.40, "mrr": 0.30, "latency_p50_ms": 5.0},
                "total_ram_mb": 200.0,
            },
        }
        winner = select_winner(candidates)
        self.assertEqual(winner["winner"], "embeddinggemma")
        self.assertEqual(winner["rankings"][0]["model"], "embeddinggemma")


class TestEndToEndMockedBenchmark(unittest.TestCase):
    """End-to-end benchmark workflow with mocked Ollama API responses."""

    @patch.object(OllamaEmbeddingClient, "check_health")
    @patch.object(OllamaEmbeddingClient, "get_running_models")
    @patch.object(OllamaEmbeddingClient, "get_available_models")
    @patch.object(OllamaEmbeddingClient, "get_available_model_names")
    @patch.object(OllamaEmbeddingClient, "get_model_info")
    @patch.object(OllamaEmbeddingClient, "unload_model")
    @patch.object(OllamaEmbeddingClient, "embed")
    def test_full_benchmark_mocked_pipeline(
        self,
        mock_embed,
        mock_unload,
        mock_info,
        mock_avail_names,
        mock_avail_models,
        mock_running,
        mock_health,
    ):
        mock_health.return_value = {"connected": True, "version": "0.3.14"}
        mock_running.return_value = set()
        mock_avail_names.return_value = {"nomic-embed-text", "nomic-embed-text:latest"}
        mock_avail_models.return_value = [
            {"name": "nomic-embed-text", "size": 274302464, "details": {"parameter_size": "137M"}}
        ]
        mock_info.return_value = {"details": {"parameter_size": "137M", "quantization_level": "F16"}}
        mock_unload.return_value = True

        # Mock embeddings: return 768-dim vectors
        def fake_embed(model, inputs, truncate=True):
            if isinstance(inputs, str):
                count = 1
            else:
                count = len(inputs)
            vectors = [[0.05 * (j % 10) for j in range(768)] for _ in range(count)]
            return vectors, 0.005

        mock_embed.side_effect = fake_embed

        results = run_embedding_benchmark(
            candidates=["nomic-embed-text", "embeddinggemma"],
            max_queries=5,
        )

        self.assertIn("nomic-embed-text", results["candidates"])
        self.assertIn("embeddinggemma", results["candidates"])

        nomic_res = results["candidates"]["nomic-embed-text"]
        self.assertEqual(nomic_res["status"], "completed")
        self.assertEqual(nomic_res["dimension"], 768)
        self.assertIn("metrics", nomic_res)
        self.assertGreater(nomic_res["total_ram_mb"], 200.0)

        gemma_res = results["candidates"]["embeddinggemma"]
        self.assertEqual(gemma_res["status"], "not_pulled")

        winner_info = results["winner_summary"]
        self.assertEqual(winner_info["winner"], "nomic-embed-text")


if __name__ == "__main__":
    unittest.main()
