"""Unit tests for the cross-encoder reranker using FAKE scorer/tokenizer/session objects.

These verify wiring and ordering logic only. They are NOT benchmark evidence for model quality.
"""

import math
import sys
import unittest
import warnings
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
for p in (HERE.parents[1], next(q for q in HERE.parents if (q / "retrieval" / "pipeline.py").is_file())):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from reranker.onnx_reranker import (  # noqa: E402
    CrossEncoderReranker, OnnxPairScorer, RerankerUnavailable, load_default_reranker)
from retrieval.pipeline import RetrievalPipeline  # noqa: E402


class FakeScorer:
    def __init__(self, fn):
        self.fn, self.calls = fn, []

    def score_pairs(self, query, passages):
        self.calls.append((query, list(passages)))
        return [self.fn(query, p) for p in passages]


def cands(n):
    return [{"id": i, "text": f"passage {i}", "confidence": 0.1} for i in range(1, n + 1)]


class TestCrossEncoderReranker(unittest.TestCase):
    def test_orders_by_score_and_annotates(self):
        r = CrossEncoderReranker(FakeScorer(lambda q, p: {"passage 1": 0.1, "passage 2": 5.0, "passage 3": 2.0}[p]))
        out = r(cands(3), query="q")
        self.assertEqual([c["id"] for c in out], [2, 3, 1])
        self.assertEqual(out[0]["rerank_score"], 5.0)

    def test_does_not_mutate_input(self):
        inp = cands(3)
        CrossEncoderReranker(FakeScorer(lambda q, p: -len(p)))(inp, query="q")
        self.assertTrue(all("rerank_score" not in c for c in inp))

    def test_ties_are_stable(self):
        out = CrossEncoderReranker(FakeScorer(lambda q, p: 1.0))(cands(5), query="q")
        self.assertEqual([c["id"] for c in out], [1, 2, 3, 4, 5])

    def test_cap_leaves_tail_untouched(self):
        sc = FakeScorer(lambda q, p: -int(p.split()[1]) * -1.0)  # higher id => higher score
        r = CrossEncoderReranker(sc, max_candidates=3)
        out = r(cands(6), query="q")
        self.assertEqual(len(sc.calls[0][1]), 3)
        self.assertEqual([c["id"] for c in out], [3, 2, 1, 4, 5, 6])
        self.assertNotIn("rerank_score", out[3])

    def test_empty_and_query_required(self):
        r = CrossEncoderReranker(FakeScorer(lambda q, p: 0.0))
        self.assertEqual(r([], query="q"), [])
        with self.assertRaises(ValueError):
            r(cands(2))
        with self.assertRaises(ValueError):
            r(cands(2), query="  ")

    def test_bad_scores_raise(self):
        for bad in ([1.0], [1.0, math.nan], [1.0, math.inf], [1.0, "x"]):
            r = CrossEncoderReranker(Mock(score_pairs=Mock(return_value=bad)))
            with self.assertRaises(ValueError):
                r(cands(2), query="q")

    def test_constructor_validation(self):
        with self.assertRaises(ValueError):
            CrossEncoderReranker(FakeScorer(lambda q, p: 0), max_candidates=31)
        with self.assertRaises(TypeError):
            CrossEncoderReranker(object())

    def test_missing_model_files(self):
        with self.assertRaises(RerankerUnavailable):
            load_default_reranker(HERE / "does-not-exist")


class FakeEnc(SimpleNamespace):
    pass


class FakeTokenizer:
    def encode_batch(self, pairs):
        out = []
        for q, p in pairs:
            ids = [1] * (len(p) % 5 + 2)
            out.append(FakeEnc(ids=ids, attention_mask=[1] * len(ids), type_ids=[0] * len(ids)))
        return out


class FakeSession:
    def __init__(self, names=("input_ids", "attention_mask", "token_type_ids"), width=1):
        self.names, self.width, self.feeds = names, width, []

    def get_inputs(self):
        return [SimpleNamespace(name=n) for n in self.names]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _outs, feed):
        self.feeds.append(feed)
        n = len(feed["input_ids"])
        return [[[float(i)] * self.width for i in range(n)]]


class TestOnnxPairScorerWithFakes(unittest.TestCase):
    def test_directml_provider_is_preferred_with_compatible_session_options(self):
        class RuntimeSession(FakeSession):
            def get_providers(self):
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

        class SessionOptions:
            enable_cpu_mem_arena = True
            enable_mem_pattern = True
            execution_mode = None

        options_class = SessionOptions

        class Runtime:
            ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
            SessionOptions = options_class

            @staticmethod
            def get_available_providers():
                return ["DmlExecutionProvider", "CPUExecutionProvider"]

            @staticmethod
            def InferenceSession(_path, options, providers):
                Runtime.options = options
                Runtime.providers = providers
                return RuntimeSession()

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"onnxruntime": Runtime}):
            model = Path(tmp) / "model.onnx"
            model.touch()
            OnnxPairScorer(model, tokenizer=FakeTokenizer())
        self.assertEqual(Runtime.providers, ["DmlExecutionProvider", "CPUExecutionProvider"])
        self.assertFalse(Runtime.options.enable_cpu_mem_arena)
        self.assertFalse(Runtime.options.enable_mem_pattern)
        self.assertEqual(Runtime.options.execution_mode, "sequential")

    def test_cpu_provider_is_used_when_directml_is_unavailable(self):
        class SessionOptions:
            enable_mem_pattern = True
            execution_mode = None

        options_class = SessionOptions

        class Runtime:
            ExecutionMode = SimpleNamespace(ORT_SEQUENTIAL="sequential")
            SessionOptions = options_class

            @staticmethod
            def get_available_providers():
                return ["CPUExecutionProvider"]

            @staticmethod
            def InferenceSession(_path, _options, providers):
                Runtime.providers = providers
                return FakeSession()

        with tempfile.TemporaryDirectory() as tmp, patch.dict(sys.modules, {"onnxruntime": Runtime}):
            model = Path(tmp) / "model.onnx"
            model.touch()
            OnnxPairScorer(model, tokenizer=FakeTokenizer())
        self.assertEqual(Runtime.providers, ["CPUExecutionProvider"])

    def test_batches_and_feeds_only_declared_inputs(self):
        sess = FakeSession(names=("input_ids", "attention_mask"))
        sc = OnnxPairScorer(session=sess, tokenizer=FakeTokenizer(), batch_size=2)
        scores = sc.score_pairs("q", ["a", "bb", "ccc", "dddd", "eeeee"])
        self.assertEqual(len(scores), 5)
        self.assertEqual(len(sess.feeds), 3)  # 2 + 2 + 1
        self.assertNotIn("token_type_ids", sess.feeds[0])

    def test_two_class_head_uses_logit_difference(self):
        class S2(FakeSession):
            def run(self, _o, feed):
                return [[[1.0, 4.0]] * len(feed["input_ids"])]
        sc = OnnxPairScorer(session=S2(width=2), tokenizer=FakeTokenizer())
        self.assertEqual(sc.score_pairs("q", ["a"]), [3.0])

    def test_unsupported_width(self):
        sc = OnnxPairScorer(session=FakeSession(width=3), tokenizer=FakeTokenizer())
        with self.assertRaises(ValueError):
            sc.score_pairs("q", ["a"])

    def test_missing_paths(self):
        with self.assertRaises(RerankerUnavailable):
            OnnxPairScorer(HERE / "nope.onnx", HERE / "nope.json")
        with self.assertRaises(ValueError):
            OnnxPairScorer(session=FakeSession(), tokenizer=FakeTokenizer(), max_length=2)


def fact(i, conf, **kw):
    return {"id": i, "text": f"passage {i}", "source": "s", "writer": "w", "created_at": "x",
            "valid_from": "x", "valid_to": "y", "status": "active", "superseded_by": None,
            "confidence": conf, "tags": "", **kw}


def make_storage(top_conf, n=40):
    st = Mock()
    st.connect.return_value = None
    st._validate_vector.side_effect = lambda v, name="v": v
    st.search_keyword.return_value = [fact(i, top_conf if i == 1 else 0.9) for i in range(1, n + 1)]
    st.search_vector.return_value = [dict(fact(i, top_conf if i == 1 else 0.9), distance=0.1) for i in range(1, n + 1)]
    return st


class TestPipelineIntegration(unittest.TestCase):
    def test_gate_open_reranks_and_caps_candidates(self):
        sc = FakeScorer(lambda q, p: float(p.split()[1]))  # last passage best
        pipe = RetrievalPipeline(make_storage(0.1), embedder=lambda q: [0.0], reranker=CrossEncoderReranker(sc),
                                 cache_size=0, max_candidates_for_rerank=10)
        res = pipe.search("query", limit=3)
        self.assertEqual(len(sc.calls[0][1]), 10)  # pipeline cap honoured
        self.assertEqual([r["id"] for r in res], [10, 9, 8])
        self.assertEqual(pipe.rerank_gate_activations, 1)
        self.assertEqual(pipe.rerank_fallbacks, 0)

    def test_gate_closed_skips_reranker(self):
        sc = FakeScorer(lambda q, p: 0.0)
        pipe = RetrievalPipeline(make_storage(0.9), embedder=lambda q: [0.0], reranker=CrossEncoderReranker(sc),
                                 cache_size=0)
        pipe.search("query", limit=3)
        self.assertEqual(sc.calls, [])
        self.assertEqual(pipe.rerank_gate_activations, 0)

    def test_scorer_failure_falls_back_to_fused_order(self):
        def boom(q, p):
            raise RuntimeError("onnx failed")
        base = RetrievalPipeline(make_storage(0.1), embedder=lambda q: [0.0], cache_size=0).search("query", 5)
        pipe = RetrievalPipeline(make_storage(0.1), embedder=lambda q: [0.0],
                                 reranker=CrossEncoderReranker(FakeScorer(boom)), cache_size=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = pipe.search("query", 5)
        self.assertEqual([r["id"] for r in res], [r["id"] for r in base])
        self.assertEqual(pipe.rerank_gate_activations, 1)
        self.assertEqual(pipe.rerank_fallbacks, 1)


class TestBenchmarkWiring(unittest.TestCase):
    def test_compare_runs_identical_queries_on_both_arms(self):
        from reranker.benchmark import compare
        sc = FakeScorer(lambda q, p: float(p.split()[1]))
        golden = [{"question": "q1", "answer": "passage 3 content", "source": "s", "id": 1},
                  {"question": "q2", "answer": "passage 4 content", "source": "s", "id": 2}]
        res = compare(make_storage(0.1), lambda q: [0.0], CrossEncoderReranker(sc), golden)
        self.assertEqual(res["no_reranker"]["queries"], res["cross_encoder"]["queries"], 2)
        self.assertEqual(res["no_reranker"]["gate_activations"], 0)
        self.assertEqual(res["cross_encoder"]["gate_activations"], 2)  # warm-up call excluded
        for k in ("recall_at_1", "recall_at_5", "mrr", "total_latency_p50_ms", "total_latency_p95_ms"):
            self.assertIn(k, res["cross_encoder"])


if __name__ == "__main__":
    unittest.main()
