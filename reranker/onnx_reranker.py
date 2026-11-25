"""Local cross-encoder reranker for the retrieval pipeline.

Targets cross-encoder/ms-marco-MiniLM-L6-v2 (Apache-2.0, ONNX int8 AVX2 graph + tokenizer.json),
but any ONNX sequence-classification graph with a single relevance logit works.

Contract with ``retrieval.pipeline.RetrievalPipeline``:
  * ``wants_query = True``  -> pipeline calls ``reranker(candidates, query=query)``.
  * The pipeline applies the low-confidence gate and the candidate cap (<=30) before the call;
    this class also enforces its own ``max_candidates`` (default 30) and returns any uncapped tail
    unchanged, so the output is always a permutation of the input.
  * Any exception (scorer failure, NaN/inf score, wrong score count) propagates; the pipeline
    catches it, counts ``rerank_fallbacks`` and keeps the fused RRF order.

Scoring is pluggable: ``scorer`` is any object with ``score_pairs(query, passages) -> list[float]``
(tests use fakes). ``OnnxPairScorer`` is the real implementation (onnxruntime + tokenizers).
"""

from __future__ import annotations

import math
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Sequence

MODEL_FILENAME = "model_quint8_avx2.onnx"
TOKENIZER_FILENAME = "tokenizer.json"
DEFAULT_MAX_CANDIDATES = 30
DEFAULT_MAX_LENGTH = 256
DEFAULT_BATCH_SIZE = 16


class RerankerUnavailable(RuntimeError):
    """Model/tokenizer file or runtime dependency is missing."""


class PairScorer(Protocol):
    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]: ...


class OnnxPairScorer:
    """Scores (query, passage) pairs with an ONNX cross-encoder graph.

    ``session`` and ``tokenizer`` may be injected (already loaded objects) instead of paths. The
    tokenizer needs ``encode_batch(list_of_pairs)`` returning items with ``.ids``,
    ``.attention_mask`` and ``.type_ids`` (the ``tokenizers`` library interface); the session needs
    ``get_inputs()`` (items with ``.name``) and ``run(None, feed)``.
    """

    def __init__(self, model_path: Optional[str | Path] = None, tokenizer_path: Optional[str | Path] = None,
                 *, session: Any = None, tokenizer: Any = None,
                 max_length: int = DEFAULT_MAX_LENGTH, batch_size: int = DEFAULT_BATCH_SIZE,
                 intra_op_threads: int = 0):
        if type(max_length) is not int or not 8 <= max_length <= 512:
            raise ValueError("max_length must be an integer from 8 to 512")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if session is None:
            if model_path is None or not Path(model_path).is_file():
                raise RerankerUnavailable(f"ONNX model not found: {model_path}")
            try:
                import onnxruntime as ort
            except ImportError as exc:
                raise RerankerUnavailable("onnxruntime is not installed") from exc
            opts = ort.SessionOptions()
            if intra_op_threads:
                opts.intra_op_num_threads = intra_op_threads
            use_directml = "DmlExecutionProvider" in ort.get_available_providers()
            providers = (["DmlExecutionProvider", "CPUExecutionProvider"] if use_directml
                         else ["CPUExecutionProvider"])
            if use_directml:
                opts.enable_cpu_mem_arena = False
                opts.enable_mem_pattern = False
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(model_path), opts, providers=providers)
        if tokenizer is None:
            if tokenizer_path is None or not Path(tokenizer_path).is_file():
                raise RerankerUnavailable(f"tokenizer.json not found: {tokenizer_path}")
            try:
                from tokenizers import Tokenizer
            except ImportError as exc:
                raise RerankerUnavailable("tokenizers is not installed") from exc
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
            tokenizer.enable_truncation(max_length=max_length, strategy="longest_first")
            tokenizer.enable_padding()  # pad to longest in batch; pad id/token come from tokenizer.json
        self.session = session
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self._input_names = {i.name for i in session.get_inputs()}
        self._run_lock = threading.Lock() if "DmlExecutionProvider" in getattr(session, "get_providers", lambda: [])() else None

    def score_pairs(self, query: str, passages: Sequence[str]) -> List[float]:
        import numpy as np

        scores: List[float] = []
        for start in range(0, len(passages), self.batch_size):
            chunk = passages[start:start + self.batch_size]
            encs = self.tokenizer.encode_batch([(query, p) for p in chunk])
            width = max(len(e.ids) for e in encs)  # tokenizer pads already; this is a safety net

            def pad(rows):
                return np.asarray([list(r) + [0] * (width - len(r)) for r in rows], dtype=np.int64)

            feed = {
                "input_ids": pad(e.ids for e in encs),
                "attention_mask": pad(e.attention_mask for e in encs),
                "token_type_ids": pad(e.type_ids for e in encs),
            }
            feed = {k: v for k, v in feed.items() if k in self._input_names}
            if self._run_lock:
                with self._run_lock:
                    output = self.session.run(None, feed)[0]
            else:
                output = self.session.run(None, feed)[0]
            logits = np.asarray(output, dtype=np.float64).reshape(len(chunk), -1)
            if logits.shape[1] == 1:  # single relevance logit (ms-marco cross-encoders)
                scores.extend(float(v) for v in logits[:, 0])
            elif logits.shape[1] == 2:  # [not-relevant, relevant] classifier head
                scores.extend(float(v) for v in logits[:, 1] - logits[:, 0])
            else:
                raise ValueError(f"unsupported output width {logits.shape[1]}")
        return scores


class CrossEncoderReranker:
    """Callable reranker: ``reranker(candidates, query=...) -> reordered candidate dicts``."""

    wants_query = True  # RetrievalPipeline passes query= only when this is exactly True

    def __init__(self, scorer: PairScorer, *, max_candidates: int = DEFAULT_MAX_CANDIDATES,
                 text_key: str = "text", score_key: str = "rerank_score"):
        if type(max_candidates) is not int or not 1 <= max_candidates <= 30:
            raise ValueError("max_candidates must be an integer from 1 to 30")
        if not callable(getattr(scorer, "score_pairs", None)):
            raise TypeError("scorer must provide score_pairs(query, passages)")
        self.scorer = scorer
        self.max_candidates = max_candidates
        self.text_key = text_key
        self.score_key = score_key

    @classmethod
    def from_paths(cls, model_path: str | Path, tokenizer_path: str | Path, *,
                   max_length: int = DEFAULT_MAX_LENGTH, batch_size: int = DEFAULT_BATCH_SIZE,
                   max_candidates: int = DEFAULT_MAX_CANDIDATES) -> "CrossEncoderReranker":
        scorer = OnnxPairScorer(model_path, tokenizer_path, max_length=max_length, batch_size=batch_size)
        return cls(scorer, max_candidates=max_candidates)

    def __call__(self, candidates: List[Dict[str, Any]], query: Optional[str] = None) -> List[Dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query is required for cross-encoder reranking")
        candidates = list(candidates)
        head, tail = candidates[:self.max_candidates], candidates[self.max_candidates:]
        if not head:
            return candidates
        passages = []
        for c in head:
            text = c.get(self.text_key)
            passages.append(text if isinstance(text, str) else "")
        scores = self.scorer.score_pairs(query, passages)
        if len(scores) != len(head):
            raise ValueError(f"scorer returned {len(scores)} scores for {len(head)} candidates")
        clean = []
        for s in scores:
            if isinstance(s, bool) or not isinstance(s, (int, float)) or not math.isfinite(s):
                raise ValueError("scorer returned a non-finite or non-numeric score")
            clean.append(float(s))
        # Stable: highest score first; ties keep incoming (fused RRF) order.
        order = sorted(range(len(head)), key=lambda i: (-clean[i], i))
        out = []
        for i in order:
            item = dict(head[i])
            item[self.score_key] = clean[i]
            out.append(item)
        return out + tail


def load_default_reranker(model_dir: str | Path, *, model_filename: str = MODEL_FILENAME,
                          max_length: int = DEFAULT_MAX_LENGTH, batch_size: int = DEFAULT_BATCH_SIZE,
                          max_candidates: int = DEFAULT_MAX_CANDIDATES) -> CrossEncoderReranker:
    d = Path(model_dir)
    return CrossEncoderReranker.from_paths(d / model_filename, d / TOKENIZER_FILENAME, max_length=max_length,
                                           batch_size=batch_size, max_candidates=max_candidates)
