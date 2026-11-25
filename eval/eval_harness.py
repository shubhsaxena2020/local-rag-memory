"""Offline corpus validation and callback-based retrieval evaluation."""

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Callable

try:
    from grounding.relevance import is_relevant as _strict_is_relevant
except ImportError:  # run as a script: put the project root on sys.path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from grounding.relevance import is_relevant as _strict_is_relevant

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_PATH = EVAL_DIR / "golden_set.jsonl"
ABSTAIN_PATH = EVAL_DIR / "abstain_probes.jsonl"
MIN_GOLDEN = 150
MIN_ABSTAIN = 40


def load_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    with Path(filepath).open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate_corpus(golden_set=None, abstain_probes=None) -> dict[str, int]:
    golden = load_jsonl(GOLDEN_PATH) if golden_set is None else golden_set
    abstain = load_jsonl(ABSTAIN_PATH) if abstain_probes is None else abstain_probes
    if len(golden) < MIN_GOLDEN or len(abstain) < MIN_ABSTAIN:
        raise ValueError(f"Need at least {MIN_GOLDEN} golden rows and {MIN_ABSTAIN} abstain probes")
    questions = [row.get("question") for row in golden]
    ids = [row.get("id") for row in golden]
    for i, row in enumerate(golden):
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "answer", "source")):
            raise ValueError(f"Golden row {i} has a missing/blank question, answer, or source")
        if not isinstance(ids[i], int):
            raise ValueError(f"Golden row {i} has an invalid id")
        if Path(row["source"]).name != row["source"]:
            raise ValueError(f"Golden row {i} source must be a basename")
    if len(set(questions)) != len(questions):
        raise ValueError("Golden questions must be unique")
    if len(set(ids)) != len(ids):
        raise ValueError("Golden ids must be unique")
    for i, row in enumerate(abstain):
        if not all(isinstance(row.get(k), str) and row[k].strip() for k in ("question", "reason")):
            raise ValueError(f"Abstain row {i} has a missing/blank question or reason")
        if type(row.get("expected_abstain")) is not bool:
            raise ValueError(f"Abstain row {i} must have a boolean expected_abstain label")
    return {"golden": len(golden), "abstain": len(abstain), "sources": len({r["source"] for r in golden})}


def evaluate_retrieval(
    golden_set: list[dict[str, Any]],
    retriever: Callable[[str, int], list[dict[str, Any]]],
    *,
    abstain_probes: list[dict[str, Any]] | None = None,
    top_k: int = 5,
    answerer: Callable[[str, list[dict[str, Any]]], list[str]] | None = None,
    claim_supported: Callable[[str, list[dict[str, Any]]], bool] | None = None,
    should_abstain: Callable[[list[dict[str, Any]]], bool] | None = None,
) -> dict[str, Any]:
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k must be an integer from 1 to 100")
    ranks: list[int | None] = []
    latencies: list[float] = []
    unsupported: list[bool] = []
    for row in golden_set:
        start = time.perf_counter()
        results = retriever(row["question"], top_k)
        latencies.append((time.perf_counter() - start) * 1000)
        results = list(results)[:top_k]
        rank = next((i + 1 for i, item in enumerate(results) if _is_relevant(item, row)), None)
        ranks.append(rank)
        if answerer is not None or claim_supported is not None:
            if answerer is None or claim_supported is None:
                raise ValueError("answerer and claim_supported must be supplied together")
            claims = answerer(row["question"], results)
            if not isinstance(claims, list) or any(not isinstance(c, str) for c in claims):
                raise ValueError("answerer must return a list of claim strings")
            for claim in claims:
                label = claim_supported(claim, results)
                if type(label) is not bool:
                    raise ValueError("claim_supported must return bool")
                unsupported.append(not label)
    abstain_precision = None
    if abstain_probes is not None and should_abstain is not None:
        true_positive = 0
        false_positive = 0
        for probe in abstain_probes:
            results = list(retriever(probe["question"], top_k))[:top_k]
            actual = should_abstain(results)
            if type(actual) is not bool:
                raise ValueError("should_abstain must return bool")
            expected = probe["expected_abstain"]
            true_positive += actual and expected
            false_positive += actual and not expected
        abstain_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else None
    return {
        "recall_at_1": sum(rank == 1 for rank in ranks) / len(ranks) if ranks else None,
        "recall_at_5": sum(rank is not None and rank <= 5 for rank in ranks) / len(ranks) if ranks else None,
        "mrr": statistics.fmean(1 / rank if rank else 0 for rank in ranks) if ranks else None,
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": _percentile(latencies, 0.95) if latencies else None,
        "unsupported_claim_rate": sum(unsupported) / len(unsupported) if unsupported else None,
        "abstain_precision": abstain_precision,
        "total_questions": len(golden_set),
        "total_claims": len(unsupported),
        "total_abstain_probes": len(abstain_probes or []),
    }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values cannot be empty")
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _is_relevant(result: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Strict rule shared with grounding/: matching `source` alone is NOT a hit."""
    return _strict_is_relevant(result, expected)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true", help="validate local corpus files; makes no model/network calls")
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate_corpus(), indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
