"""Pure metric functions for the grounded-answer evaluation. No I/O, no model calls.

Conventions
  * "abstain" is the positive class for abstain precision/recall.
  * A claim is *unsupported* if the judge verdict is `unsupported` OR `contradicted`;
    `contradicted` is also reported on its own.
  * Undefined ratios (zero denominator) are None, never 0.0, so they cannot be mistaken for results.
"""

import math
from typing import Any, Iterable

UNSUPPORTED_VERDICTS = ("unsupported", "contradicted")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def wilson_interval(k: int, n: int, z: float = 1.96) -> list[float] | None:
    """95% Wilson score interval for k successes in n trials (None when n == 0)."""
    if n <= 0:
        return None
    if not 0 <= k <= n:
        raise ValueError("k must be between 0 and n")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _rate(k: int, n: int) -> dict[str, Any]:
    return {"k": k, "n": n, "rate": ratio(k, n), "ci95": wilson_interval(k, n)}


def abstain_confusion(should_abstain: list[bool], did_abstain: list[bool]) -> dict[str, Any]:
    """Confusion metrics with abstain = positive class. Inputs must be equal-length lists of real bools."""
    if len(should_abstain) != len(did_abstain):
        raise ValueError(f"length mismatch: {len(should_abstain)} != {len(did_abstain)}")
    for name, values in (("should_abstain", should_abstain), ("did_abstain", did_abstain)):
        for i, v in enumerate(values):
            if type(v) is not bool:
                raise ValueError(f"{name}[{i}] must be a bool, got {v!r}")
    tp = sum(s and d for s, d in zip(should_abstain, did_abstain))
    fp = sum((not s) and d for s, d in zip(should_abstain, did_abstain))
    fn = sum(s and (not d) for s, d in zip(should_abstain, did_abstain))
    tn = sum((not s) and (not d) for s, d in zip(should_abstain, did_abstain))
    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(should_abstain),
        "precision": precision, "recall": recall, "f1": f1,
        "accuracy": ratio(tp + tn, len(should_abstain)),
        "false_abstain_rate": ratio(fp, fp + tn),  # answerable questions wrongly refused
        "precision_ci95": wilson_interval(tp, tp + fp), "recall_ci95": wilson_interval(tp, tp + fn),
    }


def claim_metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Claim- and response-level unsupported rates.

    Each item: {"verdicts": [str,...], "bases": [str|None,...], "abstained": bool|None, "errored": bool}.
    Abstaining and errored responses contribute no claims (errors are counted, never scored as supported).
    `not_retrieval_grounded`: supported claims whose judge-reported basis is not the retrieved facts
    (only claims with a reported basis are counted in its denominator).
    """
    claims = unsupported = contradicted = supported = 0
    responses = bad_responses = errored = 0
    basis_known = basis_not_retrieved = 0
    for item in items:
        if item.get("errored"):
            errored += 1
            continue
        if item.get("abstained"):
            continue
        verdicts = item.get("verdicts") or []
        bases = item.get("bases") or [None] * len(verdicts)
        responses += 1
        u = sum(v in UNSUPPORTED_VERDICTS for v in verdicts)
        claims += len(verdicts)
        unsupported += u
        contradicted += sum(v == "contradicted" for v in verdicts)
        supported += sum(v == "supported" for v in verdicts)
        bad_responses += u > 0
        for v, b in zip(verdicts, bases):
            if v == "supported" and b is not None:
                basis_known += 1
                basis_not_retrieved += b != "retrieved"
    return {
        "claims": claims, "supported": supported, "unsupported": unsupported, "contradicted": contradicted,
        "unsupported_claim_rate": _rate(unsupported, claims),
        "answered_responses": responses, "responses_with_unsupported": bad_responses,
        "unsupported_response_rate": _rate(bad_responses, responses),
        "errored_responses": errored,
        "not_retrieval_grounded": _rate(basis_not_retrieved, basis_known),
    }


def accuracy_metrics(items: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Answer accuracy against the expected fact.

    Each item: {"answer_correct": bool|None, "abstained": bool, "errored": bool, "string_match": bool|None}.
    Denominator n = all items: abstentions, wrong answers, and errored (empty/invalid) responses are all
    "not correct". `accuracy_when_answered` excludes abstentions and errors.
    """
    items = list(items)
    n = len(items)
    errored = sum(bool(i.get("errored")) for i in items)
    abstained = sum(bool(i.get("abstained")) and not i.get("errored") for i in items)
    answered = [i for i in items if not i.get("abstained") and not i.get("errored")]
    correct = sum(i.get("answer_correct") is True for i in answered)
    string_hits = sum(i.get("string_match") is True for i in answered)
    disagree = sum(
        1 for i in answered if i.get("string_match") is not None and i.get("answer_correct") is not None
        and i["string_match"] != i["answer_correct"]
    )
    return {
        "n": n, "answered": len(answered), "abstained": abstained, "errored": errored,
        "accuracy": _rate(correct, n),
        "accuracy_when_answered": _rate(correct, len(answered)),
        "string_match_accuracy": _rate(string_hits, n),
        "judge_vs_string_disagreements": disagree,
    }


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid {label}: {value!r}")
    return float(value)


def top_score(facts: list[dict[str, Any]]) -> float | None:
    """Query-level answerability signal for the abstention gate (highest value among facts).

    Preference order, decided once per call so that scales are never mixed:
      1. `answerability_score` (mean of the top-five dense cosine similarities, supplied by the cache
         builder; query-level, so every fact of an item normally carries the same value). If ANY fact
         carries a non-None `answerability_score`, only those values are used.
      2. otherwise `score` (pipeline retrieval rank score, i.e. RRF), then `confidence` per fact.

    0.0 when nothing was retrieved (strongest "unanswerable" signal); None when facts exist but none
    carries a score (the gate then cannot fire for that item). A present but non-numeric, bool, or
    non-finite value raises ValueError (it is never silently replaced by the RRF fallback).
    """
    if not facts:
        return 0.0
    answerability = [
        _finite_number(f["answerability_score"], "answerability score")
        for f in facts if f.get("answerability_score") is not None
    ]
    if answerability:
        return max(answerability)
    best: float | None = None
    for fact in facts:
        # Pipeline `score` is query-specific RRF relevance; `confidence` describes the
        # stored fact/source. Calibrating abstention on source confidence would ignore retrieval.
        value = fact.get("score", fact.get("confidence"))
        if value is None:
            continue
        value = _finite_number(value, "retrieval score")
        best = value if best is None else max(best, value)
    return best


def gated_abstain(item: dict[str, Any], threshold: float | None) -> bool | None:
    """System-level abstain decision: gate below `threshold`, else defer to the answerer.

    Returns None when the item has no usable answerer outcome and the gate did not fire.
    """
    if threshold is not None and item.get("score") is not None and item["score"] < threshold:
        return True
    return item.get("abstained")  # None when the response/judgement errored


def evaluate_abstention(items: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    """Abstain precision/recall over items {"should_abstain": bool, "score": float, "abstained": bool|None}.

    Items whose outcome is unknown (errored response and gate did not fire) are excluded and counted.
    """
    truth: list[bool] = []
    pred: list[bool] = []
    errored = 0
    for item in items:
        decision = gated_abstain(item, threshold)
        if decision is None:
            errored += 1
            continue
        truth.append(item["should_abstain"])
        pred.append(decision)
    result = abstain_confusion(truth, pred)
    result["threshold"] = threshold
    result["errored_excluded"] = errored
    return result


def candidate_thresholds(scores: list[float]) -> list[float]:
    """Gate candidates: below-all (gate never fires), midpoints between distinct scores, above-all."""
    distinct = sorted(set(scores))
    if not distinct:
        return []
    cands = [distinct[0] - 1.0]
    cands += [(a + b) / 2 for a, b in zip(distinct, distinct[1:])]
    cands.append(distinct[-1] + 1.0)
    return cands


def calibrate_threshold(
    dev_items: list[dict[str, Any]], *, target_precision: float = 0.9, min_abstains: int = 1
) -> dict[str, Any]:
    """Pick the retrieval-score gate on DEV items only.

    Rule: among candidates whose dev abstain precision >= target_precision (and >= min_abstains
    predicted abstains), choose max recall, then max precision, then the lowest threshold.
    If none is feasible, fall back to max F1 (then precision, then lowest threshold) and flag
    `target_met: false`. Deterministic for a given dev set.
    """
    if not 0 < target_precision <= 1:
        raise ValueError("target_precision must be in (0, 1]")
    usable = [i for i in dev_items if i.get("score") is not None]
    if not usable:
        return {"threshold": None, "target_met": False, "reason": "no dev items", "dev": None, "candidates": 0}
    scored = []
    for tau in candidate_thresholds([i["score"] for i in usable]):
        m = evaluate_abstention(dev_items, tau)
        scored.append((tau, m))
    feasible = [
        (tau, m) for tau, m in scored
        if m["precision"] is not None and m["precision"] >= target_precision and (m["tp"] + m["fp"]) >= min_abstains
    ]
    if feasible:
        tau, m = sorted(feasible, key=lambda t: (-(t[1]["recall"] or 0.0), -t[1]["precision"], t[0]))[0]
        met, reason = True, "max dev recall subject to dev precision >= target"
    else:
        tau, m = sorted(scored, key=lambda t: (-(t[1]["f1"] or 0.0), -(t[1]["precision"] or 0.0), t[0]))[0]
        met, reason = False, "no candidate reached target precision; fell back to max dev F1"
    return {"threshold": tau, "target_precision": target_precision, "target_met": met, "reason": reason,
            "dev": m, "candidates": len(scored)}
