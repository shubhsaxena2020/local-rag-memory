"""Contradiction/supersede judge benchmark harness (Round 2 hardening, W4).

Nothing in this module talks to a model unless `call_ollama_judge`,
`run_benchmark_on_file` or the CLI `--run` flag is invoked explicitly.

Labels (fact1 = stored fact, fact2 = incoming fact):
  0 compatible addition  same entity, different attribute; both remain true
  1 contradiction        same entity + attribute, conflicting value/negation -> supersede
  2 retraction           fact2 withdraws fact1 without giving a new value    -> retract
  3 unrelated            different entity or topic
  4 temporal change      both facts time-scoped to different periods; both kept
  5 ambiguous/unsure     abstain label (a model may emit it; dataset has no rows)
  6 paraphrase           same meaning / unit-equivalent restatement; NOT a contradiction

Mutation decision (binary): labels {1, 2} are mutations; everything else is not.

Hang protection: every call has a hard wall-clock deadline (worker thread +
socket timeout), a tiny `num_predict` cap, schema-constrained JSON output, an
overall batch budget, and an abort after N consecutive failures.
"""

import json
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence
from urllib.parse import urlsplit

MAX_FACT_LENGTH = 10000

LABEL_NAMES = {
    0: "compatible additions",
    1: "same fact update (contradiction/supersede)",
    2: "retract",
    3: "unrelated entities",
    4: "temporal change",
    5: "ambiguous",
    6: "paraphrase (not a contradiction)",
}
NUM_LABELS = len(LABEL_NAMES)
MUTATION_LABELS = {1, 2}
NON_MUTATION_LABELS = set(LABEL_NAMES) - MUTATION_LABELS

# Defaults for the bounded runner.
DEFAULT_MODEL = "qwen2.5:1.5b-instruct"
DEFAULT_CALL_TIMEOUT_S = 20.0
DEFAULT_BATCH_BUDGET_S = 600.0
DEFAULT_MAX_CONSECUTIVE_FAILURES = 5
DEFAULT_NUM_PREDICT = 24
DEFAULT_NUM_CTX = 1024

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"label": {"type": "integer", "enum": sorted(LABEL_NAMES)}},
    "required": ["label"],
}


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _check_label_value(label, where: str) -> None:
    if type(label) is not int or not 0 <= label < NUM_LABELS:
        raise ValueError(f"{where} must be an integer label from 0 to {NUM_LABELS - 1}")


def load_benchmark_data(filepath) -> List[Dict]:
    """Load and validate JSONL benchmark rows.

    Required: id:int (unique), fact1/fact2: non-empty str, label:int 0..6,
    category == LABEL_NAMES[label]. Optional: subtype: non-empty str.
    A duplicate (fact1, fact2) pair is rejected. Errors carry the line number.
    """
    data = []
    seen_ids = set()
    seen_pairs = set()
    with open(filepath, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            if not isinstance(record, dict):
                raise ValueError(f"Record must be an object in line {line_num}")
            for k in ("id", "fact1", "fact2", "label", "category"):
                if k not in record:
                    raise ValueError(f"Missing required field '{k}' in line {line_num}")
            if type(record["id"]) is not int:
                raise ValueError(f"Field 'id' must be integer in line {line_num}")
            for k in ("fact1", "fact2"):
                if not isinstance(record[k], str) or not record[k].strip():
                    raise ValueError(f"Field '{k}' must be non-empty string in line {line_num}")
            if type(record["label"]) is not int:
                raise ValueError(f"Field 'label' must be integer in line {line_num}")
            if not 0 <= record["label"] < NUM_LABELS:
                raise ValueError(f"Field 'label' must be integer 0-{NUM_LABELS - 1} in line {line_num}")
            if not isinstance(record["category"], str):
                raise ValueError(f"Field 'category' must be string in line {line_num}")
            expected = LABEL_NAMES[record["label"]]
            if record["category"] != expected:
                raise ValueError(
                    f"Category mismatch: expected '{expected}' for label {record['label']} "
                    f"but got '{record['category']}' in line {line_num}"
                )
            if "subtype" in record and (not isinstance(record["subtype"], str) or not record["subtype"].strip()):
                raise ValueError(f"Field 'subtype' must be non-empty string in line {line_num}")
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate id {record['id']} in line {line_num}")
            pair = (record["fact1"], record["fact2"])
            if pair in seen_pairs:
                raise ValueError(f"Duplicate fact pair in line {line_num}")
            seen_ids.add(record["id"])
            seen_pairs.add(pair)
            data.append(record)
    return data


def dataset_summary(data: Sequence[Dict]) -> Dict:
    """Counts by label/category and by (label, subtype)."""
    by_label = Counter(r["label"] for r in data)
    by_subtype = Counter((r["label"], r.get("subtype", "-")) for r in data)
    return {
        "total": len(data),
        "by_label": {k: by_label[k] for k in sorted(by_label)},
        "by_category": {LABEL_NAMES[k]: by_label[k] for k in sorted(by_label)},
        "by_label_subtype": {f"{k[0]}/{k[1]}": v for k, v in sorted(by_subtype.items())},
        "mutation_rows": sum(v for k, v in by_label.items() if k in MUTATION_LABELS),
        "non_mutation_rows": sum(v for k, v in by_label.items() if k not in MUTATION_LABELS),
    }


# --------------------------------------------------------------------------
# Response parsing (strict)
# --------------------------------------------------------------------------
def parse_label(text) -> int:
    """Accept only a lone ASCII digit 0..6 (surrounding whitespace allowed)."""
    if not isinstance(text, str):
        raise ValueError("Label response must be a string")
    stripped = text.strip()
    if stripped not in {str(i) for i in range(NUM_LABELS)}:
        raise ValueError(f"Expected single digit label (0-{NUM_LABELS - 1}), got: {text!r}")
    return int(stripped)


def parse_judge_response(text) -> int:
    """Strict parse of a model reply: JSON {"label": n} (exactly that one key,
    n an int 0..6, not a bool) or, as a fallback, a lone digit. Anything else
    (prose, extra keys, strings, floats, arrays, empty) raises ValueError."""
    if not isinstance(text, str):
        raise ValueError("Judge response must be a string")
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty judge response")
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError as e:
            raise ValueError(f"Judge response is not valid JSON: {e}")
        if not isinstance(obj, dict) or set(obj) != {"label"}:
            raise ValueError("Judge JSON must be an object with exactly the key 'label'")
        label = obj["label"]
        if type(label) is not int or not 0 <= label < NUM_LABELS:
            raise ValueError(f"Judge label must be an integer 0-{NUM_LABELS - 1}")
        return label
    return parse_label(stripped)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def _validate_label_lists(y_true, y_pred) -> None:
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between true and pred")
    for name, labels in (("y_true", y_true), ("y_pred", y_pred)):
        for i, label in enumerate(labels):
            _check_label_value(label, f"{name}[{i}]")


def _safe_div(a, b) -> float:
    return a / b if b else 0.0


def _prf(tp, fp, fn):
    p = _safe_div(tp, tp + fp)
    r = _safe_div(tp, tp + fn)
    return p, r, _safe_div(2 * p * r, p + r)


def _binary_from_bins(t: List[int], p: List[int]) -> Dict:
    tp = sum(1 for a, b in zip(t, p) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(t, p) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(t, p) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(t, p) if a == 1 and b == 0)
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_mutation_rate": _safe_div(fp, fp + tn),  # wrongly supersedes/retracts
        "missed_mutation_rate": _safe_div(fn, fn + tp),
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def compute_mutation_metrics(y_true: List[int], y_pred: List[int]) -> Dict:
    """Binary mutation metrics (positive = label in {1, 2})."""
    _validate_label_lists(y_true, y_pred)
    t = [1 if x in MUTATION_LABELS else 0 for x in y_true]
    p = [1 if x in MUTATION_LABELS else 0 for x in y_pred]
    return _binary_from_bins(t, p)


def compute_full_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Exact-class accuracy across all labels."""
    _validate_label_lists(y_true, y_pred)
    return _safe_div(sum(1 for a, b in zip(y_true, y_pred) if a == b), len(y_true))


def compute_report(y_true: List[int], y_pred: List[Optional[int]],
                   subtypes: Optional[List[Optional[str]]] = None) -> Dict:
    """Full metrics where y_pred may contain None (timeout / invalid output /
    skipped by budget). A None prediction is always scored WRONG: for exact
    class it is a miss, and for the binary decision it is forced to the
    opposite of the truth (fn for mutations, fp otherwise). Scores are over
    all planned rows, so a failing model cannot look better by failing.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch between true and pred")
    for i, label in enumerate(y_true):
        _check_label_value(label, f"y_true[{i}]")
    for i, pred_label in enumerate(y_pred):
        if pred_label is not None:
            _check_label_value(pred_label, f"y_pred[{i}]")
    if subtypes is not None and len(subtypes) != len(y_true):
        raise ValueError("Length mismatch between subtypes and labels")

    n = len(y_true)
    t_bin = [1 if x in MUTATION_LABELS else 0 for x in y_true]
    p_bin = [(1 - tb) if p is None else (1 if p in MUTATION_LABELS else 0)
             for tb, p in zip(t_bin, y_pred)]
    answered = [(a, b) for a, b in zip(y_true, y_pred) if b is not None]
    exact_correct = sum(1 for a, b in answered if a == b)

    confusion = [[0] * NUM_LABELS for _ in range(NUM_LABELS)]
    for a, b in answered:
        confusion[a][b] += 1

    per_class = {}
    for c in range(NUM_LABELS):
        support = sum(1 for a in y_true if a == c)
        tp = sum(1 for a, b in answered if a == c and b == c)
        fp = sum(1 for a, b in answered if a != c and b == c)
        p, r, f = _prf(tp, fp, support - tp)
        per_class[str(c)] = {"name": LABEL_NAMES[c], "support": support,
                             "precision": p, "recall": r, "f1": f}

    per_subtype = None
    if subtypes is not None:
        buckets: defaultdict[str, List[int]] = defaultdict(lambda: [0, 0])
        for yt, yp, s in zip(y_true, y_pred, subtypes):
            key = f"{yt}/{s or '-'}"
            buckets[key][1] += 1
            buckets[key][0] += 1 if (yp is not None and yt == yp) else 0
        per_subtype = {k: {"correct": v[0], "n": v[1], "accuracy": _safe_div(v[0], v[1])}
                       for k, v in sorted(buckets.items())}

    return {
        "n": n,
        "n_answered": len(answered),
        "exact_accuracy": _safe_div(exact_correct, n),
        "exact_accuracy_on_answered": _safe_div(exact_correct, len(answered)),
        "mutation_metrics": _binary_from_bins(t_bin, p_bin),
        "per_class": per_class,
        "confusion_matrix_rows_true_cols_pred": confusion,
        "per_subtype": per_subtype,
    }


# --------------------------------------------------------------------------
# Prompt + bounded Ollama call
# --------------------------------------------------------------------------
def ollama_judge_prompt(fact1: str, fact2: str) -> str:
    """Deliberately short prompt (cheap for 1-4B models; reply is JSON)."""
    return (
        f"Stored fact A: {fact1}\n"
        f"New fact B: {fact2}\n"
        "How does B relate to A? Answer JSON {\"label\": n}.\n"
        "0 compatible: same subject, different attribute, both true\n"
        "1 contradicts: same subject+attribute, different value/unit or negated\n"
        "2 retracts: B withdraws A (disregard/forget), no new value\n"
        "3 unrelated: different subject/topic\n"
        "4 temporal: same attribute, different explicit time periods, both true\n"
        "5 unsure\n"
        "6 paraphrase: same meaning (reworded or equivalent units)"
    )


class JudgeTimeout(TimeoutError):
    """Raised when a single judge call exceeds its hard wall-clock deadline."""


ALLOWED_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _validate_endpoint(base_url: str) -> None:
    """Validate that base_url strictly targets local loopback over HTTP."""
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("Ollama base_url must be a non-empty string")
    parsed = urlsplit(base_url.strip())
    if parsed.scheme.lower() != "http":
        raise ValueError(f"Ollama base_url must use HTTP scheme, got: '{parsed.scheme}'")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Ollama base_url must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama base_url must not contain a query or fragment")
    hostname = (parsed.hostname or "").lower()
    if hostname not in ALLOWED_LOOPBACK_HOSTS:
        raise ValueError(
            f"Ollama base_url must target a loopback host {ALLOWED_LOOPBACK_HOSTS}, got: '{hostname}'"
        )
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Ollama base_url has an invalid port") from exc
    if port == 0:
        raise ValueError("Ollama base_url port must be between 1 and 65535")


def _validate_fact(name: str, value) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_FACT_LENGTH:
        raise ValueError(f"{name} must be non-empty and at most {MAX_FACT_LENGTH} characters")


def run_with_deadline(fn: Callable[[], object], timeout_s: float):
    """Run fn() in a daemon thread; raise JudgeTimeout if it has not finished
    within timeout_s seconds. The caller is never blocked longer than that
    (the abandoned thread is a daemon and dies with the process)."""
    box: Dict[str, object] = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised in caller
            box["error"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise JudgeTimeout(f"judge call exceeded hard timeout of {timeout_s:.1f}s")
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box["value"]


def _http_generate(base_url: str, payload: dict, timeout_s: float) -> dict:
    _validate_endpoint(base_url)
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirectHandler)
    try:
        with opener.open(req, timeout=timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as e:
        raise ConnectionError(f"Failed to connect to Ollama at {base_url}: {e}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON response from Ollama: {e}")


def build_payload(fact1: str, fact2: str, model: str, num_predict: int = DEFAULT_NUM_PREDICT,
                  num_ctx: int = DEFAULT_NUM_CTX, think: Optional[bool] = False,
                  keep_alive: str = "5m") -> dict:
    """Ollama /api/generate body. `format` is a JSON schema (constrained
    decoding). think=False stops thinking-capable models burning the
    num_predict cap on hidden reasoning; pass think=None to omit the field."""
    payload = {
        "model": model,
        "prompt": ollama_judge_prompt(fact1, fact2),
        "stream": False,
        "format": RESPONSE_SCHEMA,
        "keep_alive": keep_alive,
        "options": {"temperature": 0.0, "num_predict": num_predict, "num_ctx": num_ctx},
    }
    if think is not None:
        payload["think"] = think
    return payload


def call_ollama_judge(fact1: str, fact2: str, model: str = DEFAULT_MODEL,
                      base_url: str = "http://127.0.0.1:11434",
                      timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
                      num_predict: int = DEFAULT_NUM_PREDICT,
                      transport: Optional[Callable[[str, dict, float], dict]] = None) -> int:
    """One bounded judgment call. Returns label 0..6.

    Raises ValueError (bad input / unparsable output), ConnectionError, or
    JudgeTimeout (hard deadline). `transport(base_url, payload, timeout_s)` is
    injectable so tests never touch the network. Never prints model output.
    """
    _validate_fact("fact1", fact1)
    _validate_fact("fact2", fact2)
    _validate_endpoint(base_url)
    if not (isinstance(timeout_s, (int, float)) and not isinstance(timeout_s, bool) and timeout_s > 0):
        raise ValueError("timeout_s must be positive")
    send = transport or _http_generate
    payload = build_payload(fact1, fact2, model, num_predict=num_predict)
    resp = run_with_deadline(lambda: send(base_url, payload, timeout_s), timeout_s)
    if not isinstance(resp, dict):
        raise ValueError("Ollama response was not an object")
    return parse_judge_response(resp.get("response", ""))


def _percentile(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def run_benchmark_on_file(data_file, model: str = DEFAULT_MODEL,
                          base_url: str = "http://127.0.0.1:11434",
                          per_call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
                          batch_budget_s: float = DEFAULT_BATCH_BUDGET_S,
                          max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
                          num_predict: int = DEFAULT_NUM_PREDICT,
                          limit: Optional[int] = None,
                          judge: Optional[Callable[[str, str], int]] = None) -> Dict:
    """Bounded benchmark. Total wall time <= batch_budget_s (+ tiny overhead).

    Each row gets min(per_call_timeout_s, time left in budget). Failures
    (timeout / connection / unparsable) are recorded per row and scored wrong.
    The run stops early when the budget is exhausted or after
    `max_consecutive_failures` failures in a row; unattempted rows stay None
    (scored wrong) and are reported as `skipped`. `judge(fact1, fact2)->int`
    can be injected for tests. The result contains no model text.
    """
    data = load_benchmark_data(data_file)
    if limit is not None:
        data = data[:limit]
    y_true = [r["label"] for r in data]
    subtypes = [r.get("subtype") for r in data]
    y_pred: List[Optional[int]] = [None] * len(data)
    errors: List[Optional[str]] = [None] * len(data)
    latencies: List[Optional[float]] = [None] * len(data)

    started = time.perf_counter()
    consecutive = 0
    aborted = None
    for i, item in enumerate(data):
        remaining = batch_budget_s - (time.perf_counter() - started)
        if remaining <= 0:
            aborted = "batch_budget_exhausted"
            break
        if consecutive >= max_consecutive_failures:
            aborted = f"{consecutive}_consecutive_failures"
            break
        call_timeout = max(0.05, min(per_call_timeout_s, remaining))
        t0 = time.perf_counter()
        try:
            if judge is not None:
                judge_call = partial(judge, item["fact1"], item["fact2"])
                label = run_with_deadline(judge_call, call_timeout)
                _check_label_value(label, "judge result")
                y_pred[i] = label
            else:
                y_pred[i] = call_ollama_judge(item["fact1"], item["fact2"], model=model,
                                              base_url=base_url, timeout_s=call_timeout,
                                              num_predict=num_predict)
            consecutive = 0
        except TimeoutError:  # JudgeTimeout and socket-level timeouts
            errors[i] = "timeout"
            consecutive += 1
        except ConnectionError:
            errors[i] = "connection"
            consecutive += 1
        except (ValueError, TypeError):
            errors[i] = "invalid_output"
            consecutive += 1
        except Exception as e:  # noqa: BLE001 - keep the batch bounded, record type only
            errors[i] = f"error:{type(e).__name__}"
            consecutive += 1
        latencies[i] = time.perf_counter() - t0
    total_seconds = time.perf_counter() - started

    report = compute_report(y_true, y_pred, subtypes)
    done = sorted(
        latency
        for latency, pred in zip(latencies, y_pred)
        if latency is not None and pred is not None
    )
    err_counts = Counter(e for e in errors if e)
    out = dict(report)
    out.update({
        "model": model,
        "num_examples": len(data),
        "full_accuracy": report["exact_accuracy"],  # backwards-compatible key
        "completed": report["n_answered"],
        "failed": sum(err_counts.values()),
        "failures_by_type": dict(err_counts),
        "skipped": sum(1 for latency in latencies if latency is None),
        "aborted_reason": aborted,
        "complete": report["n_answered"] == len(data),
        "config": {"per_call_timeout_s": per_call_timeout_s, "batch_budget_s": batch_budget_s,
                   "max_consecutive_failures": max_consecutive_failures, "num_predict": num_predict},
        "total_seconds": total_seconds,
        "p50_seconds": statistics.median(done) if done else 0.0,
        "p95_seconds": _percentile(done, 0.95),
        "per_example_seconds": latencies,
        "predictions": y_pred,
        "true_labels": y_true,
        "errors": errors,
    })
    return out


# --------------------------------------------------------------------------
# CLI: safe dataset summary by default; model calls only with --run
# --------------------------------------------------------------------------
def _main(argv: List[str]) -> int:
    import argparse
    import os

    ap = argparse.ArgumentParser(description="Judge benchmark harness (no model calls unless --run)")
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_data.jsonl"))
    ap.add_argument("--run", action="store_true", help="actually call the local Ollama model")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--call-timeout", type=float, default=DEFAULT_CALL_TIMEOUT_S)
    ap.add_argument("--budget", type=float, default=DEFAULT_BATCH_BUDGET_S)
    ap.add_argument("--max-failures", type=int, default=DEFAULT_MAX_CONSECUTIVE_FAILURES)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None, help="write full JSON result here")
    args = ap.parse_args(argv)

    if not args.run:
        print(json.dumps(dataset_summary(load_benchmark_data(args.data)), indent=2))
        return 0
    result = run_benchmark_on_file(args.data, model=args.model, per_call_timeout_s=args.call_timeout,
                                   batch_budget_s=args.budget, max_consecutive_failures=args.max_failures,
                                   limit=args.limit)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    brief = {k: result[k] for k in ("model", "n", "completed", "failed", "skipped", "aborted_reason",
                                    "exact_accuracy", "total_seconds", "p50_seconds", "p95_seconds")}
    brief["mutation_accuracy"] = result["mutation_metrics"]["accuracy"]
    brief["false_mutation_rate"] = result["mutation_metrics"]["false_mutation_rate"]
    print(json.dumps(brief, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
