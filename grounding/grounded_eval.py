"""Round 2 callback-based answerer + independent-judge harness (offline; no model/network calls here).

The caller supplies four callbacks, so this module never talks to a provider:
  retriever(question, top_k) -> list[dict]   facts with "text", optional "source"/"score"/"confidence"
  answerer(request: dict)    -> str          request = {"system", "prompt", "condition"}
  judge(prompt: str)         -> str          one JSON object, schema in answer_protocol.build_judge_prompt
Conditions: (a) no_memory on answerable questions, (b) grounded = retrieved context + grounding rule on the
same questions, (c) probe = grounded condition on the no-answer probes.

Split: golden rows and probes are stratified separately ("answerable" / "no_answer"), so dev and held-out
each contain both kinds. The abstain confidence threshold is tuned on dev items only; every reported metric
is computed on held-out items only (the dev section only holds the calibration record).
Callback failures (exception, non-string, empty, invalid judge JSON) are counted as `errored`, never scored
as supported/correct/abstained.
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):  # run as a script
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "grounding"

from eval.splits import assign_splits

from .answer_protocol import (
    build_answer_request,
    build_judge_prompt,
    check_independent,
    extract_claims,
    parse_judge_output,
    parse_response,
)
from .grounded_metrics import (
    accuracy_metrics,
    calibrate_threshold,
    claim_metrics,
    evaluate_abstention,
    top_score,
)
from .relevance import answer_in_text

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SEED = 20261019
CONDITIONS = ("no_memory", "grounded", "probe")


def golden_key(row: dict[str, Any]) -> str:
    return f"golden:{row['id']}"


def probe_key(row: dict[str, Any]) -> str:
    return "probe:" + hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:16]


def build_split(golden: list[dict], probes: list[dict], *, seed: int = DEFAULT_SEED, dev_fraction: float = 0.5) -> dict:
    """Reproducible manifest {"seed", "dev": [keys], "heldout": [keys]}; order-independent."""
    items = [("answerable", golden_key(r)) for r in golden] + [("no_answer", probe_key(r)) for r in probes]
    mapping = assign_splits(
        items, key=lambda it: it[1], seed=seed, namespace="grounded-eval-r2", dev_fraction=dev_fraction,
        stratum=lambda it: it[0],
    )
    manifest = {"seed": seed, "dev_fraction": dev_fraction}
    for part in ("dev", "heldout"):
        manifest[part] = sorted(k for k, v in mapping.items() if v == part)
    kinds = {part: {k.split(":")[0] for k in manifest[part]} for part in ("dev", "heldout")}
    for part, present in kinds.items():
        if present != {"golden", "probe"}:
            raise ValueError(f"{part} split must contain both answerable and no-answer items (got {sorted(present)})")
    return manifest


def _facts(retriever: Callable, question: str, top_k: int) -> list[dict[str, Any]]:
    raw = retriever(question, top_k)
    if not isinstance(raw, list) or not all(isinstance(f, dict) for f in raw):
        raise ValueError("retriever must return a list of dicts")
    out = []
    for i, fact in enumerate(raw[:top_k], 1):
        fact = dict(fact)
        fact.setdefault("fact_id", f"F{i}")
        out.append(fact)
    return out


def _call(callback: Callable, *args) -> tuple[Any, str | None]:
    try:
        return callback(*args), None
    except Exception as exc:  # noqa: BLE001 - callbacks are untrusted; record the type only
        return None, f"callback raised {type(exc).__name__}"


def run_item(
    *, condition: str, question: str, facts: list[dict], gold: str | None, expect_abstain: bool,
    answerer: Callable, judge: Callable, show_memory: bool,
) -> dict[str, Any]:
    """Answer one question and judge it. Returns a flat record; `errored` + `error` on any callback defect."""
    base = {"condition": condition, "question": question, "score": top_score(facts), "should_abstain": expect_abstain,
            "errored": True, "error": None, "abstained": None, "verdicts": [], "bases": [], "answer_correct": None,
            "string_match": None, "n_claims": 0, "kind": None}
    request = build_answer_request(condition, question, facts if show_memory else None)
    request["condition"] = condition
    raw, err = _call(answerer, request)
    parsed = parse_response(raw) if err is None else {"status": "invalid", "text": "", "kind": None}
    if err or parsed["status"] != "ok":
        base["error"] = err or f"answerer output {parsed['status']}"
        return base
    answer = parsed["text"]
    claims, _ = extract_claims(answer)
    base.update(kind=parsed["kind"], n_claims=len(claims))
    reference = f"Q: {question}\nA: {gold}" if gold is not None and not expect_abstain else None
    prompt = build_judge_prompt(
        question=question, answer=answer, claims=claims, evidence=facts, reference_text=reference,
        expected_answer=gold, expect_abstain=expect_abstain, memory_shown=show_memory,
    )
    verdict_raw, err = _call(judge, prompt)
    verdict = parse_judge_output(verdict_raw if err is None else None, len(claims), require_correct=gold is not None and not expect_abstain)
    if err or not verdict["ok"]:
        base["error"] = err or f"judge output invalid: {verdict['error']}"
        return base
    base.update(
        errored=False, abstained=verdict["abstained"], verdicts=verdict["verdicts"], bases=verdict["bases"],
        answer_correct=verdict["answer_correct"],
        string_match=None if verdict["abstained"] or gold is None or expect_abstain else answer_in_text(gold, answer),
    )
    return base


def summarize(records: list[dict[str, Any]], threshold: float | None) -> dict[str, Any]:
    """Metrics for one list of records from a single condition."""
    gated = [
        dict(r, claims_dropped=True) if threshold is not None and r["score"] is not None and r["score"] < threshold else r
        for r in records
    ]
    system_items = [dict(r, abstained=True) if r.get("claims_dropped") else r for r in gated]
    return {
        "n": len(records),
        "errored": sum(r["errored"] for r in records),
        "answerer": claim_metrics(records),
        "system_after_gate": claim_metrics(system_items),
        "accuracy": accuracy_metrics([r for r in records if not r["should_abstain"]]),
    }


def run_grounded_eval(
    golden: list[dict], probes: list[dict], *, retriever: Callable, answerer: Callable, judge: Callable,
    answerer_model: str, judge_model: str, seed: int = DEFAULT_SEED, top_k: int = 5,
    target_precision: float = 0.9, allow_same_family: bool = False, manifest: dict | None = None,
) -> dict[str, Any]:
    for name, cb in (("retriever", retriever), ("answerer", answerer), ("judge", judge)):
        if not callable(cb):
            raise ValueError(f"{name} must be callable")
    if not allow_same_family:
        check_independent(answerer_model, judge_model)
    manifest = manifest or build_split(golden, probes, seed=seed)
    part_of = {k: p for p in ("dev", "heldout") for k in manifest[p]}
    records: dict[str, list[dict]] = {p: [] for p in ("dev", "heldout")}
    for row in golden:
        part = part_of[golden_key(row)]
        facts = _facts(retriever, row["question"], top_k)
        for cond, show in (("no_memory", False), ("grounded", True)):
            rec = run_item(condition=cond, question=row["question"], facts=facts, gold=row["answer"], expect_abstain=False,
                           answerer=answerer, judge=judge, show_memory=show)
            records[part].append(dict(rec, key=golden_key(row)))
    for row in probes:
        part = part_of[probe_key(row)]
        facts = _facts(retriever, row["question"], top_k)
        rec = run_item(condition="probe", question=row["question"], facts=facts, gold=None, expect_abstain=bool(row["expected_abstain"]),
                       answerer=answerer, judge=judge, show_memory=True)
        records[part].append(dict(rec, key=probe_key(row)))

    def gate_items(part: str) -> list[dict]:
        return [r for r in records[part] if r["condition"] in ("grounded", "probe")]

    calibration = calibrate_threshold(gate_items("dev"), target_precision=target_precision)  # DEV ONLY
    tau = calibration["threshold"]
    held = records["heldout"]
    heldout = {
        "threshold": tau,
        "split_sizes": {p: {"golden": sum(k.startswith("golden:") for k in manifest[p]),
                            "probe": sum(k.startswith("probe:") for k in manifest[p])} for p in ("dev", "heldout")},
        "abstention": evaluate_abstention(gate_items("heldout"), tau),
        "abstention_answerer_only": evaluate_abstention(gate_items("heldout"), None),
        **{c: summarize([r for r in held if r["condition"] == c], tau) for c in CONDITIONS},
    }
    return {"seed": manifest["seed"], "answerer_model": answerer_model, "judge_model": judge_model, "top_k": top_k,
            "calibration_dev": calibration, "heldout": heldout, "manifest": manifest, "records": records}


# ---------------------------------------------------------------------------------------------
# Offline plumbing smoke (stub callbacks; NOT a quality result)
# ---------------------------------------------------------------------------------------------
_TOKEN = re.compile(r"[a-z0-9]+")


def _stub_retriever(golden: list[dict]) -> Callable:
    passages = [(r["source"], r["answer"]) for r in golden]
    toks = [set(_TOKEN.findall(t.lower())) for _, t in passages]

    def retrieve(question: str, top_k: int) -> list[dict]:
        q = set(_TOKEN.findall(question.lower()))
        scored = sorted(((len(q & t) / (len(q | t) or 1), i) for i, t in enumerate(toks)), reverse=True)[:top_k]
        return [{"text": passages[i][1], "source": passages[i][0], "score": s} for s, i in scored if s > 0]
    return retrieve


def _stub_answerer(request: dict) -> str:
    if request["condition"] == "no_memory":
        return "I don't know."
    lines = [ln for ln in request["prompt"].splitlines() if ln.startswith("[")]
    return lines[0].split("|", 3)[-1].strip() if lines else "unverified: no relevant facts found."


def _stub_judge(prompt: str) -> str:
    abstained = "unverified" in prompt.split("ANSWER (untrusted):")[1][:60] or "I don't know" in prompt
    n = len(re.findall(r"^\d+\. ", prompt.split("NUMBERED CLAIMS")[1], re.M))
    return json.dumps({"verdicts": [{"claim_id": i, "verdict": "supported", "basis": "retrieved"} for i in range(1, n + 1)],
                       "abstained": abstained, "answer_correct": None if abstained else True})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", action="store_true", help="print the reproducible dev/held-out split counts")
    parser.add_argument("--offline-smoke", action="store_true", help="run the plumbing with stub callbacks (no model calls)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    from eval.eval_harness import ABSTAIN_PATH, GOLDEN_PATH, load_jsonl

    golden, probes = load_jsonl(GOLDEN_PATH), load_jsonl(ABSTAIN_PATH)
    if args.split:
        m = build_split(golden, probes, seed=args.seed)
        print(json.dumps({p: {"golden": sum(k.startswith("golden:") for k in m[p]), "probe": sum(k.startswith("probe:") for k in m[p])}
                          for p in ("dev", "heldout")}, indent=2))
        return 0
    if args.offline_smoke:
        out = run_grounded_eval(golden, probes, retriever=_stub_retriever(golden), answerer=_stub_answerer, judge=_stub_judge,
                                answerer_model="cc/claude-sonnet-5-5", judge_model="ag/gemini-stub", seed=args.seed)
        out.pop("records"), out.pop("manifest")
        print(json.dumps(out, indent=2, default=str))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
