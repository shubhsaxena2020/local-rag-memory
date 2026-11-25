# Evaluation package report

## Corpus

- 165 unique question/answer/source rows, deduplicated from the prior 373 rows.
- 35 distinct source files. Every source reference resolved within the allowed Claude memory directory, and all 165 answers were verified as exact substrings of their cited source files by the orchestrator.
- 40 abstain probes carry explicit `expected_abstain: true` labels and reasons. They are hand-built lexical probes, not an independently validated challenge set; source coverage and semantic ambiguity remain limitations.
- Five conflicting duplicate question groups from the original corpus were omitted because the source did not support a reliable unambiguous resolution.

## Harness

`eval_harness.py` validates corpus structure without network/model calls. Its callback-based `evaluate_retrieval` measures source or exact-answer-text recall@1/@5, MRR, retrieval p50/p95 latency, optional claim-level unsupported rate with an independent boolean callback, and abstain precision when explicit labels and an abstain callback are supplied. Unavailable claim/abstain metrics are `null`, not fabricated zeroes. The baseline adapter is read-only, loopback-only, bounded, and uses a request timeout.

No end-to-end baseline or candidate retrieval benchmark has been run yet; this package report contains no benchmark scores.

## Verification

Run from the project root:

```powershell
python -m unittest discover -s eval/tests -v
python eval/eval_harness.py --help
python eval/eval_harness.py --validate
```

Orchestrator results: 5 unit tests passed; CLI validation reports `golden=165`, `abstain=40`, `sources=35`. Evidence is in `../evidence/eval-tests-final.txt`, `../evidence/eval-help.txt`, and `../evidence/eval-validate.txt`.
