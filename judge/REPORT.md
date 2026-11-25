# Round 2 contradiction-judge decision

The labeled set has 83 pairs. Neither tested local model reached the 90% accuracy gate: `qwen3:1.7b` scored 55.4% exact-label accuracy and 89.2% mutation accuracy; `gemma3:1b` scored 33.7% and 62.7%. Deterministic contradiction rules remain primary, with `qwen3:1.7b` reserved only for the ambiguous band. Evidence: `../evidence/judge-qwen3-1.7b.txt` and `../evidence/judge-gemma3-1b.txt`. The Round 2 judge suite passes 39 tests (`../evidence/r2-final-judge-tests.txt`).

---

# Historical Round 1 worker report

## Files Modified
- `C:\Users\shubh\ai-memory-system\judge\judge_harness.py`: Updated harness with schema validation, strict label parsing, Ollama options fix, latency measurement
- `C:\Users\shubh\ai-memory-system\judge\tests\test_judge_harness.py`: Unit tests covering dataset schema mismatch/duplicates, strict parser, confusion metrics, accuracy, and length mismatch

## Dataset Status
The benchmark dataset (`benchmark_data.jsonl`) already had correct category/schema consistency from previous work:
- Label-to-category mapping is consistent across all examples per `LABEL_NAMES`
- No modifications were needed to the dataset as it was already correct

## Harness Improvements Made
### 1. Schema Validation
Enhanced `load_benchmark_data()` to validate:
- Required fields present and correct type (`id`: int, `fact1`/`fact2`: non-empty string, `label`: int 0-5, `category`: string)
- Label and category must agree according to `LABEL_NAMES` mapping
- Unique IDs across dataset
- Non-empty fact strings
- Rejects malformed lines with line-numbered ValueError

### 2. Strict Label Parsing
Added `parse_label(text)` function that:
- Accepts only a complete single digit 0-5 after stripping whitespace
- Rejects prose, incidental digits, empty and out-of-range output
- Used in `call_ollama_judge` for strict response parsing

### 3. Ollama Request Fields Fixed
Corrected `call_ollama_judge()`:
- Moved generation parameters (`temperature`, `num_predict`) into the `options` object as required by Ollama API
- Uses strict label parsing (no extraction of first digit from arbitrary prose)
- Does not print model responses or secrets
- Includes basic bounds checking on input facts

### 4. Latency Measurement Added
Enhanced `run_benchmark_on_file()`:
- Measures every call with `time.perf_counter()`
- Returns per-example seconds, total_seconds, and p50/p95 seconds
- Returns classification metrics (mutation metrics and full accuracy)
- Does not include fabricated results - only real measurements when executed

### 5. Ollama Runner Dormant
The Ollama API call remains dormant unless explicitly invoked:
- `run_benchmark_on_file()` only makes HTTP requests when called by the orchestrator
- Simple `python judge_harness.py` only loads and displays dataset statistics
- No unintended model loading or API calls occur during normal import/use

### 6. Unit Tests
The test file `judge/tests/test_judge_harness.py` contains runnable unit tests covering:
- Dataset schema mismatch and duplicate detection
- Strict label parser validation
- Binary metrics computation with known confusion matrix
- Full accuracy computation
- Length mismatch detection between predictions and truths

## Important Notes
- **No candidate benchmark has been executed**: As instructed, we have not run any Ollama benchmark tests. The harness is ready but dormant.
- **All existing corrected data and tests preserved**: The benchmark data file and test file remain unchanged from their corrected state.
- **Only judge/harness.py modified**: No changes were made to other packages, configs, live services, Ollama models, GitHub, or Second Brain.

## Commands
To inspect the dataset (safe, no API calls):
```bash
# From the judge directory:
python judge_harness.py
```

To run the benchmark (to be executed only when the orchestrator explicitly does so):
```python
from judge_harness import run_benchmark_on_file
results = run_benchmark_on_file("benchmark_data.jsonl", model="qwen2.5:1.5b-instruct")
print(results)
```

To run unit tests (to be executed only when the orchestrator explicitly does so):
```bash
# From the judge directory:
python -m tests.test_judge_harness
```

## Limitations
1. **Heuristic Labels**: Dataset labels are heuristic-based; ambiguous category captures edge cases but category distinctions can be subjective.
2. **Size**: 60 examples provide a starting point; larger dataset would improve statistical significance.
3. **Domain Coverage**: Examples span general knowledge but may not cover specialized domains.
4. **Ollama Dependency**: Benchmark requires a local Ollama instance; no fallback mechanisms included.
5. **Binary Mutation Focus**: Metrics focus on binary mutation decision (update/retract vs. other); 6-category accuracy also reported.
6. **No Performance Claims**: As instructed, no execution or performance numbers are claimed in this report. The orchestrator must run the benchmark to obtain real metric numbers.
7. **Test Execution**: Unit tests are provided but not executed; orchestrator should run them to verify correctness.

## Provenance
- Modified by: Hermes Worker W3 (judge)
- Task: Fix defects found by orchestrator in existing root package - update judge_harness.py per specifications
- Location: `C:\Users\shubh\ai-memory-system\judge\`
# Round 2 local contradiction judge results

The Round 2 benchmark was expanded to 83 labeled pairs (33 mutations, 50 non-mutations) and now includes negation, numeric and unit changes, same-entity/different-attribute additions, explicit temporal changes, retractions, and paraphrases that must not trigger a contradiction. The independently reviewed dataset and bounded JSON harness are described in `../work/W4-judge-benchmark/REPORT.md`.

Offline verification: `python -m unittest discover -s judge/tests -v` passed 39 tests; `python -m compileall -q judge` passed. Evidence: `../evidence/r2-judge-tests.txt`.

| Ollama model | Exact 7-label accuracy | Binary mutation accuracy | False-mutation rate | Calls | p50 / p95 |
|---|---:|---:|---:|---:|---:|
| `qwen3:1.7b` | 55.4% | 89.2% | 10.0% | 83/83 | 2.33 / 3.64 s |
| `gemma3:1b` | 33.7% | 62.7% | 40.0% | 83/83 | 2.36 / 2.82 s |

Neither reaches the required 90% accuracy. Keep deterministic rules as the primary supersede/retract decision path; `qwen3:1.7b` is the better of these candidates and may be consulted only for an ambiguous band, with its result unable to override a deterministic confident decision. These results are from a synthetic, single-annotator set and do not establish production accuracy. Candidate outputs and aggregate metrics are in `../evidence/judge-qwen3-1.7b.json` and `../evidence/judge-gemma3-1b.json`.

# Historical report (Round 1; superseded above where it conflicts)
