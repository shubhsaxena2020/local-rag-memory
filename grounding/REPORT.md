# Grounding Layer Report

## Round 2 live-evaluation status

The Round 2 harness is now exercised with Sonnet 5.5 answers and an independent Gemini-family judge through a local model router. Golden no-memory and grounded answerer requests are isolated into separate worker directories; outputs from the earlier mixed-condition batches are discarded. See `../evidence/r2-answerer-output-validation.json` for the exact counts and `../evidence/r2-independent-judge-validation.json` / `../evidence/r2-grounded-eval-summary.json` for final judge and held-out evidence when available.

A calibration defect was found and fixed: top-score extraction now prefers the query-specific retrieval `score` before the source reliability `confidence`, with confidence retained only as a fallback for older callers. A regression test first reproduced the wrong .95 value instead of the retrieval .03, then passed after the fix. Current grounding suite: 45 tests passed (`../evidence/r2-final-grounding-tests.txt`). Do not interpret the earlier offline smoke metrics below as Round 2 live results.

---


## Changes Made in This Round

### grounding.py
- Added `__post_init__` validation to the `Fact` dataclass to enforce:
  - Non-empty, size-limited `fact_id`, `text`, and `source` (max 256 characters)
  - Finite `confidence` within [0.0, 1.0]
  - `status` restricted to "active", "superseded", or "retracted"
  - Optional `superseded_by` validation (non-empty, size-limited if provided)
- Modified `format_grounded_response` to:
  - Abstain (return `grounded_or_abstain: "abstain"`) for facts with status "superseded" or "retracted", regardless of confidence, and include an explicit historical warning message (e.g., "unverified: historical fact (superseded). [f1] | Source: ... | Confidence: ... | Status: superseded | Superseded by: ...")
  - For "active" facts with sufficient confidence, include the exact `fact_text` in the response message (e.g., "... | Status: active | <fact_text>")
  - Preserve existing behavior for low-confidence abstention and no-facts case
- Updated `get_instruction_block` to add a warning line: "If the fact is superseded or retracted, the answer will be unverified and include a warning that this is a historical fact."

### evaluation_adapter.py
- Replaced `compute_unsupported_claim_rate(predictions, ground_truth)` with a claim-level version:
  - New signature: `compute_unsupported_claim_rate(supported_labels: list[bool])`
  - Computes unsupported claim rate as the proportion of `False` labels in the input list (independent evaluator judgments)
  - Returns 0.0 for empty input
- Retained the original implementation as `compute_unsupported_claim_rate_old` for reference
- Added `compute_citation_support_rate(predictions, ground_truth_support)`:
  - Measures retrieval citation membership: proportion of grounded predictions where the cited `fact_id` is in the corresponding ground-truth support set
  - Validates that `len(predictions) == len(ground_truth_support)`, raising `ValueError` on mismatch
  - Returns 0.0 if no grounded predictions or empty input
- Kept `compute_abstain_precision` but added length validation:
  - Now checks `len(predictions) == len(ground_truth)` and raises `ValueError` on mismatch
- Removed the combined `compute_metrics` function; metrics are now computed separately by callers

### tests/test_grounding.py
- Added discoverable unit tests using unittest.TestCase for:
  - Active response fact text
  - No result (no facts)
  - Low confidence
  - Superseded/retracted at high and low confidence
  - Field/status/confidence validation
  - Claim metric (unsupported claim rate)
  - Citation support
  - Abstain precision and mismatched lengths
  - Fact validation for blank/oversize fields, non-finite/out-of-range confidence, invalid statuses
- Test file is discoverable and runs with `python -m unittest discover -s grounding/tests -v` from repo root

### instruction_block.txt
- Added line: "If the fact is superseded or retracted, the answer will be unverified and include a warning that this is a historical fact."

## Summary
All changes strictly implement the specified behavior:
- `Fact` validation rejects invalid inputs as required
- `format_grounded_response` handles status-based abstention with historical warnings and includes exact fact text for active facts
- Evaluation adapter provides claim-level unsupported claim rate (via independent labels) and separate citation support rate, with length validation where appropriate
- Unit tests cover all validation, behavioral, and metric cases
- Instruction block updated to warn about superseded/retracted facts

## Important Notes
- No independent model evaluator was run for this round; all verification is based on the unit tests provided.
- All edits are confined to the specified files under `C:\Users\shubh\ai-memory-system\grounding`.
## Orchestrator verification
- From the repository root, `python -m unittest discover -s grounding/tests -v` passes 21 tests.
- `python -m compileall -q grounding` passes.
- The discovered tests required package-root exports/imports; historical-status responses now take precedence over low-confidence wording, and claim support labels reject non-bool values.
- Evidence: `../evidence/grounding-tests-final.txt` and `../evidence/grounding-compileall.txt`.
