"""Shared strict relevance rule used by eval/ and grounding/ per PLAN-24H Appendix A.

A retrieved item is relevant if and only if:
  * Exact gold passage/fact ID match when explicit (target_doc_id / target_fact_id), OR
  * The expected answer text appears in the retrieved text:
    - exact match after normalization, OR
    - full normalized answer containment subject to PLAN-24H Appendix A:
      * normalized answer is >= MIN_SUBSTANTIVE_CHARS (12 characters), OR
      * raw answer is a complete entity or number token, matching with word/token boundaries.

Matching `source` alone is NEVER a hit.
Normalization: Unicode NFKC, casefold, whitespace collapsed. Pure functions, no I/O.
"""

import re
import unicodedata
from typing import Any

# PLAN-24H Appendix A: answer must be at least 12 characters or a complete entity/number
MIN_SUBSTANTIVE_CHARS = 12

_SHORT_STOPWORDS = {
    "a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to"
}
_NUMBER_ANSWER = re.compile(
    r"^[+-]?(?:\d+(?:[.,]\d+)*(?:[-_/]\d+(?:[.,]\d+)*)*|\.\d+)(?:%|ms|s|kb|mb|gb)?$",
    re.I,
)
_ENTITY_ANSWER = re.compile(
    r"^(?:[A-Z][A-Za-z0-9._:/+-]*)(?:\s+[A-Z0-9][A-Za-z0-9._:/+-]*)*$"
)

_WS = re.compile(r"\s+")


def normalize_text(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return _WS.sub(" ", unicodedata.normalize("NFKC", value).casefold()).strip()


def answer_in_text(answer: Any, text: Any) -> bool:
    """True iff `answer` matches `text` per PLAN-24H Appendix A strict relevance:
    - Exact match after normalization, OR
    - Full normalized answer phrase containment in normalized text AND:
      - normalized answer length >= 12 characters, OR
      - raw answer is a complete entity token or complete number token
        (enforced with token/punctuation boundaries to prevent short substring hits).
    """
    if not isinstance(answer, str) or not isinstance(text, str):
        return False

    raw_answer = answer.strip()
    ans = normalize_text(raw_answer)
    txt = normalize_text(text)

    if not ans or not txt:
        return False

    # Exact normalized match is always a hit
    if ans == txt:
        return True

    # Full answer phrase containment for long answers (>= 12 characters)
    if len(ans) >= MIN_SUBSTANTIVE_CHARS:
        return ans in txt

    # Complete entity or number token matching for short answers
    is_number = bool(_NUMBER_ANSWER.fullmatch(raw_answer))
    is_entity = (
        len(ans) >= 3
        and ans not in _SHORT_STOPWORDS
        and bool(_ENTITY_ANSWER.fullmatch(raw_answer))
    )

    if not (is_number or is_entity):
        return False

    # Enforce punctuation / word token boundaries so "3" does not match "30",
    # and "Codex" does not match "Codexes".
    return re.search(rf"(?<!\w){re.escape(ans)}(?!\w)", txt, flags=re.UNICODE) is not None


def is_relevant(result: Any, expected: dict[str, Any]) -> bool:
    """Strict hit test for one retrieved result against one gold row per PLAN-24H Appendix A.
    `source` alone is never a hit.
    """
    if not isinstance(result, dict) or not isinstance(expected, dict):
        return False

    # Condition 1: Exact gold passage/fact ID match when explicit
    target_id = expected.get("target_fact_id") or expected.get("target_doc_id")
    if target_id is not None:
        result_id = result.get("doc_id") or result.get("id") or result.get("fact_id")
        if result_id is not None and str(result_id) == str(target_id):
            return True
        original_ids = result.get("original_ids")
        if isinstance(original_ids, (list, tuple, set)) and target_id in original_ids:
            return True

    # Condition 2: Full normalized answer containment subject to Appendix A rule
    answer = expected.get("answer") if "answer" in expected else expected.get("target_answer")
    text = result.get("text") if "text" in result else result.get("passage")

    return answer_in_text(answer, text)
