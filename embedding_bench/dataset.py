"""
Golden set dataset loader and repo-root robust path resolver.

Loads evaluation JSONL data, validates schema and required fields,
and creates structured corpus records and query targets for benchmark execution.
"""

import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def normalize_passage(text: str) -> str:
    """Canonical key for passage deduplication / exact matching.

    NFKC-normalize, collapse all whitespace runs to one space, strip, casefold.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().casefold()


@dataclass
class CorpusRecord:
    """Document record in the corpus to be indexed."""
    doc_id: str
    text: str
    source: str = ""
    original_ids: Set[Any] = field(default_factory=set)


@dataclass
class GoldenQuery:
    """Query item with ground truth target information."""
    query_id: Any
    question: str
    target_answer: str
    target_source: str = ""
    target_doc_id: str = ""


def resolve_golden_set_path(custom_path: Optional[str | Path] = None) -> Path:
    """Resolve the path to golden_set.jsonl robustly against different working directories.

    Search precedence:
    1. Explicit custom_path if provided (checked directly and relative to repo root).
    2. GOLDEN_SET_PATH environment variable if set.
    3. <repo_root>/eval/golden_set.jsonl (derived from package location).
    4. <cwd>/eval/golden_set.jsonl.
    5. <cwd>/golden_set.jsonl.
    6. <package_dir>/golden_set.jsonl.

    Raises FileNotFoundError if no candidate exists.
    """
    pkg_dir = Path(__file__).resolve().parent
    repo_root = pkg_dir.parent
    cwd = Path.cwd()

    if custom_path:
        p = Path(custom_path)
        if p.is_file():
            return p.resolve()
        candidate_repo = repo_root / p
        if candidate_repo.is_file():
            return candidate_repo.resolve()
        candidate_cwd = cwd / p
        if candidate_cwd.is_file():
            return candidate_cwd.resolve()
        raise FileNotFoundError(f"Specified golden set path does not exist: {custom_path}")

    env_path = os.environ.get("GOLDEN_SET_PATH")
    if env_path:
        ep = Path(env_path)
        if ep.is_file():
            return ep.resolve()

    search_candidates = [
        repo_root / "eval" / "golden_set.jsonl",
        cwd / "eval" / "golden_set.jsonl",
        cwd / "golden_set.jsonl",
        pkg_dir / "golden_set.jsonl",
    ]

    for cand in search_candidates:
        if cand.is_file():
            return cand.resolve()

    candidate_strs = [str(c) for c in search_candidates]
    raise FileNotFoundError(
        f"Could not locate golden_set.jsonl in any standard repo location. Checked: {candidate_strs}"
    )


def load_golden_set(
    filepath: Optional[str | Path] = None,
    deduplicate_answers: bool = True,
) -> Tuple[List[CorpusRecord], List[GoldenQuery]]:
    """Load golden set JSONL file and return (corpus_records, queries).

    Validates:
    - Line format must be valid JSON object.
    - 'question' field must be a non-empty string.
    - 'answer' (or 'text') field must be a non-empty string.
    - Raises ValueError with line number on malformed lines.

    Corpus generation:
    - If deduplicate_answers is True, identical answer passages are grouped into a
      single canonical CorpusRecord, linking all matching query IDs and sources.
    - Each GoldenQuery records its target answer, source, and target_doc_id.
    """
    resolved_path = resolve_golden_set_path(filepath)

    corpus_map: Dict[str, CorpusRecord] = {}
    corpus_list: List[CorpusRecord] = []
    queries: List[GoldenQuery] = []

    with open(resolved_path, "r", encoding="utf-8-sig") as f:
        for line_num, line in enumerate(f, 1):
            line_str = line.strip()
            if not line_str:
                continue

            try:
                record = json.loads(line_str)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} in {resolved_path.name}: {e}")

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_num} in {resolved_path.name} must be a JSON object")

            # Extract question
            question = record.get("question")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_num} has missing or empty 'question'")
            question_clean = question.strip()

            # Extract answer / text
            answer = record.get("answer") or record.get("text") or record.get("content")
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Line {line_num} has missing or empty 'answer'")
            answer_clean = answer.strip()

            source = str(record.get("source", "")).strip()
            item_id = record.get("id", line_num)

            if deduplicate_answers:
                norm_key = normalize_passage(answer_clean)
                if norm_key in corpus_map:
                    doc = corpus_map[norm_key]
                    doc.original_ids.add(item_id)
                    target_doc_id = doc.doc_id
                else:
                    doc_id = f"doc_{len(corpus_list)}"
                    doc = CorpusRecord(
                        doc_id=doc_id,
                        text=answer_clean,
                        source=source,
                        original_ids={item_id},
                    )
                    corpus_map[norm_key] = doc
                    corpus_list.append(doc)
                    target_doc_id = doc_id
            else:
                doc_id = f"doc_{item_id}"
                doc = CorpusRecord(
                    doc_id=doc_id,
                    text=answer_clean,
                    source=source,
                    original_ids={item_id},
                )
                corpus_list.append(doc)
                target_doc_id = doc_id

            queries.append(
                GoldenQuery(
                    query_id=item_id,
                    question=question_clean,
                    target_answer=answer_clean,
                    target_source=source,
                    target_doc_id=target_doc_id,
                )
            )

    return corpus_list, queries
