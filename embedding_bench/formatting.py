"""
Per-model query/document input formatting for the embedding benchmark.

Every formatter is a pure function (no I/O, no network) so it can be unit-tested
offline. Sources are the primary model documentation (copies fetched by the
round-2 worker live in work/W2-embedding-quality/docs/):

  nomic-embed-text (Nomic Embed Text v1.5)
      https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
      "the text prompt *must* include a task instruction prefix ... embed your
      documents as `search_document: <text>` and your user queries as
      `search_query: <text>`".
  qwen3-embedding:0.6b (Qwen3-Embedding-0.6B)
      https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
      Queries: `Instruct: {one-sentence task}\\nQuery:{query}` (official
      `get_detailed_instruct`, note: no space after "Query:"). Documents:
      "No need to add instruction for retrieval documents" -> raw text.
      The model card's sentence-transformers prompts.query is
      "Instruct: Given a web search query, retrieve relevant passages that answer the query\\nQuery:".
  embeddinggemma (Google EmbeddingGemma 300M)
      https://ai.google.dev/gemma/docs/embeddinggemma/model_card (Prompt Instructions)
      Retrieval query: `task: search result | query: {content}`;
      document: `title: {title | "none"} | text: {content}`.
  bge-m3 (BAAI/bge-m3)
      https://huggingface.co/BAAI/bge-m3 (FAQ 2): "the BGE-M3 model no longer
      requires adding instructions to the queries" -> raw text for both sides;
      the card documents no document prefix either.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

QWEN3_RETRIEVAL_TASK = "Given a web search query, retrieve relevant passages that answer the query"


def _clean(text: str) -> str:
    return text.strip()


# --- raw ---------------------------------------------------------------------

def raw_query(text: str) -> str:
    return _clean(text)


def raw_document(text: str, source: str = "") -> str:
    return _clean(text)


# --- Nomic v1.5 --------------------------------------------------------------

def nomic_query(text: str) -> str:
    return f"search_query: {_clean(text)}"


def nomic_document(text: str, source: str = "") -> str:
    return f"search_document: {_clean(text)}"


# --- Qwen3-Embedding ---------------------------------------------------------

def qwen3_query(text: str, task: str = QWEN3_RETRIEVAL_TASK) -> str:
    return f"Instruct: {task}\nQuery:{_clean(text)}"


def qwen3_document(text: str, source: str = "") -> str:
    return _clean(text)  # documents are NOT prefixed


# --- EmbeddingGemma ----------------------------------------------------------

def gemma_query(text: str) -> str:
    return f"task: search result | query: {_clean(text)}"


def gemma_document(text: str, source: str = "") -> str:
    title = _clean(source) if source and source.strip() else "none"
    return f"title: {title} | text: {_clean(text)}"


@dataclass(frozen=True)
class Formatter:
    name: str
    query: Callable[[str], str]
    document: Callable[..., str]
    doc_url: str


FORMATTERS: Dict[str, Formatter] = {
    "raw": Formatter("raw", raw_query, raw_document, ""),
    "nomic": Formatter("nomic", nomic_query, nomic_document,
                       "https://huggingface.co/nomic-ai/nomic-embed-text-v1.5"),
    "qwen3": Formatter("qwen3", qwen3_query, qwen3_document,
                       "https://huggingface.co/Qwen/Qwen3-Embedding-0.6B"),
    "gemma": Formatter("gemma", gemma_query, gemma_document,
                       "https://ai.google.dev/gemma/docs/embeddinggemma/model_card"),
    "bge-m3": Formatter("bge-m3", raw_query, raw_document, "https://huggingface.co/BAAI/bge-m3"),
}


def get_formatter(name: str) -> Formatter:
    try:
        return FORMATTERS[name]
    except KeyError:
        raise ValueError(f"Unknown formatter '{name}'. Known: {sorted(FORMATTERS)}")


def format_query(formatter: str, text: str) -> str:
    return get_formatter(formatter).query(text)


def format_document(formatter: str, text: str, source: str = "") -> str:
    return get_formatter(formatter).document(text, source)


# --- Benchmark variants ------------------------------------------------------

@dataclass(frozen=True)
class Variant:
    """One benchmark run: an Ollama model + an input-formatting policy."""
    key: str                 # result key; equals the model name for the primary variant
    model: str               # Ollama model name
    formatter: str           # key into FORMATTERS
    ablation_of: Optional[str] = None  # key of the primary variant this ablates


VARIANTS: List[Variant] = [
    Variant("nomic-embed-text", "nomic-embed-text", "nomic"),
    Variant("nomic-embed-text@raw", "nomic-embed-text", "raw", ablation_of="nomic-embed-text"),
    Variant("qwen3-embedding:0.6b", "qwen3-embedding:0.6b", "qwen3"),
    Variant("embeddinggemma", "embeddinggemma", "gemma"),
    Variant("bge-m3", "bge-m3", "bge-m3"),
]

_VARIANTS_BY_KEY = {v.key: v for v in VARIANTS}


def resolve_variants(tokens: Optional[List[str]] = None) -> List[Variant]:
    """Resolve CLI/API tokens to variants.

    None -> all variants (primary runs plus the Nomic raw ablation).
    A token equal to a variant key selects that variant; any other token is treated as
    an Ollama model name with the raw formatter (custom model, no prefix assumptions).
    """
    if tokens is None:
        return list(VARIANTS)
    out: List[Variant] = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        out.append(_VARIANTS_BY_KEY.get(tok) or Variant(tok, tok, "raw"))
    return out
