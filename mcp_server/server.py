"""Local stdio MCP interface for the project's append-only memory store."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "memory.db"
EMBEDDING_MODEL = "embeddinggemma"


def resolve_db_path(value: str | None = None) -> Path:
    raw = value if value is not None else os.environ.get("AI_MEMORY_DB")
    if raw is None:
        return DEFAULT_DB_PATH
    path = Path(raw).resolve()
    try:
        path.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise ValueError("AI_MEMORY_DB must resolve to a path under project data/") from exc
    return path


class MemoryTools:
    def __init__(self, db_path: str | Path | None = None,
                 embedder: Callable[[str], list[float] | None] | None = None,
                 query_embedding_cache_size: int = 0):
        # Imports here keep the package usable when only the MCP SDK is installed.
        from storage import MemoryStorage
        from lifecycle import FactLifecycleManager
        from grounding.answer_protocol import format_facts
        from retrieval.pipeline import MAX_QUERY_EMBEDDING_CACHE_SIZE, RetrievalPipeline

        # E16 query-embedding cache size: opt-in and disabled by default (0). The bound
        # mirrors the pipeline's own 0..MAX_QUERY_EMBEDDING_CACHE_SIZE validation, so an
        # out-of-range value is rejected here, before any database is opened, instead of
        # failing deeper in the pipeline.
        if (type(query_embedding_cache_size) is not int
                or not 0 <= query_embedding_cache_size <= MAX_QUERY_EMBEDDING_CACHE_SIZE):
            raise ValueError(
                f"query_embedding_cache_size must be an integer from 0 to {MAX_QUERY_EMBEDDING_CACHE_SIZE}")

        self.storage = MemoryStorage(resolve_db_path(str(db_path) if db_path is not None else None))
        self.storage.initialize_schema()
        self.lifecycle = FactLifecycleManager(self.storage)
        self.embedder = embedder
        self._last_embedding_failed = False
        self._format_facts = format_facts
        self.pipeline = RetrievalPipeline(
            self.storage,
            embedder=self._embed_query if self.embedder is not None else None,
            context_mode="source",
            source_score_mode="max",
            query_embedding_cache_size=query_embedding_cache_size,
        )

    def _embed_query(self, query: str) -> list[float] | None:
        if self.embedder is None:
            return None
        from embedding_bench.formatting import format_query
        try:
            vector = self.embedder(format_query("gemma", query))
            if vector is None:
                self._last_embedding_failed = True
                return None
            return self.storage._validate_vector(vector, name="query_vector")
        except Exception:
            self._last_embedding_failed = True
            return None

    async def save(self, text: str, source: str | None = None, writer: str | None = None,
                   tags: list[str] | None = None, confidence: float = 1.0,
                   entity_key: str | None = None) -> dict[str, Any]:
        if tags is not None and (not isinstance(tags, list) or any(not isinstance(x, str) for x in tags)):
            raise ValueError("tags must be a list of strings")
        tag_text = ",".join(tags or [])
        self.storage._validate_fact_input(text, source, writer, tag_text, confidence)
        vector = None
        if self.embedder is not None:
            from embedding_bench.formatting import format_document
            try:
                vector = self.embedder(format_document("gemma", text, source or ""))
                if vector is not None:
                    vector = self.storage._validate_vector(vector, name="vector")
            except Exception:
                vector = None
        fact_id = self.lifecycle.create_fact(
            text, source=source, writer=writer,
            tags=tag_text, confidence=confidence,
            entity_key=entity_key, vector=vector,
        )
        result = {"fact_id": fact_id, "vectorized": vector is not None}
        if self.embedder is not None and vector is None:
            result["warning"] = "Embedding unavailable; fact saved without a dense vector."
        return result

    async def recall(self, query: str, limit: int = 10,
                     filters: dict[str, Any] | None = None) -> dict[str, Any]:
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("query must be a non-empty string of at most 1000 characters")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError("filters must be an object")
        allowed = {"source", "writer", "status", "tags"}
        if filters and set(filters) - allowed:
            raise ValueError("unsupported filter key")
        def include(row: dict[str, Any]) -> bool:
            if row.get("status") != "active":
                return False
            if not filters:
                return True
            return all(
                (row.get(key) == value if key != "tags"
                 else str(value) in (row.get("tags") or "").split(","))
                for key, value in filters.items()
            )

        self._last_embedding_failed = False
        rows = self.pipeline.search(query, limit=limit, filter_func=include)
        mode = "hybrid" if self.embedder is not None and not self._last_embedding_failed else "keyword_only"
        facts = []
        for row in rows:
            fact = {key: value for key, value in row.items() if key != "vector"}
            fact["fact_id"] = f"F{fact['id']}"
            fact["source_confidence"] = fact.get("confidence")
            # The grounding threshold must use retrieval answerability, not the stored source-confidence field.
            fact["confidence"] = fact.get("answerability_score")
            facts.append(fact)
        warning = None if mode == "hybrid" else "Dense embedding unavailable; returned keyword-only results."
        return {
            "facts": facts,
            "retrieval_mode": mode,
            "warning": warning,
            "grounding_context": self._format_facts(facts),
        }

    async def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        conn = self.storage.conn
        assert conn is not None
        rows = conn.execute(
            "SELECT id,text,source,writer,created_at,valid_from,valid_to,status,superseded_by,confidence,tags "
            "FROM facts WHERE status='active' ORDER BY id DESC LIMIT ?", (min(limit, 1000),)
        ).fetchall()
        names = ["id", "text", "source", "writer", "created_at", "valid_from", "valid_to",
                 "status", "superseded_by", "confidence", "tags"]
        return [dict(zip(names, row)) for row in rows]

    async def health(self) -> dict[str, Any]:
        return {"status": "healthy" if self.storage.health_check() else "unhealthy",
                "database": str(self.storage.db_path)}

    async def supersede(self, old_fact_id: int, text: str, context: str,
                        entity_key: str | None = None) -> dict[str, Any]:
        # Mutation approval is intentionally unavailable until a validated local judge is configured.
        raise RuntimeError("local mutation judge is not configured; no fact was changed")

    async def retract(self, fact_id: int, context: str) -> dict[str, Any]:
        raise RuntimeError("local mutation judge is not configured; no fact was changed")


def build_server(memory: MemoryTools | None = None) -> Server:
    if memory is None:
        local_embedder: Callable[[str], list[float] | None] | None = None
        if os.environ.get("AI_MEMORY_DISABLE_EMBEDDING") != "1":
            from embedding_bench.client import OllamaEmbeddingClient
            client = OllamaEmbeddingClient(timeout=10)

            def ollama_embed(text: str) -> list[float]:
                vectors, _ = client.embed(EMBEDDING_MODEL, text, keep_alive="30m")
                return vectors[0]

            local_embedder = ollama_embed

        tools = MemoryTools(embedder=local_embedder)
    else:
        tools = memory

    async def list_tools(_ctx: Any, _params: Any) -> ListToolsResult:
        return ListToolsResult(tools=[
            Tool(name="memory_save", description="Append a fact to local memory", inputSchema={"type":"object","properties":{"text":{"type":"string"},"source":{"type":"string"},"writer":{"type":"string"},"tags":{"type":"array","items":{"type":"string"}},"confidence":{"type":"number","minimum":0,"maximum":1},"entity_key":{"type":"string"}},"required":["text"]}),
            Tool(name="memory_recall", description="Search active memory facts", inputSchema={"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","minimum":1},"filters":{"type":"object"}},"required":["query"]}),
            Tool(name="memory_recent", description="Return recent active memory facts", inputSchema={"type":"object","properties":{"limit":{"type":"integer","minimum":1}}}),
            Tool(name="memory_health", description="Check local memory database health", inputSchema={"type":"object","properties":{}}),
            Tool(name="memory_supersede", description="Propose a judge-approved supersede", inputSchema={"type":"object","properties":{"old_fact_id":{"type":"integer","minimum":1},"text":{"type":"string"},"context":{"type":"string"},"entity_key":{"type":"string"}},"required":["old_fact_id","text","context"]}),
            Tool(name="memory_retract", description="Propose a judge-approved retraction", inputSchema={"type":"object","properties":{"fact_id":{"type":"integer","minimum":1},"context":{"type":"string"}},"required":["fact_id","context"]}),
        ])

    async def call_tool(_ctx: Any, params: Any) -> CallToolResult:
        name = params.name
        arguments = params.arguments or {}
        methods = {
            "memory_save": tools.save, "memory_recall": tools.recall,
            "memory_recent": tools.recent, "memory_health": tools.health,
            "memory_supersede": tools.supersede, "memory_retract": tools.retract,
        }
        if name not in methods:
            raise ValueError("unknown tool")
        result = await methods[name](**arguments)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))])

    return Server("ai-memory-system", on_list_tools=list_tools, on_call_tool=call_tool)


async def run() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    asyncio.run(run())
