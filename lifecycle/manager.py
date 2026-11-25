"""Transactional, append-only lifecycle operations over the storage facts table."""

from datetime import datetime, timezone
from typing import Any, Callable

from storage import MemoryStorage

Judge = Callable[[str, dict[str, Any], dict[str, Any] | None, str], bool]
DuplicateDetector = Callable[[str, str | None], list[int]]


class FactLifecycleManager:
    def __init__(
        self,
        storage: MemoryStorage,
        conflict_judge: Judge | None = None,
        duplicate_detector: DuplicateDetector | None = None,
    ) -> None:
        if not isinstance(storage, MemoryStorage):
            raise TypeError("storage must be a MemoryStorage instance")
        self.storage = storage
        self.conflict_judge = conflict_judge
        self.duplicate_detector = duplicate_detector
        self.storage.initialize_schema()
        self.conn = self.storage.conn
        if self.conn is None:
            raise RuntimeError("storage connection did not initialize")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS fact_entity_keys (
                fact_id INTEGER NOT NULL REFERENCES facts(id),
                entity_key TEXT NOT NULL,
                PRIMARY KEY (fact_id, entity_key)
            );
            CREATE INDEX IF NOT EXISTS idx_fact_entity_keys_key ON fact_entity_keys(entity_key);
            CREATE TABLE IF NOT EXISTS fact_audit (
                id INTEGER PRIMARY KEY,
                fact_id INTEGER NOT NULL REFERENCES facts(id),
                action TEXT NOT NULL,
                context TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_fact_audit_fact_id ON fact_audit(fact_id);
            """
        )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")

    def _get(self, fact_id: int) -> dict[str, Any] | None:
        return self.storage.get_fact(fact_id)

    def _audit(self, fact_id: int, action: str, context: str) -> None:
        self.conn.execute(
            "INSERT INTO fact_audit(fact_id, action, context, created_at) VALUES(?,?,?,?)",
            (fact_id, action, context, self._now()),
        )

    def create_fact(
        self,
        text: str,
        *,
        source: str | None = "agent",
        writer: str | None = "ai-memory-system",
        tags: str | None = "",
        confidence: float = 1.0,
        vector: list[float] | None = None,
        entity_key: str | None = None,
        context: str = "",
    ) -> int:
        self.storage._validate_fact_input(text, source, writer, tags, confidence)
        vector = self.storage._validate_vector(vector) if vector is not None else None
        if entity_key is not None and (not isinstance(entity_key, str) or not entity_key.strip() or len(entity_key) > 500):
            raise ValueError("entity_key must be non-empty and at most 500 characters")
        if not isinstance(context, str) or len(context) > 2000:
            raise ValueError("context must be a string of at most 2000 characters")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            candidates: set[int] = set()
            if entity_key:
                candidates.update(
                    row[0]
                    for row in self.conn.execute(
                        "SELECT f.id FROM facts f JOIN fact_entity_keys k ON k.fact_id=f.id "
                        "WHERE k.entity_key=? AND f.status='active'",
                        (entity_key,),
                    )
                )
            if self.duplicate_detector is not None:
                candidates.update(self.duplicate_detector(text, entity_key))

            now = self._now()
            blob = self.storage._serialize_vector(vector) if vector is not None else None
            cur = self.conn.execute(
                "INSERT INTO facts(text,source,writer,valid_from,status,confidence,tags,vector) "
                "VALUES(?,?,?,?,'active',?,?,?)",
                (text, source, writer, now, confidence, tags, blob),
            )
            fact_id = int(cur.lastrowid)
            if entity_key:
                self.conn.execute(
                    "INSERT INTO fact_entity_keys(fact_id,entity_key) VALUES(?,?)",
                    (fact_id, entity_key),
                )
            self._audit(fact_id, "created", context)
            new_fact = self._get(fact_id)
            for old_id in sorted(candidates):
                if old_id == fact_id:
                    continue
                old_fact = self._get(old_id)
                if old_fact is None or old_fact["status"] != "active":
                    continue
                # A mutation judge is called only for a concrete supersede proposal.
                if self.conflict_judge is None:
                    continue
                approved = self.conflict_judge("supersede", old_fact, new_fact, context)
                if type(approved) is not bool:
                    raise ValueError("conflict_judge must return bool")
                if approved:
                    self.conn.execute(
                        "UPDATE facts SET status='superseded', superseded_by=?, valid_to=? "
                        "WHERE id=? AND status='active'",
                        (fact_id, now, old_id),
                    )
                    self._audit(old_id, "superseded", context)
            self.conn.commit()
            return fact_id
        except Exception:
            self.conn.rollback()
            raise

    def retract(self, fact_id: int, *, context: str) -> bool:
        if type(fact_id) is not int or fact_id <= 0:
            raise ValueError("fact_id must be a positive integer")
        if not isinstance(context, str) or not context.strip() or len(context) > 2000:
            raise ValueError("context must be non-empty and at most 2000 characters")
        fact = self._get(fact_id)
        if fact is None or fact["status"] != "active" or self.conflict_judge is None:
            return False
        approved = self.conflict_judge("retract", fact, None, context)
        if type(approved) is not bool:
            raise ValueError("conflict_judge must return bool")
        if not approved:
            return False
        now = self._now()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            cur = self.conn.execute(
                "UPDATE facts SET status='retracted', valid_to=? WHERE id=? AND status='active'",
                (now, fact_id),
            )
            if cur.rowcount != 1:
                self.conn.rollback()
                return False
            self._audit(fact_id, "retracted", context)
            self.conn.commit()
            return True
        except Exception:
            self.conn.rollback()
            raise

    def audit_history(self, fact_id: int) -> list[dict[str, Any]]:
        if type(fact_id) is not int or fact_id <= 0:
            raise ValueError("fact_id must be a positive integer")
        rows = self.conn.execute(
            "SELECT action, context, created_at FROM fact_audit WHERE fact_id=? ORDER BY id",
            (fact_id,),
        ).fetchall()
        return [{"action": row[0], "context": row[1], "created_at": row[2]} for row in rows]
