-- Schema version 1: Initial schema for AI Memory System
-- Uses WAL mode, FTS5 for keyword search, and stores embeddings as BLOB for vector search (numpy fallback or sqlite-vec if available)

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- Main facts table
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    source TEXT,
    writer TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    valid_from TIMESTAMP,
    valid_to TIMESTAMP,
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by INTEGER REFERENCES facts(id),
    confidence REAL NOT NULL DEFAULT 1.0,
    tags TEXT, -- CSV or JSON, application-defined
    vector BLOB -- Embedding vector as BLOB of float32 values (4 bytes per dimension)
);

-- FTS5 virtual table for keyword search on facts.text
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text,
    content='facts',
    content_rowid='id',
    tokenize='unicode61 remove_diacritics 2'
);

-- Triggers to keep FTS5 in sync with facts table
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO facts_fts(rowid, text) VALUES(new.id, new.text);
END;

-- Schema migrations tracking
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial version if not present
INSERT OR IGNORE INTO schema_migrations (version) VALUES (1);