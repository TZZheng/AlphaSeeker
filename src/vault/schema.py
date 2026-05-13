"""SQLite schema and initialization for the persistent research vault."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.vault.paths import VaultPaths, default_vault_paths

SCHEMA_VERSION = 1

DDL = f"""
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,
  title TEXT,
  path TEXT NOT NULL,
  url TEXT,
  published_at TEXT,
  ingested_at TEXT NOT NULL,
  source_grade TEXT DEFAULT 'B',
  checksum TEXT,
  metadata_json TEXT DEFAULT '{{}}'
);

CREATE TABLE IF NOT EXISTS companies (
  ticker TEXT PRIMARY KEY,
  name TEXT,
  exchange TEXT,
  sector TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS document_companies (
  doc_id TEXT NOT NULL,
  ticker TEXT NOT NULL,
  relevance TEXT DEFAULT 'mentioned',
  PRIMARY KEY (doc_id, ticker),
  FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE,
  FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS facts (
  fact_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  statement TEXT NOT NULL,
  section TEXT,
  source_doc_id TEXT,
  source_quote TEXT,
  source_grade TEXT,
  confidence REAL DEFAULT 0.5,
  observed_at TEXT,
  created_at TEXT NOT NULL,
  supersedes_fact_id TEXT,
  status TEXT DEFAULT 'active',
  FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE,
  FOREIGN KEY (source_doc_id) REFERENCES documents(doc_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS metrics (
  metric_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  metric_name TEXT NOT NULL,
  period TEXT,
  value TEXT NOT NULL,
  unit TEXT,
  source_doc_id TEXT,
  source_grade TEXT,
  observed_at TEXT,
  created_at TEXT NOT NULL,
  status TEXT DEFAULT 'active',
  FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE,
  FOREIGN KEY (source_doc_id) REFERENCES documents(doc_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS questions (
  question_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  question TEXT NOT NULL,
  status TEXT DEFAULT 'open',
  priority TEXT DEFAULT 'normal',
  created_at TEXT NOT NULL,
  updated_at TEXT,
  FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS answers (
  answer_id TEXT PRIMARY KEY,
  question_id TEXT NOT NULL,
  answer TEXT NOT NULL,
  source_doc_ids TEXT DEFAULT '[]',
  confidence REAL DEFAULT 0.5,
  created_at TEXT NOT NULL,
  FOREIGN KEY (question_id) REFERENCES questions(question_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS conflicts (
  conflict_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  conflict_type TEXT NOT NULL,
  summary TEXT NOT NULL,
  left_ref TEXT,
  right_ref TEXT,
  severity TEXT DEFAULT 'medium',
  status TEXT DEFAULT 'open',
  created_at TEXT NOT NULL,
  resolved_at TEXT,
  FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS wiki_versions (
  version_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  path TEXT NOT NULL,
  run_id TEXT,
  created_at TEXT NOT NULL,
  summary TEXT,
  FOREIGN KEY (ticker) REFERENCES companies(ticker) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
CREATE INDEX IF NOT EXISTS idx_facts_ticker_status ON facts(ticker, status);
CREATE INDEX IF NOT EXISTS idx_metrics_ticker_name_period ON metrics(ticker, metric_name, period);
CREATE INDEX IF NOT EXISTS idx_questions_ticker_status ON questions(ticker, status);
CREATE INDEX IF NOT EXISTS idx_conflicts_ticker_status ON conflicts(ticker, status);

INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('schema_version', '{SCHEMA_VERSION}');
"""


def connect_vault(root: str | Path | None = None) -> sqlite3.Connection:
    """Open a SQLite connection for the vault, creating parent dirs if needed."""

    paths = default_vault_paths(root).ensure()
    conn = sqlite3.connect(paths.database_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_vault(root: str | Path | None = None) -> VaultPaths:
    """Create vault directories and initialize the SQLite schema."""

    paths = default_vault_paths(root).ensure()
    with sqlite3.connect(paths.database_path) as conn:
        conn.executescript(DDL)
    return paths
