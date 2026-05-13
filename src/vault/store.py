"""CRUD helpers for AlphaSeeker's persistent research vault."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any
import uuid

from src.vault.schema import connect_vault, init_vault


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _json_dumps(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True, default=str)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


@dataclass(frozen=True)
class VaultStore:
    """Small SQLite-backed store for durable company research state."""

    root: Path | str | None = None

    def __post_init__(self) -> None:
        init_vault(self.root)

    def connect(self) -> sqlite3.Connection:
        return connect_vault(self.root)

    def upsert_company(
        self,
        ticker: str,
        *,
        name: str | None = None,
        exchange: str | None = None,
        sector: str | None = None,
        updated_at: str | None = None,
    ) -> dict[str, Any]:
        ticker_norm = ticker.strip().upper()
        if not ticker_norm:
            raise ValueError("ticker is required")
        timestamp = updated_at or utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO companies(ticker, name, exchange, sector, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                  name=COALESCE(excluded.name, companies.name),
                  exchange=COALESCE(excluded.exchange, companies.exchange),
                  sector=COALESCE(excluded.sector, companies.sector),
                  updated_at=excluded.updated_at
                """,
                (ticker_norm, name, exchange, sector, timestamp),
            )
            row = conn.execute("SELECT * FROM companies WHERE ticker = ?", (ticker_norm,)).fetchone()
        result = _row_to_dict(row)
        assert result is not None
        return result

    def get_company(self, ticker: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM companies WHERE ticker = ?", (ticker.strip().upper(),)).fetchone()
        return _row_to_dict(row)

    def insert_document(
        self,
        *,
        source_type: str,
        path: str | Path,
        title: str | None = None,
        url: str | None = None,
        published_at: str | None = None,
        source_grade: str = "B",
        checksum: str | None = None,
        metadata: dict[str, Any] | None = None,
        doc_id: str | None = None,
        ingested_at: str | None = None,
    ) -> dict[str, Any]:
        resolved_doc_id = doc_id or new_id("doc")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO documents(
                  doc_id, source_type, title, path, url, published_at,
                  ingested_at, source_grade, checksum, metadata_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                  source_type=excluded.source_type,
                  title=excluded.title,
                  path=excluded.path,
                  url=excluded.url,
                  published_at=excluded.published_at,
                  ingested_at=excluded.ingested_at,
                  source_grade=excluded.source_grade,
                  checksum=excluded.checksum,
                  metadata_json=excluded.metadata_json
                """,
                (
                    resolved_doc_id,
                    source_type,
                    title,
                    str(path),
                    url,
                    published_at,
                    ingested_at or utc_now_iso(),
                    source_grade,
                    checksum,
                    _json_dumps(metadata),
                ),
            )
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (resolved_doc_id,)).fetchone()
        result = _row_to_dict(row)
        assert result is not None
        return result

    def link_document_company(self, doc_id: str, ticker: str, *, relevance: str = "mentioned") -> None:
        ticker_norm = ticker.strip().upper()
        if not ticker_norm:
            raise ValueError("ticker is required")
        self.upsert_company(ticker_norm)
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO document_companies(doc_id, ticker, relevance)
                VALUES (?, ?, ?)
                ON CONFLICT(doc_id, ticker) DO UPDATE SET relevance=excluded.relevance
                """,
                (doc_id, ticker_norm, relevance),
            )

    def list_documents(self, *, ticker: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self.connect() as conn:
            if ticker:
                rows = conn.execute(
                    """
                    SELECT d.*
                    FROM documents d
                    JOIN document_companies dc ON dc.doc_id = d.doc_id
                    WHERE dc.ticker = ?
                    ORDER BY COALESCE(d.published_at, d.ingested_at) DESC
                    LIMIT ?
                    """,
                    (ticker.strip().upper(), limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM documents
                    ORDER BY COALESCE(published_at, ingested_at) DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        return [dict(row) for row in rows]

    def add_fact(
        self,
        ticker: str,
        statement: str,
        *,
        section: str | None = None,
        source_doc_id: str | None = None,
        source_quote: str | None = None,
        source_grade: str | None = None,
        confidence: float = 0.5,
        observed_at: str | None = None,
        status: str = "active",
        fact_id: str | None = None,
    ) -> dict[str, Any]:
        ticker_norm = ticker.strip().upper()
        self.upsert_company(ticker_norm)
        resolved_id = fact_id or new_id("fact")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO facts(
                  fact_id, ticker, statement, section, source_doc_id, source_quote,
                  source_grade, confidence, observed_at, created_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fact_id) DO UPDATE SET
                  ticker=excluded.ticker,
                  statement=excluded.statement,
                  section=excluded.section,
                  source_doc_id=excluded.source_doc_id,
                  source_quote=excluded.source_quote,
                  source_grade=excluded.source_grade,
                  confidence=excluded.confidence,
                  observed_at=excluded.observed_at,
                  status=excluded.status
                """,
                (
                    resolved_id,
                    ticker_norm,
                    statement,
                    section,
                    source_doc_id,
                    source_quote,
                    source_grade,
                    confidence,
                    observed_at,
                    utc_now_iso(),
                    status,
                ),
            )
            row = conn.execute("SELECT * FROM facts WHERE fact_id = ?", (resolved_id,)).fetchone()
        result = _row_to_dict(row)
        assert result is not None
        return result

    def add_metric(
        self,
        ticker: str,
        metric_name: str,
        value: str,
        *,
        period: str | None = None,
        unit: str | None = None,
        source_doc_id: str | None = None,
        source_grade: str | None = None,
        observed_at: str | None = None,
        status: str = "active",
        metric_id: str | None = None,
    ) -> dict[str, Any]:
        ticker_norm = ticker.strip().upper()
        self.upsert_company(ticker_norm)
        resolved_id = metric_id or new_id("metric")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO metrics(
                  metric_id, ticker, metric_name, period, value, unit,
                  source_doc_id, source_grade, observed_at, created_at, status
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(metric_id) DO UPDATE SET
                  ticker=excluded.ticker,
                  metric_name=excluded.metric_name,
                  period=excluded.period,
                  value=excluded.value,
                  unit=excluded.unit,
                  source_doc_id=excluded.source_doc_id,
                  source_grade=excluded.source_grade,
                  observed_at=excluded.observed_at,
                  status=excluded.status
                """,
                (
                    resolved_id,
                    ticker_norm,
                    metric_name,
                    period,
                    value,
                    unit,
                    source_doc_id,
                    source_grade,
                    observed_at,
                    utc_now_iso(),
                    status,
                ),
            )
            row = conn.execute("SELECT * FROM metrics WHERE metric_id = ?", (resolved_id,)).fetchone()
        result = _row_to_dict(row)
        assert result is not None
        return result

    def add_question(
        self,
        ticker: str,
        question: str,
        *,
        priority: str = "normal",
        status: str = "open",
        question_id: str | None = None,
    ) -> dict[str, Any]:
        ticker_norm = ticker.strip().upper()
        self.upsert_company(ticker_norm)
        resolved_id = question_id or new_id("question")
        timestamp = utc_now_iso()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO questions(question_id, ticker, question, status, priority, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(question_id) DO UPDATE SET
                  ticker=excluded.ticker,
                  question=excluded.question,
                  status=excluded.status,
                  priority=excluded.priority,
                  updated_at=excluded.updated_at
                """,
                (resolved_id, ticker_norm, question, status, priority, timestamp, timestamp),
            )
            row = conn.execute("SELECT * FROM questions WHERE question_id = ?", (resolved_id,)).fetchone()
        result = _row_to_dict(row)
        assert result is not None
        return result

    def add_conflict(
        self,
        ticker: str,
        conflict_type: str,
        summary: str,
        *,
        left_ref: str | None = None,
        right_ref: str | None = None,
        severity: str = "medium",
        status: str = "open",
        conflict_id: str | None = None,
    ) -> dict[str, Any]:
        ticker_norm = ticker.strip().upper()
        self.upsert_company(ticker_norm)
        resolved_id = conflict_id or new_id("conflict")
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO conflicts(
                  conflict_id, ticker, conflict_type, summary, left_ref, right_ref,
                  severity, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (resolved_id, ticker_norm, conflict_type, summary, left_ref, right_ref, severity, status, utc_now_iso()),
            )
            row = conn.execute("SELECT * FROM conflicts WHERE conflict_id = ?", (resolved_id,)).fetchone()
        result = _row_to_dict(row)
        assert result is not None
        return result

    def company_context(self, ticker: str, *, limit: int = 20) -> dict[str, Any]:
        ticker_norm = ticker.strip().upper()
        with self.connect() as conn:
            company = _row_to_dict(conn.execute("SELECT * FROM companies WHERE ticker = ?", (ticker_norm,)).fetchone())
            documents = [
                dict(row)
                for row in conn.execute(
                    """
                    SELECT d.*
                    FROM documents d
                    JOIN document_companies dc ON dc.doc_id = d.doc_id
                    WHERE dc.ticker = ?
                    ORDER BY COALESCE(d.published_at, d.ingested_at) DESC
                    LIMIT ?
                    """,
                    (ticker_norm, limit),
                ).fetchall()
            ]
            facts = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM facts WHERE ticker = ? AND status = 'active' ORDER BY created_at DESC LIMIT ?",
                    (ticker_norm, limit),
                ).fetchall()
            ]
            metrics = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM metrics WHERE ticker = ? AND status = 'active' ORDER BY created_at DESC LIMIT ?",
                    (ticker_norm, limit),
                ).fetchall()
            ]
            questions = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM questions WHERE ticker = ? AND status = 'open' ORDER BY created_at DESC LIMIT ?",
                    (ticker_norm, limit),
                ).fetchall()
            ]
            conflicts = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM conflicts WHERE ticker = ? AND status = 'open' ORDER BY created_at DESC LIMIT ?",
                    (ticker_norm, limit),
                ).fetchall()
            ]
        return {
            "company": company,
            "documents": documents,
            "facts": facts,
            "metrics": metrics,
            "questions": questions,
            "conflicts": conflicts,
        }
