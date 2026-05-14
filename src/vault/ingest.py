"""Document ingestion helpers for the persistent research vault."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

from pypdf import PdfReader

from src.vault.paths import default_vault_paths
from src.vault.store import VaultStore

_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".htm"}


def checksum_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _extract_pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
    return "\n\n".join(pages)


def extract_text(path: str | Path) -> str:
    """Extract readable text from a local document.

    MVP support is intentionally small: Markdown/text-like files and PDFs.
    """

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(source)
    if suffix in _TEXT_SUFFIXES:
        try:
            return source.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return source.read_text(encoding="latin-1")
    raise ValueError(f"Unsupported vault ingest file type: {suffix or '<none>'}")


def ingest_file(
    path: str | Path,
    *,
    ticker: str | None = None,
    source_type: str | None = None,
    title: str | None = None,
    source_grade: str = "B",
    source_grade_rationale: str | None = None,
    url: str | None = None,
    published_at: str | None = None,
    metadata: dict[str, Any] | None = None,
    root: str | Path | None = None,
    store: VaultStore | None = None,
) -> dict[str, Any]:
    """Copy a local source into the vault and register it in SQLite."""

    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    active_store = store or VaultStore(root)
    vault_paths = default_vault_paths(root).ensure()
    checksum = checksum_file(source)
    doc_id = f"doc_{checksum[:16]}"
    doc_dir = vault_paths.document_dir(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)

    original_name = f"original{source.suffix.lower()}" if source.suffix else "original"
    original_path = doc_dir / original_name
    extracted_path = doc_dir / "extracted.md"
    metadata_path = doc_dir / "metadata.json"

    if not original_path.exists() or checksum_file(original_path) != checksum:
        shutil.copy2(source, original_path)
    text = extract_text(source)
    extracted_path.write_text(text, encoding="utf-8")

    metadata_payload = {
        "source_path": str(source),
        "original_path": str(original_path),
        "extracted_path": str(extracted_path),
        **(metadata or {}),
    }
    if source_grade_rationale:
        metadata_payload["source_grade_rationale"] = source_grade_rationale
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")

    document = active_store.insert_document(
        doc_id=doc_id,
        source_type=source_type or source.suffix.lower().lstrip(".") or "file",
        title=title or source.stem,
        path=extracted_path,
        url=url,
        published_at=published_at,
        source_grade=source_grade,
        checksum=checksum,
        metadata=metadata_payload,
    )
    if ticker:
        active_store.link_document_company(doc_id, ticker, relevance="primary")
    return {
        "doc_id": doc_id,
        "document": document,
        "document_dir": str(doc_dir),
        "original_path": str(original_path),
        "extracted_path": str(extracted_path),
        "metadata_path": str(metadata_path),
        "text_chars": len(text),
    }


def ingest_text(
    text: str,
    *,
    ticker: str | None = None,
    source_type: str = "text",
    title: str | None = None,
    source_grade: str = "B",
    source_grade_rationale: str | None = None,
    url: str | None = None,
    published_at: str | None = None,
    metadata: dict[str, Any] | None = None,
    root: str | Path | None = None,
    store: VaultStore | None = None,
) -> dict[str, Any]:
    """Register raw text as a vault document without requiring a source file."""

    active_store = store or VaultStore(root)
    vault_paths = default_vault_paths(root).ensure()
    digest_input = "\n".join([source_type, title or "", url or "", published_at or "", text])
    checksum = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()
    doc_id = f"doc_{checksum[:16]}"
    doc_dir = vault_paths.document_dir(doc_id)
    doc_dir.mkdir(parents=True, exist_ok=True)
    extracted_path = doc_dir / "extracted.md"
    metadata_path = doc_dir / "metadata.json"
    extracted_path.write_text(text, encoding="utf-8")
    metadata_payload = {"extracted_path": str(extracted_path), **(metadata or {})}
    if source_grade_rationale:
        metadata_payload["source_grade_rationale"] = source_grade_rationale
    metadata_path.write_text(json.dumps(metadata_payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")

    document = active_store.insert_document(
        doc_id=doc_id,
        source_type=source_type,
        title=title,
        path=extracted_path,
        url=url,
        published_at=published_at,
        source_grade=source_grade,
        checksum=checksum,
        metadata=metadata_payload,
    )
    if ticker:
        active_store.link_document_company(doc_id, ticker, relevance="primary")
    return {
        "doc_id": doc_id,
        "document": document,
        "document_dir": str(doc_dir),
        "extracted_path": str(extracted_path),
        "metadata_path": str(metadata_path),
        "text_chars": len(text),
    }
