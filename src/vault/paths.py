"""Path helpers for the persistent research vault."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VaultPaths:
    """Resolved filesystem locations for one vault root."""

    root: Path

    @property
    def database_path(self) -> Path:
        return self.root / "vault.sqlite"

    @property
    def documents_dir(self) -> Path:
        return self.root / "documents"

    @property
    def companies_dir(self) -> Path:
        return self.root / "companies"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    def document_dir(self, doc_id: str) -> Path:
        return self.documents_dir / doc_id

    def company_dir(self, ticker: str) -> Path:
        return self.companies_dir / ticker.upper()

    def ensure(self) -> "VaultPaths":
        self.root.mkdir(parents=True, exist_ok=True)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.companies_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        return self


def default_vault_paths(root: str | Path | None = None) -> VaultPaths:
    """Return the default project-local vault paths.

    The default root is intentionally repo-relative so Obsidian can open
    `data/research_vault/` directly as a local Markdown vault.
    """

    return VaultPaths(Path(root) if root is not None else Path("data") / "research_vault")
