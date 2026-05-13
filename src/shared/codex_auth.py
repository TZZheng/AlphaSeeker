"""Codex OAuth token manager for native ChatGPT Codex backend access.

Reads tokens written by LingTai/TUI (``~/.lingtai-tui/codex-auth.json``),
checks expiry, and auto-refreshes via the OpenAI OAuth endpoint. This lets
AlphaSeeker use Ted's Codex subscription without an ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import httpx
from filelock import FileLock

TOKEN_URL = "https://auth.openai.com/oauth/token"
CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
REFRESH_BUFFER_SECONDS = 300


class CodexAuthError(Exception):
    """Raised when Codex OAuth tokens cannot be refreshed."""


class CodexTokenManager:
    """Manage Codex OAuth tokens stored on disk by LingTai/TUI."""

    def __init__(self, token_path: str | None = None) -> None:
        if token_path is None:
            tui_dir = os.environ.get("LINGTAI_TUI_DIR", "~/.lingtai-tui")
            token_path = str(Path(tui_dir).expanduser() / "codex-auth.json")
        self._path = Path(token_path)
        self._lock_path = self._path.with_suffix(".json.lock")
        self._cache: dict | None = None
        self._cache_mtime: float = 0.0

    def is_authenticated(self) -> bool:
        """Return True if the token file exists and contains a refresh token."""
        try:
            data = self._read()
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return False
        return bool(data.get("refresh_token"))

    def get_access_token(self) -> str:
        """Return a valid access token, refreshing automatically if needed."""
        data = self._read()
        expires_at = data.get("expires_at", 0)
        if time.time() + REFRESH_BUFFER_SECONDS >= expires_at:
            self._refresh(data)
            data = self._read()
        return data["access_token"]

    def _read(self) -> dict:
        try:
            mtime = self._path.stat().st_mtime
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Codex token file not found: {self._path}. "
                "Authenticate in LingTai/TUI first so ~/.lingtai-tui/codex-auth.json exists."
            )

        if self._cache is not None and mtime == self._cache_mtime:
            return self._cache

        with self._path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        self._cache = data
        self._cache_mtime = mtime
        return data

    def _refresh(self, data: dict) -> None:
        lock = FileLock(self._lock_path, timeout=30)
        with lock:
            fresh = self._read()
            if fresh.get("expires_at", 0) > time.time() + REFRESH_BUFFER_SECONDS:
                return

            refresh_token = fresh.get("refresh_token") or data.get("refresh_token")
            if not refresh_token:
                raise CodexAuthError("No refresh_token available in Codex token file.")

            response = httpx.post(
                TOKEN_URL,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CLIENT_ID,
                },
                timeout=30,
            )
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (401, 403):
                    raise CodexAuthError(
                        "Codex session expired. Re-authenticate in LingTai/TUI to refresh ~/.lingtai-tui/codex-auth.json."
                    ) from exc
                raise

            result = response.json()
            fresh["access_token"] = result["access_token"]
            if "refresh_token" in result:
                fresh["refresh_token"] = result["refresh_token"]
            fresh["expires_at"] = result.get(
                "expires_at", int(time.time()) + result.get("expires_in", 3600)
            )

            tmp_path = self._path.with_suffix(".json.tmp")
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(fresh, f, indent=2)
            tmp_path.replace(self._path)

            self._cache = None
            self._cache_mtime = 0.0
