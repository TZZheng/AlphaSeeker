#!/usr/bin/env python3
"""Simulate the future frontend harness for a LingTai-native ticker team.

A ticker team should live in its own ticker-local LingTai network:

    vault/companies/<TICKER>/.lingtai/

Given a ticker and a natural-language request, this script simulates the
runtime side of the future frontend harness: it queues the request through the
ticker-local ``human`` pseudo-agent to the ticker-local
``<TICKER>_orchestrator``. Existing long-lived agents are not re-prompted on
every request; they keep their own memory, pad, and habits.

For first-time team setup or explicit reset, the same script can render the
thin team/role templates into per-agent ``init.json.comment`` text and patch
existing ticker-local ``init.json`` files with ``--apply-comments``.

Defaults are conservative: no init.json mutation unless ``--apply-comments``
is passed, and no mailbox write unless ``--send`` is passed.
"""

from __future__ import annotations

import argparse
import json
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

ROLE_TEMPLATES = {
    "orchestrator": "orchestrator.system.md",
    "source": "source_maintainer.system.md",
    "writer": "writer.system.md",
    "reviewer": "reviewer.system.md",
}

TEMPLATES_ROOT = Path("docs/research_platform/lingtai_native/templates")


def company_root(root: Path, *, ticker: str) -> Path:
    return root / "vault" / "companies" / ticker.upper()


def network_root(root: Path, *, ticker: str) -> Path:
    return company_root(root, ticker=ticker) / ".lingtai"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _render(text: str, *, ticker: str) -> str:
    return text.replace("<TICKER>", ticker.upper()).strip()


def _strip_heading(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
        while lines and not lines[0].strip():
            lines = lines[1:]
    return "\n".join(lines).strip()


def render_comment(root: Path, *, ticker: str, role: str) -> str:
    role = role.lower()
    if role not in ROLE_TEMPLATES:
        known = ", ".join(sorted(ROLE_TEMPLATES))
        raise SystemExit(f"unknown role {role!r}; expected one of: {known}")

    templates_root = root / TEMPLATES_ROOT
    policy = _strip_heading(_render(_read(templates_root / "policy.template.md"), ticker=ticker))
    role_body = _strip_heading(_render(_read(templates_root / "roles" / ROLE_TEMPLATES[role]), ticker=ticker))
    return f"{policy}\n\n{role_body}\n"


def roles_from_arg(raw: str) -> Iterable[str]:
    if raw == "all":
        return ROLE_TEMPLATES.keys()
    return [part.strip().lower() for part in raw.split(",") if part.strip()]


def agent_name(ticker: str, role: str) -> str:
    return f"{ticker.upper()}_{role.lower()}"


def init_json_path(root: Path, *, ticker: str, role: str) -> Path:
    return network_root(root, ticker=ticker) / agent_name(ticker, role) / "init.json"


def update_init_comment(root: Path, *, ticker: str, role: str) -> Path:
    path = init_json_path(root, ticker=ticker, role=role)
    if not path.exists():
        raise SystemExit(f"missing init.json for {agent_name(ticker, role)}: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    data["comment"] = render_comment(root, ticker=ticker, role=role).strip()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def ensure_vault_dirs(root: Path, *, ticker: str) -> list[Path]:
    base = company_root(root, ticker=ticker)
    paths = [
        base / "wiki",
        base / "team" / "raw",
        base / "team" / "drafts",
        base / "team" / "published",
        base / "team" / "published" / "versions",
        base / ".lingtai" / "human" / "mailbox" / "outbox",
        base / ".lingtai" / "human" / "mailbox" / "inbox",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _human_identity(root: Path, *, ticker: str) -> dict[str, object]:
    path = network_root(root, ticker=ticker) / "human" / ".agent.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data)
    return {"address": "human", "agent_name": "human", "via": "ticker-local-harness", "ticker": ticker.upper()}


def _mailbox_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(2)


def enqueue_request(root: Path, *, ticker: str, request: str, subject: str = "") -> Path:
    recipient = agent_name(ticker, "orchestrator")
    net = network_root(root, ticker=ticker)
    recipient_dir = net / recipient
    if not recipient_dir.exists():
        raise SystemExit(f"missing ticker-local orchestrator directory for {recipient}: {recipient_dir}")

    outbox = net / "human" / "mailbox" / "outbox"
    outbox.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc)
    mail_id = _mailbox_id(now)
    message_dir = outbox / mail_id
    message_dir.mkdir(parents=False, exist_ok=False)
    payload = {
        "id": mail_id,
        "_mailbox_id": mail_id,
        "from": "human",
        "to": [recipient],
        "cc": [],
        "subject": subject,
        "message": request,
        "type": "normal",
        "received_at": now.isoformat().replace("+00:00", "Z"),
        "identity": _human_identity(root, ticker=ticker),
    }
    (message_dir / "message.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return message_dir / "message.json"


def human_inbox(root: Path, *, ticker: str) -> Path:
    return network_root(root, ticker=ticker) / "human" / "mailbox" / "inbox"


def list_human_inbox(root: Path, *, ticker: str) -> list[Path]:
    inbox = human_inbox(root, ticker=ticker)
    if not inbox.exists():
        return []
    return sorted(inbox.glob("*/message.json"))


def print_human_inbox(root: Path, *, ticker: str) -> None:
    messages = list_human_inbox(root, ticker=ticker)
    if not messages:
        print(f"no ticker-local human inbox messages for {ticker.upper()}")
        return
    for path in messages:
        payload = json.loads(path.read_text(encoding="utf-8"))
        sender = payload.get("from", "")
        subject = payload.get("subject", "")
        received = payload.get("received_at", "")
        body = str(payload.get("message", ""))
        preview = body.replace("\n", " ")[:240]
        print(f"--- {path}")
        print(f"from: {sender}")
        print(f"subject: {subject}")
        print(f"received_at: {received}")
        print(preview)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="Ticker symbol, e.g. TSLA")
    parser.add_argument("request", nargs="?", help="Natural-language request for <TICKER>_orchestrator.")
    parser.add_argument("--root", default=".", help="AlphaSeeker repository root. Ticker network is vault/companies/<TICKER>/.lingtai under this root.")
    parser.add_argument(
        "--role",
        default="all",
        help="Setup-only role selection: orchestrator, source, writer, reviewer, or comma-separated list. Default: all.",
    )
    parser.add_argument("--render-comments", action="store_true", help="Print rendered init comments for setup review.")
    parser.add_argument("--apply-comments", action="store_true", help="Setup/reset mode: patch rendered comments into existing init.json files.")
    parser.add_argument("--ensure-dirs", action="store_true", help="Setup mode: create wiki/raw/drafts/published directories for the ticker.")
    parser.add_argument("--send", action="store_true", help="Runtime mode: queue the request to <TICKER>_orchestrator via ticker-local human outbox.")
    parser.add_argument("--read-human", action="store_true", help="Gateway mode: list messages sent to the ticker-local human inbox.")
    parser.add_argument("--subject", default="", help="Optional internal-mail subject for --send.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    ticker = args.ticker.upper()
    selected_roles = list(roles_from_arg(args.role))

    if args.ensure_dirs:
        for path in ensure_vault_dirs(root, ticker=ticker):
            print(f"ensured {path}")

    if args.apply_comments or args.render_comments:
        for role in selected_roles:
            if args.apply_comments:
                path = update_init_comment(root, ticker=ticker, role=role)
                print(f"updated {path}")
            else:
                print(f"--- {agent_name(ticker, role)} comment ---")
                print(render_comment(root, ticker=ticker, role=role).rstrip())

    if args.send:
        if not args.request:
            raise SystemExit("--send requires a request argument")
        path = enqueue_request(root, ticker=ticker, request=args.request, subject=args.subject)
        print(f"queued request to {agent_name(ticker, 'orchestrator')}: {path}")

    if args.read_human:
        print_human_inbox(root, ticker=ticker)

    if not (args.send or args.read_human or args.apply_comments or args.render_comments or args.ensure_dirs):
        raise SystemExit("nothing to do; use --send for an existing team, --read-human for replies, or --render-comments/--apply-comments for setup")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
