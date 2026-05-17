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
import hashlib
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

TEMPLATES_ROOT = Path("templates/lingtai_native")

HUMAN_ENDPOINT_NOTE = (
    "Ticker-local human endpoint. Outer harness/codex reads this mailbox and "
    "relays selected messages to the real human UI."
)

HUMAN_INBOX_ACCESS = {
    "mailbox": {
        "inbox": {
            "nirvana": True,
        },
    },
}


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


def human_manifest(ticker: str) -> dict[str, object]:
    return {
        "address": "human",
        "agent_name": "human",
        "nickname": f"{ticker.upper()} frontend harness",
        "ticker": ticker.upper(),
        "note": HUMAN_ENDPOINT_NOTE,
        # Explicit null keeps this directory a pseudo-human endpoint rather
        # than a runnable agent; the kernel/TUI both treat admin:null as human.
        "admin": None,
        # The outer/admin harness may inspect the ticker-local human inbox.
        # Keep this as data rather than changing admin, so normal ticker
        # avatars do not gain karma/nirvana powers and the endpoint remains
        # a pseudo-agent.
        "access": HUMAN_INBOX_ACCESS,
    }


def _with_human_inbox_access(data: dict[str, object], *, ticker: str) -> dict[str, object]:
    updated = dict(data)
    updated["address"] = "human"
    updated["agent_name"] = "human"
    updated.setdefault("nickname", f"{ticker.upper()} frontend harness")
    updated["ticker"] = ticker.upper()
    updated.setdefault("note", HUMAN_ENDPOINT_NOTE)
    updated["admin"] = None

    access = updated.get("access")
    if not isinstance(access, dict):
        access = {}
    mailbox = access.get("mailbox")
    if not isinstance(mailbox, dict):
        mailbox = {}
    inbox = mailbox.get("inbox")
    if not isinstance(inbox, dict):
        inbox = {}
    inbox["nirvana"] = True
    mailbox["inbox"] = inbox
    access["mailbox"] = mailbox
    updated["access"] = access
    return updated


def ensure_human_endpoint(root: Path, *, ticker: str) -> list[Path]:
    human = network_root(root, ticker=ticker) / "human"
    paths = [
        human,
        human / "mailbox",
        human / "mailbox" / "outbox",
        human / "mailbox" / "inbox",
        human / "mailbox" / "sent",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = human / ".agent.json"
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = _with_human_inbox_access(data, ticker=ticker)
    else:
        manifest = human_manifest(ticker)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return paths + [manifest_path]


def ensure_vault_dirs(root: Path, *, ticker: str) -> list[Path]:
    base = company_root(root, ticker=ticker)
    paths = [
        base / "wiki",
        base / "team" / "raw",
        base / "team" / "drafts",
        base / "team" / "published",
        base / "team" / "published" / "versions",
    ]
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
    return paths + ensure_human_endpoint(root, ticker=ticker)


def _human_identity(root: Path, *, ticker: str) -> dict[str, object]:
    path = network_root(root, ticker=ticker) / "human" / ".agent.json"
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return _with_human_inbox_access(data, ticker=ticker)
    return human_manifest(ticker)


def _mailbox_id(now: datetime) -> str:
    return now.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(2)


def enqueue_request(root: Path, *, ticker: str, request: str, subject: str = "") -> Path:
    ensure_human_endpoint(root, ticker=ticker)

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


def latest_md_path(root: Path, *, ticker: str) -> Path:
    return company_root(root, ticker=ticker) / "team" / "published" / "latest.md"


def latest_state_path(root: Path, *, ticker: str) -> Path:
    return company_root(root, ticker=ticker) / "team" / "published" / ".latest_probe_state.json"


def latest_synthesis_state_path(root: Path, *, ticker: str) -> Path:
    return company_root(root, ticker=ticker) / "team" / "published" / ".latest_synthesis_state.json"


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latest_probe_question(root: Path, *, ticker: str) -> str:
    latest = latest_md_path(root, ticker=ticker)
    return f"""The harness observed that `{latest}` has changed since the last probe.

Do not treat this as a request for a checklist. Treat the current `latest.md` as the artifact under live adversarial review.

Free-form probe:

> If you walked into an investment committee with the current `latest.md`, what is the single question from a skeptical IC chair that would most embarrass the team because the answer could plausibly have been prepared within the current public-source/tool boundary — and why has that not already changed `latest.md`?

Answer that question first. If the answer reveals a material feasible upgrade, route the work and update `latest.md`. If it does not, explain why the answer belongs to true frontier rather than feasible work.
""".strip()


def enqueue_latest_probe_if_changed(root: Path, *, ticker: str) -> Path | None:
    latest = latest_md_path(root, ticker=ticker)
    current_hash = file_sha256(latest)
    if current_hash is None:
        raise SystemExit(f"missing latest.md for {ticker.upper()}: {latest}")

    state_path = latest_state_path(root, ticker=ticker)
    previous_hash = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_hash = state.get("latest_sha256")

    if previous_hash == current_hash:
        return None

    probe = enqueue_request(
        root,
        ticker=ticker,
        request=latest_probe_question(root, ticker=ticker),
        subject="Harness probe after latest.md update",
    )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state_path.write_text(
        json.dumps(
            {
                "ticker": ticker.upper(),
                "latest_path": str(latest),
                "latest_sha256": current_hash,
                "last_probe_queued_at": now,
                "last_probe_message": str(probe),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return probe


def latest_synthesis_question(root: Path, *, ticker: str) -> str:
    latest = latest_md_path(root, ticker=ticker)
    return f"""The harness observed that `{latest}` is ready for institutional-grade synthesis review.

Stop patching. Do not add another marginal diligence item merely because one can be found. Read the whole investment memo and judge the artifact as a capital-allocation document.

Core gate question:

> Read the whole investment memo. Do you think it is an institutional-grade memo that you would be willing to allocate capital with? If not, list what you could improve and improve it.

Interpretation:

- If the memo is not coherent enough for real capital-allocation discussion, rewrite / reorganize it into a committee-readable memo rather than appending another checklist section.
- First extract the core investment thesis spine, then consolidate the discovered diligence points under 3-5 decisive underwriting variables.
- Keep the main body focused on evidence that actually changes the recommendation; move secondary checks, edge risks, and monitoring details into an appendix / risk register / diligence backlog.
- If a new fact is truly necessary to reach institutional-grade quality, add it only if it changes the capital-allocation judgment; otherwise prioritize synthesis and structure.

Required output:

1. State whether the current memo is institutional-grade enough to allocate capital with, and why.
2. If not, name the improvements needed and perform them.
3. State the coherent thesis in one paragraph.
4. Name the 3-5 decisive variables that drive the recommendation.
5. Explain which existing sections were merged, demoted to appendix, or deleted as duplicative.
6. Update `latest.md` and the version archive/draft if you can improve coherence without weakening evidence.
7. If you cannot safely rewrite, explain exactly what blocks synthesis.

Recommendation discipline: preserve the actual investment conclusion unless the synthesis changes the conclusion for a clearly stated reason.
""".strip()


def enqueue_latest_synthesis_if_changed(root: Path, *, ticker: str) -> Path | None:
    latest = latest_md_path(root, ticker=ticker)
    current_hash = file_sha256(latest)
    if current_hash is None:
        raise SystemExit(f"missing latest.md for {ticker.upper()}: {latest}")

    state_path = latest_synthesis_state_path(root, ticker=ticker)
    previous_hash = None
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        previous_hash = state.get("latest_sha256")

    if previous_hash == current_hash:
        return None

    request = enqueue_request(
        root,
        ticker=ticker,
        request=latest_synthesis_question(root, ticker=ticker),
        subject="Harness synthesis after latest.md expansion",
    )
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    state_path.write_text(
        json.dumps(
            {
                "ticker": ticker.upper(),
                "latest_path": str(latest),
                "latest_sha256": current_hash,
                "last_synthesis_queued_at": now,
                "last_synthesis_message": str(request),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return request


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
    parser.add_argument(
        "--probe-latest-if-changed",
        action="store_true",
        help="Runtime mode: if team/published/latest.md changed since the last probe, queue one free-form IC-chair probe to <TICKER>_orchestrator.",
    )
    parser.add_argument(
        "--synthesize-latest-if-changed",
        action="store_true",
        help="Runtime mode: if team/published/latest.md changed since the last synthesis, queue a coherence/distillation request instead of another marginal diligence probe.",
    )
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

    if args.probe_latest_if_changed:
        path = enqueue_latest_probe_if_changed(root, ticker=ticker)
        if path is None:
            print(f"latest.md unchanged since last probe for {ticker}; no probe queued")
        else:
            print(f"queued latest.md update probe to {agent_name(ticker, 'orchestrator')}: {path}")

    if args.synthesize_latest_if_changed:
        path = enqueue_latest_synthesis_if_changed(root, ticker=ticker)
        if path is None:
            print(f"latest.md unchanged since last synthesis for {ticker}; no synthesis queued")
        else:
            print(f"queued latest.md synthesis request to {agent_name(ticker, 'orchestrator')}: {path}")

    if args.read_human:
        print_human_inbox(root, ticker=ticker)

    if not (args.send or args.probe_latest_if_changed or args.synthesize_latest_if_changed or args.read_human or args.apply_comments or args.render_comments or args.ensure_dirs):
        raise SystemExit("nothing to do; use --send for an existing team, --probe-latest-if-changed after a latest.md update, --synthesize-latest-if-changed for coherence distillation, --read-human for replies, or --render-comments/--apply-comments for setup")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
