# Ticker Team Frontend Plan

Last updated: 2026-05-16

## Goal

Build a frontend for managing ticker-local LingTai research teams under:

```text
vault/companies/<TICKER>/.lingtai/
```

The intended operator flow is:

1. Search for a ticker.
2. Press Enter to open that ticker's management screen.
3. Use one pane to talk to the ticker team's orchestrator.
4. Use another pane to inspect the files the team produced, especially `team/published/latest.md` and version archives.
5. Trigger the controller actions that the current harness already exposes: create/prepare a ticker workspace, send a message, read team replies, queue memo grill/probe/synthesis loops, and eventually create a new ticker team end-to-end.

## Current state from inspection

### AlphaSeeker harness

The current bridge is `scripts/lingtai_ticker_harness.py`. It is deliberately filesystem-first and does **not** require activating a process to queue a request.

It currently supports:

- Resolving company and network roots:
  - `vault/companies/<TICKER>/`
  - `vault/companies/<TICKER>/.lingtai/`
- Ensuring vault/team directories and the ticker-local pseudo-human endpoint exist.
- Rendering role comments from `templates/lingtai_native/`.
- Patching existing ticker-local `init.json` files with role comments via `--apply-comments`.
- Writing an outbound pseudo-human mailbox message to `<TICKER>_orchestrator` via `--send`.
- Reading replies from the ticker-local `human` inbox via `--read-human`.
- Queueing two hash-guarded latest-memo loops:
  - `--probe-latest-if-changed`
  - `--synthesize-latest-if-changed`

Important limitation: the script prepares directories and patches existing agents, but it does not yet create a complete new four-agent ticker team from scratch. The frontend's “one-click new ticker” should therefore be implemented as a controller capability, not just a wrapper around current CLI flags.

### LingTai TUI / portal architecture

The LingTai Go repo confirms a clean boundary worth preserving:

- TUI and portal communicate with agents through the filesystem only: `.agent.json`, `.agent.heartbeat`, mailbox folders, logs, signal files, and `.notification/` payloads.
- TUI process launch is isolated in `tui/internal/process/launcher.go` (`python -m lingtai run <agent-dir>`).
- TUI mailbox writes are isolated in `tui/internal/fs/mail.go` (`WriteMail`). Pseudo-human sends write to `human/mailbox/outbox/<id>/message.json`, which is the same shape AlphaSeeker's harness writes.
- TUI chat rendering lives in `tui/internal/tui/mail.go`, but it is tightly coupled to Bubble Tea, the root app model, slash commands, a single selected orchestrator, and a project-root `.lingtai/` assumption.
- Portal already uses a React frontend + Go HTTP API over the same filesystem, but its domain is network visualization/replay rather than ticker-vault management.

Conclusion: reuse the **protocol and visual/interaction patterns** from LingTai TUI/portal first. Directly importing the current Go code into AlphaSeeker is not a good first step because most reusable pieces are under Go `internal/` packages and the UI models assume a single project-root network.

## Recommended architecture

Create an AlphaSeeker ticker-team controller with a UI on top.

```text
frontend UI
  ├─ ticker search / picker
  ├─ orchestrator chat pane
  ├─ file tree + markdown viewer
  └─ action buttons / loop controls

controller API / service layer
  ├─ ticker discovery and status
  ├─ mailbox read/write bridge
  ├─ process lifecycle controls
  ├─ one-click ticker-team setup
  ├─ latest.md probe/synthesis loop controls
  └─ safe file-tree reader

vault + ticker-local LingTai filesystem
  ├─ vault/companies/<TICKER>/wiki/
  ├─ vault/companies/<TICKER>/team/{raw,drafts,published}/
  └─ vault/companies/<TICKER>/.lingtai/
```

The first implementation can be either:

1. **Web frontend**: a small local HTTP server plus React/Svelte/Vite UI. This is best for file-tree browsing and markdown preview, and mirrors `lingtai-portal`'s architecture.
2. **Terminal frontend**: a new TUI using Bubble Tea/Textual-like split panes. This matches `lingtai-tui`'s feel but should still call a controller layer rather than embed the old `MailModel` directly.

Given the file-tree requirement, a web frontend is probably the faster path to a useful first version. A terminal version can come later if the operator wants the exact TUI feel.

## Process lifecycle decision

Do **not** make “open ticker management screen” imply “start the agent process.” Opening a ticker should be read-only/attach-by-default.

Separate these concepts explicitly:

| Concept | Meaning | UI action |
|---|---|---|
| Queue message | Write pseudo-human outbox mail to `<TICKER>_orchestrator`. Works even if no agent process is active. | Send / Queue |
| Wake process | Start or revive `python -m lingtai run <orchestrator-dir>`. Needed for immediate consumption of queued mail. | Wake / Start team |
| Send and wake | Queue the message, then ensure the orchestrator process is running. | Send & wake |
| Read replies | Read ticker-local `human/mailbox/inbox`. Works whether process is currently active or not. | Refresh messages |

This preserves the harness behavior Terry called out: messages can be queued without activation. It also gives users a clear control when they want immediate action.

The UI should show at least three status fields:

- Process status: running / stale heartbeat / suspended / absent.
- Mail status: queued outbound messages, latest inbound reply time.
- Artifact status: latest published memo hash and whether probe/synthesis loops have already been queued for that hash.

## Proposed controller API

These can be implemented as Python functions first, then surfaced through HTTP/CLI/UI.

### Ticker discovery

- `list_tickers(query: str | None) -> list[TickerSummary]`
  - Scans `vault/companies/*`.
  - Returns ticker, company root, whether `.lingtai/` exists, whether `team/published/latest.md` exists, latest modified times.

- `get_ticker_status(ticker) -> TickerStatus`
  - Reads filesystem status, orchestrator presence, heartbeat/process state, message counts, latest memo state hashes.

### Team setup

- `ensure_ticker_dirs(ticker)`
  - Equivalent to current `--ensure-dirs`.

- `ensure_human_endpoint(ticker)`
  - Equivalent to current pseudo-human endpoint creation.

- `create_ticker_team(ticker, preset/source_config)`
  - New capability. Creates `<TICKER>_orchestrator`, `<TICKER>_source`, `<TICKER>_writer`, `<TICKER>_reviewer` init files and manifests using `templates/lingtai_native/`.
  - Should be explicit about source preset/model and admin powers.
  - Should not auto-start processes unless the user chooses “create and start.”

- `apply_role_comments(ticker, roles=all)`
  - Equivalent to `--apply-comments`.

### Messaging

- `send_to_orchestrator(ticker, body, subject="") -> message_path`
  - Equivalent to current `--send`.
  - Queue-only by default.

- `read_human_inbox(ticker) -> list[Message]`
  - Equivalent to `--read-human`, but structured.

- `read_chat_timeline(ticker) -> Timeline`
  - Merge pseudo-human outbound messages, human inbox replies, and optionally orchestrator `logs/events.jsonl`/`history` for a TUI-like transcript.
  - This can be simple in v1: just sent/received mailbox messages.

### Process controls

- `process_status(ticker) -> ProcessStatus`
  - Check heartbeat freshness and optionally `ps` for `lingtai run <orchestrator-dir>`.

- `wake_orchestrator(ticker)`
  - Launch `python -m lingtai run <orchestrator-dir>` using the same runtime venv strategy as LingTai TUI.

- `suspend_orchestrator(ticker)` / `wake_all_team(ticker)` / `suspend_all_team(ticker)`
  - Later controls; keep v1 focused on orchestrator.

### File tree

- `list_files(ticker, subtree)`
  - Whitelist roots only: `wiki/`, `team/raw/`, `team/drafts/`, `team/published/`.
  - Hide `.lingtai/`, `.env`, caches, and guard files by default.

- `read_file(ticker, path)`
  - Read text/markdown files for preview.
  - Size limit and binary guard.

### Memo loops

- `queue_probe_if_latest_changed(ticker)`
  - Equivalent to `--probe-latest-if-changed`.

- `queue_synthesis_if_latest_changed(ticker)`
  - Equivalent to `--synthesize-latest-if-changed`.

- `configure_loop(ticker, loop_type, enabled, interval)`
  - Later: a persistent controller loop, probably backed by a small scheduler/daemon or cron. Store state outside `.lingtai/`, e.g. `vault/companies/<TICKER>/team/control/loops.json`.

## Milestones

### Milestone 1: Extract current harness into importable code

Move the core logic from `scripts/lingtai_ticker_harness.py` into an importable module, for example:

```text
src/vault/lingtai_team.py
```

Keep `scripts/lingtai_ticker_harness.py` as a thin CLI wrapper. Add unit tests for:

- `ensure_vault_dirs`
- `ensure_human_endpoint`
- `enqueue_request`
- `list_human_inbox`
- latest hash guard behavior

This makes the future frontend call Python functions instead of shelling out to a script.

### Milestone 2: Controller API + minimal UI

Implement a local controller API around the extracted module. First UI can be simple:

- left: ticker search/list
- center/right: file tree and markdown preview
- bottom/right: orchestrator message composer + reply list
- buttons: Queue message, Wake, Send & wake, Probe latest, Synthesize latest

### Milestone 3: One-click new ticker team

Add real team creation. This is the largest missing backend feature. The controller should create the network from templates and presets, validate all expected agent dirs/init files exist, then optionally wake the orchestrator.

### Milestone 4: Rich TUI-like transcript

Once the basic chat works, add session-log ingestion similar to LingTai TUI's `SessionCache`:

- mailbox messages
- orchestrator `logs/events.jsonl`
- soul/inquiry logs if useful
- notification/status footer

Do not block v1 on this. Mailbox-only is enough to operate the team.

### Milestone 5: Persistent grill/probe loop scheduler

Turn the hash-guarded one-shot operations into a visible loop controller:

- enabled/disabled
- last hash seen
- last queued message
- last successful wake/process status
- next scheduled check

For the first loop, reuse the current `latest.md` hash guard exactly so behavior stays predictable.

## Direct LingTai changes to consider later

If AlphaSeeker's frontend proves useful, there are two upstream LingTai refactors worth considering:

1. Extract a reusable, non-`internal` Go package for filesystem mailbox/session/process primitives.
2. Add a generic “attach to nested network” mode to `lingtai-tui`/`lingtai-portal`, so a caller can point at `vault/companies/<TICKER>/.lingtai/` without assuming it is the main project network.

Neither is required for the first AlphaSeeker frontend. The safer first move is to keep AlphaSeeker's controller in AlphaSeeker and mirror the stable filesystem protocol.

## Key design rule

The frontend should be a controller and observer, not an implicit lifecycle manager. It may queue work while agents are stopped; it may wake agents when explicitly asked; it should not secretly start long-lived processes merely because the user opened a ticker page.
