# Minimal Ticker Team File Structure

## Contract

For each ticker team, AlphaSeeker v0 prescribes only a ticker-local LingTai network plus a few stable file surfaces:

```text
vault/companies/<TICKER>/
  .lingtai/              # ticker-local LingTai network, gitignored runtime state
    human/               # frontend/harness endpoint for this ticker
    <TICKER>_orchestrator/
    <TICKER>_source/
    <TICKER>_writer/
    <TICKER>_reviewer/
  wiki/                  # maintained factual company wiki
  team/
    raw/                 # source-material landing zone
    drafts/              # working drafts and notes
    published/
      latest.md          # accepted human-facing output
      versions/          # optional accepted-output history
```

This is the whole v0 filesystem contract. LingTai owns mail, logs, memory, pads, lifecycle, and per-avatar working state inside the ticker-local `.lingtai/` network.

## `.lingtai/`

The ticker-local `.lingtai/` contains the team's avatars and its local `human` pseudo-agent. It is runtime state and should not be committed.

The outer harness writes requests to:

```text
vault/companies/<TICKER>/.lingtai/human/mailbox/outbox/
```

The ticker team replies to:

```text
vault/companies/<TICKER>/.lingtai/human/mailbox/inbox/
```

The outer harness decides what to relay from this ticker-local inbox to the real project-level human/TUI.

## `raw/`

`raw/` is the material landing zone and shared shelf. It is intentionally lightly structured and may start empty.

Examples:

```text
raw/
  tsla_2025_10k.html
  tsla_2026_q1_10q.pdf
  sec_submissions.json
  yfinance_snapshot_2026-05-15.json
  investor_day_transcript.txt
  notes_from_manual_source.md
```

There is no required material ID scheme, no `meta.json`, no evidence-map schema, and no required directory layout. The source maintainer may gather files into `raw/`, create folders, or add notes if that helps, but AlphaSeeker v0 does not require it.

## `wiki/`

`wiki/` is the maintained factual company layer. It is where the source maintainer maps raw material into durable facts the writer and reviewer can start from.

The wiki should stay factual and source-linked. It may include peer or market context when that helps understand `<TICKER>`, but valuation debate, memo logic, final recommendations, and reviewer comments belong in drafts/reviews, not in the wiki.

## `drafts/`

`drafts/` is a working surface for memo drafts, blocker notes, and intermediate research notes that should not be mistaken for accepted output.

The writer owns normal draft work here. Writing to `published/latest.md` is reserved for orchestrator-approved publication after the institutional-grade gate passes.

## `published/`

`published/` is the human-facing output surface. This is the one place AlphaSeeker should keep organized because the human needs a stable place to read accepted results.

Recommended layout:

```text
published/
  latest.md
  versions/
    2026-05-15.md
```

Only `latest.md` is conceptually required. `versions/` is recommended so the team can preserve accepted historical outputs.

## What v0 does not prescribe

Do not create these as part of the v0 contract:

```text
status.md
status.json
timeline.jsonl
materials/state/
requests/
logs/
workspaces/orchestrator/
workspaces/source/
workspaces/writer/
workspaces/reviewer/
index.json
evidence_map.json
freshness.md
source_notes.md
```

Reasons:

- LingTai already has logs and histories inside the ticker-local network.
- Requests should be ticker-local LingTai mail.
- Per-avatar working state should live in each avatar's own LingTai space and pad.
- Source state should emerge from the source maintainer's practice, not from a premature external schema.
- Real runs should reveal what extra structure is truly needed.

## Optional copyable skeleton

`examples/minimal_team_skeleton/` contains only the oldest raw/published skeleton. New setup should prefer:

```bash
python3 scripts/lingtai_ticker_harness.py <TICKER> --ensure-dirs
```
