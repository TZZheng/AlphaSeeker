# Minimal Ticker Team File Structure

## Contract

For each ticker team, AlphaSeeker v0 prescribes only two directories:

```text
vault/companies/<TICKER>/team/
  raw/
  published/
```

This is the whole v0 filesystem contract.

## `raw/`

`raw/` is the material landing zone and shared shelf. It is intentionally unstructured and may start empty.

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

This avoids premature boundary decisions such as "what counts as R0001 vs. R0002?" Raw material is simply raw material, and the first run may begin with none of it preloaded.

## `published/`

`published/` is the human-facing output surface. This is the one place AlphaSeeker should keep organized because the human needs a stable place to read results.

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

- LingTai already has logs and histories.
- Requests should be LingTai mail.
- Per-avatar working state should live in each avatar's own LingTai space and pad.
- Source state should emerge from the source maintainer's practice, not from a premature external schema.
- The first manual run should reveal what structure is truly needed.

## Optional copyable skeleton

`examples/minimal_team_skeleton/` contains only:

```text
raw/.gitkeep
published/.gitkeep
published/versions/.gitkeep
```

Copy it to `vault/companies/<TICKER>/team/` for the first manual run.
