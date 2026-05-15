# LingTai-Native AlphaSeeker Teams

This folder defines the first minimal design for running AlphaSeeker research as long-lived LingTai ticker teams.

The v0 contract is intentionally small:

> AlphaSeeker provides raw material and a human-facing published folder. LingTai avatars do all intermediate reasoning, communication, logging, memory, and organization themselves.

In v0 there are **no new tools**, **no panel**, **no setup CLI**, and **no in-process AlphaSeeker team runtime**. The goal is to manually run one ticker team using LingTai's existing primitives and learn what actually needs to be formalized later.

## What is here

- `architecture.md` — boundary between LingTai and AlphaSeeker.
- `minimal_team_structure.md` — the only v0 filesystem contract: `raw/` and `published/`.
- `first_manual_run_playbook.md` — how to manually run a ticker team once.
- `roles/` — design notes for the four initial avatars.
- `templates/` — English prompt/policy templates to seed LingTai avatars.
- `examples/minimal_team_skeleton/` — copyable `raw/` + `published/` skeleton.

## What is deliberately not here

- Source extraction tools.
- Draft/review/publish CRUD tools.
- Material IDs or evidence-map schemas.
- Requests, logs, timelines, status files, or per-role workspaces.
- A panel.
- Automatic avatar creation.
- Any integration with `memo_run` or the experimental in-process v9 team simulator.
