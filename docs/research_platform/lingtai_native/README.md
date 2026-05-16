# LingTai-Native AlphaSeeker Teams

This folder defines the first minimal design for running AlphaSeeker research as long-lived LingTai ticker teams.

The v0 contract is intentionally small:

> AlphaSeeker provides ticker/team setup context, a source-material landing zone, and a human-facing published folder. LingTai avatars do all intermediate reasoning, communication, logging, memory, and organization themselves.

The runtime path is also small: for an existing ticker team, the harness sends a natural-language request to `<TICKER>_orchestrator`. It does not rebuild prompts on every request; long-lived agents keep their own memory, pad, and self-organization.

## What is here

- `architecture.md` — boundary between LingTai and AlphaSeeker.
- `minimal_team_structure.md` — the v0 filesystem contract: source material, wiki, drafts, and published output.
- `first_manual_run_playbook.md` — how to manually run or simulate a ticker team.
- `source_data_tools.md` — first LingTai-registerable yfinance/SEC source tools.
- `automatic_orchestrator_questions.md` — minimal publication-gate question for orchestrator judgment.
- `roles/` — design notes for the four initial avatars.
- `templates/` — thin policy/role templates used at ticker-team initialization time.
- `examples/minimal_team_skeleton/` — copyable `raw/` + `published/` skeleton.
- `../../../scripts/lingtai_ticker_harness.py` — small frontend-harness simulator: render/apply init comments for setup, or send a request to an existing orchestrator.

## Prompt layering

General LingTai behavior belongs to LingTai itself: covenant, tools, procedures, mail, memory, lifecycle, and autonomy. AlphaSeeker should not duplicate those in ticker prompts.

AlphaSeeker contributes only a few lines of domain/team context:

1. ticker focus — the team primarily studies `<TICKER>`, while using peers/markets/macro when useful;
2. team standard — default to institutional-grade research for real capital allocation unless the human asks narrower;
3. file surfaces — raw source material, factual wiki, working drafts, and accepted published output;
4. role responsibility — orchestrator/source/writer/reviewer each get a few role-specific lines.

For new team setup, templates under `templates/` are expanded with `<TICKER>` and written into each agent's own `init.json.comment`. Comments are per-agent, not network-shared. For an existing long-lived team, do not reassemble or overwrite comments on every request; just send the request.

## What is deliberately not here

- Draft/review/publish CRUD tools.
- Material IDs or evidence-map schemas.
- Requests, logs, timelines, status files, or per-role workspaces.
- A panel.
- A separate AlphaSeeker runtime that replaces LingTai autonomy.
- Any integration with `memo_run` or the experimental in-process v9 team simulator.
