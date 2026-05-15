# First Manual Run Playbook

This playbook tests the LingTai-native design without adding any AlphaSeeker tools or runtime code.

## Goal

Run one ticker team manually and observe whether four LingTai avatars can coordinate using only:

- LingTai mail;
- LingTai pad/memory;
- LingTai file and bash capabilities;
- a minimal AlphaSeeker `raw/` + `published/` directory;
- English role prompts.

The goal is not to beat v8.1 memo quality on the first try. The goal is to learn what structure or tools are actually necessary.

## Setup

1. Pick a ticker, e.g. `TSLA`.
2. Create the minimal team directory:

   ```bash
   mkdir -p vault/companies/TSLA/team/raw
   mkdir -p vault/companies/TSLA/team/published/versions
   ```

   Or copy `examples/minimal_team_skeleton/`.

3. `raw/` may start empty. It is the landing zone where the source maintainer will place raw material it gathers during the run. If you already have local filings, snapshots, transcripts, or manual notes, you may put them there, but preloading raw material is not required.
4. Create four LingTai avatars using the existing LingTai workflow:

   ```text
   TSLA_orchestrator
   TSLA_source
   TSLA_writer
   TSLA_reviewer
   ```

5. Seed each avatar with the matching English system template from `templates/roles/` and the shared `policy.template.md`, replacing `<TICKER>` placeholders.

## Suggested first task

The human sends the orchestrator a message like:

```text
Please coordinate a first TSLA research memo. The raw landing zone is vault/companies/TSLA/team/raw/ and it may be empty at the start. Ask the source maintainer to gather or identify necessary raw material, ask the writer for a draft, ask the reviewer to challenge it, then as orchestrator cold-read the finished artifact against this original objective before publishing to vault/companies/TSLA/team/published/latest.md.
```

## Expected team flow

1. Orchestrator reads the policy and asks source maintainer to inspect `raw/` and gather missing material if needed.
2. Source maintainer explores `raw/`, gathers/records useful raw material if the directory is empty or incomplete, and mails writer/orchestrator with a source brief.
3. Orchestrator asks writer to draft.
4. Writer drafts using the source brief and raw files as needed.
5. Writer mails reviewer with the draft or draft path.
6. Reviewer checks the draft, asks source maintainer for support if needed, and mails issues to writer.
7. Writer revises or dissents.
8. Reviewer sends discovery feedback: material issues, strengths, and whether another cycle could materially improve the conclusion. Reviewer does not authorize publication and does not use `accept with caveats` as a verdict.
9. Orchestrator cold-reads the finished artifact against the original objective in natural language. If another cycle could materially improve a decisive gap, orchestrator routes it; otherwise final output is written to `published/latest.md` and optionally copied to `published/versions/<date>.md` with limitations/confidence stated plainly.
10. Orchestrator reports back to the human.

## What to observe

Record observations after the run:

- Did the avatars understand their roles?
- Did the source maintainer need a formal material index, or was raw exploration enough?
- Did writer/reviewer communication work naturally through mail?
- Did anyone need a prescribed workspace, or did each avatar self-organize?
- Was `published/latest.md` enough as the human-facing contract?
- Did the orchestrator cold-read the final artifact against the original objective rather than relying on reviewer approval?
- Did anyone use `accept with caveats` / `yes with caveats` to launder an unmet objective into publication?
- Which failures would be fixed by prompts, and which require tools?

## Escalation rule

If any avatar is blocked, it should mail the orchestrator with:

- what it tried;
- what is missing;
- whether it needs a teammate or the human;
- its recommended next action.
