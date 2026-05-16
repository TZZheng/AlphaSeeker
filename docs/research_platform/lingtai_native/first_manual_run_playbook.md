# First Manual Run Playbook

This playbook tests the LingTai-native design without adding a separate AlphaSeeker agent runtime.

## Goal

Run one ticker-local team and observe whether four long-lived LingTai avatars can coordinate using only:

- ticker-local LingTai mail;
- LingTai pad/memory;
- LingTai file and bash capabilities;
- thin per-agent `init.json.comment` role text;
- a minimal AlphaSeeker vault surface;
- an outer harness that sends requests and reads ticker-local `human` replies.

The goal is not to beat v8.1 memo quality on the first try. The goal is to learn what structure or tools are actually necessary.

## Setup / initialization

1. Pick a ticker, e.g. `TSLA`.
2. Create or verify the minimal vault and ticker-local mailbox directories:

   ```bash
   python3 scripts/lingtai_ticker_harness.py TSLA --ensure-dirs
   ```

3. Create four LingTai avatars inside the ticker-local network if they do not already exist:

   ```text
   vault/companies/TSLA/.lingtai/TSLA_orchestrator
   vault/companies/TSLA/.lingtai/TSLA_source
   vault/companies/TSLA/.lingtai/TSLA_writer
   vault/companies/TSLA/.lingtai/TSLA_reviewer
   ```

   The ticker-local `vault/companies/TSLA/.lingtai/human/` is the frontend/harness endpoint for TSLA. It is not the project-level Terry/TUI.

4. For first-time setup or explicit reset, render the thin templates into each avatar's own `init.json.comment`:

   ```bash
   # inspect first
   python3 scripts/lingtai_ticker_harness.py TSLA --render-comments

   # apply only when creating/resetting the team
   python3 scripts/lingtai_ticker_harness.py TSLA --apply-comments
   ```

   Do **not** do this before every request. Existing long-lived agents keep their own memory and self-organization.

## Runtime request simulation

For an existing team, the frontend/harness simulator should only send a natural-language request to the ticker-local orchestrator:

```bash
python3 scripts/lingtai_ticker_harness.py TSLA \
  "帮我看看最近的新闻，有没有值得投资的地方" \
  --send
```

The script queues a human-style internal mail message via:

```text
vault/companies/TSLA/.lingtai/human/mailbox/outbox/<id>/message.json
```

The ticker-local LingTai kernel delivers it to `TSLA_orchestrator`.

To inspect what the ticker team sent back to the ticker-local human endpoint:

```bash
python3 scripts/lingtai_ticker_harness.py TSLA --read-human
```

The outer harness/codex layer decides what, if anything, to relay from that ticker-local human inbox to the real project-level human/TUI.

For a full report request:

```bash
python3 scripts/lingtai_ticker_harness.py RKLB \
  "写一份对 RKLB 的研究报告" \
  --send
```

## Expected team behavior

1. Orchestrator interprets the request in natural language.
2. If source/wiki context is missing or stale, orchestrator asks source to update it.
3. Writer drafts or revises from the wiki/source base.
4. Reviewer challenges whether the draft supports its conclusion.
5. Orchestrator answers: is this institutional-grade for a real capital-allocation decision?
6. If yes, orchestrator publishes to `vault/companies/<TICKER>/team/published/latest.md` and reports to ticker-local `human`.
7. If no, orchestrator states the needed improvement, assigns it to the right teammate, and keeps the run open unless the human redirects or stops it.

## What to observe

Record observations after the run:

- Did the avatars understand their roles from only a few comment lines?
- Did ticker-local mail prevent internal team chatter from directly surfacing in the project-level TUI?
- Did the source maintainer need more structure than raw/wiki?
- Did writer/reviewer communication work naturally through mail?
- Did anyone need a prescribed workspace, or did each avatar self-organize?
- Was `published/latest.md` enough as the human-facing contract?
- Did the orchestrator make the publication judgment rather than relying on reviewer approval?
- Did the team avoid both lower-grade publication and blocker-note-as-final-output?
- Which failures would be fixed by prompts, and which require tools?
