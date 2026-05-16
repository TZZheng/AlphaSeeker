# AlphaSeeker on LingTai — Minimal Architecture

## Thesis

AlphaSeeker should not reimplement LingTai. LingTai is the runtime. AlphaSeeker is the domain recipe plus a thin frontend/harness gateway.

```text
AlphaSeeker = ticker-local LingTai placement + comment templates + vault file surfaces + request/relay gateway
LingTai     = avatar runtime + mail + pad + memory + molt + logs + tools + autonomy
```

## Ownership boundary

### LingTai owns agent life

LingTai already provides the things a long-lived research team needs:

- avatar identity and lifecycle;
- per-agent `init.json.comment` injection;
- internal mail between avatars and the local human pseudo-agent;
- per-avatar pad and durable memory;
- history, logs, and timeline-like audit trails inside each `.lingtai/` network;
- context molt and recovery;
- tool registration and MCP/capability hosting;
- idleness, wake-by-mail, and liveness.

AlphaSeeker v0 does not create replacements for any of these. In particular, v0 does **not** create team timelines, request queues, status dashboards, or per-avatar workspaces in the AlphaSeeker vault.

### AlphaSeeker owns the research contract and gateway

AlphaSeeker provides only the human/domain contract:

- where the ticker-local LingTai network lives;
- which ticker the team focuses on;
- which avatars form the team;
- what each avatar's short role responsibility is;
- where raw source material, factual wiki, working drafts, and human-facing published output live;
- how a frontend/harness sends a natural-language request to the ticker-local orchestrator;
- how the outer layer reads ticker-local `human` replies and decides what to relay to the real human UI.

The avatars decide how to organize intermediate thoughts, notes, disagreements, and logs using LingTai mail, pad, and their own working areas.

## Ticker-local network topology

A ticker team should live inside the ticker folder:

```text
vault/companies/<TICKER>/
  .lingtai/
    human/
    <TICKER>_orchestrator/
    <TICKER>_source/
    <TICKER>_writer/
    <TICKER>_reviewer/
  wiki/
  team/
    raw/
    drafts/
    published/
```

This avoids exposing every internal orchestrator/source/writer/reviewer mail to the project-level TUI. Inside a ticker-local network, `human` means the frontend/harness endpoint for that ticker. It is not automatically the real Terry-facing project-level `human`.

The outer harness/codex layer is responsible for:

1. writing the real user's request into `vault/companies/<TICKER>/.lingtai/human/mailbox/outbox/...`, addressed to `<TICKER>_orchestrator`;
2. reading `vault/companies/<TICKER>/.lingtai/human/mailbox/inbox/...` for ticker-team replies;
3. relaying only appropriate final/status messages to the real human UI;
4. leaving internal ticker-team chatter inside the ticker-local network.

## Prompt/comment layering

The kernel already supplies LingTai's general behavior. AlphaSeeker role text should therefore be only a few lines.

For **new ticker-team initialization**, render the templates under `templates/` with `<TICKER>` and write the resulting short text into each ticker-local agent's own `init.json.comment`:

```text
vault/companies/<TICKER>/.lingtai/<TICKER>_orchestrator/init.json  comment = team context + orchestrator role
vault/companies/<TICKER>/.lingtai/<TICKER>_source/init.json        comment = team context + source role
vault/companies/<TICKER>/.lingtai/<TICKER>_writer/init.json        comment = team context + writer role
vault/companies/<TICKER>/.lingtai/<TICKER>_reviewer/init.json      comment = team context + reviewer role
```

All of these agents share the ticker-local `.lingtai/` network and still have distinct comments. The network/mail namespace is shared inside the ticker; each agent's `init.json` is not.

For **existing long-lived agents**, the runtime harness should not reassemble prompts. It should simply send the new request to `<TICKER>_orchestrator`; the team keeps its memory, pad, and learned self-organization.

## v0 team shape

A ticker team starts with four LingTai avatars:

```text
<TICKER>_orchestrator
<TICKER>_source
<TICKER>_writer
<TICKER>_reviewer
```

- The frontend/harness normally talks only to the orchestrator through the ticker-local `human` mailbox.
- The orchestrator arranges the team through LingTai mail.
- The source maintainer keeps the factual source/wiki layer useful.
- The writer owns working drafts, not publication approval.
- The reviewer challenges whether evidence supports the conclusion.
- The orchestrator owns the final publication judgment.

Future roles such as valuation, risk, or news can be added only after the four-role loop works in real use.

## v0 file contract

The important AlphaSeeker file surfaces are:

```text
vault/companies/<TICKER>/.lingtai/                 # ticker-local LingTai team network
vault/companies/<TICKER>/wiki/                     # maintained factual company wiki
vault/companies/<TICKER>/team/raw/                 # source-material landing zone
vault/companies/<TICKER>/team/drafts/              # working drafts / notes
vault/companies/<TICKER>/team/published/latest.md  # accepted human-facing output
```

Everything else belongs to LingTai unless experience proves otherwise.

## Why this is so small

The design intentionally trusts LingTai avatars. If an avatar can read files, write notes, mail teammates, maintain its pad, and learn over time, AlphaSeeker should not pre-build a rigid schema for every intermediate state.

A larger schema can be added later only when real runs reveal a repeated failure mode:

- If avatars cannot find raw material, add a lightweight index.
- If humans cannot see final output, improve `published/` or relay logic.
- If source extraction is painful, add focused source tools.
- If team status is hard to inspect, build a panel that reads ticker-local LingTai state instead of replacing it.
