# AlphaSeeker on LingTai — Minimal Architecture

## Thesis

AlphaSeeker should not reimplement LingTai. LingTai is the runtime. AlphaSeeker is the domain recipe plus a thin frontend/harness layer.

```text
AlphaSeeker = ticker-team comment templates + vault file surfaces + request sender
LingTai     = avatar runtime + mail + pad + memory + molt + logs + tools + autonomy
```

## Ownership boundary

### LingTai owns agent life

LingTai already provides the things a long-lived research team needs:

- avatar identity and lifecycle;
- per-agent `init.json.comment` injection;
- internal mail between avatars and the human;
- per-avatar pad and durable memory;
- history, logs, and timeline-like audit trails inside `.lingtai/`;
- context molt and recovery;
- tool registration and MCP/capability hosting;
- idleness, wake-by-mail, and liveness.

AlphaSeeker v0 does not create replacements for any of these. In particular, v0 does **not** create team timelines, request queues, status dashboards, or per-avatar workspaces in the AlphaSeeker vault.

### AlphaSeeker owns the research contract

AlphaSeeker provides only the human/domain contract:

- which ticker the team focuses on;
- which avatars form the team;
- what each avatar's short role responsibility is;
- where raw source material, factual wiki, working drafts, and human-facing published output live;
- how a frontend/harness sends a natural-language request to the orchestrator.

The avatars decide how to organize intermediate thoughts, notes, disagreements, and logs using LingTai mail, pad, and their own working areas.

## Prompt/comment layering

The kernel already supplies LingTai's general behavior. AlphaSeeker role text should therefore be only a few lines.

For **new ticker-team initialization**, render the templates under `templates/` with `<TICKER>` and write the resulting short text into each agent's own `init.json.comment`:

```text
.lingtai/<TICKER>_orchestrator/init.json  comment = team context + orchestrator role
.lingtai/<TICKER>_source/init.json        comment = team context + source role
.lingtai/<TICKER>_writer/init.json        comment = team context + writer role
.lingtai/<TICKER>_reviewer/init.json      comment = team context + reviewer role
```

All of these agents can live in the same `.lingtai/` network and still have distinct comments. The network/mail namespace is shared; each agent's `init.json` is not.

For **existing long-lived agents**, the runtime harness should not reassemble prompts. It should simply send the new request to `<TICKER>_orchestrator`; the team keeps its memory, pad, and learned self-organization.

## v0 team shape

A ticker team starts with four LingTai avatars:

```text
<TICKER>_orchestrator
<TICKER>_source
<TICKER>_writer
<TICKER>_reviewer
```

- The human/frontend normally talks only to the orchestrator.
- The orchestrator arranges the team through LingTai mail.
- The source maintainer keeps the factual source/wiki layer useful.
- The writer owns working drafts, not publication approval.
- The reviewer challenges whether evidence supports the conclusion.
- The orchestrator owns the final publication judgment.

Future roles such as valuation, risk, or news can be added only after the four-role loop works in real use.

## v0 file contract

The important AlphaSeeker file surfaces are:

```text
vault/companies/<TICKER>/wiki/                    # maintained factual company wiki
vault/companies/<TICKER>/team/raw/                # source-material landing zone
vault/companies/<TICKER>/team/drafts/             # working drafts / notes
vault/companies/<TICKER>/team/published/latest.md # accepted human-facing output
```

Everything else belongs to LingTai unless experience proves otherwise.

## Why this is so small

The design intentionally trusts LingTai avatars. If an avatar can read files, write notes, mail teammates, maintain its pad, and learn over time, AlphaSeeker should not pre-build a rigid schema for every intermediate state.

A larger schema can be added later only when real runs reveal a repeated failure mode:

- If avatars cannot find raw material, add a lightweight index.
- If humans cannot see final output, improve `published/`.
- If source extraction is painful, add focused source tools.
- If team status is hard to inspect, build a panel that reads LingTai state instead of replacing it.
