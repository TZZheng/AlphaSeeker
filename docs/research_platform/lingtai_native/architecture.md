# AlphaSeeker on LingTai — Minimal Architecture

## Thesis

AlphaSeeker should not reimplement LingTai. LingTai is the runtime. AlphaSeeker is the domain recipe.

For v0, the design is intentionally sparse:

```text
AlphaSeeker = ticker-team recipe + raw material location + published output location + role prompts
LingTai     = avatar runtime + mail + pad + memory + molt + logs + timelines + tools + autonomy
```

## Ownership boundary

### LingTai owns agent life

LingTai already provides the things a long-lived research team needs:

- avatar identity and lifecycle;
- internal mail between avatars and the human;
- per-avatar pad and durable memory;
- history, logs, and timeline-like audit trails inside `.lingtai/`;
- context molt and recovery;
- tool registration and MCP/capability hosting;
- idleness, soul flow, wake-by-mail, and liveness.

AlphaSeeker v0 does not create replacements for any of these. In particular, v0 does **not** create team timelines, request queues, status dashboards, or per-avatar workspaces in the AlphaSeeker vault.

### AlphaSeeker owns the research contract

AlphaSeeker provides only the human/domain contract:

- where raw ticker material lives;
- where human-facing published output lives;
- which avatars form the team;
- what each avatar's responsibility is;
- what the team policy is.

The avatars decide how to organize intermediate thoughts, notes, drafts, requests, disagreements, and logs using LingTai mail, pad, and their own working areas.

## v0 team shape

A ticker team starts with four LingTai avatars:

```text
<TICKER>_orchestrator
<TICKER>_source
<TICKER>_writer
<TICKER>_reviewer
```

- The human normally talks only to the orchestrator.
- The orchestrator arranges the team through LingTai mail.
- The source maintainer explores and explains raw material.
- The writer drafts research output.
- The reviewer challenges drafts and decides whether output is publishable.

Future roles such as valuation, risk, or news can be added only after the four-role loop works in real use.

## v0 file contract

The only required AlphaSeeker team directory is:

```text
vault/companies/<TICKER>/team/
  raw/
  published/
```

`raw/` is unstructured source material. `published/` is the human-facing output surface.

Everything else belongs to LingTai unless experience proves otherwise.

## Why this is so small

The design intentionally trusts LingTai avatars. If an avatar can `ls`, read files, write notes, mail teammates, and maintain its pad, then AlphaSeeker should not pre-build a rigid schema for every possible intermediate state.

A larger schema can be added later only when the manual run reveals a repeated failure mode:

- If avatars cannot find raw material, add a lightweight index.
- If humans cannot see final output, improve `published/`.
- If source extraction is painful, register source extraction tools into LingTai.
- If team status is hard to inspect, build a panel that reads LingTai state instead of replacing it.
