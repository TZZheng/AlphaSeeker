# Source Layer Principles: Evidence Envelope, Not Rigid Schema

## Purpose

This note captures the intended boundary for AlphaSeeker's source layer after the TSLA formal IC-packet run exposed a natural next question: if better research requires more source effort, how much data infrastructure belongs in this repository, and how should source maintainers share evidence without losing ticker-by-ticker flexibility?

The answer is deliberately narrow:

> **AlphaSeeker should define source-layer research discipline and a thin evidence envelope. It should not impose a global business-field schema or become the private data-infrastructure runtime.**

This document is a design contract, not an implementation mandate. It protects the v0 source maintainer's exploratory style while giving writers, reviewers, and orchestrators enough provenance to ask: "What source supports this, how reliable is it, and what would change the recommendation?"

## Non-goals

The source layer is **not**:

- a universal data warehouse schema for all companies;
- a requirement that every ticker produce the same metrics;
- a material-ID system that source maintainers must adopt before doing useful work;
- a high-frequency scraping runtime;
- a paid-data storage layer;
- a place to commit raw scraped artifacts, licensed datasets, credentials, PII, receipts, survey data, or expert-call notes;
- a replacement for LingTai avatars' judgment, mail, memory, and exploratory research.

The v0 rule from `roles/source_maintainer_prompt_design.md` still stands: the source maintainer should not wait for a fixed state schema, should not invent a material ID system unless it helps the current team, and should let source work be driven by the investment question.

## Core principle

> **Schema should be discovered, not imposed.**

Do not start by forcing every ticker into shared business fields such as `revenue_per_user`, `unit_cost`, `churn`, `fleet_size`, or `regulatory_status`. TSLA, SOFI, RKLB, a bank, a biotech, a SaaS company, and a commodity producer require different public-source shapes.

Instead, keep the cross-ticker layer thin:

> **Do not standardize what every source must say. Standardize the minimum context needed to trace, cite, audit, refresh, and use whatever the source maintainer found.**

## Three-layer model

### Layer 0 — raw source, fully flexible

A raw source may be anything useful:

- HTML pages;
- PDFs;
- SEC filings;
- government records;
- CSVs;
- screenshots;
- source-maintainer notes;
- analyst-written source memos;
- paid/vendor files stored outside this repo;
- expert-call summaries stored in a controlled location;
- survey or receipt material stored only with explicit consent and privacy controls.

The discipline at this layer is simple: preserve enough raw artifact/provenance that a teammate can verify the claim later. Do not require the source maintainer to make messy evidence clean before it is allowed to exist.

### Layer 1 — thin evidence envelope, cross-ticker

The evidence envelope is the only cross-ticker structure this repo should define. It is a metadata wrapper around arbitrary source material or source-maintainer judgment.

A conceptual envelope may look like:

```yaml
evidence_id: optional stable local identifier
source_name: human-readable source name
source_type: filing | webpage | pdf | api | paid | manual | expert | survey | other
ticker: TSLA
as_of_date: 2026-05-16
collected_at: 2026-05-16T20:00:00Z
artifact_path_or_url: raw artifact path, URL, or controlled external reference
license_or_compliance_flag: public | tos_sensitive | paid_license | pii | expert | unknown
confidence: high | medium | low | unknown
category: hard_fact | proxy | assumption | judgment | unknown
summary: short plain-English statement of what this evidence says
payload: free-form ticker/source-specific content
maintainer_notes: free-form caveats, conflicts, next-source ideas
```

Only the envelope is shared discipline. The `payload` remains free-form and may be a table, narrative note, CSV path, JSON blob, image/OCR extraction, model-ready datapoint, or "not yet structured; read the raw artifact."

The envelope exists so agents can answer:

- Where did this come from?
- When was it collected?
- Is it public, ToS-sensitive, paid, PII-bearing, expert-derived, or unknown?
- Is it a hard fact, proxy, assumption, or judgment?
- How confident is the source maintainer?
- Where is the raw material?
- What caveats did the source maintainer preserve?

It does **not** say that every ticker must have the same metrics.

### Layer 2 — optional typed datapoints, ticker/domain-specific

Typed datapoints should appear only after a repeated source shape has proven useful.

For TSLA, mature typed datapoints might eventually include:

```text
robotaxi_permit_status
nhtsa_investigation_status
vehicle_registration_weekly
inventory_price_snapshot
fsd_price_observation
factory_permit_event
```

For SOFI, they might instead include:

```text
deposit_growth
funding_cost
loan_origination_mix
charge_off_rate
delinquency_rate
app_rank_observation
```

For RKLB:

```text
launch_event
launch_delay
contract_award
backlog_update
neutron_milestone
manufacturing_capacity_signal
```

These are ticker/domain plugins, not global AlphaSeeker obligations. A typed datapoint earns its shape by repeated use, auditability, and demonstrated model relevance.

## Source maintainer freedoms to preserve

A good source maintainer must retain freedom to:

1. **Choose what matters.** The platform must not require every ticker to monitor the same categories.
2. **Choose extraction shape.** Some sources are best represented as a table, some as a memo, some as a qualitative signpost, and some only as raw artifacts plus caveats.
3. **Keep messy evidence.** Premature normalization can destroy context and overstate confidence.
4. **Evolve structure over time.** A source shape should become typed only after it is useful and repeatable.
5. **Say "missing" clearly.** Missing evidence is itself decision-relevant when it caps confidence or prevents an upgrade.

## Research discipline to standardize

The repo should standardize discipline, not business fields.

When evidence affects a recommendation, confidence level, implementation stance, or IC packet claim, the source maintainer should make clear:

- the raw source or artifact;
- whether the item is hard fact, proxy, assumption, or judgment;
- the confidence level and major caveats;
- the compliance/license/privacy flag;
- what claim or model variable it supports;
- what it does **not** prove;
- what source would most improve or invalidate the conclusion.

This is enough to improve institutional quality without freezing ticker-specific source discovery.

## Repo boundary: contract here, data infrastructure elsewhere

This repository should remain focused on the research platform: workflow, contracts, templates, role design, publication gates, and lightweight reference examples.

### Appropriate for this repo

- Source-layer principles and source-maintainer contract.
- Evidence-envelope examples.
- Ticker source-map templates.
- Evidence category taxonomy: hard fact / proxy / assumption / judgment.
- Compliance flag taxonomy.
- Example public-source backlog for a ticker such as TSLA.
- Minimal reference scripts that use mock or public toy data.
- Export conventions for evidence books, source maps, and appendices consumed by ticker teams.

### Not appropriate for this repo

- Real raw scraped data or large raw artifact archives.
- Paid/vendor data.
- Credentials, API keys, cookies, browser profiles, or proxy configuration.
- High-frequency crawler runtime, anti-bot infrastructure, or job queues.
- PII-bearing receipts, surveys, account screenshots, or crowdsourced user data.
- Expert-call notes or licensed research reports.
- Anything whose redistribution rights are unclear.

### Integration boundary

A clean architecture is:

```text
AlphaSeeker GitHub repo
  = research workflow + role design + source principles + evidence envelope + templates

Private data infrastructure / service
  = crawlers + scheduler + storage + paid connectors + compliance controls

Local or private data store
  = raw artifacts + optional typed datapoints + evidence envelopes + alerts

Ticker LingTai agents
  = consume exported source maps, evidence books, alerts, and model-input notes
```

The GitHub repo defines what good source outputs should look like. The private data layer may decide how to fetch, license, store, and refresh them.

## Minimal viable source layer

The first useful version should not be a full alternative-data platform. It can be a manual-first source layer:

1. For each ticker, identify 5–10 decision-relevant source gaps.
2. Let the source maintainer collect raw materials freely under the ticker's `team/raw/` surface or a controlled external store.
3. Attach a thin evidence envelope or plain-English source note to each item that matters.
4. Preserve hard fact / proxy / assumption / judgment separation.
5. Record what model variable, thesis pillar, or recommendation trigger the evidence affects.
6. Export an evidence book, source map, or source-maintainer memo that writer/reviewer/orchestrator can cite.

Automation should come later, after the team knows which sources repeatedly matter.

## TSLA vertical-slice example

For TSLA, a source-platform MVP could begin with public or low-risk sources:

- NHTSA / ODI / recall / investigation records.
- CA DMV / CPUC autonomous vehicle permits.
- City-level permit, hearing, depot, and incident records.
- Tesla price, incentives, and inventory snapshots, subject to ToS review.
- China/EU registration and public auto-market signals.
- Energy project and utility filing records.
- Jobs, factory permits, patents, and supplier-commentary diffs for Robotaxi/Cybercab readiness.

Higher-risk or higher-cost sources should require explicit human approval:

- app/fare scraping;
- receipt or ride crowdsourcing;
- surveys involving personal data;
- credit-card/app panels;
- licensed sell-side or market-positioning datasets;
- expert calls;
- teardown, satellite, or vendor alternative data.

The first TSLA deliverable should be a source map/evidence book and source-maintainer note, not a production crawler fleet.

## Suggested document artifacts

If this design is implemented further, prefer names that communicate flexibility:

- `source_layer_principles.md` — this document.
- `source_evidence_envelope.md` — concrete envelope examples and validation hints.
- `source_maintainer_contract.md` — how source maintainers communicate with writer/reviewer/orchestrator.
- `ticker_source_map_template.md` — optional template for source-gap backlogs.
- `evidence_provenance_standard.md` — if provenance rules grow beyond this note.

Avoid naming the first artifact `schema.md` unless it is clear that it is only a metadata envelope, not a universal business schema.

## Design test

Before adding any source-layer feature, ask:

1. Does it help a teammate verify a claim or understand a source gap?
2. Does it preserve source maintainer freedom to explore ticker-specific evidence?
3. Does it distinguish hard fact, proxy, assumption, and judgment?
4. Does it avoid committing private/licensed/PII-bearing data to the repo?
5. Does it bind evidence to research impact only when that binding is actually known?

If the answer to any of these is no, the feature is probably too rigid, too operational, or outside this repo's scope.
