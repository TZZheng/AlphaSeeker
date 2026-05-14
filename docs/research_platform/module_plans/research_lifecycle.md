# Research Lifecycle Module Plan

Last updated: 2026-05-13
Planner: `research_lifecycle_planner`

Scope: research object / question / conflict / freshness lifecycle. This is planning only. No code changes.

## 1. Module purpose

This module is the durable company research-state layer between source ingestion and downstream outputs. It replaces durable freeform `llm_research_state.md` as the center of truth with source-grounded, inspectable, versioned objects: facts, claims, metrics, guidance, catalysts, risks, thesis points, questions, answers, conflicts, review items, freshness warnings, and diff logs.

Its job is to preserve what the platform currently knows, what it only suspects, what is contradicted, what is stale, and what requires human judgment. Markdown pages remain useful, but they should be generated views over structured state, not the durable truth.

## 2. Owned responsibilities

- Maintain lifecycle state for research objects by company/ticker.
- Define statuses and transitions for questions, answers, claims/facts, conflicts, review items, and freshness warnings.
- Persist evidence links from each durable object to source document(s), quotes, dates, and source grades.
- Track object-level versioning: created, updated, superseded, stale, rejected, resolved.
- Create and maintain a unified review queue for machine flags and human decisions.
- Produce queryable state for wiki rendering, memo context assembly, daily monitoring, and meeting prep.
- Enforce that LLM output enters as cited candidates unless policy or human review promotes it.
- Keep immutable diff/event logs so changes are auditable.

## 3. Explicit non-responsibilities

- Fetching filings, news, reports, market data, transcripts, or web sources; this belongs to input/retrieval.
- Raw document storage, checksums, entity linking, OCR/PDF parsing, and initial source registry; this belongs to vault ingestion/source registry.
- LLM provider orchestration, prompt execution, retries, and token budgeting.
- Final Markdown layout, memo/deck rendering, and artifact cleanup.
- Investment decisions: long/short/hold, model changes, target price updates, and thesis acceptance require human/reviewed decision records.
- Treating freeform LLM prose as durable truth.

## 4. Inputs / upstream ports

From ingestion/source registry:

- `CompanyRef`: ticker, name, exchange, sector.
- `SourceDocumentRef`: doc_id, title, path, URL, source_type, source_grade, published_at, ingested_at, checksum, metadata, linked ticker/relevance.

From deterministic extraction:

- `ExtractedObjectCandidate`: ticker, object_type, normalized fields, source_doc_id, quote/location, source_grade, observed_at, extraction_method, confidence, stable key.

From LLM extraction/synthesis:

- `LLMExtractionBatch`: run_id, model, prompt/template version, source bundle manifest, candidate objects, evidence refs, confidence, rationale, uncertainty note.
- LLM candidates must include valid source refs/quotes unless they are explicitly questions/gaps.

From humans/orchestrator:

- `DecisionEvent`: actor, action, target refs, decision note, timestamp. Actions include accept, reject, supersede, resolve conflict, create/answer question, assign owner, change priority, or update thesis.

From monitoring:

- `MonitoringEvent`: new source, price move, earnings near, catalyst due, source stale, model changed, metric changed.

## 5. Outputs / downstream ports

Expose `CompanyResearchContext` for task-specific assembly:

- active facts, metrics, guidance, risks, thesis points, catalysts;
- candidate/unreviewed claims with labels;
- open and answered questions;
- unresolved conflicts;
- freshness/staleness warnings;
- review queue summary;
- recent diff log;
- evidence map.

Expose renderer-ready views for:

- `source_index.md`;
- `question_list.md`;
- `conflicts.md`;
- `review_queue.md` / `status_patrol.md`;
- `catalysts.md`;
- company wiki sections;
- meeting prep briefs;
- daily monitoring digests;
- memo context packages.

Generated views should include render timestamp, object/diff watermark, and warning banners for stale/conflicted/unreviewed state.

## 6. Persistent artifacts and schema needs

Current schema already has `documents`, `companies`, `document_companies`, `facts`, `metrics`, `questions`, `answers`, `conflicts`, and `wiki_versions`. It is a useful prototype but too thin for lifecycle state.

Recommended schema direction: add a generic lifecycle envelope plus evidence/review/diff tables, while preserving compatibility with current tables.

Core `research_objects` fields:

- object_id, ticker, object_type;
- statement/title;
- status: candidate, active, needs_review, rejected, superseded, stale, resolved, closed;
- confidence, source_grade_best;
- created_at, updated_at, observed_at, effective_at, stale_after_at, last_checked_at;
- created_by: deterministic, llm, human, orchestrator;
- extraction_method, extraction_run_id;
- review_state, reviewer, reviewed_at;
- supersedes_object_id, superseded_by_object_id;
- object_hash, topic_tags_json, metadata_json.

Evidence table:

- evidence_id, object_id, doc_id, quote, location_json, source_grade, support_type, confidence, created_at.

Subtype payloads should cover:

- metrics: metric_name, period, value_text, value_numeric, unit, currency/scale, normalized_key;
- claims/facts: claim_kind, section, polarity;
- guidance: issuer, guided_metric, target_period, low/high/value_text, issued_at;
- catalysts: event_name, expected_date, verification_status;
- thesis points: thesis_kind, conviction, human_decision_required;
- questions: question_text, priority, owner, status, next_action, due_at, source;
- answers: question_object_id, answer_text, answer_route, answer_status, confidence.

Add first-class `review_items`, `freshness_policies`, `freshness_checks`, and `research_diff_log` tables.

Durable artifacts: source registry, research objects, evidence, questions/answers, conflicts, review items, freshness checks, diff logs, decision logs. Generated artifacts: wiki/source/question/conflict/status Markdown, meeting prep, memo context, `llm_research_state.md` if retained. Temporary artifacts: prompts, source bundles, scratch JSON, raw search pages, agent workspaces.

## 7. Deterministic code vs LLM decision boundary

Deterministic code owns:

- IDs, idempotency, schema validation;
- source/evidence existence checks;
- exact numeric parsing where source format is deterministic;
- source grade propagation;
- object hashes and diff logs;
- freshness/staleness policy checks;
- allowed status transitions;
- exact conflict checks for comparable fields;
- generated view rendering;
- preventing unreviewed LLM candidates from becoming active truth silently.

LLM may assist with:

- candidate claim/fact/guidance/risk/catalyst/thesis extraction;
- topic/section classification;
- question and gap generation;
- semantic conflict candidate detection;
- preliminary answers with quotes;
- meeting prep and synthesis drafts.

LLM must not, without human/policy promotion:

- create uncited active truth;
- resolve ambiguous conflicts;
- update thesis, model assumptions, ratings, or target price;
- hide uncertainty in polished prose;
- overwrite old state without supersession/diff records.

Promotion rule: deterministic A-grade exact extraction may become active machine-validated state; deterministic B-grade or derived extraction should be labeled support or needs_review; cited LLM output enters as candidate/needs_review; uncited factual LLM output is rejected or review-itemed; human acceptance creates active human-reviewed state.

## 8. Failure modes and human escalation points

Failure modes:

- uncited or weakly cited LLM claims;
- citation quote not present in source;
- duplicate object explosion on reruns;
- false conflicts from mismatched units, periods, currencies, or scopes;
- missed semantic conflicts;
- stale guidance/metrics/TP/catalysts appearing current;
- review backlog making outputs untrusted;
- ambiguous source grade;
- multiple active versions without clear supersession;
- generated wiki drift from DB state;
- preliminary answer mistaken for confirmed answer;
- entity-linking contamination.

Escalate to human/Ted/orchestrator when:

- A-grade sources conflict;
- same-grade sources conflict and recency does not clearly resolve;
- B-grade evidence contradicts A-grade but may reveal alpha;
- thesis point, exit condition, or investment implication changes;
- stale object affects memo/deck/meeting output;
- source grade policy is unclear;
- high/critical review item remains open;
- catalyst date arrives without verification;
- model or valuation assumption changes materially;
- module boundary is ambiguous.

## 9. Versioned/staged coding plan

### v0: interfaces/contracts only

- Freeze object/status/evidence/review/diff contracts.
- Define generated-vs-durable policy.
- Treat current `facts`, `metrics`, `questions`, `conflicts`, `status_patrol` as compatibility backing.
- Define `source_index`, `question_list`, and `review_queue` as the minimal lifecycle surface.
- Define query ports to context assembly and wiki rendering.
- No complex ontology yet; no implementation until orchestrator merge.

### v1: first useful implementation

- Add or migrate to `research_objects`, `object_evidence`, `review_items`, and `research_diff_log`.
- Extend question lifecycle with owner, priority, status, evidence refs, answer refs, next action.
- Add answer insertion/lifecycle support.
- Validate citations and source refs.
- Persist LLM synthesis as candidates + evidence + review items, not durable freeform truth.
- Render question list, conflicts, and review queue from structured state.
- Ensure memo context labels candidate/conflicted/stale state.

### v2+: enhancements

- Add structured registries for claims, guidance, catalysts, risks, and thesis points.
- Add richer semantic duplicate/conflict detection.
- Integrate daily monitoring: new sources, stale warnings, catalyst due checks, price/valuation moves, daily diff digest.
- Add meeting prep/follow-up: question packets, preliminary vault answers, post-meeting answer backfill, unanswered/dodged/conflicting statuses.
- Add output gates: high-severity conflicts/stale state must be acknowledged before memo/deck use.

## 10. Tests / validation gates

- Schema migration preserves current data.
- Status transition tests for questions, conflicts, answers, objects, and freshness warnings.
- Evidence validation rejects invalid doc refs and uncited factual candidates.
- Idempotency tests prevent duplicate questions/conflicts/metrics on rerun.
- Resolved conflict does not reopen on identical inputs.
- Freshness tests cover stale, unknown date, catalyst due, and refreshed source.
- Generated views rebuild from durable state without `llm_research_state.md`.
- Mocked LLM outputs: valid cited candidates persist as candidates; uncited factual claims fail safe; malformed JSON leaves state unchanged.
- Memo context includes warnings for unresolved high-severity conflicts and stale critical objects.

## 11. Open questions for Ted/orchestrator

1. Should v1 use a generic `research_objects` envelope or incrementally extend current tables?
2. Which object types require explicit human acceptance before appearing as active memo context?
3. Can any LLM-extracted cited A-grade statement become active automatically, or always candidate?
4. Who owns source-grade assignment changes after ingestion?
5. What priority vocabulary should questions/review items use: high/normal/low or P0/P1/P2?
6. Should conflict auto-resolution ever use A-over-B/newer-over-older rules, or only flag for human review?
7. What default stale windows apply to valuation, guidance, TP, earnings, news, thesis, risks, and model metrics?
8. Should memo generation block on high-severity unresolved items or only warn?
9. Should raw LLM prompts/responses be retained indefinitely, temporarily, or deleted after candidate persistence?
10. Does meeting-specific question list ownership belong here or to a separate meeting module?

## 12. Interactions with other modules

- **Input/retrieval:** retrieval fetches sources; lifecycle records how new sources change object state and emits retrieval needs from stale/gap items.
- **Vault ingestion/source registry:** ingestion owns document identity, checksums, source grade, and company links; lifecycle owns research objects derived from those sources.
- **LLM synthesis/extraction:** LLM proposes cited candidates; lifecycle validates, persists, labels, and queues review.
- **Context assembly:** lifecycle provides filtered company research context; assembly chooses task-specific subset and token budget.
- **Memo/output harness:** lifecycle supplies active/candidate/conflicted/stale state and evidence refs; harness generates final deliverable and should return output manifest/claim refs later.
- **Wiki/view renderer:** renderer owns Markdown layout; lifecycle owns state/query semantics.
- **Daily monitoring:** monitoring observes events; lifecycle converts them into objects, stale warnings, questions, review items, and diffs.
- **Meeting prep/follow-up:** meeting module packages and collects Q&A; lifecycle persists durable questions, answers, evidence, and cascade candidates.
