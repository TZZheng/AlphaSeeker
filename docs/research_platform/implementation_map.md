# Research Platform Implementation Map

Last updated: 2026-05-13

This map aligns the current AlphaSeeker research-vault code with the larger Librarian-style equity research platform described in `xiaohongshu_librarian_2026-05-12/librarian_xhs_extracted_report.md`.

## Product north star

The platform should not be a generic "chat with documents" system. The durable product object is a living, source-aware company wiki backed by first-class database records for documents, facts, metrics, questions, answers, conflicts, decisions, and versions.

The machine should do the mechanical work:

- ingest and classify sources;
- entity-link documents to companies;
- extract facts, metrics, guidance, catalysts, and questions;
- rank source reliability;
- compare claims across sources;
- detect stale or missing evidence;
- produce review queues and wiki diffs;
- prepare meeting questions with vault context.

The human should judge source credibility in context, decide whether a conflict is alpha or noise, update thesis, and choose investment actions.

## Desired modules vs current code

| Desired module | Why it matters | Current code | Gap / next action |
|---|---|---|---|
| Company/ticker wiki as central artifact | Entity-centric durable research state beats generic chat threads. | `src/vault/wiki.py` renders a simple company `wiki.md`; `src/vault/onboard.py` calls it after ingestion/extraction. | Expand from MVP 10-section page toward lifecycle sections: snapshot, valuation, key takeaways, guidance, risks, open questions, conflicts, source registry, patrol status. |
| Source registry and lineage | Every number/claim needs source type, grade, date, and path/URL. | `documents`, `document_companies`, `facts.source_doc_id`, `metrics.source_doc_id`, `source_index.md`. | Improve display and citation ergonomics; avoid raw IDs in user-facing text where possible; add source-detail summaries and source health. |
| First-class facts and metrics | Claims/numbers must be queryable and cross-checkable, not just text chunks. | `facts` and `metrics` tables; deterministic SEC snippets and yfinance-derived metrics; A-grade SEC companyfacts capital-return metrics. | Add richer claim categories: guidance, thesis, catalyst, estimate/forecast, model metric. |
| Source grading | A-grade primary evidence should override B-grade support evidence unless explicitly escalated. | `source_grade` on documents/facts/metrics; A-vs-B metric comparison and support notes. | Generalize beyond capital-return metrics; make unsupported B-grade rows visible in a review queue. |
| Conflict detection and arbitration | Conflicts are often alpha entry points and should become human review items. | `conflicts` table; B-vs-A metric mismatch conflicts; resolved/open lifecycle. | Add side-by-side conflict rendering with refs and human action prompts; broaden to stale guidance, forecast disagreements, duplicate conflicting claims. |
| Question list workflow | Research should track what we know, what we asked, what was answered, and what remains missing. | `questions` and `answers` tables exist; default questions seeded; `question_list.md` renders open questions. | Convert questions into a section/gap/conflict review queue; add answer backfill flow later. |
| Daily monitoring | Living wiki requires recurring news/regulatory/competitor/announcement refresh. | Not implemented. | Future: query templates + dedupe + source ingestion; do not start here until workflow spine is clear. |
| Model registry and diffs | Model-derived numbers need Excel lineage and change logs. | Not implemented; only derived yfinance support metrics. | Future: model registry table and snapshot diff pipeline. |
| Status patrol / stale checks | QC should happen continuously, not as end-stage cleanup. | Not implemented as a separate layer; some questions/support notes encode missing confirmation. | Next vertical slice: deterministic status patrol that emits review questions and a patrol page. |
| Wiki versioning / diff logs | Updates should be traceable. | Existing wiki copy archive in `companies/<ticker>/versions/` and `wiki_versions` table. | Add render summary / diff summary to the wiki and support pages; later make diff logs first-class. |
| Meeting prep and follow-up | Question list should compound across analyst/company meetings. | Not implemented beyond seeded questions. | Future: generate question list with vault context; add per-question answer backfill and wiki cascade. |
| Memo/deck generation | Final outputs should pull from already-QC'd wiki state. | Not implemented. | Future only after wiki state is trustworthy. |

## Current implemented primitives worth keeping

The recent XOM vault work is a useful lower-level QC primitive, not the whole product:

1. Persistent local vault (`SQLite` + Markdown/Obsidian-compatible files).
2. SEC source ingestion and source registry.
3. Company wiki generation.
4. B-grade derived market-data metrics.
5. A-grade SEC companyfacts capital-return metrics.
6. Metric source grading and B-vs-A comparison.
7. Stable conflict IDs with resolved/open lifecycle.
8. Targeted confirmation questions when B-grade metrics lack A-grade support.
9. Wiki support notes that disappear only when same metric/period has A-grade confirmation.
10. Basic wiki version archive.

These pieces validate the process-QC direction. The mistake would be to keep drilling into one more numeric mapping before connecting the broader workflow.

## Next vertical slice: onboarding workflow spine v1

Goal: make onboarding produce a reviewable research state, not only a set of extracted metrics.

### Scope

Implement a deterministic status-patrol/review-queue layer attached to onboarding and wiki rendering.

Minimum output for one ticker:

1. `status_patrol.md` support page with checks, severity, status, and action.
2. Wiki section summarizing patrol warnings.
3. Questions generated from patrol gaps/conflicts so the review queue is actionable.
4. Conflict page that shows side-by-side refs and suggested human action.
5. Source index remains available as lineage backbone.

### Initial deterministic patrol checks

Start simple and testable:

- `missing_a_grade_valuation_support`: valuation support metrics exist but lack same metric/period A-grade confirmation.
- `unresolved_conflicts`: open conflicts exist and need human judgment.
- `no_recent_sec_source`: no linked SEC filing source exists for the company.
- `stale_source_inventory`: latest linked source is older than a configurable threshold, when dates are available.
- `no_open_questions`: no open questions exist after onboarding, which likely means the review queue failed to capture gaps.

### Out of scope for this slice

- Full daily WebSearch/news ingestion.
- Excel model registry and diffing.
- Meeting ASR / answer backfill.
- Memo/deck generation.
- More A-grade valuation numeric mappings unless needed to test the patrol workflow.

## Milestone discipline

For each proper implementation milestone:

1. Keep changes scoped.
2. Add/adjust unit tests.
3. Run focused and adjacent tests.
4. Run full suite before commit when code behavior changes materially.
5. Commit only intended files; preserve unrelated dirty worktree.
6. Ask daemon for independent review before declaring the platform slice clean.
