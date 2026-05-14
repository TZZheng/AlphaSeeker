# Research Platform Implementation Map

Last updated: 2026-05-13

This map aligns AlphaSeeker with the larger Librarian-style equity research platform described in `xiaohongshu_librarian_2026-05-12/librarian_xhs_extracted_report.md`, with one correction after review: the **minimum MVP is LLM-assisted research-state generation**, not deterministic wiki/status-page polishing.

## Product north star

The platform should not be a generic "chat with documents" system, and it should not be a deterministic wiki formatter. It should preserve AlphaSeeker's original product promise — **given an investment prompt, build an investment memo** — while adding a durable company research state underneath it.

The durable product object is a living, source-aware company wiki / research state. The user-facing deliverable remains an investment memo when the user asks for one.

The MVP proof is:

> Given a user investment prompt, a company/ticker, and a small batch of source materials, AlphaSeeker uses an LLM to extract and synthesize an analyst-usable company research state with citations/open questions, then feeds that state into the existing invest-memo flow so the run produces both a memo and a reusable company wiki/question list.

The intended flow is:

```text
user prompt + source docs/live data
  -> LLM-generated company research state / wiki with citations
  -> existing multi-agent investment memo synthesis
  -> memo deliverable + persisted wiki/questions for the next run
```

Deterministic code is scaffolding and guardrail only:

- ingest/select source materials;
- keep source metadata and lineage;
- pass compact source bundles to the LLM;
- persist generated sections, citations, and open questions;
- render the result in files a human can inspect;
- optionally run cheap QC checks that point out missing evidence.

The human should judge source credibility in context, decide whether conflicts matter, update thesis, and choose investment actions. The LLM should do the expensive synthesis work: turn scattered source documents into a first usable research state.

## Minimum MVP acceptance criteria

For one demo prompt/company/ticker, a clean command or harness run should produce:

1. A source bundle assembled from a small set of local/available materials.
2. An LLM-generated company wiki / research state with at least:
   - company / business summary;
   - key metrics or valuation points when present in the sources;
   - guidance / management commentary when present;
   - risks and counterevidence;
   - thesis / key takeaways;
   - open questions / gaps.
3. Source citations for generated claims in a simple inspectable form: document title/path plus quote snippet is enough for MVP.
4. A readable Markdown output under the company vault, plus a question list.
5. An investment memo generated through the existing AlphaSeeker harness that explicitly consumes or cites the generated research state.
6. Tests around prompt/input/output plumbing and citation persistence where feasible.

A successful MVP demo is not "the wiki has the prettiest deterministic sections." It is "the system can read a source bundle, produce a cited analyst starting point, and use that durable state to help generate the investment memo AlphaSeeker was originally built to deliver."

## Desired modules vs current code

| Desired module | MVP role | Current code | Gap / next action |
|---|---|---|---|
| LLM source-bundle synthesis | **Core MVP substrate.** Convert raw source material into wiki sections, citations, and questions. | `src/vault/synthesis.py` provides a first minimal source-bundle -> LLM JSON -> `llm_research_state.md` flow. The harness already has LLM runtime and evidence-oriented types. | Use this generated research state as a reusable, cited starting point for memo runs. |
| Existing investmemo harness | **Core user-facing MVP.** AlphaSeeker's original promise is prompt -> investment memo. | `src/harness.run_harness()` already creates multi-agent runs and root `publish/final.md`; `src/vault/memo_flow.py` now attaches generated research state as root context before running it. | Preserve this as the final deliverable and validate with a small live memo demo. |
| Company/ticker wiki as central artifact | Durable substrate for future memo runs, not a replacement for memo output. | `src/vault/wiki.py` renders deterministic `wiki.md`; `src/vault/synthesis.py` now writes `llm_research_state.md`. | Keep wiki simple and source-aware; use it as reusable context for memo synthesis. |
| Source registry and lineage | Required guardrail: every generated claim should trace to source material. | `documents`, `document_companies`, `facts.source_doc_id`, `metrics.source_doc_id`, `source_index.md`. | Add/persist LLM citations as document path/title + quote snippets. Avoid raw IDs in user-facing text. |
| First-class generated research sections | Needed to store LLM output beyond numeric facts. | `facts`, `metrics`, `questions`, `answers`, `conflicts`; no first-class wiki-section table yet. | For MVP, simple Markdown + optional fact/question rows is acceptable; do not build a complex ontology first. |
| Question list workflow | Core output: the LLM should surface what remains unknown. | `questions` and `answers` tables exist; default/status questions can render. | Have LLM synthesis emit open questions/gaps and persist them. Answer-backfill can come later. |
| Source grading / QC | Guardrail, not product core. | `source_grade`; A-vs-B metric comparison; status patrol checks. | Keep existing QC, but do not broaden deterministic rules unless directly needed to validate LLM output. |
| Conflict detection and arbitration | Useful later; can reveal alpha but is not the minimum proof. | `conflicts` table; B-vs-A metric mismatch conflicts; conflict page with side-by-side refs/actions. | Defer broad conflict arbitration until after LLM wiki MVP works. |
| Status patrol / stale checks | Guardrail already implemented. | `src/vault/status.py`, `status_patrol.md`, patrol question seeding. | Treat as infrastructure. Do not continue polishing patrol lifecycle before LLM synthesis demo. |
| Daily monitoring | Future product workflow. | Not implemented. | Defer. Needs the LLM wiki state first. |
| Model registry and diffs | Future workflow for analyst models. | Not implemented; only derived market-data support metrics. | Defer. |
| Meeting prep and follow-up | Future workflow that should consume the generated wiki/question list. | Not implemented beyond seeded questions. | Defer until source-bundle synthesis can create a useful wiki/question list. |
| Memo/deck generation | Future output layer. | Existing harness can produce final memos per run. | Later: generate from already-cited wiki state. |

## Current implemented primitives worth keeping

The recent vault work is useful infrastructure, but it is not the MVP center:

1. Persistent local vault (`SQLite` + Markdown/Obsidian-compatible files).
2. Source ingestion/registry and company linking.
3. Company wiki generation path.
4. B-grade derived market-data metrics.
5. A-grade SEC companyfacts capital-return metrics.
6. Metric source grading and B-vs-A comparison.
7. Stable conflict IDs with resolved/open lifecycle.
8. Targeted confirmation questions when B-grade metrics lack A-grade support.
9. Wiki support notes that disappear only when same metric/period has A-grade confirmation.
10. Status patrol / review queue (`status_patrol.md`) and conflict refs/actions.
11. Basic wiki version archive.

These pieces can support the LLM demo by providing persistence, source lineage, and guardrails. The mistake would be to keep expanding deterministic patrol/metric coverage before proving LLM-assisted wiki generation.

## Next vertical slice: vault-backed memo MVP

Goal: make one company memo run produce both the original AlphaSeeker investment memo and a cited, reusable analyst starting point from source materials.

### Scope

Minimum output for one prompt/ticker:

1. Select/assemble a compact source bundle from local/vault documents.
2. Use an LLM to produce a cited research-state wiki with sections:
   - business overview;
   - key metrics / valuation points from sources;
   - guidance / management commentary;
   - risks / counterevidence;
   - thesis / key takeaways;
   - open questions.
3. Require citations with quote snippets for factual claims.
4. Persist/render the result as Markdown under `data/research_vault/companies/<TICKER>/` and persist open questions where practical.
5. Attach that generated research-state Markdown as a root harness context file.
6. Run the existing AlphaSeeker memo harness with the user's original investment prompt plus a short instruction to read the attached research state first.
7. Produce the memo at the normal root `publish/final.md` path.

### Suggested implementation order

1. Keep the LLM research-state synthesizer tiny: local text/markdown source bundle first, strict JSON output, readable Markdown render.
2. Add root `HarnessRequest.context_files` support so the generated wiki can be copied into the root agent `context/` instead of pasted into `user_prompt`.
3. Add a thin wrapper command/function that runs synthesis first, then calls `run_harness()` with the generated wiki as a context file.
4. Keep the existing `vault_synthesize_research_state` harness skill for later research-agent use, but do not rely on the root orchestrator choosing it for the first MVP demo.
5. Run one XOM or other known-company demo and inspect both `llm_research_state.md` and root `publish/final.md`.

### Out of scope for this slice

- More A-grade valuation numeric mappings.
- More deterministic status-patrol checks or lifecycle cleanup.
- Daily WebSearch/news ingestion.
- Excel model registry and diffing.
- Meeting ASR / answer backfill.
- Deck generation.
- Full conflict arbitration beyond existing guardrails.

## Milestone discipline

For each proper implementation milestone:

1. Keep changes scoped to the MVP proof.
2. Add/adjust unit tests around the new behavior.
3. Run focused and adjacent tests.
4. Run full suite before commit when code behavior changes materially.
5. Commit only intended files; preserve unrelated dirty worktree.
6. Ask daemon for independent review before declaring a product slice clean.
