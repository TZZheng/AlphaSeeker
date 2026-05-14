# AlphaSeeker Research Platform — v1 Decision Boundary Checklist

Last updated: 2026-05-14 08:10 CDT
Owner: `codex`
Purpose: clear v1 vertical-slice boundaries before implementation.

## Status

**V1 boundary gate is cleared.** Ted answered the final Claude Code grill escalations in email `10169290-8d47-4d70-b82e-954fced89af2`.

Claude Code v1 boundary grill report:

- `docs/research_platform/module_plans/V1_CLAUDE_BOUNDARY_GRILL.md`

v0 commit:

- `54748bf Add research platform v0 contracts`

## Final Ted rulings

1. **Cross-layer orchestrator home:** use `src/research_platform/memo_run.py`.
2. **Manual-file default grade:** use `B` by default. Also update prototype `src/vault/memo_flow.py` so its manual-file default becomes B too.
3. **Harness skill pack policy:** use option 2, soft preference. Keep equity tools available, but the prompt template must tell the harness to prefer attached curated context, only call tools for specific verification, and preserve/cite context package/source_index citations where possible.

## Intended v1 vertical slice

Target: first useful deterministic XOM-style research-platform flow.

```text
ResearchTask / ticker / optional manual files
  -> retrieval of manual files + SEC filings + company profile + financial snapshot + market snapshot
  -> deterministic vault ingestion / registry
  -> source_index + question_list / review artifacts
  -> MemoContextPackage
  -> runtime assembles harness prompt from Markdown prompt template + context package
  -> existing memo harness final.md
  -> manifest + status with product cleanup policy
```

No durable canonical `llm_research_state.md` should be introduced.

## Cleared v1 boundaries

### 1. v1 entrypoint and user interface

Use new cross-layer research-platform entrypoint:

```bash
python -m src.research_platform.memo_run --ticker XOM --prompt "..." --manual-file path/to/file.md --vault-root vault
```

The command internally calls retrieval -> vault ingestion -> context package -> existing harness. Keep `src.vault.memo_flow` as prototype/backward-compatible scaffold, but update its manual-file default grade to B.

### 2. v1 source requirements: required vs best-effort

- Required inputs for a normal run: at least one source among manual files or SEC filing.
- Best-effort but expected: latest 10-K/10-Q, company profile, financial snapshot, market snapshot.
- Missing best-effort sources do **not** abort; they create warnings/caveats in `MemoContextPackage` and manifest.
- Strict mode can come later.

### 3. SEC filing scope

Default SEC retrieval:

- latest 10-K;
- latest 10-Q if newer than 10-K;
- latest 8-K optional/not required for v1.

Source grade: SEC filings are A.

Engineering default: compare `filed_at` ISO timestamps; if either is missing, include 10-K only and emit a caveat.

### 4. Profile / financial / market snapshot providers

Use existing deterministic Python tool modules where available under `src/tools/equity/` or harness skill logic. Likely provider path: current yfinance/existing equity tool functions for company profile, financials, and market data.

Source grade: B by default for vendor/API-derived snapshots, with provider name and timestamp in metadata.

If provider fails or returns empty data, continue with caveat.

### 5. Manual-file grading metadata

- CLI accepts repeatable `--manual-file path`; default grade B with `manual file, provenance not machine-verified` rationale.
- Optional later syntax: `--manual-grade A|B|C` and `--manual-title` can be added simply.
- Prototype `src.vault.memo_flow` should also default manual files to B.

### 6. Vault file layout and relative paths

Use relative paths under `vault/`:

```text
vault/
  companies/<TICKER>/
    sources/
      manual/
      sec_filings/
      profile/
      financials/
      market/
      news/
    research/
      source_index.md
      question_list.md
      review_queue.md
      conflict_list.md
      context_packages/<run_id>.json
      memos/<run_id>/final.md
      memos/<run_id>/manifest.json
      memos/<run_id>/status.json
      memos/<run_id>/source_index.md
      memos/<run_id>/question_list.md
```

Contracts store paths relative to `vault/`. `run_id` must be unique per invocation; collision raises rather than clobbering artifacts.

### 7. SQLite schema changes vs artifact-first v1

- v1 uses existing vault DB documents table for registered documents where possible.
- Safe v1 DB writes: `documents`, `document_companies`, and `companies` only.
- New lifecycle objects (`ClaimRecord`, `QuestionRecord`, `ConflictRecord`, `MemoContextPackage`) are written as JSON/Markdown artifacts first.
- Do not write v1 lifecycle objects to existing `questions`, `facts`, `metrics`, `conflicts`, or `wiki_versions` tables; avoid DB migration until v1.1/v2.
- Add/propagate `source_grade_rationale` into `metadata_json` for ingested docs.

### 8. Claim extraction in v1

- v1 does **not** build full claim registry.
- v1 may generate `source_index.md`, `question_list.md`, caveats, and a `MemoContextPackage` with citation/evidence components.
- LLM qualitative active-claim lifecycle starts in v2 unless needed for the memo demo.

### 9. MemoContextPackage to harness bridge and prompt assembly

- Write a structured `MemoContextPackage` JSON for machine validation.
- Store the harness prompt as a Markdown prompt-template file, not hard-coded Python text.
- Runtime assembles the actual harness prompt from:
  1. Markdown prompt template;
  2. user prompt / ticker / run metadata;
  3. rendered context/source bundle derived from `MemoContextPackage`;
  4. warnings/caveats/questions/citations.
- Put the run-level template under `src/research_platform/prompts/memo_user.md`.
- Render with a strict helper that fails if any `{{placeholder}}` remains unresolved.
- Use soft preference skill policy: keep equity tools available, but the prompt must instruct the harness to prefer attached curated context files and only call tools for targeted verification.

Required template variables:

- `ticker`
- `company_name`
- `user_prompt`
- `run_id`
- `source_index_path`
- `question_list_path`
- `context_package_path`
- `required_sources_summary`
- `missing_sources_block`
- `caveats_block`
- `freshness_block`
- `citation_usage_instructions`

Pass `MemoContextPackage` JSON + source/question files via `HarnessRequest.context_files`; do not inline large JSON in the prompt.

### 10. Citation rendering in v1 memo

- v1 should attempt readable Obsidian-compatible links in source index/context bundle where deterministic path components exist.
- The final memo should at least cite using readable source title + grade + date; if easy, render `[[vault_relative_path#heading|display_title]]`.
- Contract truth remains structured citations in package/manifest, not prose parsing.
- Manifest should retain the full structured citation list so prose can be audited later.

### 11. Cleanup / retention

Always keep product/reproducibility artifacts:

- `final.md`
- `manifest.json`
- `status.json`
- `source_index.md`
- `question_list.md`
- structured `MemoContextPackage` JSON

Clean harness scratch/workspaces by default unless failure/debug mode. Cleanup must not delete vault originals or context package artifacts.

### 12. Tests and live demo gate

Validation before calling v1 useful:

- unit tests for retrieval request assembly, source records, manifest/context package rendering, prompt-template assembly, and non-blocking missing-source caveats;
- mocked provider tests only in default pytest, no live network in unit tests;
- one manual/live XOM demo command after tests pass, explicitly reported as a demo artifact inspection.

Claude test additions:

- prompt template golden render + missing placeholder failure;
- `SourceRecord` -> `DocumentRef` mapping;
- optional source failure -> caveat, not abort;
- zero required sources -> abort before harness;
- manual-file grade B + rationale stored in `documents.metadata_json`;
- mocked SEC 403/empty provider failures -> caveats;
- manifest/status JSON round trip;
- cleanup keeps vault originals;
- context files reach harness agent;
- inline prompt budget around 32KB and context bundle sanity cap around 1MB.

## Implementation order

1. Create `src/research_platform/` package and entrypoint.
2. Add Markdown harness prompt template and strict runtime prompt assembly from template + context package.
3. Extend `ingest_file` / `ingest_text` to accept/store `source_grade_rationale` in metadata.
4. Update prototype `src.vault.memo_flow` manual default grade to B.
5. Wire deterministic/manual source ingestion and path layout.
6. Wire SEC/profile/financial/market provider adapters using existing tool code.
7. Render `source_index.md`, `question_list.md`, and `MemoContextPackage` JSON.
8. Bridge context package into existing memo harness via `HarnessRequest.context_files` and soft-preference prompt.
9. Write manifest/status/cleanup policy.
10. Add unit tests, then run XOM demo if Ted approves live/provider calls.
