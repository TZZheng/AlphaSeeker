# V1 Claude Boundary Grill

Reviewer: Claude Code (daemon grill pass)
Date: 2026-05-14
Inputs reviewed:
- `docs/research_platform/module_plans/V1_DECISION_BOUNDARY_CHECKLIST.md`
- `docs/research_platform/module_plans/CONSOLIDATED_PLAN.md`
- `src/retrieval/types.py`, `src/vault/contracts.py`, `src/harness/context_types.py`
- `src/vault/memo_flow.py`, `src/vault/ingest.py`, `src/vault/schema.py`
- `src/harness/prompt_builder.py`, `src/harness/prompts/{system,task,tools}.md`

## 1. Executive summary

**Conditional yes.** The Ted-approved boundaries answer the big architectural questions (entrypoint location, required vs best-effort sources, artifact-first lifecycle, prompt-template assembly, retention). They are clear enough to start coding the v1 vertical slice *if* engineering decides four small operational defaults (listed in §3) and Ted clarifies three boundary points where the checklist is genuinely silent or contradicts existing code (listed in §2). The biggest concrete risk is not architectural — it is that v1 reuses `src/vault/ingest.py` which currently hardcodes manual files at grade "A", contradicting the v1 default grade "B" rule, and stores no `source_grade_rationale` field. That is a real implementation seam, not a doc question.

## 2. Escalate to Ted before implementation

1. **Module home for the v1 orchestrator.** The checklist proposes `src.retrieval.memo_run`. But the Consolidated Plan §1.1 explicitly scopes `src/retrieval/` as the input/source-acquisition layer that "does not own" memo generation or harness orchestration. `memo_run` is the cross-layer orchestrator (retrieval → vault → context → harness). Putting it inside `src.retrieval` violates the layering Ted approved on 2026-05-13. Reasonable options:
   - `src/research_platform/memo_run.py` (new package; matches doc naming)
   - `src/orchestration/memo_run.py`
   - top-level `src/memo_run.py`
   - keep `src.retrieval.memo_run` and accept the layering exception (least clean but matches checklist literally)
   Worth a one-line ruling.

2. **Manual file default grade.** The v1 checklist says manual files default to `B` with rationale "manual file, provenance not machine-verified". The Consolidated Plan §8 open question #8 ("Should manual files default to A-grade if uploaded by the human?") is *not* in §9 resolved decisions. Existing `src/vault/memo_flow.py:51` hardcodes `source_grade="A"` for manual files. This is a deliberate behavior change. Confirm: v1 overrides manual-file grade to `B` and adds an explicit `source_grade_rationale`, even when the same files would have been graded `A` under the prototype `memo_flow`. (Claude's reading is that the checklist's `B` default is correct and supersedes the prototype; just need an explicit yes.)

3. **Is the existing harness allowed to re-fetch sources during a v1 memo run?** The plan supplies a curated `MemoContextPackage`, but the existing `equity` skill still gives the agent SEC/profile/financial tools. If the agent re-runs SEC search, we get duplicate context, drift between curated grades and re-fetched data, and unbounded prompt/tool cost. Options:
   - Hard: strip the equity skill pack for v1 memo runs (only `core`).
   - Soft: keep tools but instruct in prompt template "prefer attached context_files; only call tools to verify a specific claim".
   - Wide-open: keep current `DEFAULT_SKILL_PACKS = ["core", "equity", "macro", "commodity"]`.
   The checklist is silent. Default would be soft, but Ted should confirm.

## 3. Engineering defaults — decide locally, no Ted needed

These are answered (or clearly enough implied) by the approved boundaries:

- **CLI flags:** `--manual-file` is repeatable (action="append"). `--vault-root` is optional; falls back to `default_vault_paths()`. `--prompt` is required. `--run-id` optional.
- **Source freshness comparison for "10-Q newer than 10-K":** compare `filed_at` ISO timestamps from SEC; if either missing, include 10-K only and emit a caveat. No fuzzy logic.
- **`source_index.md` placement:** per-company file under `vault/companies/<TICKER>/research/source_index.md`, *plus* a frozen copy under `memos/<run_id>/source_index.md` for reproducibility. The checklist layout shows the per-company file; per-run snapshot is a reasonable engineering default and Ted's "manifest" definition implies per-run reproducibility.
- **MemoContextPackage transport:** pass `context_packages/<run_id>.json`, `source_index.md`, and `question_list.md` as `HarnessRequest.context_files` (existing mechanism, already used in `memo_flow.py:64`). Do not inline JSON in the prompt body.
- **Provider tool reuse:** call existing functions in `src/tools/equity/` directly from `memo_run`; do not add a new abstraction layer in v1. v1.1 can extract a `SourceAdapter` protocol after the shape is proven.
- **Required-source rule on zero inputs:** if no manual files and SEC retrieval fails or returns empty, the run aborts with a non-zero status, writes `status.json` with `failed` and a clear `errors` entry, and does not call the harness. (Matches §2's "at least one source among manual files or SEC filing".)
- **Where the prompt template lives:** see §5.

## 4. Critique: `src.retrieval.memo_run` vs extending `src.vault.memo_flow`

**A new entrypoint is correct. The proposed location is debatable but acceptable if Ted insists.**

Reasons to keep `memo_flow` as prototype scaffold and write a new entrypoint:

- `memo_flow.run_vault_backed_memo` is built around `synthesize_company_research_state`, which produces a *freeform Markdown wiki* (`llm_research_state.md`-style). The Consolidated Plan §1.3 explicitly says that file pattern is being deprecated. Forcing v1 into the same function would either (a) keep generating the wiki we are trying to retire, or (b) carve `synthesize_company_research_state` out of memo_flow entirely, which is a bigger and riskier change than adding a sibling.
- `memo_flow` hardcodes `source_grade="A"` for manual files (`memo_flow.py:51`), the opposite of v1 policy. Forking off a clean entrypoint avoids litigating the legacy behavior.
- `memo_flow` has *no* concept of `RetrievalRequest`/`RetrievalBatch`/`SourceRecord`/`MemoContextPackage`. It would need a near-total rewrite to honor v0 contracts. At that point it is not "extending"; it is replacing.
- Keeping `memo_flow` callable preserves backward compatibility for any caller (CI, scripts) that already uses it.

The one cost of the split: two memo entrypoints to maintain. Engineering should add a one-line README note that `memo_flow` is "prototype, will be removed in v2 once `memo_run` ships and is dogfooded." That is cheaper than the rewrite.

**Location concern:** see §2 item 1. The layering argument is real but not blocking — pick a location and document it.

## 5. Critique: prompt-template / runtime assembly

**Approved direction is right; concrete decisions still owed.**

### Where the `.md` template should live

The existing prompts are under `src/harness/prompts/` and rendered by `src/harness/prompt_builder.py` using the `{{key}}` replacement convention (see `prompt_builder.py:61`). Two options:

- **Option A — colocate with harness prompts** at `src/harness/prompts/research/memo_user.md`. Pro: stays inside the package that already owns prompt rendering; same `{{key}}` convention; tests can import the existing render helper. Con: this template is consumed *before* `HarnessRequest`, by the orchestrator, not by the per-agent prompt builder — putting it in `harness/prompts/` blurs the boundary (per-agent vs run-level prompt).
- **Option B — colocate with orchestrator** at `src/research_platform/prompts/memo_user.md` (or wherever `memo_run` lives). Pro: matches "owned by memo_run". Con: needs a local copy of the template render helper, or import `_render_prompt_template` from `prompt_builder`.

**Recommendation:** Option B, with the template rendered by a small `assemble_memo_prompt(template_path, package, ...)` helper that internally reuses `prompt_builder._render_prompt_template` (it is already a simple pure function on `{{key}}`). This keeps the harness package focused on agent-level prompts and gives the new entrypoint a clean home.

### Variables that must be explicit in the template contract

The renderer should fail loudly on unknown placeholders and on missing required keys. Required:

- `{{ticker}}`
- `{{company_name}}`  *(empty string if absent — must be allowed)*
- `{{user_prompt}}`
- `{{run_id}}`
- `{{source_index_path}}`         (vault-relative)
- `{{question_list_path}}`        (vault-relative)
- `{{context_package_path}}`      (vault-relative)
- `{{required_sources_summary}}`  (rendered bullet list of source titles + grades)
- `{{missing_sources_block}}`     ("None" or bullet list)
- `{{caveats_block}}`             ("None" or bullet list)
- `{{freshness_block}}`           ("None" or bullet list — for v1 mostly "None")
- `{{citation_usage_instructions}}` (static guidance block; lives in the template, not assembled from context)

Optional / future:
- `{{open_questions_summary}}` for inline preview (the file path covers the long form)
- `{{document_count}}`, `{{a_grade_count}}`, `{{b_grade_count}}` for at-a-glance source mix

### Tests that should lock this down

- **Golden render test:** sample `MemoContextPackage` + fixed template → exact-string expected output (snapshot under `tests/unit/research_platform/golden/`).
- **Missing variable raises:** template referencing `{{nonexistent}}` → `KeyError` or explicit `UnrenderedPlaceholderError`. The current `_render_prompt_template` *silently passes through* unreplaced `{{...}}`, which would let template bugs leak into real prompts. v1 should add a post-render assertion that no `{{` remains.
- **Empty-list blocks render as "None":** verify `missing_sources_block` and `caveats_block` never produce a bare empty section.
- **Citation key format invariant:** every citation in the package has a `citation_key` matching `^src_[a-z0-9_]+$` (or chosen pattern), and the rendered prompt mentions each key at least once in the `{{required_sources_summary}}` block so the model has something to anchor against.
- **Prompt length budget:** rendered prompt + context files attached must stay under a configurable byte budget (suggest 32KB for the inline prompt, files passed separately via `context_files`). Test asserts the inline portion stays under budget for a representative XOM-sized package.
- **Obsidian link rendering:** when a `Citation` has heading_path, the rendered `source_index.md` line uses `[[vault_relative_path#heading|display_title]]`; when no heading, falls back to `[[vault_relative_path|display_title]]`. Golden test on both branches.

## 6. Critique: artifact-first vs SQLite

**Right call for v1. The pitfalls are at the seams with the existing DB writer.**

### Minimal DB writes that are safe in v1

Reuse existing `src/vault/ingest.py` for raw document registration. That gives us:

- `INSERT INTO documents` — works; `metadata_json` field is the carrier for the new `source_grade_rationale`, `source_id`, `retrieval_method`, `fetcher_name`, `fetcher_version`.
- `INSERT INTO document_companies` — works.
- `INSERT INTO companies` — auto-created by `VaultStore.link_document_company`.

### What must NOT touch the DB in v1

The schema *already has* `questions`, `answers`, `conflicts`, `facts`, `metrics`, `wiki_versions` tables. v1 must **not** write to them, even though it would be tempting to insert the `QuestionRecord` JSON into the `questions` table. Two truths (JSON artifact + SQLite row) will drift the moment v2 changes the JSON shape. v1 writes only to `documents` + link tables; everything else stays in JSON/MD.

### Concrete implementation risks at the DB seam

1. **`ingest_file` does not record `source_grade_rationale`.** Current signature (`src/vault/ingest.py:55`) takes `source_grade` as a single character. v1 needs to thread rationale through. Either: (a) extend `ingest_file` to accept `source_grade_rationale: str | None`, stored under `metadata["source_grade_rationale"]`; or (b) wrap `ingest_file` in `memo_run` and patch the metadata JSON post-hoc. (a) is cleaner and a one-line change.
2. **`ingest_file` requires the file on disk and only supports a small suffix set** (`_TEXT_SUFFIXES`, PDF). For SEC filings fetched as text, use `ingest_text` (already exists). For yfinance JSON snapshots, write a synthetic Markdown file via `ingest_text` rather than trying to ingest a `.json` directly (the `documents.path` field points to extracted text; JSON should also be saved as a sibling artifact, not as the canonical extracted text).
3. **`extract_text` raises `ValueError` on unsupported suffixes** (`ingest.py:52`). v1 manual-file handling must catch this and either reject the manual file with a clear error or downgrade to "binary attachment, not parsed" with a caveat. The plan does not specify — engineering default: reject with explicit error since at least one required source must be readable.
4. **Default schema `documents.source_grade` is `'B'`** (`schema.py:28`). Good — matches v1 default. But `memo_flow` already inserts manual files at `'A'`, so any existing vault from prototype runs has `'A'`-grade manual entries. v1 ingestion idempotency uses checksum-based `doc_id`; a re-ingested file *will not* overwrite the prior grade unless we explicitly update. Tradeoff: leave legacy A-grade alone (simplest) and document. Do not introduce a "regrade on ingest" path in v1.

### Artifact layout sanity check

The proposed layout in §6 of the checklist is fine. Two small concerns:

- `research/context_packages/<run_id>.json` lives outside `memos/<run_id>/`. Reproducibility is preserved by `manifest.json` referencing it. Acceptable.
- `memos/<run_id>/` contains both product and reproducibility artifacts. Confirm `final.md` will not be regenerated under the *same* run_id on retry (would clobber). Engineering default: `run_id` is mandatory unique per invocation; collision raises.

## 7. Critique: provider/retrieval adapters — mock vs live, failure seams

### Mock by default in unit tests

All four external providers must be mocked in pytest:
- SEC filings (`src.tools.equity.sec_filings.search_and_read_filings`)
- Company profile (`src.tools.equity.company_profile.fetch_company_profile`)
- Financials (`src.tools.equity.financials.fetch_financial_metrics`)
- Market data (`src.tools.equity.market_data.fetch_historical_data`)

Tests should inject mocks via either monkeypatch or a thin `SourceAdapters` dataclass passed into `memo_run`. The dataclass approach is cleaner and would also be the natural seam to extract in v1.1.

### Likely failure seams

- **SEC EDGAR rate limits / user-agent requirements.** SEC requires a User-Agent string; missing or generic UAs return 403. If existing `sec_filings` already handles this, fine; otherwise v1 will hit it on the first live run. Worth confirming with one mocked-403 test that the failure converts to a caveat + missing-source entry instead of crashing.
- **yfinance returning empty / NaN snapshots after market hours or for thinly traded tickers.** XOM is fine; the generic case is not. v1 should treat "empty payload" as a non-error caveat, not a hard failure.
- **10-Q vs 10-K date comparison.** If either filing has a missing `filed_at`, do not synthesize a comparison — emit caveat "could not compare 10-K/10-Q recency, included latest 10-K only".
- **Provider returns content that `extract_text` cannot parse.** Mostly relevant for SEC's exhibit blobs. Tests should include one provider mock that returns an `.html` body (supported) and one that returns an unsupported binary blob (must caveat, not crash).
- **Manual file outside vault root / path traversal.** `SourceRecord.vault_relative_path` validator already blocks `..` and absolute paths, but `--manual-file` is an *input* path that may live anywhere. Resolve to absolute path at CLI time; copy/ingest into vault; never store the original absolute path as `vault_relative_path`. (The existing `ingest_file` does this correctly; just confirm in tests.)
- **Run idempotency on partial failure.** If SEC succeeds but yfinance fails mid-run, `documents` table already has the SEC row inserted. A retry with the same `run_id` should pick up where it left off — but the simpler v1 contract is "retries get a new `run_id`; vault dedupes by checksum on documents". Document this clearly.

### Live demo gate

Only after all unit tests pass. The XOM live demo command should be documented but not exercised in CI. Outputs to inspect: `final.md` exists, `manifest.json` references at least one SEC filing with grade `A` and at least one B-grade snapshot, `status.json == "succeeded"`, scratch is cleaned, vault originals are present.

## 8. Critique: MemoContextPackage → existing harness bridge

### Duplicate context risk

The existing harness skill pack includes `equity`, which gives agents SEC/profile/financial tools. With a curated package attached, the agent now has *two* paths to the same data. Risks:

- **Drift:** agent re-fetches SEC, gets a 10-K filed_at slightly different from the curated one due to a same-day amendment, and writes a memo with both grades floating around.
- **Cost:** unnecessary tool calls inflate latency and tokens for v1 where the curated context should be enough.
- **Citation drift:** agent cites a snippet it just re-fetched rather than the curated `Citation.citation_key`, breaking the structured citation contract.

Mitigations (pick one — flag to Ted as §2 item 3):
- (a) Strip `equity` from `available_skill_packs` for v1 memo runs. Memo runs become pure context-consumers.
- (b) Keep equity tools, but prompt template explicitly orders: "prefer attached `context_files`; call tools only to verify a specific claim, and cite using the same `citation_key` you find in the context package".
- Personal recommendation: (b) for v1 (allows verification, fewer changes), revisit in v1.1.

### Citation loss risk

`Citation` objects carry structured `citation_key`/`document_id`/`vault_relative_path`. The agent writes `final.md` as freeform prose. **There is no automated post-check that the citations in the prose match the citations in the context package.** The plan says "Contract truth remains structured citations in package/manifest, not prose parsing" — fine for v1, but at least one test should assert that the manifest *contains* the full citation list so reviewers can grep prose against it. Defer prose validation to v2.

### Cleanup risk

`MemoContextPackage` JSON lives under `vault/companies/<TICKER>/research/context_packages/<run_id>.json`. The harness scratch lives under `run_root` (a separate temp tree managed by harness `artifacts.py`). Cleanup policy `product_final_only` deletes scratch — *not* the vault path. Safe. One test must assert this invariant after a successful run and a failed run.

### Prompt length risk

Inline `MemoContextPackage` JSON for an XOM-style run with 10-K + 10-Q + profile + financials + market + 2 manual files could easily be 50–200KB. **Do not** inline. Strategy already in the plan: attach via `context_files`, render only summaries inline. Add a hard budget test: rendered inline prompt stays under 32KB; total attached context under 1MB (rough sanity cap).

### Tool catalog seam

`HarnessRequest.context_files` is a documented field (`src/harness/types.py:266`). It is already exercised by `memo_flow.py:64`. Confirm — by reading or stepping through `runtime.py` — that attached files are made readable to the agent in its context directory. (Spot-check passed: `prompt_builder._list_context_files` enumerates them in the prompt, and `runtime.py` writes them in.)

## 9. Minimal implementation checklist before coding

1. Pick the orchestrator's package location (§2 #1). Create the package.
2. Resolve manual-file default grade (§2 #2). Document the decision in the new module's docstring.
3. Decide skill-pack policy for v1 memo runs (§2 #3). Bake into `memo_run` defaults.
4. Add `source_grade_rationale: str | None = None` to `ingest_file` and `ingest_text` (one-line change; stored under `metadata["source_grade_rationale"]`).
5. Tighten `_render_prompt_template` (or write a stricter wrapper) to raise on any leftover `{{...}}` placeholders after substitution.
6. Decide the prompt template path (§5).
7. Pin the `manifest.json` schema: `{run_id, ticker, prompt, source_documents: [{document_id, title, source_type, source_grade, checksum, vault_relative_path}], final_path, context_package_path, status, created_at}`.
8. Pin the `status.json` schema: `{run_id, status: "succeeded"|"failed"|"partial", started_at, finished_at, errors: [...], warnings: [...]}`.
9. Pin the `source_index.md` and `question_list.md` rendering format (one-liner spec each).
10. Write the test list:
    - prompt template render (golden + failure)
    - `SourceRecord` → `DocumentRef` mapping
    - `RetrievalBatch` with missing optional source → caveat, not error
    - `RetrievalBatch` with no required source → run aborts before harness
    - manual-file grade B + rationale ends up in `documents.metadata_json`
    - mocked SEC failure → caveat
    - mocked yfinance empty → caveat
    - manifest.json round-trip
    - cleanup keeps vault originals
    - context_files reach the agent (smoke)
11. Decide where unit tests live (`tests/unit/research_platform/` is the natural home).
12. Then, and only then, write `memo_run.py`.

## 10. Final lists

### Escalate to Ted
- Module home for the orchestrator: `src.retrieval.memo_run` (per checklist) vs `src.research_platform.memo_run` (per layering in Consolidated Plan §1.1). One-line ruling needed.
- Confirm manual-file default grade `B` with rationale overrides the prototype `memo_flow` behavior of grade `A`.
- v1 memo-run skill pack policy: keep `equity` tools available (with prompt guidance to prefer curated context), or strip to `core` only?

### Can decide now
- `--manual-file` repeatable, `--vault-root` optional, `--prompt` required, `--run-id` optional.
- 10-K/10-Q recency comparison uses `filed_at`; missing dates → 10-K only + caveat.
- `source_index.md` and `question_list.md` written per-company *and* snapshotted per-run under `memos/<run_id>/`.
- Pass `MemoContextPackage` JSON + source/question files via `HarnessRequest.context_files`; never inline the JSON.
- Reuse `src/tools/equity/` functions directly; defer adapter abstraction to v1.1.
- Run aborts pre-harness if zero required sources retrieved.
- `run_id` must be unique per invocation; collision raises.
- v1 writes only to `documents` + `document_companies` + `companies`; no `questions`/`facts`/`metrics`/`conflicts`/`wiki_versions` writes.
- Tests use mocked providers; live XOM is a manual demo after CI passes.
- Add `source_grade_rationale` to `ingest_file`/`ingest_text` and to `metadata_json`.
- Tighten `{{...}}` placeholder render to fail on missing keys.
- Cap inline prompt at ~32KB; context bundle at ~1MB (engineering cap, revisable).
