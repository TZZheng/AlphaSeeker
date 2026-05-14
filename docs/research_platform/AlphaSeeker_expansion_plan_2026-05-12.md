# AlphaSeeker Expansion Plan — Toward a Librarian-Style Equity Research OS

Date: 2026-05-12 PT
Repo inspected: https://github.com/TZZheng/AlphaSeeker
Local clone used for inspection: `/tmp/AlphaSeeker`

## Executive takeaway

AlphaSeeker already has the hard part: a file-based async multi-agent research harness that fetches live data, writes every run to disk, uses deterministic skill packs, and already has per-run concepts like `SourceCard`, `FactIndexRecord`, `CoverageMatrix`, and an `evidence_ledger`.

The Xiaohongshu post suggests a different product shape: not “one run -> one memo,” but “every run updates a persistent company/ticker research wiki and fact registry.”

So the right expansion is **not** to rebuild the harness. Keep the harness. Add a persistent layer underneath/alongside it:

- raw document vault;
- SQLite/DuckDB registry;
- company/ticker wiki pages;
- source/fact/metric/question/answer/conflict tables;
- harness skills that read/write this persistent layer;
- workflows for onboarding, meeting prep, post-meeting update, conflict scan, and daily monitor.

## What AlphaSeeker already has

Useful existing pieces:

- `README.md` describes current system as “question to comprehensive investment memo” and file-based async multi-agent runtime.
- `src/harness/README.md` shows the runtime: `run_harness()` creates a run root; workers execute tools; children publish files; root synthesizes final report.
- `src/harness/artifacts.py` defines run layout and per-agent workspaces under `data/harness_runs/<run_id>/`.
- `src/harness/types.py` already defines `SourceCard`, `FactIndexRecord`, `SectionBrief`, `CoverageMatrix`, `EvidenceItem`, and `HarnessState`.
- `src/harness/retrieval.py` already performs deterministic corpus-building/reduction for a single run.
- `src/harness/skills/equity.py` already wraps deterministic equity tools: market data, company profile, financials, SEC filings, earnings calls, insider activity, peers.
- `src/harness/registry.py` has a simple skill-pack registry, so adding a new `vault` or `research_os` skill pack is straightforward.

Current limitation:

- Most knowledge is **per-run**. A run can produce excellent artifacts, but the next run does not automatically inherit a durable company wiki/fact base unless the user manually points it at old files.

The upgrade is therefore: **promote selected per-run outputs into a persistent company-centric vault.**

## Proposed minimal architecture

Add this persistent tree:

```text
data/research_vault/
├── vault.sqlite                  # durable registry
├── documents/                    # raw imported source files, content-addressed or dated
│   └── <doc_id>/
│       ├── original.<ext>
│       ├── extracted.txt
│       └── metadata.json
├── companies/
│   └── <TICKER>/
│       ├── wiki.md               # living company homepage
│       ├── question_list.md      # active/open questions
│       ├── conflicts.md          # human review queue
│       ├── catalyst_calendar.md
│       └── versions/
│           └── wiki_YYYYMMDD_HHMMSS.md
└── exports/
    └── <run_id>/                 # optional snapshots used by harness context
```

Use SQLite first. DuckDB can come later if analytics on large tabular data becomes important.

## Minimal DB schema

Start with these tables. Keep fields simple; do not over-engineer embeddings on day one.

```sql
CREATE TABLE documents (
  doc_id TEXT PRIMARY KEY,
  source_type TEXT NOT NULL,      -- pdf, markdown, note, web, transcript, excel, xhs_ocr, sec, model
  title TEXT,
  path TEXT NOT NULL,
  url TEXT,
  published_at TEXT,
  ingested_at TEXT NOT NULL,
  source_grade TEXT DEFAULT 'B',  -- A/B/C, human-overridable
  checksum TEXT,
  metadata_json TEXT
);

CREATE TABLE companies (
  ticker TEXT PRIMARY KEY,
  name TEXT,
  exchange TEXT,
  sector TEXT,
  updated_at TEXT
);

CREATE TABLE document_companies (
  doc_id TEXT NOT NULL,
  ticker TEXT NOT NULL,
  relevance TEXT DEFAULT 'mentioned',
  PRIMARY KEY (doc_id, ticker)
);

CREATE TABLE facts (
  fact_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  statement TEXT NOT NULL,
  section TEXT,
  source_doc_id TEXT,
  source_quote TEXT,
  source_grade TEXT,
  confidence REAL DEFAULT 0.5,
  observed_at TEXT,
  created_at TEXT NOT NULL,
  supersedes_fact_id TEXT,
  status TEXT DEFAULT 'active'    -- active, stale, disputed, rejected
);

CREATE TABLE metrics (
  metric_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  metric_name TEXT NOT NULL,      -- revenue, EBITDA, TP, PE, EV/EBITDA, guidance, etc.
  period TEXT,
  value TEXT NOT NULL,
  unit TEXT,
  source_doc_id TEXT,
  source_grade TEXT,
  observed_at TEXT,
  created_at TEXT NOT NULL,
  status TEXT DEFAULT 'active'
);

CREATE TABLE questions (
  question_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  question TEXT NOT NULL,
  status TEXT DEFAULT 'open',     -- open, answered, stale
  priority TEXT DEFAULT 'normal',
  created_at TEXT NOT NULL,
  updated_at TEXT
);

CREATE TABLE answers (
  answer_id TEXT PRIMARY KEY,
  question_id TEXT NOT NULL,
  answer TEXT NOT NULL,
  source_doc_ids TEXT,            -- JSON list
  confidence REAL DEFAULT 0.5,
  created_at TEXT NOT NULL
);

CREATE TABLE conflicts (
  conflict_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  conflict_type TEXT NOT NULL,    -- metric_mismatch, stale_tp, source_disagreement, guidance_change
  summary TEXT NOT NULL,
  left_ref TEXT,
  right_ref TEXT,
  severity TEXT DEFAULT 'medium',
  status TEXT DEFAULT 'open',     -- open, resolved, ignored
  created_at TEXT NOT NULL,
  resolved_at TEXT
);

CREATE TABLE wiki_versions (
  version_id TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  path TEXT NOT NULL,
  run_id TEXT,
  created_at TEXT NOT NULL,
  summary TEXT
);
```

## Code changes by file

### 1. Add vault package

```text
src/vault/
├── __init__.py
├── paths.py          # resolve data/research_vault paths
├── schema.py         # DDL and migrations
├── store.py          # SQLite helpers and CRUD
├── ingest.py         # file/url/text ingestion -> documents + extracted text
├── wiki.py           # render/update company wiki markdown
├── questions.py      # question list and Q&A archive helpers
└── conflicts.py      # deterministic stale/conflict checks
```

This keeps persistent research state separate from the harness runtime.

### 2. Add a harness skill pack

```text
src/harness/skills/vault.py
```

First public skills:

- `vault_ingest_document(path_or_url, ticker=None, source_type=None, title=None, source_grade='B')`
- `vault_search(ticker, query=None, section=None, max_results=20)`
- `vault_get_company_context(ticker)` — returns wiki summary + open questions + open conflicts + latest docs.
- `vault_update_company_wiki(ticker, proposed_markdown, run_id=None)` — writes versioned wiki update.
- `vault_log_question(ticker, question, priority='normal')`
- `vault_answer_question(question_id, answer, source_doc_ids=[], confidence=0.5)`
- `vault_scan_conflicts(ticker)` — writes/updates conflict queue.

Integration points:

- `src/harness/skills/__init__.py`: export `VAULT_SKILLS`.
- `src/harness/registry.py`: include `VAULT_SKILLS` in `build_skill_registry()`.
- `src/harness/types.py`: add `"vault"` to `ALL_PACKS` validation.
- `src/harness/presets.py`: make `vault_*` visible to `research`, `orchestrator`, and maybe `evaluator`; hide write/update skills from `source_triage` if you want safety.

### 3. Add CLI commands later, not first

Current `main.py` is a thin Typer wrapper around the TUI. Do not start by building a full CLI suite. First make the vault usable through harness skills and tests.

Later useful commands:

```bash
uv run python -m src.vault.ingest --ticker XOM ./docs/xom_report.pdf
uv run python -m src.vault.wiki --ticker XOM --render
uv run python -m src.vault.conflicts --ticker XOM
```

## Workflow mapping from the Xiaohongshu post

### A. New company onboarding

Input: ticker + dumped docs.

1. User places PDFs/notes/transcripts/models into a folder.
2. Harness run calls `vault_ingest_document` on each source.
3. Research agents use existing equity skills (`fetch_company_profile`, `fetch_financials`, `search_sec_filings`, `research_earnings_call`, etc.).
4. Agents extract facts/metrics/questions into vault.
5. `wiki.py` renders `data/research_vault/companies/<TICKER>/wiki.md`.
6. `vault_scan_conflicts` creates a first `conflicts.md` and DB rows.

Deliverable: company wiki + open gaps/questions.

### B. Meeting prep / analyst call prep

Input: ticker + question list.

1. User writes questions or calls `vault_log_question`.
2. `vault_get_company_context(ticker)` loads wiki + docs + open conflicts.
3. Agent answers each question from vault evidence first.
4. Unknown or contradictory items become follow-up questions.
5. Output: `question_list.md` with preliminary answers and suggested follow-ups.

### C. Post-meeting update

Input: meeting notes/transcript.

1. Ingest meeting note as high-grade source if it is direct management/analyst communication.
2. Extract new facts/metrics and mark old facts as superseded if needed.
3. Update wiki sections.
4. Resolve or create conflicts.
5. Archive Q&A.

### D. Daily monitor

Later, not MVP day one.

1. Search latest news/filings/company IR pages.
2. Ingest new sources.
3. Compare new metrics/facts against active facts/metrics.
4. Update daily key takeaways and conflict queue.

## Suggested wiki template

Use a stable markdown template so the system compounds knowledge:

```markdown
# <TICKER> — <Company Name>

Last updated: <timestamp>
Source policy: every factual claim should have source_doc_id or link.

## 1. One-line thesis
## 2. Business overview
## 3. Segment economics
## 4. Key guidance / management commentary
## 5. Known truths
## 6. Recent thesis changes
## 7. Valuation snapshot
## 8. Model / estimates registry
## 9. Catalysts calendar
## 10. Risks and counterevidence
## 11. Peer / competitor dynamics
## 12. Open questions
## 13. Conflicts / items needing human judgment
## 14. Source index
```

This mirrors the post’s “living company homepage” idea without copying an overly complex 14-section structure.

## Implementation order

### Step 1 — Persistent vault skeleton, tests first

Files:

- `src/vault/paths.py`
- `src/vault/schema.py`
- `src/vault/store.py`
- `tests/unit/test_vault_store.py`

Acceptance:

- `init_vault()` creates `data/research_vault/vault.sqlite`.
- Can insert/read document, company, fact, metric, question, conflict.
- Tests pass offline.

### Step 2 — Document ingestion

Files:

- `src/vault/ingest.py`
- `tests/unit/test_vault_ingest.py`

Support:

- markdown/text first;
- PDF via existing `pypdf` dependency;
- web via existing `trafilatura`/web read path later;
- Excel/model files later.

Acceptance:

- Given a local markdown/PDF path and ticker, copy original to `documents/<doc_id>/`, write `extracted.txt`, insert DB rows.

### Step 3 — Vault harness skills

Files:

- `src/harness/skills/vault.py`
- modify `src/harness/skills/__init__.py`
- modify `src/harness/registry.py`
- modify `src/harness/types.py`
- maybe modify `src/harness/presets.py`
- tests under `tests/unit/test_harness_vault_skills.py`

Acceptance:

- A harness skill can ingest a doc and retrieve company context.
- Skill outputs remain small; bulky content goes to files.

### Step 4 — Company wiki renderer

Files:

- `src/vault/wiki.py`
- `tests/unit/test_vault_wiki.py`

Acceptance:

- Given facts/metrics/questions/conflicts in DB, render `companies/<TICKER>/wiki.md`.
- Previous wiki versions are saved under `versions/`.

### Step 5 — Question workflow

Files:

- `src/vault/questions.py`
- harness skills for log/answer question.

Acceptance:

- Open questions appear in wiki.
- Answers link back to source docs.

### Step 6 — Conflict scanner

Files:

- `src/vault/conflicts.py`

Start with deterministic rules:

- same `ticker + metric_name + period` but different values from different sources;
- target price older than N days;
- active fact superseded by newer source but still in wiki;
- source grades conflict: A-grade filing/model vs B-grade web/sell-side note;
- valuation metric stale relative to latest price date.

Acceptance:

- Scanner creates `conflicts` rows and `companies/<TICKER>/conflicts.md`.

## What not to do first

Avoid these in the first iteration:

- full web app;
- vector DB as the core data model;
- complicated ontology;
- fully automated daily cron;
- trying to perfectly parse every broker PDF/table/model;
- making agents directly mutate wiki without a versioned proposed-update path.

The post’s key lesson is workflow and QC, not a fancy frontend.

## Recommended first user-facing prompt after implementation

```text
Onboard XOM into the research vault. Use existing equity skills and any local documents under ./data/inbox/xom. Build or update the company wiki, log open questions, and run a first conflict scan. Publish a summary of what changed and what still needs human judgment.
```

## Short answer

AlphaSeeker should evolve from:

```text
prompt -> multi-agent run -> final memo
```

to:

```text
source docs + live data -> persistent company vault -> multi-agent run -> wiki updates + conflicts + memo
```

The current repo is already close because it is file-first and has structured evidence/retrieval artifacts. The missing piece is persistence across runs and a company/ticker-centered schema.
