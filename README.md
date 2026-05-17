# AlphaSeeker

AlphaSeeker is now a vault-first research workspace for company-centered investment work.  The old subprocess harness and Textual TUI have been removed.  The retained backend focuses on three things:

1. **Persistent company vaults** under `vault/companies/<TICKER>/`.
2. **Deterministic source ingestion and wiki/status rendering** in `src/vault/`, backed by reusable market/source tools in `src/tools/` and shared model/reliability utilities in `src/shared/`.
3. **A ticker-local LingTai team bridge** (`scripts/lingtai_ticker_harness.py`) that lets a frontend or operator send questions to long-lived ticker teams and read their replies through the ticker-local `human` mailbox.

This is a breaking cleanup: legacy `src/harness`, `src/cli`, `src/research_platform`, `src/retrieval`, `main.py`, and their tests/docs are intentionally gone.

## Current repository shape

```text
config/models.yaml                         # model role configuration for synthesis/shared LLM calls
scripts/lingtai_ticker_harness.py          # bridge into ticker-local LingTai teams
src/shared/                                # model config, Codex auth, retry/cache, web/text utilities
src/tools/                                 # reusable equity/macro/commodity source tools
src/vault/                                 # company vault schema, ingest, extraction, wiki/status/synthesis
vault/companies/<TICKER>/                  # persistent company artifacts and ticker-local teams
templates/lingtai_native/
                                            # role/policy templates consumed by the ticker harness
tests/unit/test_vault_*.py                 # retained vault tests
tests/unit/test_sec_filings.py             # retained source-tool tests
tests/unit/test_reliability.py             # retained shared utility tests
```

## Vault backend

The vault backend stores company research as durable files plus a SQLite index.
Key modules:

- `src/vault/schema.py` / `src/vault/store.py` — initialize and query the vault database.
- `src/vault/ingest.py` — ingest text, markdown, PDFs, and local files into a company vault with source metadata.
- `src/vault/sec_import.py` — import SEC filing text through the retained equity filing tools.
- `src/vault/extract.py` — deterministic extraction helpers for structured company records.
- `src/vault/status.py` — status-patrol/open-question helpers.
- `src/vault/wiki.py` — render company wiki/support pages from vault state.
- `src/vault/synthesis.py` — build a source bundle and call the configured LLM for a company research-state synthesis.
- `src/vault/onboard.py` — high-level onboarding flow that pulls profile/financial/SEC context and renders initial vault pages.

Example direct usage:

```bash
uv run python - <<'PY'
from src.vault.ingest import ingest_text
from src.vault.store import VaultStore

result = ingest_text(
    "Company-provided source text or an analyst note.",
    ticker="XOM",
    title="XOM source note",
    source_type="manual_file",
    source_grade="B",
    source_grade_rationale="manual note; provenance checked by operator",
    root="vault",
)
print(result)
print(VaultStore("vault").company_context("XOM")["documents"])
PY
```

Run the synthesis layer only when a model backend is configured and expected to be used:

```bash
uv run python - <<'PY'
from src.vault.synthesis import synthesize_company_research_state

print(synthesize_company_research_state("XOM", root="vault"))
PY
```

## Ticker-local LingTai bridge

Ticker teams live under:

```text
vault/companies/<TICKER>/.lingtai/
```

The bridge script simulates the future frontend/runtime boundary.  It does not run the team itself; it writes mailbox messages from the ticker-local pseudo-human endpoint to `<TICKER>_orchestrator`, and it can read replies sent back to the ticker-local `human` inbox.

Common commands:

```bash
# Ensure vault/team directories and ticker-local human endpoint exist.
uv run python scripts/lingtai_ticker_harness.py TSLA --root . --ensure-dirs

# Review rendered role comments from the native templates.
uv run python scripts/lingtai_ticker_harness.py TSLA --root . --render-comments

# Patch rendered comments into existing ticker-local init.json files.
uv run python scripts/lingtai_ticker_harness.py TSLA --root . --apply-comments

# Send a natural-language request to TSLA_orchestrator through the pseudo-human outbox.
uv run python scripts/lingtai_ticker_harness.py TSLA "Please refresh the latest TSLA memo." --root . --send --subject "Refresh latest memo"

# Read messages that the ticker-local team sent to the pseudo-human inbox.
uv run python scripts/lingtai_ticker_harness.py TSLA --root . --read-human

# Queue one adversarial IC-chair probe if latest.md changed since the last probe.
uv run python scripts/lingtai_ticker_harness.py TSLA --root . --probe-latest-if-changed

# Queue an institutional-grade synthesis review if latest.md changed since the last synthesis.
uv run python scripts/lingtai_ticker_harness.py TSLA --root . --synthesize-latest-if-changed
```

The ticker-local `human` directory is intentionally a pseudo-agent endpoint (`admin: null`) with metadata allowing the outer/admin harness to inspect its inbox.  Do not turn it into a runnable admin avatar just to read/write mailbox traffic; use a separate controller if one is needed.

## Model configuration

`config/models.yaml` controls model assignments used by retained shared/vault LLM calls.  The current default is native Codex subscription access:

```yaml
harness:
  agent: "codex/gpt-5.5"
  condense: "codex/gpt-5.5"
```

The key name `harness` is currently a compatibility label used by `src.shared.model_config` and `src.vault.synthesis`; it no longer refers to the removed subprocess harness.  Override with environment variables such as:

```bash
export ALPHASEEKER_MODEL_HARNESS_AGENT="minimax/MiniMax-M2.5"
```

Provider key requirements are derived by `src.shared.model_config`.  `codex/*` uses OAuth tokens from `~/.lingtai-tui/codex-auth.json`; OpenAI-compatible providers use their provider-specific API keys.

## Development and validation

Install dependencies with `uv`, then run:

```bash
uv run python -m compileall -q src scripts
uv run pytest -q
uv run python -m py_compile scripts/lingtai_ticker_harness.py
```

The CI workflow performs the same compile check and the offline pytest suite.  There is no separate live legacy-harness CI job after the cleanup.

## Notes on generated artifacts

- `vault/companies/<TICKER>/team/` contains published memos, drafts, raw team outputs, and version archives.
- `vault/companies/<TICKER>/.lingtai/` contains live agent process state and mailboxes.  Treat this as runtime state; do not commit indiscriminately unless the project owner explicitly wants a network snapshot.
- `.latest_probe_state.json` and `.latest_synthesis_state.json` under a published memo folder are guard files used by the bridge script to avoid re-queuing the same probe/synthesis request for an unchanged `latest.md`.
