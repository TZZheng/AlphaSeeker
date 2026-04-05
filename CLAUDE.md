# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaSeeker is a multi-agent quantitative research system for equity, macro, and commodity analysis. A root orchestrator agent routes user prompts to domain-specific child agents, which gather real data through code and external APIs, then synthesize their outputs into a single coherent report.

## Commands

```bash
# Install dependencies
uv sync

# Compile check (syntax validation)
uv run python -m compileall -q src main.py

# Run offline tests (unit + component)
uv run pytest -m "not live"

# Run live integration tests (requires API keys)
uv run pytest -m "live"

# Run a single test file or function
uv run pytest tests/unit/test_harness_types_and_verifier.py
uv run pytest tests/unit/test_harness_types_and_verifier.py::test_function_name -v

# Run AlphaSeeker CLI
uv run python main.py "Analyze AAPL"
uv run python main.py  # interactive mode
```

## Architecture

### Harness Runtime

Harness is a file-based async multi-agent runtime. The root agent runs as a subprocess supervised by an async kernel. The LLM decides when to spawn child agents, which skills to call, and when to publish results. File-based handoff between agents provides transparency and recoverability.

```
User Prompt → Root Orchestrator Agent (subprocess)
    → spawns child agents (research, writer, synthesizer, etc.)
    → child agents call deterministic skill tools (fetch_market_data, fetch_futures_curve, etc.)
    → publish/ files → parent reads → synthesizes final report
```

Key components:
- `src/harness/runtime.py` — async supervisor kernel, launches agent workers, watches heartbeats
- `src/harness/agent_worker.py` — long-lived worker process for root and child agents
- `src/harness/executor.py` — file-first agent tools: `spawn_subagent`, `list_children`, `bash`, `write_file`, `edit_file`
- `src/harness/transport.py` — MiniMax Anthropic/OpenAI transport adapters with transcript replay
- `src/harness/commenter.py` — paired commenter sidecar that reads agent workspace and injects advisory comments
- `src/harness/skills/` — deterministic skill library reused as tools (equity, macro, commodity, core)

### Skill Packs

Agents use skill packs to access domain-specific tools:
- **core** — `search_in_files`, `get_current_datetime`, `search_web`, `read_web_pages`, `condense_context`, `read_file`, `retrieve_sources`
- **equity** — `fetch_market_data`, `plot_price_history`, `fetch_company_profile`, `fetch_financials`, `search_sec_filings`, `fetch_insider_activity`, `research_earnings_call`, `analyze_peers`
- **macro** — `fetch_macro_indicators`, `fetch_world_bank_indicators`
- **commodity** — `fetch_eia_inventory`, `fetch_cot_report`, `fetch_futures_curve`

### Shared Layer

- `src/shared/model_config.py`: Provider-agnostic model selection with `ALPHASEEKER_MODEL_<AGENT>_<ROLE>` env var overrides
- `src/shared/llm_manager.py`: LLM invocation abstraction
- `src/shared/web_search.py`: Web research capability (DDG + trafilatura)
- `src/shared/report_filename.py`: Report filename building

### Tool Library

`src/tools/` contains domain-specific data-fetching tools used by harness skills:
- `src/tools/equity/` — market data, financials, SEC filings, earnings calls, insider trading, peers, visualization
- `src/tools/macro/` — FRED, World Bank indicators
- `src/tools/commodity/` — EIA, CFTC COT, futures curve

### Model Configuration

Models are defined in `config/models.yaml` by role. Provider prefixes: `sf/` (SiliconFlow), `kimi-`, `minimax/`, `gpt-`, `gemini-`, `claude-`. Each requires a corresponding API key env var.

### Runtime Outputs

- `data/harness_runs/<run_id>/` — all run artifacts, agent workspaces, transcripts, progress
- `reports/` — generated Markdown reports (legacy, git-ignored)
- `charts/` — generated chart images (git-ignored)

## Testing

Pytest markers: `unit` (deterministic, no network), `component` (mocked dependencies), `live` (real APIs), `network` (external network required).

## File Organization

```
src/tools/                               # Shared data-fetching tool library
src/harness/                            # Harness runtime (main/only execution mode)
src/shared/                             # Shared utilities (LLM, model config, web search)
src/utils/                              # General utilities
tests/{unit,component,live}/             # Three-tier test structure
```
