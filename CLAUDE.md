# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaSeeker is a multi-agent quantitative research system for equity, macro, and commodity analysis. A supervisor-led architecture routes user prompts to domain-specific specialist agents, which gather real data through code and external APIs, then synthesizes their outputs into a single coherent report.

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
uv run pytest tests/unit/test_model_config.py
uv run pytest tests/unit/test_model_config.py::test_function_name -v

# Run AlphaSeeker CLI
uv run python main.py
```

## Architecture

### Supervisor-Led Multi-Agent Flow

```
User Prompt → Supervisor (classify intent) → Intent Router
    → [Equity Agent] ─┐
    → [Macro Agent]  ─┼─→ (parallel via LangGraph Send API)
    → [Commodity Agent] ┘
    → Synthesizer → Final Response
```

The **supervisor graph** (`src/supervisor/graph.py`) is a LangGraph StateGraph that:
1. Classifies user intent via LLM
2. Routes to appropriate sub-agents in parallel using the `Send` API
3. Synthesizes results from all agents

Each **sub-agent** is itself a LangGraph application:
- **Equity** (`src/agents/equity/graph.py`): yfinance, SEC filings, web search
- **Macro** (`src/agents/macro/`): FRED, World Bank data
- **Commodity** (`src/agents/commodity/`): EIA, CFTC, futures data

### Shared Layer

- `src/shared/model_config.py`: Provider-agnostic model selection with `ALPHASEEKER_MODEL_<AGENT>_<ROLE>` env var overrides
- `src/shared/llm_manager.py`: LLM invocation abstraction
- `src/shared/web_search.py`: Web research capability
- `src/shared/node_contracts.py`: Result validation for graph nodes

### Harness Runtime

An alternative execution mode (`--runtime harness`) with its own executor, agent worker, and transport layer in `src/harness/`.

### Model Configuration

Models are defined in `config/models.yaml` by agent and role. Provider prefixes: `sf/` (SiliconFlow), `kimi-`, `minimax/`, `gpt-`, `gemini-`, `claude-`. Each requires a corresponding API key env var.

### Runtime Outputs

- `data/` — fetched datasets and debug artifacts
- `reports/` — generated Markdown reports
- `charts/` — generated chart images

These directories are git-ignored.

## Testing

Pytest markers: `unit` (deterministic, no network), `component` (mocked dependencies), `live` (real APIs), `network` (external network required).

## File Organization

```
src/agents/{equity,macro,commodity}/   # Domain sub-agents (each is a LangGraph app)
src/supervisor/                         # Supervisor orchestration layer
src/shared/                             # Shared utilities (LLM, model config, web search)
src/harness/                            # Alternative runtime harness system
src/utils/                              # General utilities
tests/{unit,component,live}/             # Three-tier test structure
```
