# AlphaSeeker

[![CI](https://github.com/TZZheng/AlphaSeeker/actions/workflows/ci.yml/badge.svg)](https://github.com/TZZheng/AlphaSeeker/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/package%20manager-uv-5C6AC4)
![License](https://img.shields.io/badge/license-MIT-green)

**Multi-agent quantitative research — from question to comprehensive investment memo.**

AlphaSeeker is a file-based async multi-agent runtime. Give it a financial question — it spawns specialist agents, fetches real market data, builds charts, and synthesizes everything into a single well-structured report. The entire process is traceable: every agent's workspace, every data fetch, every model call is written to disk.

No black boxes. No hallucinated numbers. The research pipeline that actually shows its work.

## Table of Contents

- [Overview](#overview)
- [Why AlphaSeeker](#why-alphaseeker)
- [What It Can Do](#what-it-can-do)
- [How a Query Flows](#how-a-query-flows)
- [Quickstart](#quickstart)
- [Configuration](#configuration)
- [Example Prompts](#example-prompts)
- [Example Outputs](#example-outputs)
- [Project Structure](#project-structure)
- [Testing](#testing)
- [Current Limits](#current-limits)
- [Additional Reading](#additional-reading)
- [Contributing](#contributing)
- [License](#license)

## Overview

AlphaSeeker runs a **root orchestrator agent** as a supervised subprocess. The orchestrator decides — based on the prompt — when to delegate to child agents, which skill tools to call, and when the report is ready to publish.

Child agents are **deterministic research workers**: they don't guess at numbers, they fetch them from live data sources (market data, SEC filings, FRED, EIA, CFTC). The orchestrator synthesizes their outputs into a final memo.

Key design choices:

- **File-based handoff** — agents communicate by writing and reading files, not in-memory state. You can inspect every intermediate result.
- **Subprocess isolation** — a crashed or stalled child agent cannot block the pipeline.
- **Skill packs** — deterministic tools organized by domain: core (search, file, time), equity, macro, commodity.
- **Commenter sidecar** — a paired reviewer reads each agent's workspace and injects advisory notes back into the next turn.

## Why AlphaSeeker

| Typical finance research | AlphaSeeker |
|---|---|
| Relies on a single model's training data | Fetches live market data, SEC filings, EIA, FRED |
| You can't see how the answer was built | Every step is written to disk — inspect it all |
| One flat response | Multi-agent parallel research synthesized into one memo |
| Model can hallucinate numbers | Deterministic tools pull facts from external APIs |
| Fragile on slow I/O | Subprocess isolation — one slow tool doesn't block the pipeline |

## What It Can Do

| Domain | Skill tools | Typical output |
|---|---|---|
| Equity | Market data, company profile, financials, SEC filings, insider activity, peer analysis, earnings calls | Equity research memo with valuation, risk factors, peer context |
| Macro | FRED indicators, World Bank cross-country data | Macro brief covering growth, inflation, rates |
| Commodity | EIA inventory, CFTC COT positioning, futures curve | Commodity report with supply-demand analysis |
| Core | Web search, file read/write, context condensation, datetime | Used by all domains |

The orchestrator decides which combination of skills to invoke based on the prompt — no manual routing required.

## How a Query Flows

```
User Prompt
    │
    ▼
Root Orchestrator Agent (subprocess)
    │  LLM decides: delegate? which skills? when done?
    ├─► spawns child agents (research, writer, synthesizer...)
    │       │
    │       ▼
    │   Skill Tools (deterministic — fetch real data)
    │       │  market data, SEC filings, FRED, EIA...
    │       ▼
    │   publish/  ← agent writes results here
    │
    ▼
Root reads child publish/ files → synthesizes final memo
    │
    ▼
Final Report + full run trace on disk
```

Every agent turn, every tool call, every fetched file is written to `data/harness_runs/<run_id>/`. You can replay the entire research process.

## Quickstart

### 1. Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Fill only the keys required by the model providers and data sources you actually use.

### 4. Run AlphaSeeker

```bash
uv run python main.py
```

You will be prompted for a research question in the terminal.

## Configuration

### Model providers

Model assignments live in `config/models.yaml` and can be overridden with environment variables.

At startup, AlphaSeeker checks whether the API keys required by your active model choices are present.

| Provider model prefix | Required env var |
|---|---|
| `sf/` | `SILICONFLOW_API_KEY` |
| `kimi-` | `KIMI_API_KEY` |
| `minimax/` or `MiniMax-*` | `MINIMAX_API_KEY` |
| `gpt-` or `o*` | `OPENAI_API_KEY` |
| `gemini-` | `GOOGLE_API_KEY` |
| `claude-` | `ANTHROPIC_API_KEY` |

If you use MiniMax, the backend term `base URL` means the root HTTP endpoint your client sends API requests to. AlphaSeeker defaults to `https://api.minimaxi.com/v1`, and you can override it with `MINIMAX_BASE_URL` if you need a different endpoint.

### Data source keys

These are only required when the skills they power are invoked:

- `FRED_API_KEY` for macro indicator fetches
- `EIA_API_KEY` for commodity inventory fetches
- `FMP_API_KEY` for insider-trading data

### Model override order

Model selection follows this priority:

1. `ALPHASEEKER_MODEL_<AGENT>_<ROLE>` environment variable override
2. `config/models.yaml`
3. Fallback defaults in `src/shared/model_config.py`

Example:

```bash
export ALPHASEEKER_MODEL_HARNESS_AGENT="kimi-k2.5"
```

The harness runtime currently uses two model roles: `agent` (the orchestrator and child agents) and `condense` (context compression).

## Example Prompts

- `Analyze AAPL from valuation and risk perspective`
- `US macro outlook for the next 12 months`
- `Crude oil supply-demand and futures curve outlook`
- `How do higher rates affect JPM and bank margins?`
- `How would a weaker dollar affect gold miners and the gold price?`

## Example Outputs

These were all generated by the harness runtime:

- [AAPL one-page showcase](docs/examples/aapl_equity_one_pager.md) — concise equity brief with charts
- [AAPL curated case study](docs/examples/aapl_equity_case_study.md) — detailed walkthrough
- [AAPL full generated report](docs/examples/AAPL_analysis_report.md) — complete output with full trace

Every run also writes its complete workspace to `data/harness_runs/<run_id>/` — you can replay exactly what the model saw and did.

## Project Structure

```text
AlphaSeeker/
├── main.py                      # CLI entry point
├── config/
│   └── models.yaml              # Model assignments by role
├── src/
│   ├── harness/                 # Runtime: orchestrator, workers, skills
│   │   ├── runtime.py           # Async supervisor kernel
│   │   ├── agent_worker.py      # Long-lived worker process
│   │   ├── executor.py          # File-first tools (spawn, bash, read/write)
│   │   ├── transport.py         # MiniMax / OpenAI API adapters
│   │   ├── commenter.py         # Paired reviewer sidecar
│   │   └── skills/              # Deterministic skill adapters
│   │       ├── core.py          # search_in_files, read_file, condense...
│   │       ├── equity.py        # fetch_market_data, fetch_financials...
│   │       ├── macro.py         # fetch_macro_indicators...
│   │       └── commodity.py     # fetch_eia_inventory, fetch_cot_report...
│   ├── tools/                   # Shared data-fetching library
│   │   ├── equity/             # yfinance, SEC, peers, visualization
│   │   ├── macro/              # FRED, World Bank
│   │   └── commodity/          # EIA, CFTC COT, futures curve
│   └── shared/                  # LLM manager, model config, web search
├── data/harness_runs/          # Every run's full workspace (git-ignored)
├── reports/                     # Final generated memos (git-ignored)
├── charts/                      # Generated chart images (git-ignored)
└── tests/
    ├── unit/                   # Deterministic logic
    ├── component/              # Multi-function flows
    └── live/                  # Real API runs
```

## Testing

AlphaSeeker uses a layered pytest setup:

- `unit` tests for deterministic logic
- `component` tests for multi-function flows with mocked dependencies
- `live` tests for full runs against real providers

Run the local quality gate:

```bash
uv run python -m compileall -q src main.py
uv run pytest -m "not live"
```

Run the live suite:

```bash
uv run pytest -m "live"
```

GitHub Actions runs the offline suite on pushes and pull requests, and supports live test runs through manual dispatch.

If you are not used to the word "suite": it just means a grouped set of tests.

## Current Limits

- Terminal-only interface — no web UI.
- Live data sources (market data, SEC, FRED, EIA) require valid API keys.
- Generated analysis should be reviewed by a human before use in investment decisions.

## Additional Reading

- [Harness runtime deep dive](src/harness/README.md) — architecture, workspace protocol, public interface
- [Harness task model](src/harness/TASK.md) — how tasks, artifacts, and skill state work
- [Roadmap and next milestones](TODO.md)
- [Contribution guide](CONTRIBUTING.md)
- [Security policy](SECURITY.md)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
