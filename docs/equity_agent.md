# AlphaSeeker: Equity Research Sub-Agent

The **Equity Research Sub-Agent** is a specialized module within the broader AlphaSeeker Multi-Agent System. Its sole purpose is to analyze individual public companies and generate CFA-standard initiation reports. It uses LLMs for reasoning and synthesis, while strictly executing Python code for data retrieval — ensuring zero hallucination on financial figures.

## Core Philosophy

1. **Deterministic Execution** — All equity data (prices, financials, ownership) is fetched via code, never hallucinated.
2. **Structured Reasoning** — Queries decompose into structured `AnalysisPlan`s via Pydantic models.
3. **Deep Research** — 50+ web queries, full-page article reading, and SEC filing ingestion feed a multi-stage analysis.

## Sub-Agent Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                   Equity Research Sub-Agent Graph                      │
│                                                                        │
│  planner → fetch_data → chart → profile → financials → peers           │
│       → research_qualitative (50+ queries, full-page read)             │
│       → synthesize_research (MAP-REDUCE LLM extraction)                │
│       → generate_section (×6 loop) → generate_summary                  │
│       → verify → save                                                  │
└──────────────────────────────────────────────────────────────────────────┘
```

### Data Collection Layer (Tools)

| Module | Purpose | Source |
|--------|---------|--------|
| `market_data.py` | OHLCV price history | `yfinance` |
| `visualization.py` | Technical price chart | `matplotlib` |
| `company_profile.py` | Identity, ownership, institutional holders | `yfinance` |
| `financials.py` | Income, balance sheet, cash flow (annual + quarterly + TTM + ratios) | `yfinance` |
| `peers.py` | Data-driven peer discovery and comparison metrics | `yfinance` |
| `web_search.py` | DDG text search, DDG news search, parallel full-page reading | `ddgs`, `trafilatura` |
| `sec_filings.py` | SEC EDGAR filing search and text extraction (10-K, 10-Q, 8-K) | SEC EFTS API |
| `analysis.py` | Analysis utilities | — |

### Agent Layer (Graph Nodes)

| Node | Description |
|------|-------------|
| `planner` | Decomposes user query → `AnalysisPlan` (ticker, subtasks) |
| `fetch_data` | Downloads 1Y OHLCV data to CSV |
| `generate_chart` | Plots price + volume chart (PNG) |
| `fetch_company_profile` | Saves company description, ownership, top holders to Markdown |
| `fetch_financials` | Saves annual/quarterly financials + TTM + key ratios to Markdown |
| `analyze_peers` | Discovers peers by sector/industry/market cap, saves comparison table |
| `research_qualitative` | 50+ queries across 8 categories → full-page reads → 5 SEC filings |
| `synthesize_research` | MAP: chunk raw text → extract facts; REDUCE: per-section research briefs |
| `generate_section` | Loops 6× to generate each CFA-standard section with structured output |
| `generate_summary` | Synthesizes all sections into investment summary + target price |
| `verify_content` | Validates report completeness |
| `save_report` | Writes final Markdown report to `reports/` |

### LLM Manager (`llm_manager.py`)

A centralized model registry with rate-limit resilience:

- **`get_llm(model_name)`** — Returns a configured LangChain `ChatModel` instance, lazy-initialized and cached.
- **`RateLimitWrapper`** — Wraps Gemini models with automatic 429 retry + model fallback. Uses a Factory Pattern so `with_structured_output` and `bind_tools` bindings are correctly reconstructed when switching models.
- **Fallback chain** — `gemini-3-flash-preview` → `gemini-2.5-flash` → `gemini-2.5-pro` → `gemini-2.0-flash` → `gemini-exp-1206`.
- **Provider routing** — Prefix determines provider: `gemini-*` → Google, `kimi-*` → Moonshot AI, `sf/*` → SiliconFlow.

### Intelligent Context Condensation

Replaces hard truncation (`text[:N]`) with LLM-driven condensation (`condense_context`). When text exceeds a character budget, the LLM extracts core facts, numbers, and named entities — preventing information loss at the tail end of long documents. Used in:

- Follow-up query generation (company profile + initial snippets)
- Data file reading for section generation
- Investment summary generation

### Schema Layer

All internal communication uses **Pydantic models** (`src/schemas.py`):

- `AnalysisPlan` — Planner output (ticker, subtasks, peers)
- `ResearchSection` — Title + Markdown content with `[n]` citations
- `ResearchReport` — Full CFA-standard report (summary, thesis, 6 sections, references)
- `AgentState` — LangGraph `TypedDict` carrying all data between nodes

## Deep Research Pipeline

The three-layer research architecture solves the "snippet problem" — where search engines return 200-char abstracts that lack critical facts. A **two-phase query system** ensures both breadth and domain-specific depth for any ticker:

### Phase 1: Generic Queries (Industry-Agnostic)

~40 templated queries using only `{company_name}`, `{ticker}`, `{sector}`, `{industry}` — no hardcoded domain jargon. Works identically for a GPU hyperscaler, a pharma company, or a regional bank.

### Phase 2: LLM Follow-Up Queries (Company-Specific)

After Phase 1 collects initial snippets, the LLM reads the company profile + snippets and generates 15-25 **targeted follow-up queries** using specific names, products, and terms it learned. For example:
- For **CRWV**: "NVIDIA equity stake CoreWeave percentage", "CoreWeave Denton TX data center delays"
- For **NVO**: "Ozempic patent cliff timeline Novo Nordisk", "Wegovy insurance coverage expansion 2025"
- For **JPM**: "JPMorgan Chase First Republic acquisition integration", "Jamie Dimon succession planning"

```
Phase 1: GENERIC SEARCH         Phase 2: LLM FOLLOW-UP         Layer 3: SYNTHESIZE
┌──────────────────────┐        ┌──────────────────────┐        ┌──────────────────┐
│ ~40 template queries │──────▶│ LLM reads snippets + │──────▶│ MAP: extract     │
│ ~15 news queries     │       │ profile → generates  │        │   facts per chunk │
│ (works for ANY co.)  │       │ 15-25 targeted       │        │ REDUCE: brief    │
│                      │       │ follow-up queries    │        │   per section    │
└──────────────────────┘        └──────────────────────┘        └──────────────────┘
  Breadth: 8 categories          Depth: domain-specific          6 focused briefs
  ~120 URLs                      ~75 URLs                        ~33 KB synthesized
```

### Query Categories (8)

1. **Business Strategy** — business model, products, technology, competitive advantage
2. **Financial Performance** — earnings, backlog, margins, debt, capex, guidance
3. **Ownership & Governance** — shareholders, strategic investors, insiders, board
4. **Competitive Landscape** — vs rivals, pricing, benchmarks
5. **Risks & Headwinds** — lawsuits, delays, downgrades, customer concentration
6. **Catalysts & Events** — product launches, partnerships, capacity expansion
7. **Analyst Sentiment** — price targets, bull/bear cases, short interest
8. **Industry & Macro** — TAM, supply/demand, power constraints

## Report Output

CFA Institute Equity Research standard sections:

1. **Investment Summary** (generated last, presented first)
   - Rating + Target Price
   - Mispricing Thesis
   - Key Catalysts
2. **Business Description** — Economics, revenue models, product mix
3. **Industry Analysis** — Porter's 5 Forces, moat, competitive positioning
4. **Financial Analysis** — Quality of earnings, TTM metrics, debt structure
5. **Valuation Analysis** — Relative (vs peers) + DCF scenario analysis
6. **Investment Risks** — Operational, financial, regulatory
7. **ESG Analysis** — Environmental, social, governance factors
8. **References** — Cited data sources

## Equity Sub-Agent Tech Stack

| Component | Technology |
|-----------|------------|
| Sub-Agent Execution | LangGraph |
| LLM — Report Writing | Moonshot AI (`kimi-k2.5`) |
| LLM — Extraction / Condensation | SiliconFlow (`Qwen3-VL-32B-Instruct`) |
| LLM — Fallback | Google Gemini (5-model fallback chain) |
| LLM Management | Custom `RateLimitWrapper` + model registry |
| Schema Validation | Pydantic |
| Market Data | `yfinance` |
| Web Research | `ddgs`, `trafilatura` |
| SEC Filings | SEC EDGAR EFTS API (no key needed) |
| Visualization | `matplotlib` |
| Runtime | Python 3.11+, managed by `uv` |

## Model Assignments

Each pipeline step is mapped to a specific LLM. Edit `graph.py` to swap:

| Step | Default Model | Rationale |
|------|--------------|-----------|
| Planning | `sf/Qwen/Qwen3-VL-32B-Instruct` | Trivial extraction |
| Condensation | `sf/Qwen/Qwen3-VL-32B-Instruct` | Summarization |
| Follow-up Queries | `sf/Qwen/Qwen3-VL-32B-Instruct` | Search query generation |
| MAP (fact extraction) | `sf/Qwen/Qwen3-VL-32B-Instruct` | Bulk fact extraction |
| REDUCE (brief synthesis) | `sf/Qwen/Qwen3-VL-32B-Instruct` | Organize facts |
| Section Writing | `kimi-k2.5` | Professional report quality |
| Investment Summary | `kimi-k2.5` | Synthesis quality |

