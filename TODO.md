# Actionable Items for Multi-Agent AlphaSeeker

This document tracks the steps required to transition AlphaSeeker into a Supervisor-led multi-agent system. See `README.md` for architecture overview.

## Phase 1: Supervisor Architecture Foundation
- [x] **Define Sub-Agent Interface:** `SubAgentRequest` and `SubAgentResponse` in `src/shared/schemas.py`.
- [x] **Refactor Existing Pipeline:** Equity graph split into thin `graph.py` (wiring) + `nodes.py` (all node logic).
- [x] **Build Supervisor Graph:** `src/supervisor/graph.py` with intent classification → parallel fan-out → synthesis.
- [x] **Build Synthesizer Node:** `src/supervisor/synthesizer.py` with `run_synthesis()` dispatching single/multi-agent modes.
- [x] **Build Intent Router:** `src/supervisor/router.py` with `classify_user_prompt()` → `ClassificationResult` schema.
- [x] **Update Entry Point:** `main.py` now routes through supervisor instead of directly invoking equity agent.
- [x] **Centralize Web Search:** `src/shared/web_search.py` — all agents import directly, no per-agent wrappers.
- [x] **Model Config System:** `config/models.yaml` + `src/shared/model_config.py` — 3-tier resolution (env → YAML → defaults).
- [x] **Shared Text Utils:** `src/shared/text_utils.py` — `condense_context()` and `read_file_safe()` used by all agents.
- [x] **Partial Results Design:** `_extract_report()` in supervisor/graph.py — 4-level fallback for mid-pipeline failures.
- [x] **Agent Results Merge:** `Annotated[Dict, operator.ior]` on `SupervisorState.agent_results` for safe parallel merge.

## Phase 2: Enhancing the Equity Sub-Agent
- [ ] **Earnings Call Tool:** Integrate a tool to fetch and parse recent earnings call transcripts.
- [ ] **Insider & Institutional Tracking Tool:** Add API calls for Form 4 (insider trades) and 13F (institutional holdings).
- [ ] **Supply Chain & Customer Discovery:** Analyze 10-K revenue concentration disclosures.

## Phase 3: Developing New Domain Sub-Agents

### Macro & Nation Agent (`src/agents/macro/`)
- [x] **Schemas:** `MacroPlan`, `MacroSection`, `MacroReport`, `MacroState` in `macro/schemas.py`.
- [x] **Graph Wiring:** Pipeline in `macro/graph.py` (planner → fetch → research → synthesize → sections → verify → save).
- [x] **Node Stubs:** All node function stubs with docstrings in `macro/nodes.py`.
- [x] **FRED Tool Skeleton:** `macro/tools/fred.py` — series IDs, `fetch_fred_series()`, `fetch_macro_indicators()`.
- [x] **World Bank Tool Skeleton:** `macro/tools/world_bank.py` — indicator codes, `fetch_world_bank_indicators()`.
- [x] **Implement FRED Tool:** Fill in `fetch_fred_series()` and `fetch_macro_indicators()` with actual API calls.
- [x] **Implement World Bank Tool:** Fill in `fetch_world_bank_indicators()` with actual API calls.
- [x] **Implement Macro Nodes:** Fill in all node functions in `macro/nodes.py`.

### Commodity Agent (`src/agents/commodity/`)
- [x] **Schemas:** `CommodityPlan`, `CommoditySection`, `CommodityReport`, `CommodityState` in `commodity/schemas.py`.
- [x] **Graph Wiring:** Pipeline in `commodity/graph.py` (planner → EIA → COT → futures → research → sections → verify → save).
- [x] **Node Stubs:** All node function stubs with docstrings in `commodity/nodes.py`.
- [x] **EIA Tool Skeleton:** `commodity/tools/eia.py` — series IDs, `fetch_eia_series()`, `fetch_eia_inventory()`.
- [x] **CFTC Tool Skeleton:** `commodity/tools/cftc.py` — market codes, `fetch_cot_report()`.
- [x] **Futures Tool Skeleton:** `commodity/tools/futures.py` — ticker mappings, `fetch_futures_curve()`.
- [x] **Implement EIA Tool:** Fill in `fetch_eia_series()` and `fetch_eia_inventory()` with actual API calls.
- [x] **Implement CFTC Tool:** Fill in `fetch_cot_report()` — download/parse CFTC CSV data.
- [x] **Implement Futures Tool:** Fill in `fetch_futures_curve()` — yfinance continuous contracts.
- [x] **Implement Commodity Nodes:** Fill in all node functions in `commodity/nodes.py`.

## Phase 4: Supervisor Implementation
- [x] **Router skeleton:** `classify_user_prompt()`, `get_agent_nodes()`, `validate_classification()` stubs.
- [x] **Synthesizer skeleton:** `format_single_result()`, `synthesize_multi_agent()`, `run_synthesis()` stubs.
- [x] **Wire synthesizer:** `synthesize_results` node in graph.py calls `synthesizer.run_synthesis()`.
- [ ] **Implement Router:** Fill in `classify_user_prompt()` with LLM structured output call.
- [ ] **Implement Synthesizer:** Fill in `format_single_result()` and `synthesize_multi_agent()`.
- [ ] **Implement Sub-Agent Runners:** Fill in `run_equity_agent()`, `run_macro_agent()`, `run_commodity_agent()`.


## Phase 6: Testing & Documentation
- [ ] **Tests directory:** Create `tests/` with test_supervisor.py, test_equity_agent.py, etc.
- [ ] **Macro docs:** Write `docs/macro_agent.md`.
- [ ] **Commodity docs:** Write `docs/commodity_agent.md`.

## Polish Items
- [ ] **Polish SECTION_PROMPTS for macro/commodity** — refine domain-specific guidance after initial implementation.
- [ ] **Model Config Documentation** — add `config/models.yaml` usage instructions to README.
