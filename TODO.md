# Actionable Items for Multi-Agent AlphaSeeker

This document tracks the steps required to transition AlphaSeeker into a Supervisor-led multi-agent system, as outlined in `SYSTEM_VISION.md`.

## Phase 1: Supervisor Architecture Foundation
- [ ] **Define Sub-Agent Interface:** Create a standard Pydantic schema (e.g., `SubAgentRequest`, `SubAgentResponse`) in `src/schemas.py` that all sub-agents will conform to. This ensures the Supervisor can uniformly communicate with any agent.
- [ ] **Refactor Existing Pipeline:** Wrap the current AlphaSeeker equity structure (the existing LangGraph) into a discrete, callable `EquitySubAgent` node that accepts the new interface.
- [ ] **Build Supervisor Graph:** Create `src/agent/supervisor_graph.py` using LangGraph. Implement an Intent Router node that classifies user queries into an `EntityType` (Equity, Macro, Commodity, etc.) and routes to the appropriate sub-agent node.
- [ ] **Build Synthesizer Node:** Implement a node at the end of the Supervisor graph that takes outputs from the sub-agents and uses an LLM to weave them into a final answer.

## Phase 2: Enhancing the Equity Sub-Agent
These steps improve the quality of a single ticker analysis (as discussed previously).
- [ ] **Earnings Call Tool:** Integrate a tool to fetch and parse recent earnings call transcripts. LLMs are excellent at extracting management tone and sentiment shifts from Q&A sessions.
- [ ] **Insider & Institutional Tracking Tool:** Add scraping or API calls for Form 4 (insider trades) and 13F (institutional holdings) to follow "smart money" movements.
- [ ] **Supply Chain & Customer Discovery:** Add logic (perhaps analyzing 10-K revenue concentration disclosures) to identify key suppliers and customers, enriching the business moat analysis.

## Phase 3: Developing New Domain Sub-Agents
- [ ] **Build Macro Agent (`src/agent/macro_graph.py`):** 
  - Sub-task: Identify and integrate a robust data source like the FRED API.
  - Sub-task: Create tools to fetch interest rates, GDP, CPI, and employment data.
  - Sub-task: Define a dynamic report schema (`MacroReportSchema`) for the LLM to fill.
- [ ] **Build Commodity Agent (`src/agent/commodity_graph.py`):**
  - Sub-task: Identify APIs for futures curves and inventory data (e.g., EIA for energy).
  - Sub-task: Create tools for contango/backwardation math.
  - Sub-task: Define a `CommodityReportSchema`.

## Phase 4: Institutional Knowledge Base (Currently Paused)
*Note: This phase is paused until the live-data agents are fully functional.*
- [ ] **Offline Ingestion Pipeline:** Create a script (`ingest_reports.py`) to parse (LlamaParse/PyMuPDF), chunk, metadata-tag, and embed the 4,000+ pages of professional research reports into a local Vector DB (e.g., Chroma or FAISS).
- [ ] **Document QA Sub-Agent:** Build the RAG Sub-Agent in LangGraph that can perform semantic search across these reports to answer queries, acting as another tool the Supervisor can call.
