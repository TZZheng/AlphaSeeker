# AlphaSeeker: Multi-Agent Quantitative Research System

## Core Vision
AlphaSeeker is evolving from a single linear equity research pipeline into a **Multi-Agent Orchestration System**. The system will feature a top-level **Supervisor Agent** that acts as the intelligent routing and synthesis engine, managing a fleet of specialized **Sub-Agents**, where each sub-agent is a domain expert in a specific asset class or research methodology.

This architecture ensures scalability, high cohesion (sub-agents do one thing perfectly), and loose coupling (new asset classes or data sources can be easily added as independent agents).

## Architecture: Supervisor & Sub-Agents Pattern

```mermaid
graph TD
    User([User Prompt]) --> Supervisor[Supervisor Agent]
    Supervisor --"Classifies Intent"--> Router{Intent Router}
    
    Router --Equity--> Agent1[Equity Sub-Agent]
    Router --Macro/Nation--> Agent2[Macro Sub-Agent]
    Router --Commodity--> Agent3[Commodity Sub-Agent]
    
    Agent1 --> Synthesizer[Synthesizer Node]
    Agent2 --> Synthesizer
    Agent3 --> Synthesizer
    
    Synthesizer --> Final([Final Integrated Response])
```

### 1. Supervisor Agent
**Role:** The system's brain and orchestrator. It does not pull data directly. Instead, it:
1. Takes the natural language input.
2. Understands the underlying intent and required data constraints.
3. Delegates tasks to one or more appropriate sub-agents.
4. Synthesizes their individual outputs into a coherent final response.
- **Intent Router:** Determines whether the prompt is about equities, macroeconomic trends, commodities, or requires historical institutional research.
- **Synthesizer:** Merges reports. For example, if a user asks about "The impact of rising interest rates on JPMorgan," the Supervisor can call the **Macro Agent** (for rate trends) and the **Equity Agent** (for JPM financials) and merge their findings.

### 2. Specialized Sub-Agents

**Sub-Agent 1: Equity Research Agent (Current AlphaSeeker)**
- **Focus:** Single public companies (stocks). *(See [docs/equity_agent.md](docs/equity_agent.md) for full architecture and capabilities of this sub-agent)*
- **Capabilities:** Fetches pricing, company profiles, financials, SEC filings; conducts web research; generates CFA-standard initiation reports.
- **Upcoming Upgrades:** Earnings call analysis, insider trading tracking.

**Sub-Agent 2: Macro & Nation Agent (Planned)**
- **Focus:** Global economics, interest rates, inflation, employment, national policies.
- **Data Sources:** FRED API (Federal Reserve Economic Data), World Bank API, OECD.
- **Output:** Macro-outlook briefs and economic indicator summaries.

**Sub-Agent 3: Commodity Agent (Planned)**
- **Focus:** Physical assets like Crude Oil, Gold, Copper, Agriculture.
- **Data Sources:** EIA (Energy Information Administration) inventory reports, CFTC Commitments of Traders (COT), futures curve data (contango/backwardation).
- **Output:** Supply/demand imbalances and price trend analysis.
