# LingTai-Native Source Data Tools

## Purpose

The TSLA manual run proved that a LingTai ticker team can gather source material with generic tools, but it also exposed two repeated data gaps:

1. market/valuation facts such as current price, market cap, shares, enterprise value, and price history;
2. SEC source retrieval that should be reliable, polite, and easy for the source maintainer to save into `raw/`.

AlphaSeeker should provide these as **LingTai-registerable source tools**, not as a replacement runtime.

```text
LingTai owns: avatar life, mail, memory, orchestration, tool hosting.
AlphaSeeker owns: finance-domain source helpers that LingTai avatars may call.
```

## Agent-facing tool surface

The source-data server is `alphaseeker-source-data`, implemented in:

- `src/research_platform/lingtai_tools/source_data.py`
- `src/research_platform/lingtai_tools/server.py`

The MCP surface is intentionally small. It exposes research actions, not every low-level API primitive:

| Tool | Use |
|---|---|
| `as_market_context` | Gather current market/valuation context: quote/profile snapshot plus optional price history saved into `raw/market/`. |
| `as_sec_source_pack` | Save a compact SEC source pack into `raw/sec/`: company resolution, recent filings, fetched filing documents, and optional companyfacts. |
| `as_investment_source_pack` | Gather the baseline pack for moving toward a **full investment conclusion**: market context plus SEC context saved under `raw/`. |

The lower-level Python helpers still exist for tests and composition (`yfinance_snapshot`, `yfinance_history`, `sec_find_company`, `sec_recent_filings`, `sec_companyfacts_snapshot`, `sec_fetch_filing`, `sec_save_source_pack`). They are not all exposed as separate MCP tools because that makes the agent choose mechanical API steps instead of research actions.

The server is factual. It does not decide whether a memo is good, which filings matter most, or whether a team should publish. Those judgments belong to the LingTai ticker team.

## Registration sketch

A LingTai agent can mount the server as a stdio MCP. The exact registration mechanism may be managed by the orchestrator, but the server command is:

```bash
cd /Users/tianzhezheng/Documents/AlphaSeeker
.venv/bin/python -m src.research_platform.lingtai_tools.server
```

Example `mcp/servers.json` entry for a quick local experiment:

```json
{
  "alphaseeker_source_data": {
    "type": "stdio",
    "command": "/Users/tianzhezheng/Documents/AlphaSeeker/.venv/bin/python",
    "args": ["-m", "src.research_platform.lingtai_tools.server"],
    "env": {
      "PYTHONPATH": "/Users/tianzhezheng/Documents/AlphaSeeker"
    }
  }
}
```

For durable deployment, prefer LingTai's registry route (`mcp_registry.jsonl` + `init.json` activation) rather than committing agent-local `mcp/servers.json` files. Empirically, the quick TSLA experiment mounted successfully through `mcp/servers.json`; the custom `init.json -> mcp` attempt did not surface the tools in that team's refreshed tool list.

## Source maintainer usage pattern

A source maintainer should treat these tools as optional accelerators toward a conclusion:

1. If asked to move toward a full investment conclusion and baseline external evidence is missing, call `as_investment_source_pack` into `raw/`.
2. If only market/valuation context is missing, call `as_market_context` into `raw/market/`.
3. If SEC material is missing or stale, call `as_sec_source_pack` into `raw/sec/`.
4. Write a plain-English note under `raw/notes/` only if helpful; no formal evidence ID system is required.
5. Tell writer/reviewer/orchestrator what was gathered, what it supports, and what remains missing via LingTai mail.

Do not make the tool output itself the published memo. The team still has to reason. Honest caveats are necessary, but they are not a substitute for taking the smallest available evidence-gathering step when the mission is a full investment conclusion.
