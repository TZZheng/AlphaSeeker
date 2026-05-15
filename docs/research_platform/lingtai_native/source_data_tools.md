# LingTai-Native Source Data Tools

## Purpose

The TSLA manual run proved that a LingTai ticker team can gather source material with generic tools, but it also exposed two repeated data gaps:

1. market/valuation facts such as current price, market cap, shares, and price history;
2. SEC source retrieval that should be reliable, polite, and easy for the source maintainer to save into `raw/`.

AlphaSeeker should provide these as **LingTai-registerable source tools**, not as a replacement runtime.

```text
LingTai owns: avatar life, mail, memory, orchestration, tool hosting.
AlphaSeeker owns: finance-domain source helpers that LingTai avatars may call.
```

## v1 tool surface

The first tool server is `alphaseeker-lingtai-source-data`, implemented in:

- `src/research_platform/lingtai_tools/source_data.py`
- `src/research_platform/lingtai_tools/server.py`

It exposes the following MCP tools:

| Tool | Use |
|---|---|
| `as_yfinance_snapshot` | Compact quote/valuation/profile snapshot from yfinance. |
| `as_yfinance_history` | OHLCV history from yfinance; can save CSV into `raw/market/`. |
| `as_sec_find_company` | Resolve ticker to SEC CIK and title. |
| `as_sec_recent_filings` | List recent filings from SEC submissions JSON. |
| `as_sec_companyfacts_snapshot` | Fetch companyfacts and optionally save full JSON + compact metrics snapshot. |
| `as_sec_fetch_filing` | Fetch one SEC filing/document URL and optionally save raw document. |
| `as_sec_save_source_pack` | Save a compact SEC source pack into `raw/sec/`. |

The server is intentionally factual and low-level. It does not decide whether a memo is good, which filings matter, or whether a team should publish. Those judgments belong to the LingTai ticker team.

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

For durable deployment, prefer LingTai's registry route (`mcp_registry.jsonl` + `init.json` activation) rather than committing agent-local `mcp/servers.json` files.

## Source maintainer usage pattern

A source maintainer should treat these tools as optional accelerators:

1. If `raw/` is empty, call `as_sec_save_source_pack` into `raw/sec/` and `as_yfinance_history` into `raw/market/` if market context is needed.
2. Call `as_yfinance_snapshot` when the team needs current price/market cap/share-related context.
3. Write a plain-English note under `raw/notes/` only if helpful; no formal evidence ID system is required.
4. Tell writer/reviewer what was gathered and what remains missing via LingTai mail.

Do not make the tool output itself the published memo. The team still has to reason.
