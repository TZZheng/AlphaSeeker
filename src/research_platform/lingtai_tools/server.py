"""MCP server exposing AlphaSeeker source-data tools to LingTai avatars."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from . import source_data

log = logging.getLogger("alphaseeker_lingtai_tools")

SERVER_INSTRUCTIONS = (
    "AlphaSeeker source-data MCP: minimal yfinance and SEC helpers for "
    "LingTai-native ticker teams. These tools fetch or save raw material; they "
    "do not replace LingTai's avatar runtime, mail, memory, or orchestration."
)

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "as_yfinance_snapshot": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. TSLA."},
            "include_profile": {"type": "boolean", "default": True},
        },
        "required": ["ticker"],
    },
    "as_yfinance_history": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "period": {"type": "string", "default": "1y"},
            "interval": {"type": "string", "default": "1d"},
            "output_dir": {"type": "string", "description": "Optional directory to save CSV into."},
        },
        "required": ["ticker"],
    },
    "as_sec_find_company": {
        "type": "object",
        "properties": {"ticker": {"type": "string"}},
        "required": ["ticker"],
    },
    "as_sec_recent_filings": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "form_types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "default": 10},
        },
        "required": ["ticker"],
    },
    "as_sec_companyfacts_snapshot": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "output_dir": {"type": "string", "description": "Optional directory to save full companyfacts JSON and snapshot."},
        },
        "required": ["ticker"],
    },
    "as_sec_fetch_filing": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "SEC filing/document URL."},
            "output_dir": {"type": "string", "description": "Optional directory to save raw filing document."},
            "max_bytes": {"type": "integer", "default": 20000000},
        },
        "required": ["url"],
    },
    "as_sec_save_source_pack": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string"},
            "output_dir": {"type": "string", "description": "Directory to save SEC source files into, usually team/raw/sec."},
            "form_types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "default": 6},
            "include_companyfacts": {"type": "boolean", "default": True},
        },
        "required": ["ticker", "output_dir"],
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "as_yfinance_snapshot": "Fetch a compact yfinance quote/valuation/profile snapshot for a ticker.",
    "as_yfinance_history": "Fetch yfinance OHLCV history and optionally save it as CSV.",
    "as_sec_find_company": "Resolve a ticker to SEC CIK and company title.",
    "as_sec_recent_filings": "List recent SEC filings for a ticker from SEC submissions JSON.",
    "as_sec_companyfacts_snapshot": "Fetch SEC companyfacts and optionally save full JSON plus a compact metrics snapshot.",
    "as_sec_fetch_filing": "Fetch a SEC filing/document URL and optionally save the raw document.",
    "as_sec_save_source_pack": "Save a compact SEC source pack for a ticker into a raw/sec directory.",
}


def _ok(result: Any) -> list[types.TextContent]:
    return [types.TextContent(type="text", text=json.dumps(result, ensure_ascii=False, default=str))]


def _error(exc: Exception) -> list[types.TextContent]:
    return _ok({"status": "error", "error_type": type(exc).__name__, "error": str(exc)})


def build_server() -> Server:
    server = Server("alphaseeker-source-data", instructions=SERVER_INSTRUCTIONS)

    @server.list_tools()
    async def _list_tools() -> list[types.Tool]:
        return [
            types.Tool(name=name, description=TOOL_DESCRIPTIONS[name], inputSchema=TOOL_SCHEMAS[name])
            for name in TOOL_SCHEMAS
        ]

    @server.call_tool()
    async def _call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
        args = arguments or {}
        try:
            if name == "as_yfinance_snapshot":
                result = await asyncio.to_thread(
                    source_data.yfinance_snapshot,
                    args["ticker"],
                    include_profile=bool(args.get("include_profile", True)),
                )
            elif name == "as_yfinance_history":
                result = await asyncio.to_thread(
                    source_data.yfinance_history,
                    args["ticker"],
                    period=str(args.get("period", "1y")),
                    interval=str(args.get("interval", "1d")),
                    output_dir=args.get("output_dir"),
                )
            elif name == "as_sec_find_company":
                result = await asyncio.to_thread(source_data.sec_find_company, args["ticker"])
            elif name == "as_sec_recent_filings":
                result = await asyncio.to_thread(
                    source_data.sec_recent_filings,
                    args["ticker"],
                    form_types=args.get("form_types"),
                    limit=int(args.get("limit", 10)),
                )
            elif name == "as_sec_companyfacts_snapshot":
                result = await asyncio.to_thread(
                    source_data.sec_companyfacts_snapshot,
                    args["ticker"],
                    output_dir=args.get("output_dir"),
                )
            elif name == "as_sec_fetch_filing":
                result = await asyncio.to_thread(
                    source_data.sec_fetch_filing,
                    args["url"],
                    output_dir=args.get("output_dir"),
                    max_bytes=int(args.get("max_bytes", 20_000_000)),
                )
            elif name == "as_sec_save_source_pack":
                result = await asyncio.to_thread(
                    source_data.sec_save_source_pack,
                    args["ticker"],
                    output_dir=args["output_dir"],
                    form_types=args.get("form_types"),
                    limit=int(args.get("limit", 6)),
                    include_companyfacts=bool(args.get("include_companyfacts", True)),
                )
            else:
                raise ValueError(f"unknown tool: {name!r}")
            return _ok(result)
        except Exception as exc:
            log.exception("tool call failed: %s", name)
            return _error(exc)

    return server


async def serve() -> None:
    server = build_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
