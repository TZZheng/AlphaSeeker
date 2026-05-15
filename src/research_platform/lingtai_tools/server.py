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
    "AlphaSeeker source-data MCP: small, research-intent-oriented yfinance "
    "and SEC helpers for LingTai-native ticker teams. These tools fetch or save "
    "raw material; they do not replace LingTai's avatar runtime, mail, memory, "
    "or orchestration, and they do not make investment conclusions."
)

TOOL_SCHEMAS: dict[str, dict[str, Any]] = {
    "as_market_context": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. TSLA."},
            "output_dir": {
                "type": "string",
                "description": "Optional directory to save market files into, usually team/raw/market.",
            },
            "history_period": {"type": "string", "default": "1y"},
            "history_interval": {"type": "string", "default": "1d"},
            "include_profile": {"type": "boolean", "default": True},
            "include_history": {"type": "boolean", "default": True},
        },
        "required": ["ticker"],
    },
    "as_sec_source_pack": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. TSLA."},
            "output_dir": {
                "type": "string",
                "description": "Directory to save SEC source files into, usually team/raw/sec.",
            },
            "form_types": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": "integer", "default": 6},
            "include_companyfacts": {"type": "boolean", "default": True},
        },
        "required": ["ticker", "output_dir"],
    },
    "as_investment_source_pack": {
        "type": "object",
        "properties": {
            "ticker": {"type": "string", "description": "Ticker symbol, e.g. TSLA."},
            "output_dir": {
                "type": "string",
                "description": "Directory to save the baseline pack into, usually team/raw.",
            },
            "focus": {
                "type": "string",
                "description": "Optional research focus; defaults to full investment conclusion.",
            },
            "include_market": {"type": "boolean", "default": True},
            "include_sec": {"type": "boolean", "default": True},
            "sec_form_types": {"type": "array", "items": {"type": "string"}},
            "sec_limit": {"type": "integer", "default": 6},
        },
        "required": ["ticker", "output_dir"],
    },
}

TOOL_DESCRIPTIONS: dict[str, str] = {
    "as_market_context": (
        "Gather current market/valuation context for a ticker: quote/profile snapshot "
        "plus optional price history saved into raw/market."
    ),
    "as_sec_source_pack": (
        "Save a compact SEC source pack for a ticker into raw/sec: company resolution, "
        "recent filings, fetched filing documents, and optional companyfacts snapshot."
    ),
    "as_investment_source_pack": (
        "Gather the baseline source pack for moving toward a full investment conclusion: "
        "market context plus SEC context saved under raw/."
    ),
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
            if name == "as_market_context":
                result = await asyncio.to_thread(
                    source_data.market_context,
                    args["ticker"],
                    output_dir=args.get("output_dir"),
                    history_period=str(args.get("history_period", "1y")),
                    history_interval=str(args.get("history_interval", "1d")),
                    include_profile=bool(args.get("include_profile", True)),
                    include_history=bool(args.get("include_history", True)),
                )
            elif name == "as_sec_source_pack":
                result = await asyncio.to_thread(
                    source_data.sec_save_source_pack,
                    args["ticker"],
                    output_dir=args["output_dir"],
                    form_types=args.get("form_types"),
                    limit=int(args.get("limit", 6)),
                    include_companyfacts=bool(args.get("include_companyfacts", True)),
                )
            elif name == "as_investment_source_pack":
                result = await asyncio.to_thread(
                    source_data.investment_source_pack,
                    args["ticker"],
                    output_dir=args["output_dir"],
                    focus=args.get("focus"),
                    include_market=bool(args.get("include_market", True)),
                    include_sec=bool(args.get("include_sec", True)),
                    sec_form_types=args.get("sec_form_types"),
                    sec_limit=int(args.get("sec_limit", 6)),
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
