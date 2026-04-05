# AlphaSeeker TODO

This file tracks open work against the current repo layout (`src/supervisor/`, `src/agents/`, `src/harness/`, `docs/`, `tests/`). Completed bootstrap work has been removed so this stays focused on unfinished items.

## Documentation
- [ ] Write `docs/macro_agent.md`.
- [ ] Write `docs/commodity_agent.md`.
- [ ] Document the `--runtime harness` flow in `README.md`, including where reports and run traces are saved.

## Equity Follow-Ups
- [ ] Add explicit 13F change tracking for institutional ownership. The repo already has Yahoo holder snapshots in `src/agents/equity/tools/company_profile.py` and Form 4 insider flow in `src/agents/equity/tools/insider_trading.py`, but it still lacks quarter-over-quarter 13F movement analysis.
- [ ] Harden `src/agents/equity/tools/earnings_calls.py` with transcript-first source ranking and dedicated tests. The tool exists today, but it still relies on generic web-search discovery.

## Macro & Commodity Quality
- [ ] Polish `SECTION_PROMPTS` in `src/agents/macro/nodes.py` and `src/agents/commodity/nodes.py`.
- [ ] Downgrade broken EIA fetches to `partial` or `failed` when the saved artifact is only error text.
- [ ] Investigate `fetch_macro_indicators()` live stalls and add tighter timeouts plus clearer logging.

## Harness Follow-Ups
- [ ] Harden verifier structured-output parsing. The judge can omit `decision` or return `improvement_instructions` in the wrong shape.
- [ ] Stop truncating verifier evidence to a fixed tail. Pass all cited IDs or build a citation-targeted evidence subset.
- [ ] Add a writer timeout budget or stronger prompt-size control in `src/harness/writer.py`.
- [ ] Tighten citation quality so material claims prefer filings, company releases, and earnings-call notes over headline-only search evidence.

## Reliability, Measurement, and Ops
- [ ] Expand the shared retry/cache layer in `src/shared/reliability.py` to the remaining uncached external paths (`yfinance`, SEC reads, DDG/trafilatura fetches) so timeout and backoff behavior is consistent across the stack.
- [ ] Add LangSmith tracing for graph and harness runs to capture node-level latency, prompts, responses, and token/cost data.
- [ ] Add LangSmith eval datasets for routing and report quality, then gate prompt and graph changes on evals in addition to `pytest`.
- [ ] Finish security and operations hardening: managed secrets, stage/prod separation, release/rollback workflow, spend guardrails, and retention policies.
