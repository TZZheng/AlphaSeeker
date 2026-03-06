---
trigger: always_on
---

# Project: AlphaSeeker (Quantitative Analysis Agent)

## Core Philosophy
1. **Deterministic Outputs**: All analysis must be done via Python code execution, NOT by LLM hallucination.
2. **Structured Data**: All internal communication must use Pydantic models.
3. **Defense in Depth**: Always assume external APIs (Yahoo Finance) might fail; implement retries and error handling.

## Tech Stack
- Python 3.10+
- LangGraph (for orchestration)
- Pydantic (for schema validation)
- yfinance (data source)

## Coding Standards
- Type hint everything.
- Add type check for Python codes.
- Use docstrings for all functions.
- No loose scripts; use modular design.
- Clean up test codes after verification.
- update README.md to record the update of the project
- look for TODO.md to find non-finished actionable items, and tick the item when finish one.
## Other standards
- If I prohibit you from doing something, write a knowledge base about that.