"""
Shared cross-agent Pydantic models.

Defines the uniform SubAgentRequest / SubAgentResponse contract that the
Supervisor uses to invoke any sub-agent identically, regardless of domain.

Design principle (from README):
  "All sub-agents accept SubAgentRequest and return SubAgentResponse,
   so the Supervisor can call any agent identically."
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# SubAgentRequest — Supervisor → Sub-Agent
# ---------------------------------------------------------------------------

class SubAgentRequest(BaseModel):
    """
    Uniform input contract for every sub-agent.
    The Supervisor populates this from the classified_entities in SupervisorState.

    Each sub-agent reads only the fields relevant to its domain and ignores the rest:
      - Equity agent reads: ticker
      - Macro agent reads:  topic
      - Commodity agent reads: asset
    """
    user_prompt: str = Field(
        ...,
        description="The original natural language question from the user. "
                    "Provides context so the sub-agent can tailor its output."
    )
    ticker: str = Field(
        default="",
        description="Stock ticker symbol for the equity sub-agent. E.g. 'AAPL'."
    )
    topic: str = Field(
        default="",
        description="Macro/economic topic for the macro sub-agent. "
                    "E.g. 'US interest rate outlook', 'China GDP growth'."
    )
    asset: str = Field(
        default="",
        description="Physical commodity name for the commodity sub-agent. "
                    "E.g. 'crude oil', 'gold', 'copper'."
    )
    extra: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional domain-specific overrides or config flags. "
                    "Used sparingly to avoid breaking the uniform interface."
    )


# ---------------------------------------------------------------------------
# SubAgentResponse — Sub-Agent → Supervisor
# ---------------------------------------------------------------------------

class SubAgentResponse(BaseModel):
    """
    Uniform output contract from every sub-agent back to the Supervisor.
    Each sub-agent serializes its full output into report_md before returning.
    """
    agent_type: str = Field(
        ...,
        description="Which agent produced this response. One of: 'equity', 'macro', 'commodity'."
    )
    report_md: str = Field(
        ...,
        description="The complete research output as a Markdown string. "
                    "May include embedded image paths (e.g. ![chart](path)), "
                    "tables, and section headers. This is what the Synthesizer receives."
    )
    success: bool = Field(
        default=True,
        description="False if the sub-agent encountered an unrecoverable error."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Human-readable error description if success=False."
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional diagnostic metadata (e.g. data sources used, "
                    "report file path, execution time). Not used by the Synthesizer."
    )
