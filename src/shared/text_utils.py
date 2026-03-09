"""
Shared text utilities — used by all agent node pipelines.

Functions:
  - condense_context: LLM-based intelligent text condensation
  - read_file_safe: Safely reads a file with optional LLM condensation
"""

import os
import time
from typing import Optional

from langchain_core.messages import HumanMessage

from src.shared.llm_manager import get_llm
from src.shared.model_config import get_model


def condense_context(
    text: str,
    max_chars: int,
    agent: str = "equity",
    purpose: str = "research analysis",
    focus_areas: str = "",
) -> str:
    """
    Intelligently handles long text: returns as-is if short enough,
    or calls the LLM to extract core information if too long.

    Replaces hard truncation (text[:N]) which loses information.

    Args:
        text: The input text to potentially condense.
        max_chars: Character budget. If text is shorter, return unchanged.
        agent: Agent name for model config lookup (e.g. "equity", "macro").
        purpose: Why the text is being condensed (helps LLM focus).
        focus_areas: Optional specific topics to prioritize.

    Returns:
        Original text (if short enough) or LLM-condensed version.
    """
    if not text or len(text) <= max_chars:
        return text

    model_name = get_model(agent, "condense")
    target_chars = int(max_chars * 0.9)
    focus_instruction = ""
    if focus_areas:
        focus_instruction = f"\nPay special attention to: {focus_areas}"

    condense_prompt = f"""You are condensing a long document for {purpose}.
The original is {len(text):,} characters but the budget is ~{target_chars:,} characters.

RULES:
- Preserve ALL specific numbers, financial figures, dates, percentages, and dollar amounts
- Preserve ALL named entities (people, companies, products, locations)
- Preserve ALL facts about ownership stakes, contracts, partnerships, lawsuits
- Remove boilerplate, repeated information, and filler text
- Keep the most important and unique information
- Output in the same format (markdown/bullet points) as the input{focus_instruction}

CONDENSE THIS:
{text}
"""

    current_prompt = condense_prompt

    for attempt in range(2):
        try:
            response = get_llm(model_name).invoke([HumanMessage(content=current_prompt)])
            condensed = response.content

            if len(condensed) <= max_chars:
                print(f"  Condensed {len(text):,} → {len(condensed):,} chars ({purpose[:100]}...)")
                return condensed

            print(f"  Warning: Condensation (Attempt {attempt+1}) output {len(condensed):,} chars > limit {max_chars:,}")

            timestamp = int(time.time())
            debug_dir = "data/debug"
            os.makedirs(debug_dir, exist_ok=True)
            debug_file = f"{debug_dir}/condensation_fail_{timestamp}_{attempt}.txt"
            with open(debug_file, "w", encoding="utf-8") as f:
                f.write(f"--- ORIGINAL TEXT ({len(text)} chars) ---\n")
                f.write(text)
                f.write(f"\n\n--- FAILED CONDENSATION ({len(condensed)} chars) ---\n")
                f.write(condensed)
            print(f"  Saved debug context to {debug_file}")

            current_prompt = f"""
Your previous summary was {len(condensed):,} characters, which exceeds the limit of {max_chars:,} characters.
Please shorten it significantly while keeping the key facts.

Previous Output:
{condensed}
"""
        except Exception as e:
            print(f"  Condensation attempt {attempt+1} failed ({e})")
            break

    print(f"  Condensation failed after retries, falling back to strict truncation")
    return text[:max_chars]


def read_file_safe(
    path: Optional[str],
    max_chars: int = 5000,
    agent: str = "equity",
    condense_purpose: str = "research section generation",
) -> str:
    """
    Safely reads a file. If content exceeds max_chars, uses LLM condensation.

    Args:
        path: File path to read (returns "N/A" if None or missing).
        max_chars: Max characters before condensation kicks in.
        agent: Agent name for model config lookup.
        condense_purpose: Purpose string passed to condense_context.

    Returns:
        File content, condensed content, or "N/A".
    """
    if not path or not os.path.exists(path):
        return "N/A"
    try:
        with open(path, "r") as f:
            content = f.read()
        if len(content) > max_chars:
            return condense_context(
                content, max_chars=max_chars,
                agent=agent,
                purpose=condense_purpose,
                focus_areas="financial figures, ratios, company metrics, key data points",
            )
        return content
    except Exception:
        return "N/A"
