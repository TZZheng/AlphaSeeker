# Automatic Orchestrator Questions

## Principle

The TSLA test showed that improving the system by adding more role-prompt checklist text is the wrong default. In a stable LingTai-native team, the better steering mechanism is to **ask the orchestrator a small number of general questions** and let the team decide how to respond.

The automatic questioner is not a hidden analyst, not a valuation engine, and not a prompt-length multiplier. It should not assume the system already has specific support for consensus estimates, regulatory datasets, competitor models, or legal/news dockets.

## v0 questions

The v0 automatic question set has only two core questions:

1. **Is the current memo sufficient to support an investment conclusion?**
2. **If not, does the team need to look outside the current materials for new evidence?**

These questions are deliberately general. They ask the orchestrator for a self-assessment of the current published output and evidence base. They do not tell the orchestrator which source to fetch, which valuation method to use, or which section to rewrite.

## Expected orchestrator response

The orchestrator can answer in several valid ways:

- “Yes, the memo supports a conclusion,” with a short explanation of why the evidence is sufficient.
- “No, the memo does not support a conclusion, but the team can improve it using existing raw material.”
- “No, the memo does not support a conclusion, and the source maintainer should seek external material.”
- “No, the memo should explicitly remain a source-pack-limited update rather than an investment memo.”

The orchestrator may then choose to mail the source maintainer, writer, or reviewer; gather more material; revise `published/latest.md`; or report the limitation to the human.

## Non-goals

The automatic questioner should not:

- ask a long checklist of sector-specific questions;
- prescribe yfinance, SEC, consensus, competitor, legal, or regulatory sources;
- bypass the orchestrator and direct individual team members;
- edit `published/` itself;
- create a parallel AlphaSeeker runtime.

## Why this matches LingTai

LingTai avatars already have autonomy, memory, mail, and tools. A stable team should be steered by good questions and observed through its outputs, not micromanaged by ever-expanding initial prompts.
