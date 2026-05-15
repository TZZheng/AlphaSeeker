# Automatic Orchestrator Questions

## Principle

The TSLA test showed that improving the system by adding more role-prompt checklist text is the wrong default. In a stable LingTai-native team, the better steering mechanism is to **ask the orchestrator a small number of general questions** and let the team decide how to respond.

The automatic questioner is not a hidden analyst, not a valuation engine, and not a prompt-length multiplier. It should not assume the system already has specific support for consensus estimates, regulatory datasets, competitor models, or legal/news dockets.

At the same time, the questioner must make the target state clear: the default research direction is a **full investment conclusion**, not merely a source-pack-limited preliminary update. Honest caveats are required, but caveats are not a stopping condition by themselves.

## v1 questions

The v1 automatic question set has three short questions/actions:

1. **Is the current memo sufficient to support a full investment conclusion, not merely a source-pack-limited preliminary update?**
2. **If not, what is the smallest concrete evidence-gathering action needed to move it toward a full investment conclusion?**
3. **If that action is within the team's current tools and scope, start it; otherwise explain the blocker.**

These questions are deliberately general. They ask the orchestrator for a self-assessment of the current published output and evidence base, then require one minimal next action when action is possible. They do not tell the orchestrator which source to fetch, which valuation method to use, or which section to rewrite.

## Expected orchestrator response

The orchestrator can answer in several valid ways:

- “Yes, the memo supports a full investment conclusion,” with a short explanation of why the evidence is sufficient.
- “No, the memo does not support a full investment conclusion, but the team can improve it using existing raw material,” followed by a focused writer/reviewer task.
- “No, the memo does not support a full investment conclusion, and the source maintainer should gather the smallest missing external evidence category,” followed by a focused source task.
- “No, the next necessary evidence is outside current tools/scope,” with a clear blocker report to the human.

A source-pack-limited memo can be a valid intermediate artifact, but it is not the terminal state when the mission is to reach a full investment conclusion.

## Non-goals

The automatic questioner should not:

- ask a long checklist of sector-specific questions;
- prescribe yfinance, SEC, consensus, competitor, legal, or regulatory sources;
- bypass the orchestrator and direct individual team members;
- edit `published/` itself;
- create a parallel AlphaSeeker runtime.

## Why this matches LingTai

LingTai avatars already have autonomy, memory, mail, and tools. A stable team should be steered by good questions and observed through its outputs, not micromanaged by ever-expanding initial prompts. The “smallest concrete evidence action” clause prevents the orchestrator from treating honest caveats as an excuse for inaction.
