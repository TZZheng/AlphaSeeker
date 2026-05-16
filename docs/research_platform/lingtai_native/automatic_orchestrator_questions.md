# Automatic Orchestrator Questions

## Principle

The TSLA test showed that improving the system by adding more role-prompt checklist text is the wrong default. In a stable LingTai-native team, the better steering mechanism is to **ask the orchestrator a small number of general questions** and let the team decide how to respond.

The automatic questioner is not a hidden analyst, not a valuation engine, and not a prompt-length multiplier. It should not assume the system already has specific support for consensus estimates, regulatory datasets, competitor models, or legal/news dockets.

At the same time, the questioner must make the target state clear: the default research direction is a **full investment conclusion**, not merely a source-pack-limited preliminary update. Honest caveats are required, but caveats are not a stopping condition by themselves.

## v1 questions

The v1 automatic question set has a few short questions/actions. They are natural-language prompts, not a schema:

1. **What was the original objective? Quote it verbatim before judging the memo.**
2. **As a cold reader, would you sign the current memo as achieving that original objective, not merely a source-pack-limited preliminary update or a narrower objective the team drifted into?**
3. **If not, what single issue most prevents publication?**
4. **Is the memo's conclusion strength proportional to its evidence depth?** A high-confidence Buy/Sell needs stronger support than a medium-confidence stance or Neutral/Hold.
5. **Could another source/writer/reviewer cycle materially improve the blocker, conclusion strength, recommendation, confidence, or framing?**
6. **If that action is within the team's current tools and scope, start it; otherwise explain why the remaining gap is irreducible, why more work would not materially help, or why the conclusion strength should be lowered.**

These questions are deliberately general. They ask the orchestrator for a self-assessment of the current published output and evidence base, then require one minimal next action when action is possible. They do not tell the orchestrator which source to fetch, which valuation method to use, or which section to rewrite. If the argument naturally requires a model, consensus comparison, peer frame, scenario analysis, transcript review, regulatory diligence, customer/competitive check, or another artifact, that need should emerge from the claimed conclusion and the current price's embedded expectations.

## Expected orchestrator response

The orchestrator can answer in several valid ways:

- “Yes, I would sign this as meeting the original objective,” with a short explanation of why the conclusion strength is proportional to the evidence depth and why remaining limitations do not materially change the conclusion or confidence.
- “No, I would not sign this yet, but the team can improve it using existing raw material,” followed by a focused writer/reviewer task.
- “No, I would not sign this yet, and the source maintainer should gather the smallest missing external evidence category,” followed by a focused source task.
- “No, the next necessary evidence is outside current tools/scope,” with a clear blocker report to the human.

A source-pack-limited memo can be a valid intermediate artifact, but it is not the terminal state when the mission is to reach a full investment conclusion. Do not use `accept with caveats` or `yes with caveats` as a final answer; answer whether the original objective is met, then discuss limitations separately.

## Non-goals

The automatic questioner should not:

- ask a long checklist of sector-specific questions;
- prescribe yfinance, SEC, consensus, competitor, legal, or regulatory sources;
- bypass the orchestrator and direct individual team members;
- edit `published/` itself;
- create a parallel AlphaSeeker runtime.

## Why this matches LingTai

LingTai avatars already have autonomy, memory, mail, and tools. A stable team should be steered by good questions and observed through its outputs, not micromanaged by ever-expanding initial prompts. The “smallest concrete evidence action” clause prevents the orchestrator from treating honest caveats as an excuse for inaction.
