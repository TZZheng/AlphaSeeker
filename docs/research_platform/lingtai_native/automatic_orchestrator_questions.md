# Automatic Orchestrator Questions

## Principle

The TSLA test showed that adding more checklist text is the wrong default. A stable LingTai-native team should be steered by a few direct questions to the orchestrator, then trusted to involve source, writer, and reviewer.

The automatic questioner is not a hidden analyst, valuation engine, or prompt-length multiplier. It should not assume every ticker needs the same DCF, consensus table, peer table, scenario model, legal docket, or regulatory dataset.

## v1 questions

The harness asks the orchestrator only:

1. **What was the original objective? Quote it before judging the memo.**
2. **Does the current memo achieve that objective, or did the team drift into a narrower/preliminary answer?**
3. **Is this memo institutional-grade for a real capital-allocation decision? Answer directly.**

If yes, publish.
If not, identify what prevents it from being an institutional-grade investment memo, and improve it with the team.
If the remaining gaps are infeasible, outside scope, or low materiality, publish at the honest grade earned, with explanation.

## Expected orchestrator response

The orchestrator should answer in plain language, not by satisfying a separate checklist. A good response says whether the memo is institutional-grade, what prevents that grade if it is not, what the team will improve now, or why the remaining gaps do not justify another cycle.

A source-pack-limited memo can be a valid intermediate artifact, but it is not the terminal state when the mission is to reach a full investment conclusion. Do not use `accept with caveats` or `yes with caveats` as a final answer; answer whether the original objective is met, then discuss limitations separately.

## Non-goals

The automatic questioner should not:

- ask a long checklist of sector-specific questions;
- prescribe yfinance, SEC, consensus, competitor, legal, or regulatory sources;
- bypass the orchestrator and direct individual team members;
- edit `published/` itself;
- create a parallel AlphaSeeker runtime.

## Why this matches LingTai

LingTai avatars already have autonomy, memory, mail, and tools. A stable team should be steered by good questions and observed through its outputs, not micromanaged by ever-expanding initial prompts.
