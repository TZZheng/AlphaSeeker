# Automatic Orchestrator Questions

## Principle

The TSLA test showed that adding more checklist text is the wrong default. A stable LingTai-native team should be steered by a few direct questions to the orchestrator, then trusted to involve source, writer, and reviewer.

The automatic questioner is not a hidden analyst, valuation engine, or prompt-length multiplier. It should not assume every ticker needs the same DCF, consensus table, peer table, scenario model, legal docket, or regulatory dataset.

## v1 questions

The harness asks the orchestrator only:

1. **What was the original objective? Quote it before judging the memo.**
2. **Does the current memo achieve that objective, or did the team drift into a narrower/preliminary answer?**
3. **Is this memo institutional-grade for a real capital-allocation decision? Answer directly: yes or no.**

If yes, publish as accepted research.
If not, do not publish as accepted research. Identify what prevents it from being institutional-grade and improve it with the team. If more source is needed, source must gather or map more source. If more analysis is needed, writer/reviewer must improve the analysis. Repeat until the gate passes or the human explicitly changes the objective/stops the run.

For a full-investment-conclusion request, do **not** create an autonomous lower-grade exit such as “publish as idea memo,” “publish at honest grade,” or “accepted with caveats.” A lower-grade note may be delivered only if the human explicitly asks for that downgraded deliverable.

## Expected orchestrator response

The orchestrator should answer in plain language, not by satisfying a separate checklist. A good response says whether the memo is institutional-grade, what prevents that grade if it is not, and what the team will improve now. It should not decide on its own that remaining gaps do not justify another cycle for a full-investment-conclusion request.

A source-pack-limited memo can be a valid intermediate artifact, but it is not the terminal state when the mission is to reach a full investment conclusion. Do not use `accept with caveats`, `yes with caveats`, `yes, if...`, or similar as a final answer. Caveats can explain a no; they cannot turn a no into a yes.

## Non-goals

The automatic questioner should not:

- ask a long checklist of sector-specific questions;
- prescribe yfinance, SEC, consensus, competitor, legal, or regulatory sources;
- bypass the orchestrator and direct individual team members;
- edit `published/` itself;
- create a parallel AlphaSeeker runtime.

## Why this matches LingTai

LingTai avatars already have autonomy, memory, mail, and tools. A stable team should be steered by good questions and observed through its outputs, not micromanaged by ever-expanding initial prompts.
