# Automatic Orchestrator Question

## Principle

The TSLA/SOFI tests showed that adding more checklist text is the wrong default. A stable LingTai-native team should be steered by a direct publication question to the orchestrator, then trusted to involve source, writer, and reviewer.

The runtime harness is not a hidden analyst, valuation engine, or prompt-length multiplier. It should not assume every ticker needs the same DCF, consensus table, peer table, scenario model, legal docket, or regulatory dataset.

## Runtime question

For each memo or update, the harness/orchestrator loop centers on one question:

> Is this institutional-grade for a real capital-allocation decision? Answer directly: yes or no.

If yes, publish as accepted research at `vault/companies/<TICKER>/team/published/latest.md`.

If no, do not publish. State the improvement needed, assign it to the right teammate, and keep the run open unless the human redirects or stops it.

For a full-investment-conclusion request, do **not** create an autonomous lower-grade exit such as “publish as idea memo,” “publish at honest grade,” or “accepted with caveats.” A lower-grade note may be delivered only if the human explicitly asks for that downgraded deliverable. A blocker note is an open question for the human, not the finished deliverable.

## Non-goals

The automatic harness should not:

- ask a long checklist of sector-specific questions;
- prescribe yfinance, SEC, consensus, competitor, legal, or regulatory sources;
- rebuild or overwrite prompts for existing long-lived agents on every request;
- bypass the orchestrator and direct individual team members;
- edit `published/` itself;
- create a parallel AlphaSeeker runtime.

## Why this matches LingTai

LingTai avatars already have autonomy, memory, mail, tools, and long-lived self-organization. A stable team should be steered by clear role comments and good runtime requests, not micromanaged by ever-expanding initial prompts.
