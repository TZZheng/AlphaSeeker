# Automatic Orchestrator Questions

## Principle

The TSLA test showed that improving the system by adding more role-prompt checklist text is the wrong default. In a stable LingTai-native team, the better steering mechanism is to **ask the orchestrator a small number of general questions** and let the team decide how to respond.

The automatic questioner is not a hidden analyst, not a valuation engine, and not a prompt-length multiplier. It should not assume the system already has specific support for consensus estimates, regulatory datasets, competitor models, or legal/news dockets.

At the same time, the questioner must make the target state clear: the default research direction is a **full investment conclusion**, not merely a source-pack-limited preliminary update. Honest caveats are required, but caveats are not a stopping condition by themselves.

## v1 questions

The current automatic question set has a few short questions/actions. They are natural-language prompts, not a sector checklist. The harness asks the orchestrator only; the orchestrator may then involve source, writer, and reviewer.

1. **What was the original objective? Quote it verbatim before judging the memo.**
2. **As a cold reader, would you sign the current memo as achieving that original objective, not merely a source-pack-limited preliminary update or a narrower objective the team drifted into?**
3. **Is this memo institutional-grade for a real capital-allocation decision? Answer directly. If not, what prevents it from being an institutional-grade investment memo? Improve it with your teams until either the feasible material gaps are addressed, or the remaining gaps are infeasible, outside scope, low-materiality, too costly for likely value, or unlikely to change the decision. Then publish only at the honest grade earned.**
4. **Is the memo's conclusion strength proportional to its evidence depth and claimed grade?** A high-confidence Buy/Sell or institutional-grade memo needs stronger support than a medium-confidence stance, PM briefing note, or discovery memo.
5. **Could another source/writer/reviewer cycle materially improve a blocker, conclusion strength, recommendation, confidence, risk/reward, time horizon, position sizing, or framing?**
6. **If that action is within the team's current tools and scope and could materially change the decision, start it; otherwise explain why the remaining gap is infeasible, outside scope, low materiality, too costly for likely value, unlikely to change the decision, or why the memo should publish only at a lower honest grade. Do not stop merely because a weaker conclusion is already defensible.**

These questions are deliberately general. They ask the orchestrator for a self-assessment of the current published output and evidence base, then require one minimal next action when action is possible. They do not tell the orchestrator which source to fetch, which valuation method to use, or which section to rewrite. If the argument naturally requires a model, consensus comparison, peer frame, scenario analysis, transcript review, regulatory diligence, customer/competitive check, or another artifact, that need should emerge from the claimed conclusion and the current price's embedded expectations.

## Expected orchestrator response

The orchestrator must answer the institutional-grade question directly, using one of the four labels below as supporting policy:

- `Yes — institutional-grade`, with a short explanation of the decision the memo supports, horizon, risk/reward, thesis-breakers, evidence depth, and why remaining limitations do not justify another cycle.
- `No — IC-ready but not institutional-grade`, with the missing institutional-grade gaps, why the current lower grade is honest, and whether any feasible material gap should be routed now.
- `No — PM briefing/discovery-grade only`, with the gap list and a focused source/writer/reviewer task if a feasible material gap remains.
- `No — do not publish yet`, followed by the next concrete evidence or revision cycle.
- `No, the next necessary evidence is outside current tools/scope`, with a clear blocker report to the human and a lower honest grade if an intermediate memo is still published.

A source-pack-limited memo can be a valid intermediate artifact, but it is not the terminal state when the mission is to reach a full investment conclusion. Do not use `accept with caveats` or `yes with caveats` as a final answer; answer whether the original objective is met, then discuss limitations separately.

## Non-goals

The automatic questioner should not:

- ask a long checklist of sector-specific questions;
- prescribe yfinance, SEC, consensus, competitor, legal, or regulatory sources;
- bypass the orchestrator and direct individual team members; the harness asks the orchestrator, and the orchestrator decides which teammate to involve;
- edit `published/` itself;
- create a parallel AlphaSeeker runtime.

## Why this matches LingTai

LingTai avatars already have autonomy, memory, mail, and tools. A stable team should be steered by good questions and observed through its outputs, not micromanaged by ever-expanding initial prompts. The “smallest concrete evidence action” clause prevents the orchestrator from treating honest caveats as an excuse for inaction.
