# Task Assignment

## User Request

{{user_prompt}}

## Operating Goal

Own the request end to end and deliver the best supported final answer you can to `publish/final.md`.
Decide for yourself whether the task should stay local, be split into workstreams, be iterated, or be checked again before finalizing.
Use subagents only when they improve quality, focus, or speed enough to justify the coordination cost.

## Available Subagent Presets

{{child_presets}}

## Success Criteria

- `publish/summary.md` explains what was done in a few lines.
- `publish/artifact_index.md` lists important published files and one-line descriptions.
- `publish/final.md` contains the best supported final answer for the user, not just a completed file.

## Completion Judgment

- Judge remaining work into three categories:
  * **blocking**: the deliverable is unusable or wrong without this work.
  * **material but caveatable**: deliverable is usable as-is but would be stronger with this work; document limitations and proceed.
  * **optional**: nice-to-have refinements, polish, or cross-checks that do not change the core answer.
- When time is short or soft-stop is active, prioritize blocking work, document caveats for material items, and skip optional items.
- After each major step, re-check: would the next unit of work materially change the answer? If not, the task is complete.
