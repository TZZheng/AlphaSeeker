Yes. I would structure the evaluator around **specific quality areas**, not one vague “quality” score.

**Core Answer**
Evaluate every meaningful report version, but judge the final report as the product outcome.

A “version” should mean a candidate report snapshot, not every small file edit. Concretely: whenever the root agent writes or finishes `publish/final.md`, copy it to something like:

`data/eval_runs/<eval_id>/<case_id>/versions/v001.md`

Then the evaluator produces:

`v001.eval.json`
`v002.eval.json`
`pairwise_v001_v002.json`
`trajectory.json`

That gives us both:
- final quality: “Was the final answer good?”
- improvement quality: “Did more time make it better?”

**Define Good By Area**
For broad prompts, we should generate an **eval contract** before judging. By “contract,” I mean a small machine-readable checklist describing what this answer should be good at.

For AlphaSeeker reports, I’d start with these areas:

1. **Factual correctness**
   Material claims are true relative to cited or freshly retrieved evidence.

2. **Freshness**
   Time-sensitive claims use data recent enough for the task. For example, market prices should have an “as of” date; SEC filing data should identify the latest filing period; EIA/FRED data should not silently use stale releases.

3. **Evidence grounding**
   Important claims are backed by evidence. “Important” means claims that affect the investment conclusion, valuation, macro view, risk/reward, or factual company description.

4. **Logical soundness**
   Conclusions follow from premises. For example, “XOM is cheap” should be backed by valuation multiples, cash flow, peers, history, or scenario math, not just asserted.

5. **Completeness**
   The report answers the user’s actual prompt: valuation, risks, bull/bear case, macro backdrop, commodity drivers, etc.

6. **Numerical discipline**
   Ratios, percentages, prices, dates, and units are internally consistent. If the report says “FCF yield is 8%,” the evaluator should check whether the numerator/denominator are visible and plausible.

7. **Decision usefulness**
   The answer gives a clear investment view, key uncertainties, and what would change the conclusion.

I would make factual correctness and freshness high weight, but also add **critical fail flags**. A report with a major false number should not score highly just because it is well written.

**How To Check Factuality And Freshness**
Use a claim-checking pipeline:

1. Extract atomic claims from the report.
   Example: “XOM trades at 11x forward earnings,” “Q1 2026 earnings are on May 1, 2026,” “WTI curve is backwardated.”

2. Classify claims:
   - market data
   - financial statement data
   - filing/company fact
   - macro/commodity data
   - valuation/math claim
   - qualitative judgment

3. For each important claim, find evidence:
   - cited artifact in the run
   - child agent published files
   - deterministic tool output
   - fresh web/API retrieval if needed

4. Judge each claim as:
   - `supported`
   - `partially_supported`
   - `unsupported`
   - `contradicted`
   - `stale`
   - `not_checkable`

5. Roll that up into section scores and final scores.

This avoids requiring a single golden answer. The evaluator does not need to know “the correct memo.” It needs to know whether the memo’s claims are supported.

**How To Check Logical Soundness**
I’d avoid trying to fully formalize logic. Keep it practical:

For each major conclusion, ask the evaluator to extract:

```json
{
  "conclusion": "XOM is a hold at current prices",
  "supporting_claims": [...],
  "counterarguments_considered": [...],
  "missing_links": [...],
  "logic_score": 0.0
}
```

Then judge whether the supporting claims actually imply the conclusion. This is especially important in finance, because many bad reports contain true facts but weak inference.

Example failure:

> XOM has a strong balance sheet, therefore the stock is undervalued.

The balance sheet claim may be true, but it does not by itself prove undervaluation. The evaluator should flag that as a missing valuation bridge.

**How To Define Regression**
I agree with you: “worse” should not be vague.

A newer version is a regression if one of these happens:

1. It introduces a new critical factual error.
2. It removes important correct evidence.
3. It becomes less complete against the eval contract.
4. Pairwise judge prefers the older version by a meaningful margin.
5. The overall area scores drop more than a threshold, say 0.5 on a 10-point scale.

So `regression_count` is not “score went down by 0.1.” It is “materially worse in at least one defined area.”

**Quality Per Minute**
I would not start with `quality_per_minute`. It sounds precise but is probably noisy.

Better early metrics:

- `best_score_seen`
- `final_score`
- `versions_evaluated`
- `time_to_first_acceptable`
- `time_to_best`
- `regressions`
- `critical_errors_remaining`
- `pairwise_win_rate_vs_previous`

Later, `quality_per_minute` can be derived from score trajectory, but I would not make it central.

**Evaluator Scope**
Default behavior:

- Evaluate every root report version.
- Use child outputs and artifacts as evidence.
- Do not separately score every child output unless debugging.

This keeps the evaluator focused on what the user sees, while still using the full trace to verify whether claims are grounded.

**Recommended First Implementation**
Add a quality runner with this loop:

1. Run one eval case with budget checkpoints.
2. Snapshot each root `final.md`.
3. Generate eval contract from the prompt.
4. Score each version by rubric.
5. Pairwise compare adjacent versions.
6. Write a trajectory summary.

That gives you the experiment you want: not “did the system run longer,” but “did answer quality improve in specific, defensible areas.”