{{user_prompt}}

# AlphaSeeker Research Platform Context

You are writing an investment memo for **{{ticker}}**{{company_name}}.

A curated research-platform source package has already been assembled for this run. Prefer the attached context files and the structured citations below. Keep AlphaSeeker tools available only for targeted verification of a specific issue; do not broadly re-fetch SEC filings, profile, financials, or market data unless the attached context is insufficient or contradictory.

## Run Metadata

- Run ID: `{{run_id}}`
- Source index: `{{source_index_path}}`
- Question list: `{{question_list_path}}`
- Context package: `{{context_package_path}}`

## Curated Sources

{{required_sources_summary}}

## Missing / Best-Effort Sources

{{missing_sources_block}}

## Caveats and Warnings

{{caveats_block}}

## Freshness Notes

{{freshness_block}}

## Citation Instructions

{{citation_usage_instructions}}

## Output Contract

Write the final memo to `publish/final.md`. Be explicit about material source gaps or caveats. Use the curated source index and citation keys when citing evidence. If you call tools for verification, reconcile any new finding against the attached curated context instead of silently replacing it.
