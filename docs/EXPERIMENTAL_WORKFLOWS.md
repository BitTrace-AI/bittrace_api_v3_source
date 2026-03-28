# Experimental Workflows

This repo keeps one public package and one CLI only:

- package: `bittrace`
- CLI: `bittrace`

Everything in this document is intentionally outside the supported commercial lane.

## Command Surface

- `bittrace experimental backend-comparison`
- `bittrace experimental frontend-capacity-check`
- `bittrace experimental seed-sweep`
- `bittrace experimental leanlean-max-search`
- `bittrace experimental leanlean-deep-layer-max-search`
- `bittrace experimental leanlean-ceiling-search`
- `bittrace experimental leandeep-max-search`

## Package Surface

- `bittrace.experimental.backend_architecture_comparison`
- `bittrace.experimental.frontend_capacity_check`
- `bittrace.experimental.leanlean_seed_sweep`
- `bittrace.experimental.leanlean_max_search`
- `bittrace.experimental.leanlean_ceiling_search`
- `bittrace.experimental.leandeep_max_search`

## Config Layout

- Stable configs remain under `configs/`
- Experimental configs live under `configs/experimental/`
- The frontend-capacity source-profile template lives under `configs/experimental/source_profiles/`

## Important Template Rule

Several experimental configs consume artifacts from previous runs and therefore include `REPLACE_WITH_RUN_ID` placeholders under `runs/...`.

By default, experimental runs still materialize under the same canonical run layout:

- `runs/<config-stem>/<run-id>/`

Before running these workflows:

1. Produce the prerequisite stable or experimental run.
2. Replace the placeholder run id in the config with the real run directory.
3. Use `--prepare-only` first to confirm the graph of inputs is correct.

## Stability Rule

These commands exist so in-house research stays in the canonical repo/package/CLI. They do not have stability guarantees for config schema, artifact schema, report wording, or workflow semantics.
