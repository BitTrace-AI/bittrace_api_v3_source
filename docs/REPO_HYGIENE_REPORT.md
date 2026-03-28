# Repo Hygiene Report

Date: `2026-03-28`

## Inventory

Classified as debris:

- repo-local virtualenv state under `.venv_source/`
- Python bytecode and cache directories under `src/` and `scripts/`
- generated packaging metadata under `src/bittrace_api_v3_source.egg-info/`
- a stray archive copy at `runs/release_smoke/release_smoke_20260328_093041.zip`

Intentionally kept:

- `runs/` as the canonical working-output root referenced by stable and experimental workflows
- `release_artifacts/launch_candidate_2026-03-28/` as the centralized launch-candidate archive area
- `configs/experimental/` templates that still use `REPLACE_WITH_RUN_ID`
- internal artifact-kind/schema strings that still contain `bittrace_bearings_v3_source` for compatibility with prior artifacts

## Cleanup Applied

- removed local virtualenv, bytecode caches, and stale `.egg-info` packaging debris
- moved the stray release-smoke zip into `release_artifacts/launch_candidate_2026-03-28/archive/`
- expanded `.gitignore` so local build/cache/archive areas stay out of source-control review
- removed internal absolute virtualenv-path references from stable docs
- removed stable-lane `REPLACE_WITH_RUN_ID` placeholders from shipped persistence configs by requiring an explicit runtime `--source-run-root` when needed

## Notes

- Stable lane behavior is unchanged for the documented command path, which already passes `--source-run-root` explicitly.
- Experimental workflows remain available and intentionally documented under `bittrace experimental ...`.
- Historical run directories under `runs/` were left in place because they are ignored, operationally useful, and not part of the tracked release bundle.
