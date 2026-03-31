## Summary

This PR is a stabilization and hardening pass on the current BitTrace API v3
source repository.

The stable public surface is unchanged:

- import namespace: `bittrace`
- CLI: `bittrace`
- stable commands:
  - `bittrace campaign`
  - `bittrace verify`
  - `bittrace deployment-candidate`
  - `bittrace persistence`
- experimental commands remain under `bittrace experimental ...`

## Why This Change Is Needed

- The repo did not have a visible top-level automated test tree.
- The repo did not have GitHub Actions enforcing installability and smoke
  coverage on `push` and `pull_request`.
- Release hygiene artifacts were missing or scattered.
- The public docs did not clearly explain the relationship between the BitTrace
  API v3 repository generation and the Python package version line.

## What Changed

- Added `tests/` with pytest coverage for:
  - stable CLI help smoke
  - `python -m bittrace --help`
  - import smoke for `bittrace`, `bittrace.source`,
    `bittrace.experimental`, and `bittrace.v3`
  - stable config/schema loading and enforcement
- Added `scripts/run_ci_smoke.py` as a lighter automated smoke path.
- Kept `scripts/run_release_smoke.py` as the documented manual release gate.
- Added `.github/workflows/ci.yml` for Python 3.12 editable-install CI.
- Added `CHANGELOG.md`, `KNOWN_ISSUES.md`, a release checklist, a PR template,
  and release-ready markdown.
- Clarified the version story across `README.md`, `docs/HANDBOOK.md`,
  `pyproject.toml`, and `CHANGELOG.md`.
- Cleaned the README docs table and removed the stray `Need | Home` wording.

## Intentionally Not Changed

- No package rename.
- No CLI rename.
- No change to the stable command names.
- No expansion of the supported commercial lane.
- No redesign or 4.0-style rewrite.

## Verification Performed

- `python -m pytest`
- `python scripts/run_ci_smoke.py`

## Remaining Follow-ups / Risks

- The canonical source profile still references a repo-external raw dataset
  path, so the full end-to-end release smoke remains a manual gate.
- `scripts/run_release_smoke.py` still depends on `.[gpu]` and a release-ready
  local environment.
