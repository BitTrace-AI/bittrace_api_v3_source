# Release Checklist

Use this checklist before cutting a public release from the BitTrace API v3
source repository.

## Required

- Confirm the stable public surface is unchanged:
  `bittrace`, `python -m bittrace`, `bittrace campaign`, `bittrace verify`,
  `bittrace deployment-candidate`, and `bittrace persistence`.
- Run `python -m pytest`.
- Run `python scripts/run_ci_smoke.py`.
- Run `python scripts/run_release_smoke.py` from the repo root in the
  repo-local `.venv_source`.
- Inspect the release-smoke summary under `runs/release_smoke/<smoke_run_id>/`.
- Confirm `README.md`, `docs/HANDBOOK.md`, `SUPPORTED_SCOPE.md`, and
  `DEPLOYMENT_BOUNDARY.md` still agree.
- Confirm `CHANGELOG.md`, `KNOWN_ISSUES.md`, and the release notes for the
  target tag are up to date.
- Confirm experimental commands and configs are still fenced under
  `bittrace experimental ...` and `configs/experimental/`.

## Release Output

- Prepare the PR using `.github/pr_bodies/` content or an equivalent body.
- Prepare the GitHub release body from `.github/releases/`.
- Archive release evidence outside git if needed. `runs/`, `.venv_source/`, and
  `release_artifacts/` remain ignored on purpose.
