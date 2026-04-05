# Changelog

This changelog tracks public changes to the `bittrace-api-v3-source`
distribution.

Versioning note:

- `BitTrace API v3` identifies the frozen repository and product generation.
- The Python distribution `bittrace-api-v3-source` uses its own semver release
  line inside that v3 generation.
- Package version `0.3.1` is a stabilization release for the same v3 public
  surface.

## Unreleased

### Changed

- Clarified that BitTrace core is the packed-bit framework surfaced by the
  `bittrace` CLI and import namespace, while the historical
  `temporal_threshold_36` + `Lean-Lean` path is retained as a
  legacy/reference lane.
- Renamed the old canonical temporal Paderborn configs into explicit legacy
  reference configs to avoid presenting them as universal API truth.
- Updated stable docs, examples, and assistant guidance so front-gate
  definition is treated as project-owned unless a lane is explicitly frozen
  and documented as supported.
- Updated stable workflow wording and examples to use explicit project config
  paths for campaign and deployment-candidate execution.

### Compatibility

- The stable public import namespace remains `bittrace`.
- The stable public CLI remains `bittrace`.
- The stable stable-lane commands remain `campaign`, `verify`,
  `deployment-candidate`, and `persistence`.
- The current documented supported stable lane remains the frozen
  `temporal_threshold_36` + `Lean-Lean` reference path.

## 0.3.1 - 2026-03-31

This entry documents the first release-ready hardening pass for the current
BitTrace API v3 source repository.

### Added

- A visible top-level `tests/` tree with CLI, import, config, and smoke
  coverage.
- A lightweight `scripts/run_ci_smoke.py` path for CI-safe compile, import, and
  public-help validation.
- GitHub Actions CI on `push` and `pull_request` for Python 3.12 editable
  installs and automated checks.
- `KNOWN_ISSUES.md`, a release checklist, a PR template, and release-note
  source material.

### Changed

- Clarified the relationship between the BitTrace API v3 repository generation
  and the `bittrace-api-v3-source` package version line.
- Cleaned the README documentation map and removed the stray `Need | Home`
  wording.
- Kept `scripts/run_release_smoke.py` as the manual release gate and documented
  the lighter automated smoke path separately.

### Compatibility

- The stable public import namespace remains `bittrace`.
- The stable public CLI remains `bittrace`.
- The stable commands remain `campaign`, `verify`, `deployment-candidate`, and
  `persistence`.
- Experimental commands remain under `bittrace experimental ...`.
