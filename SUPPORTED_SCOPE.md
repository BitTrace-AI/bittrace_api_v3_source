# Supported Scope

## Supported

- Python `3.12.x`
- One supported CLI surface:
  - `bittrace-source`
  - `python -m bittrace_bearings_v3_source`
- One frontend lane: `temporal_threshold_36`
- One model lane: `Lean-Lean`
- One canonical deployment-candidate config: `configs/canonical_deployment_candidate.yaml`
- One canonical source profile for freeze/export: `configs/canonical_source_profile.yaml`
- Two persistence profiles only:
  - `configs/persistence_quiet_scout.yaml`
  - `configs/persistence_aggressive.yaml`
- Verification/parity path via `bittrace-source verify`
- Canonical six-stage freeze/export path via `bittrace-source campaign`
- Release smoke workflow via `python scripts/run_release_smoke.py`

## Unsupported

- The generic `bittrace.v3` wrapper as a commercial/source-lane customer interface
- Lean-Deep research workflows
- Max-search or ceiling-search workflows
- Deep-layer search variants beyond the frozen deployment-candidate lane
- Frontend sweep or capacity-comparison tooling
- Architecture-comparison user workflows
- 64-bit delta-augmentation experiments
- Other exploratory or internal-only research paths

## Release Rule

Anything outside the files, commands, and exact fixed profiles above is not part of the supported source/commercial launch.
