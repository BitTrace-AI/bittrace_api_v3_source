# Supported Scope

## Stable / Supported

- Python `3.12.x`
- One public CLI only:
  - `bittrace`
  - `python -m bittrace`
- One public package namespace only: `bittrace`
- Stable lane helpers under `bittrace.source`
- Shared contracts/artifacts under `bittrace.v3`
- Top-level stable commands only:
  - `bittrace campaign`
  - `bittrace verify`
  - `bittrace deployment-candidate`
  - `bittrace persistence`
- One frontend lane: `temporal_threshold_36`
- One backend lane: `Lean-Lean`
- Stable configs under `configs/`
- One canonical deployment-candidate config: `configs/canonical_deployment_candidate.yaml`
- One canonical source profile: `configs/canonical_source_profile.yaml`
- Two persistence profiles only:
  - `configs/persistence_quiet_scout.yaml`
  - `configs/persistence_aggressive.yaml`
- Release smoke workflow via `python scripts/run_release_smoke.py` from a clean repo-local `.venv_source`

## Experimental / In-House Available

- Experimental helpers under `bittrace.experimental`
- Experimental commands only under `bittrace experimental ...`
- Retained research configs under `configs/experimental/`
- Retained research lanes currently exposed through the canonical API:
  - backend architecture comparison
  - frontend capacity check
  - Lean-Lean seed sweep
  - Lean-Lean max-search
  - Lean-Lean deep-layer max-search
  - Lean-Lean ceiling-search
  - Lean-Deep max-search

## No Stability Guarantees

- Anything under `bittrace.experimental`
- Anything under `bittrace experimental ...`
- Anything under `configs/experimental/`
- Experimental config templates that depend on prior run artifacts and use `REPLACE_WITH_RUN_ID`
- Experimental artifact names, schema strings, report layouts, and search semantics

## Explicitly Not Allowed

- A second public top-level package name
- A second public CLI name
- Treating experimental commands as part of the commercial/support boundary
- Mixing experimental configs into the stable top-level commands without explicitly opting into the experimental lane
