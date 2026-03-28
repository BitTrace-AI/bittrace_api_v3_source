# BitTrace API V3 Source Lane

Frozen source/commercial lane for the supported BitTrace V3 shipping path.

This package exposes one supported user interface for the commercial source lane:

- executable of record: `bittrace-source`
- equivalent module entrypoint: `python -m bittrace_bearings_v3_source`

Python support for this lane is intentionally narrow: Python `3.12.x` only.

## Canonical Lane

- Frontend: `temporal_threshold_36`
- Backend: `Lean-Lean`
- Persistence profiles:
  - `quiet_scout`: `i1 / d1 / y6 / r12 / no-latch`
  - `aggressive`: `i1 / d1 / y3 / r7 / no-latch`

This repo keeps the canonical campaign/freeze-export path, the frozen Lean-Lean deployment-candidate path, the two supported persistence profiles, and the parity/golden-vector verification path.

## Quick Start

Install the package into a Python 3.12 environment:

```bash
python -m pip install -e .
bittrace-source --help
```

Run the canonical deployment-candidate preparation path:

```bash
bittrace-source deployment-candidate \
  --config configs/canonical_deployment_candidate.yaml \
  --run-id manual_candidate_prepare \
  --prepare-only
```

Run the bundled release smoke workflow:

```bash
python scripts/run_release_smoke.py
```

## Top-Level Docs

- `SUPPORTED_SCOPE.md`: supported vs unsupported surface
- `DEPLOYMENT_BOUNDARY.md`: ownership boundary for source/commercial release
- `docs/VERIFICATION_WORKFLOW.md`: parity, golden vectors, freeze/export, and release hardening flow
- `docs/RELEASE_WORKFLOW.md`: install, smoke, and canonical release commands
