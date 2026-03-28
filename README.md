# BitTrace Canonical API

This repository is the canonical BitTrace API source tree.

Public identity:

- package: `bittrace`
- CLI: `bittrace`
- module entrypoint: `python -m bittrace`

There is one public package surface only:

- `bittrace`
- `bittrace.source` for the supported/commercial lane
- `bittrace.experimental` for retained in-house research workflows
- `bittrace.v3` for the shared V3 contracts and artifacts

Python support remains intentionally narrow: Python `3.12.x` only.

## Supported Lane

Top-level supported commands stay unchanged:

- `bittrace campaign`
- `bittrace verify`
- `bittrace deployment-candidate`
- `bittrace persistence`

Supported stable behavior:

- frontend: `temporal_threshold_36`
- backend: `Lean-Lean`
- persistence profiles:
  - `quiet_scout`: `i1 / d1 / y6 / r12 / no-latch`
  - `aggressive`: `i1 / d1 / y3 / r7 / no-latch`

Stable configs live directly under `configs/`.

## Experimental Lane

Retained research workflows now live in the same repo/package/CLI, but only under the experimental lane:

- `bittrace experimental backend-comparison`
- `bittrace experimental frontend-capacity-check`
- `bittrace experimental seed-sweep`
- `bittrace experimental leanlean-max-search`
- `bittrace experimental leanlean-deep-layer-max-search`
- `bittrace experimental leanlean-ceiling-search`
- `bittrace experimental leandeep-max-search`

Experimental configs live under `configs/experimental/`.
Several of the retained research configs intentionally contain `REPLACE_WITH_RUN_ID` placeholders because they consume prior run artifacts and are not part of the stable release bundle.

## Quick Start

Do not reuse a previously created or shared BitTrace virtualenv for validation. Create a clean repo-local `.venv_source` so the installed `bittrace` package comes from this checkout.

```bash
rm -rf .venv_source
python3.12 -m venv .venv_source
. .venv_source/bin/activate
python -m pip install -e .
bittrace --help
bittrace experimental --help
python -m bittrace --help
python -c "import bittrace, bittrace.source, bittrace.experimental, bittrace.v3"
```

Supported deployment-candidate preparation:

```bash
bittrace deployment-candidate \
  --config configs/canonical_deployment_candidate.yaml \
  --run-id manual_candidate_prepare \
  --prepare-only
```

Experimental backend comparison preparation:

```bash
bittrace experimental backend-comparison \
  --config configs/experimental/backend_architecture_comparison.yaml \
  --run-id manual_backend_compare \
  --prepare-only
```

Bounded release smoke:

```bash
python scripts/run_release_smoke.py
```

The smoke runner recreates `.venv_source` when needed and installs `.[gpu]` there because the canonical source profile requests the GPU backend.

## Docs

- `SUPPORTED_SCOPE.md`: stable lane versus experimental lane
- `DEPLOYMENT_BOUNDARY.md`: commercial deployment boundary versus retained research tooling
- `docs/VERIFICATION_WORKFLOW.md`: supported verification workflow and packaging smoke
- `docs/RELEASE_WORKFLOW.md`: stable release workflow
- `docs/EXPERIMENTAL_WORKFLOWS.md`: retained research commands and config templates
