# User Guide

This guide is the operational cheat sheet for three questions:

1. Which command do I run?
2. Which YAML goes with that command?
3. How do I load the frozen runtime directly from Python?

Technical truth lives in [`HANDBOOK.md`](HANDBOOK.md). If another document
conflicts with the handbook, the handbook wins.

## Quick Surface

- import namespace: `bittrace`
- CLI: `bittrace`
- module fallback: `python -m bittrace`
- installed distribution name: `bittrace-api-v3-source`
- stable workflows:
  `bittrace campaign`, `bittrace verify`,
  `bittrace deployment-candidate`, `bittrace persistence`

For environment setup and smoke validation, use [`QUICKSTART.md`](QUICKSTART.md).
For support and ownership boundaries, use
[`../SUPPORTED_SCOPE.md`](../SUPPORTED_SCOPE.md) and
[`../DEPLOYMENT_BOUNDARY.md`](../DEPLOYMENT_BOUNDARY.md).

## Command Cookbook

Project campaign freeze/export:

```bash
bittrace campaign \
  --config configs/<project_source_profile>.yaml \
  --run-id <campaign_run_id> \
  --runs-root runs \
  --campaign-seed 31
```

Verify a completed campaign run root:

```bash
bittrace verify runs/<project_source_profile>/<campaign_run_id>
```

Project deployment candidate:

```bash
bittrace deployment-candidate \
  --config configs/<project_deployment_candidate>.yaml \
  --run-id <deployment_run_id> \
  --runs-root runs \
  --search-seed 7100
```

Quiet scout persistence:

```bash
bittrace persistence \
  --config configs/persistence_quiet_scout.yaml \
  --source-run-root runs/<project_deployment_candidate>/<deployment_run_id> \
  --run-id <quiet_run_id>
```

Aggressive persistence:

```bash
bittrace persistence \
  --config configs/persistence_aggressive.yaml \
  --source-run-root runs/<project_deployment_candidate>/<deployment_run_id> \
  --run-id <aggressive_run_id>
```

The shipped persistence YAMLs intentionally leave
`source_deployment_run_root` blank, so normal stable usage should pass
`--source-run-root` explicitly.

## YAML Map

| File | Used by | Normal use |
| --- | --- | --- |
| `configs/<project_source_profile>.yaml` | `bittrace campaign` | Project-owned source profile. Define staging, labels, splits, and front-gate assumptions explicitly for your data. |
| `configs/<project_deployment_candidate>.yaml` | `bittrace deployment-candidate` | Project-owned deployment-candidate config. It must point at the project source profile you actually intend to ship. |
| `configs/persistence_quiet_scout.yaml` | `bittrace persistence` | Quiet-scout persistence policy. Pass `--source-run-root` explicitly. |
| `configs/persistence_aggressive.yaml` | `bittrace persistence` | Aggressive persistence policy. Pass `--source-run-root` explicitly. |

Legacy reference configs for the old Paderborn temporal-threshold run still
exist under explicitly named `configs/legacy_paderborn_reference_*.yaml`
files. They are historical reference material, not the supported API identity
or the default front gate for new work. They can still be useful when you need
the retained reference profile, but they are not universal BitTrace truth.

For section-by-section config meaning and safe-edit boundaries, use the
handbook source-profile reference.

## What To Inspect First

| Command | Run root | Check first |
| --- | --- | --- |
| `bittrace campaign` | `runs/<project_source_profile>/<campaign_run_id>/` | `bt3.campaign_request.json`, `bt3.campaign_result.json` |
| `bittrace verify` | `runs/<project_source_profile>/<campaign_run_id>/07_parity_verification/` | `bt3.stage_request.json`, `bt3.verification_kit_manifest.json`, `bt3.golden_vector_manifest.json`, `bt3.parity_report.json` |
| `bittrace deployment-candidate` | `runs/<project_deployment_candidate>/<deployment_run_id>/` | `leanlean_deployment_candidate_summary.json`, `summary.md`, `persistence_prep/leanlean_persistence_tuning_prep.json` |
| `bittrace persistence` | `runs/<project_deployment_candidate>/<deployment_run_id>/persistence_tuning/<run_id>/` | `*_summary.json`, `*_profile.json`, `*_examples.json` |

For the full handoff artifact set and artifact semantics, use the handbook
artifact reference.

## Direct Runtime Use

You need a completed campaign run root.

Use the supported public helpers:

```python
from bittrace.source import resolve_s6_artifacts
from bittrace.v3 import (
    load_frozen_s6_golden_reference,
    load_frozen_s6_runtime,
    load_json_artifact_ref,
)
```

Minimal example:

```python
from pathlib import Path
import json

from bittrace.source import resolve_s6_artifacts
from bittrace.v3 import (
    load_frozen_s6_golden_reference,
    load_frozen_s6_runtime,
    load_json_artifact_ref,
)

run_root = Path("runs/<project_source_profile>/<campaign_run_id>")
resolved = resolve_s6_artifacts(run_root)

runtime = load_frozen_s6_runtime(
    freeze_export_manifest_ref=resolved.freeze_export_manifest_ref,
    deep_anchor_artifact_ref=resolved.deep_anchor_artifact_ref,
    frontend_export_reference_ref=resolved.frontend_export_reference_ref,
)

golden = load_frozen_s6_golden_reference(
    freeze_export_manifest_ref=resolved.freeze_export_manifest_ref,
    deep_anchor_artifact_ref=resolved.deep_anchor_artifact_ref,
    frontend_export_reference_ref=resolved.frontend_export_reference_ref,
)

frontend_export = load_json_artifact_ref(resolved.frontend_export_reference_ref)
handoff_path = Path(frontend_export.frontend_lineage.handoff_manifest_path)
record = json.loads(handoff_path.read_text(encoding="utf-8"))["records"][0]

frontend = runtime.frontend_infer(record)
deep = runtime.deep_infer(canonical_input=record)
end_to_end = runtime.end_to_end_infer(record)
expected = golden.expected_end_to_end_output(record)

print(frontend.packed_row_int)
print(deep.predicted_class, deep.reject)
print(end_to_end.predicted_class, end_to_end.reject)
print(expected.predicted_class, expected.reject)
```

Use `frontend.payload`, `deep.payload`, and the golden helpers when you need
the full artifact-backed details for comparison. Do not rely on private
modules when the public helpers already cover the job.

## Help Checklist

Use the handbook troubleshooting table first.

For CLI help requests, send:

1. repository root
2. working directory
3. exact command
4. exact YAML path
5. run root
6. full stdout and stderr
7. key artifact paths from the failing stage

For direct-runtime help requests, also send:

1. the exact Python snippet
2. the campaign run root
3. the resolved artifact paths
4. the handoff manifest path
5. a sample record or payload
6. the exact traceback or output
