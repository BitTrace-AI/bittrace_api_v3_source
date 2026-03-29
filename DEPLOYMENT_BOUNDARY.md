# Deployment Boundary

This file defines what BitTrace ships, what the supported commercial lane does
and does not cover, and what remains the integrator's responsibility.
BitTrace is source-available, not OSI open source.

For the canonical technical reference, see
[`docs/HANDBOOK.md`](docs/HANDBOOK.md). For the launch-facing command cookbook
and config-to-command mapping, see [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).
For the supported technical surface, see
[`SUPPORTED_SCOPE.md`](SUPPORTED_SCOPE.md). For license rights, see
[`LICENSE.md`](LICENSE.md).

## What BitTrace Ships

BitTrace ships:

- the source-available repository
- the public `bittrace` CLI and `bittrace` import namespace
- the supported stable workflow family:
  - `bittrace campaign`
  - `bittrace verify`
  - `bittrace deployment-candidate`
  - `bittrace persistence`
- canonical freeze/export and parity-verification tooling
- the fixed commercial deployment lane:
  - frontend `temporal_threshold_36`
  - backend `Lean-Lean`
  - quiet scout persistence `i1 / d1 / y6 / r12 / no-latch`
  - aggressive persistence `i1 / d1 / y3 / r7 / no-latch`

## MCU Scope In The Base Source Lane

Supported in the base source lane:

- generate, export, and verify the BitTrace artifacts
- run the canonical workflows
- produce validated deployment-ready outputs

Not included in the base source lane:

- chip-specific firmware integration
- flashing workflows
- board support packages
- vendor IDE or toolchain instructions for every MCU

## What The Supported Commercial Lane Means

For launch, the supported commercial deployment lane is the fixed
`temporal_threshold_36` + `Lean-Lean` path represented by:

- `configs/canonical_deployment_candidate.yaml`
- `configs/persistence_quiet_scout.yaml`
- `configs/persistence_aggressive.yaml`

`bittrace campaign` and `bittrace verify` remain supported stable workflows, but
they serve the canonical source-lane freeze/export and parity-evidence path.
They are not an alternate commercial deployment lane.

The deployment-candidate workflow may report:

- `frontend_regime=temporal_threshold_36`
- `semantic_bit_length=36`
- `comparison_bundle_bit_length=64`

That `64` bit-length value is a packed transport detail. It does not change the
supported semantic frontend identity away from `temporal_threshold_36`.

## What BitTrace Does Not Automatically Grant

BitTrace does not automatically grant:

- unrestricted commercial rights
- unrestricted production rights
- approval for every target environment
- certification for regulated or safety-critical deployment
- support for experimental commands
- support for unsupported frontend/backend combinations
- legal advice

Evaluation use is allowed under `LICENSE.md`. Commercial or production use
requires a separate written commercial license.

## What Customers And Integrators Own

Customers and integrators own:

- target-hardware adapter implementation
- BSP, drivers, build, flash, packaging, and deployment mechanics
- environment-specific validation and acceptance testing
- operational monitoring, rollback, and observability
- compliance and regulatory obligations
- final decisions about operational use

BitTrace provides the canonical artifacts and workflows. It does not assume
responsibility for the surrounding operating environment.

## What Stays Outside The Boundary

Outside the supported commercial boundary:

- everything under `bittrace experimental ...`
- everything under `configs/experimental/`
- research-only architecture comparisons and search expansions
- alternate persistence logic not defined by the two shipped profiles
- ad hoc reinterpretations of the supported lane

These paths may still be useful in-house. They are not part of the supported
commercial lane and should not be described as such.

## Deployment-Candidate Boundary

Treat a deployment candidate as:

- a controlled output for review and verification
- a candidate package or summary for customer-side integration work
- something that still requires environment-specific acceptance

"Deployment candidate" does not mean blanket approval for production use in any
environment.

## Verification Boundary

The supported verification boundary is the parity and golden-vector output from
`bittrace verify` on a completed canonical campaign run root.

Expected verification output directory:

- `<campaign_run_root>/07_parity_verification/`

Key files:

- `bt3.stage_request.json`
- `bt3.verification_kit_manifest.json`
- `bt3.golden_vector_manifest.json`
- `bt3.parity_report.json`

Those artifacts are the handoff for customer-side parity and evidence review.

## Deployable-Ready Handoff Package

For this repo, a deployable-ready handoff package means the integration team
receives the full artifact set needed to port and validate the model on the
target, not just a single summary file.

Minimum handoff:

- exact repo revision or source snapshot
- exact install command
- exact commands run
- source profile YAML
- deployment-candidate YAML
- persistence YAML if persistence is required
- `bt3.campaign_request.json`
- `bt3.campaign_result.json`
- `bt3.freeze_export_manifest.json`
- `bt3.deep_anchor_artifact.json`
- `bt3.frontend_export_reference.json`
- `bt3.stage_request.json`
- `bt3.verification_kit_manifest.json`
- `bt3.golden_vector_manifest.json`
- `bt3.parity_report.json`
- `leanlean_deployment_candidate_plan.json`
- `leanlean_deployment_candidate_summary.json`
- `summary.csv`
- `summary.md`

If persistence is required for the application, add:

- `persistence_prep/leanlean_persistence_tuning_prep.json`
- `persistence_prep/leanlean_window_outputs_template.json`
- `persistence_prep/leanlean_window_outputs.json`
- the selected persistence `*_summary.json`
- the selected persistence `*_summary.csv`
- the selected persistence `*_summary.md`
- the selected persistence `*_profile.json`
- the selected persistence `*_examples.json`

That package is ready for customer-side integration work. It is not a promise
that board support, flashing, vendor IDE setup, or system acceptance are
already complete.

## Patent And License Boundary

Use restrained factual patent wording only:

> Patent Pending. Certain core BitTrace symbolic packed-bit classification
> methods and systems are the subject of one or more pending U.S. patent
> applications.

The patent notice is informational only. It does not expand or reduce the
rights granted by `LICENSE.md`.

Launch docs must not imply:

- that source-available means OSI-style open-source commercial rights
- that evaluation use equals production permission
- that patent-pending status changes license rights by itself
- that experimental commands are commercially supported
