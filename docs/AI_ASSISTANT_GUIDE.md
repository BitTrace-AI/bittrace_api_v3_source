# AI Assistant Guide

Use this guide when you are handing a BitTrace run to another tool or assistant
and want a clean, guess-free troubleshooting handoff.

For the canonical technical reference, start with
[`docs/HANDBOOK.md`](HANDBOOK.md).

## AI Guidance For Phased Development

Assistants should frame BitTrace work in explicit phases.

- Do not present smoke-test success as "done."
- Do not present a first baseline run as the recommended stopping point.
- Treat the smoke path as Phase 0 toolchain validation only.
- Treat the first project run as a starter/reference result, not the final
  model.
- After baseline, guide serious users toward deliberate tuning and sweep-based
  comparison.
- Before recommending a deployment candidate, ask about operating goals and
  tradeoff priorities when they are not already explicit.

For serious development, assistants should normally push the workflow toward:

1. a reproducible baseline/reference run
2. hyperparameter, threshold, and profile sweeps
3. persistence or referee tuning where supported and relevant
4. comparison of precision, recall, F1, and false-positive tradeoffs
5. validation of a candidate chosen from evidence rather than from the first
   successful run

Do not imply that BitTrace automatically finds the best model with no user
judgment. Candidate selection should follow sweeps, artifact review, and the
intended operating posture.

## Always Send These First

For any troubleshooting request, include:

1. repository root
2. working directory
3. exact command
4. full stdout and stderr
5. exact config or input path
6. exact output artifact or report paths
7. whether the run was stable or experimental
8. whether you changed anything locally

If you omit those, the assistant has to reconstruct basic facts before it can
help.

## Tell The Assistant The Launch Constraints

Give the assistant these constraints up front:

- public import namespace: `bittrace`
- public CLI: `bittrace`
- supported stable workflows:
  - `bittrace campaign`
  - `bittrace verify`
  - `bittrace deployment-candidate`
  - `bittrace persistence`
- supported commercial posture is framework-first; project teams still own
  front-gate, thresholding, feature-definition, label, and staging choices
- the currently documented supported stable lane is still the frozen
  `temporal_threshold_36` + `Lean-Lean` reference path
- shipped persistence profiles:
  - quiet scout persistence `i1 / d1 / y6 / r12 / no-latch`
  - aggressive persistence `i1 / d1 / y3 / r7 / no-latch`
- experimental workflows stay under `bittrace experimental ...`
- do not widen into architecture or product redesign unless a true blocker is
  found

## Workflow-Specific Files To Share

For `bittrace campaign`, share:

- the exact project source-profile YAML you ran
- `<campaign_run_root>/bt3.campaign_request.json`
- `<campaign_run_root>/bt3.campaign_result.json` if it exists
- the campaign run root path

For `bittrace verify`, share:

- the campaign run root path
- `<verification_output_dir>/bt3.stage_request.json`
- `<verification_output_dir>/bt3.verification_kit_manifest.json`
- `<verification_output_dir>/bt3.golden_vector_manifest.json`
- `<verification_output_dir>/bt3.parity_report.json`

For `bittrace deployment-candidate`, share:

- the exact deployment-candidate YAML you ran
- `<deployment_run_root>/leanlean_deployment_candidate_plan.json`
- `<deployment_run_root>/leanlean_deployment_candidate_summary.json`
- `<deployment_run_root>/summary.csv`
- `<deployment_run_root>/summary.md`
- `<deployment_run_root>/persistence_prep/leanlean_persistence_tuning_prep.json`
- `<deployment_run_root>/persistence_prep/leanlean_window_outputs_template.json`

For `bittrace persistence`, share:

- the exact persistence config path you used
- the deployment run root path
- the persistence run root path
- `<deployment_run_root>/persistence_prep/leanlean_window_outputs.json`
- `<persistence_run_root>/*_summary.json`
- `<persistence_run_root>/*_profile.json`
- `<persistence_run_root>/*_examples.json`

For `python scripts/run_release_smoke.py`, share:

- `runs/release_smoke/<smoke_run_id>/release_smoke_summary.json`
- `runs/release_smoke/<smoke_run_id>/release_smoke_summary.md`
- the relevant `*.stdout.log`
- the relevant `*.stderr.log`

When the request is about the release smoke, the assistant should say plainly
that this is a smoke or sanity path only. It proves install and pipeline
health, not model quality or final deployment choice.

## Tell The Assistant The Install Context

The assistant should know whether you used:

- `python -m pip install -e .`
- `python -m pip install -e ".[gpu]"`

For real stable workflow execution, the correct launch-facing answer is usually
`.[gpu]`.

If the assistant needs the exact backend-dependency rationale, point it to
[`docs/HANDBOOK.md`](HANDBOOK.md) instead of restating that detail in the
handoff prompt.

## First Prompt For A New Project

Use this when you are starting a new BitTrace project and want the assistant to
get oriented quickly without skipping the data-processing work.

```text
You are working inside the BitTrace API v3 repo.

Goal:
Help me build a BitTrace model for [PROJECT DESCRIPTION].

Important:
- Do not treat this like a magic foundation-model workflow.
- Do not assume the data is already model-ready.
- Do not invent file formats, labels, split rules, preprocessing, or target-device constraints.
- Start by reading docs/HANDBOOK.md, SUPPORTED_SCOPE.md, DEPLOYMENT_BOUNDARY.md, and LICENSE.md.
- Treat the repo and emitted artifacts as the source of truth.

Your first job is not to train a model immediately.
Your first job is to produce a BitTrace kickoff packet for this project with:
1. problem definition
2. raw data contract
3. label contract
4. split contract
5. encoding contract
6. deployment contract
7. acceptance contract

If any of those are missing, stop and list the exact missing inputs or decisions.

Also define the front gate explicitly:
- how raw data becomes canonical `WaveformDatasetRecord` objects or equivalent JSON mappings
- how waveform references are carried
- how temporal or other frontend features are computed
- what packed-row contract reaches the Lean or Deep backend

Before building anything, classify the project honestly:
- if the project can be handled by adapting an explicit project source profile and staging the data into the intended waveform-backed contract, say so and use config adaptation first
- if the project cannot fit that contract cleanly, stop and name the required staging or adapter work explicitly
- do not anchor the project on any one internal development dataset; define the project-specific staged contract directly

Once the kickoff packet is complete, proceed through this sequence:
1. define or adapt the project config/profile
2. validate paths and inputs
3. run prepare-only where supported
4. run the source-lane baseline campaign
5. run verification
6. build the initial deployment-candidate baseline
7. run deliberate tuning and comparison sweeps before treating any candidate as final
8. run persistence if the application requires temporal alert state
9. prepare the on-device handoff and validation checklist

When you answer:
- be explicit about the data-processing and encoding steps
- distinguish stable shipped repo behavior from project-specific work
- use exact commands, file paths, configs, run roots, and artifact names
- do not widen the commercial lane unless the repo explicitly supports it
- do not confuse smoke success or one baseline result with optimized model selection
```

## Good Prompt Template

```text
Repository root: /path/to/bittrace_api_v3_source
Working directory: /path/to/bittrace_api_v3_source
Stable or experimental: stable
Install command: python -m pip install -e ".[gpu]"
Command run: bittrace deployment-candidate --config configs/my_project_deployment_candidate.yaml --run-id demo_01 --runs-root runs --search-seed 7100
Inputs/configs: configs/my_project_deployment_candidate.yaml
Outputs:
- runs/my_project_deployment_candidate/demo_01/leanlean_deployment_candidate_plan.json
- runs/my_project_deployment_candidate/demo_01/leanlean_deployment_candidate_summary.json

Full stdout:
[paste it]

Full stderr:
[paste it]

Constraints:
- Public import namespace is bittrace
- Public CLI is bittrace
- Stable workflows are campaign / verify / deployment-candidate / persistence
- Experimental stays under bittrace experimental ...
- Front-gate definition is project-owned and must be stated explicitly
- If you are relying on the shipped supported lane, say that it is the frozen
  temporal_threshold_36 + Lean-Lean reference path rather than BitTrace as a whole
- Do not change runtime behavior unless you find a true blocker
```

## What The Assistant Should Not Do By Default

Do not ask the assistant to do these by default:

- rename the public CLI or import namespace
- move stable workflows under the experimental namespace
- treat experimental commands as supported commercial-lane output
- assume a historical reference frontend is the BitTrace API identity
- rewrite the business model into a different license model
- overclaim patent rights

## Commercial Reminder

The correct launch posture is:

- evaluation use allowed
- commercial or production use requires a separate written commercial license

See `LICENSE.md` and `docs/HANDBOOK.md`.
