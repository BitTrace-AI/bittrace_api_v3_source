# BitTrace Handbook

This handbook is the only canonical technical reference for the current
BitTrace API v3 source repository.

If another repo document conflicts with this handbook, this handbook wins. The
other docs are intentionally shorter and role-specific.

Use it to answer:

- what BitTrace supports today
- what must be true before model building starts
- what you edit in the source profile
- what exact workflow you run
- what artifacts matter
- what BitTrace owns versus what integration still owns
- when you are still inside the stable framework boundary versus outside it

The goal of this document is practical: a reasonably trained technical user,
working with the repo and an AI tool or agent, should be able to produce a
deployable-ready BitTrace model package inside the supported boundary. Data
definition, staging, and target integration still need to be done explicitly.

## Recommended Working Style

BitTrace can look simple at the CLI surface while still being operationally
detailed in practice.

Work from exact artifacts, not from vague intentions:

- exact commands
- exact YAML paths
- exact run roots
- exact emitted artifacts
- exact parity outputs

BitTrace is not a foundation-model workflow where raw project material becomes
a deployable model automatically. Front-gate definition, staging, labels,
splits, and target constraints must already be settled.

## Public Surface

BitTrace ships one public import namespace and one public CLI:

- import namespace: `bittrace`
- CLI: `bittrace`
- module fallback: `python -m bittrace`

Supported stable commands:

- `bittrace campaign`
- `bittrace verify`
- `bittrace deployment-candidate`
- `bittrace persistence`

Experimental commands remain under:

- `bittrace experimental ...`

The editable install metadata name in `pyproject.toml` is
`bittrace-api-v3-source`. That does not change the supported public import
namespace or CLI surface.

## Versioning Note

`BitTrace API v3` identifies the frozen repository and product generation.

The Python distribution `bittrace-api-v3-source` uses its own semver release
line inside that v3 generation. A package version such as `0.3.1` is a source
distribution release for the same stable v3 public surface. It does not rename
the supported `bittrace` import namespace or the `bittrace` CLI.

## Commercial Posture

BitTrace is source-available, not OSI open source.

- Evaluation use is allowed under [`LICENSE.md`](../LICENSE.md).
- Commercial use requires a separate written commercial license.
- Production use requires a separate written commercial license.
- Experimental workflows are outside support and compatibility guarantees unless
  a separate written agreement says otherwise.

Preferred patent wording:

> Patent Pending. Certain core BitTrace symbolic packed-bit classification
> methods and systems are the subject of one or more pending U.S. patent
> applications.

## Supported Commercial Lane

The repo currently ships one supported stable commercial lane:

- frontend: `temporal_threshold_36`
- backend: `Lean-Lean`
- quiet scout persistence: `i1 / d1 / y6 / r12 / no-latch`
- aggressive persistence: `i1 / d1 / y3 / r7 / no-latch`

Important boundaries:

- the stable workflow is binary at the CLI workflow level
- experimental commands are not part of the supported commercial lane
- custom adapters, custom label postures, or custom front gates are project
  work unless the repo and docs explicitly freeze them as supported

## Reference Lane vs API Core

This handbook documents the supported lane in detail because supportability
requires a frozen lane.

BitTrace core is the packed-bit training, search, artifact, freeze/export, and
verification framework surfaced by the public `bittrace` CLI and `bittrace`
import namespace.

The supported stable lane is not the whole theoretical BitTrace modeling
surface.

The current supported stable lane is the documented reference path:

- frontend: `temporal_threshold_36`
- backend: `Lean-Lean`

Do not confuse "fixed in the canonical supported profile" with "universally
required by BitTrace."

Custom front gates, custom adapters, custom label postures, and other
project-specific modeling work may exist, but they are outside the supported
stable lane unless they are separately frozen and documented as supported.

## Start Here

Read in this order for a first pass:

1. [`README_DOC_MAP.md`](README_DOC_MAP.md)
2. [`SUPPORTED_SCOPE.md`](../SUPPORTED_SCOPE.md)
3. this handbook
4. [`DEPLOYMENT_BOUNDARY.md`](../DEPLOYMENT_BOUNDARY.md)
5. [`USER_GUIDE.md`](USER_GUIDE.md)
6. [`AI_ASSISTANT_GUIDE.md`](AI_ASSISTANT_GUIDE.md)

If you are starting a new project, focus first on these handbook sections:

- `Kickoff Packet`
- `Supported Input Contract`
- `Canonical Record Contract`
- `Temporal Feature Contract`
- `Labels And Binary Mapping`
- `Source Profile Reference`

## Document Goal

The handbook is written so that a reasonably trained practitioner with basic
AI or ML background can use BitTrace and an AI assistant to:

- define a project kickoff packet
- stage or adapt project data into the supported reference front gate or into a
  clearly project-defined front gate
- run the stable BitTrace workflow
- inspect the right artifacts
- produce a deployable-ready handoff package

The handbook does not promise:

- board bring-up
- flashing workflows
- vendor IDE projects for every target
- a full MCU firmware solution
- automatic target-side persistence implementation
- system acceptance testing

## Kickoff Packet

Before modeling starts, the project needs an explicit kickoff packet.

### Required Topics

| Topic | Why BitTrace Needs It |
| --- | --- |
| Objective | Determines what success means and what the model must decide |
| Record unit | Defines what one canonical record actually is |
| Split unit | Prevents leakage and unstable evaluation |
| Label contract | Defines what labels mean and how they map into the modeling lane |
| Source data | Defines what exists before staging or adaptation |
| Deployment target | Sets the real deploy boundary |
| Runtime constraints | Carries memory, latency, throughput, and operating limits |
| Acceptance criteria | Defines how model quality is judged |
| Parity expectations | Defines what must match between repo artifacts and target behavior |
| Persistence expectations | Defines whether alert replay is optional or part of the system contract |

### Required Packet Content

For each project, explicitly state:

- objective:
  what the model should detect or classify, what the output is used for, and
  whether false positives or false negatives are more costly
- record unit:
  whether one record is a window, event, segment, staged payload, or another
  deterministic unit
- split unit:
  what must stay together and how train, val, and test are assigned
- label contract:
  raw labels, modeled labels, exclusions, and whether the stable binary lane is
  acceptable
- source data:
  where it lives, what format it is in, and what staging or adapter work
  already exists
- deployment target:
  MCU or device family, environment, and integration constraints
- runtime constraints:
  memory, latency, throughput, power, and sampling limits if they matter
- acceptance criteria:
  which metrics matter and which split is authoritative
- parity expectations:
  what target-side results must match the repo outputs
- persistence expectations:
  whether persistence is mandatory and whether repo replay must be reproduced
  exactly

### Questions BitTrace Cannot Answer For You

BitTrace does not decide these automatically:

- what the correct record unit is
- what the correct split unit is
- what the labels should mean
- what should be excluded from modeling
- what the deployment target can actually support
- what error tradeoffs are acceptable
- whether persistence is part of the real system contract

### Signs The Project Needs Adapter Work Before Modeling

- the raw source is not waveform-backed and has no explicit front gate yet
- the raw files cannot be deterministically resolved into record-level payloads
- split assignment is still a debate instead of a rule
- label mapping is still vague
- the waveform channel and temporal feature story are not settled
- the project expects `bittrace campaign` to invent preprocessing on its own

## Supported Input Contract

This section defines what BitTrace expects before model building starts.

### Raw Source Data

Raw source data is the project-owned measurement material before BitTrace turns
it into canonical modeling inputs.

Raw source data may be:

- waveform files
- sensor capture files
- exported windows from another system
- project-local artifacts produced by your own staging or adapter layer

Raw source data is not automatically model-ready.

### Staged Dataset

A staged dataset is the deterministic, pre-modeled view of the project data
that you have decided BitTrace will consume.

A staged dataset settles:

- which records are included
- which labels they carry
- which split each record belongs to
- which waveform payload each record resolves to
- which channel names and metadata are available

The staged dataset may be represented either by:

- a raw-root layout that the current source bridge can resolve deterministically
- a custom adapter that emits canonical records directly

Deterministic replay depends on keeping these stable:

- record membership
- `source_record_id` assignment
- split assignment
- label assignment
- waveform payload resolution
- channel naming
- staging-time filtering and exclusion rules

Practical rule:

- source-profile selection should narrow a known staged dataset
- it should not compensate for missing staging logic

Common staging mistakes to avoid:

- treating the raw dump as if it were already the staged dataset
- changing labels or splits without changing the staged dataset definition
- letting selection filters hide labeling or split problems
- allowing duplicate `source_record_id` values
- allowing staged metadata to point at missing waveform payloads

### Canonical Record

A canonical record is the single modeling unit BitTrace consumes before
frontend encoding.

At minimum, a canonical record must define:

- `source_record_id`
- `split`
- `state_label`
- `waveforms`

The canonical record is the boundary between project-owned staging work and
BitTrace-owned encoding and modeling work.

### Waveform Payload Reference

A waveform payload reference is the link from a canonical record to the actual
waveform payload it depends on.

Each waveform reference must resolve through exactly one of these forms:

- `waveform_path`
- `waveform_payload_ref`

### Encoded Packed Row

An encoded packed row is the deterministic integer row emitted by the frontend
after a canonical record has passed through the chosen frontend contract.

For any given project:

- the semantic frontend identity must be declared explicitly in the project
  contract
- the packed transport row format is `packed_int_lsb0`
- the shared backend bundle is carried in a packed 64-bit transport shape

### What Must Exist Before `bittrace campaign`

Before running `bittrace campaign`, the project must already have:

- a source profile YAML that resolves to real inputs
- accessible raw data or a completed adapter or staging output
- deterministic label mapping
- deterministic split assignment
- canonical waveform references that resolve for every included record
- temporal feature settings aligned to the staged waveform channel if the
  stable temporal frontend is being used
- healthy and unhealthy coverage in every canonical split if you are using the
  supported stable binary lane

If any of those are unsettled, the project is still in kickoff or adapter work,
not in stable campaign execution.

### What BitTrace Does Not Do Automatically

BitTrace does not automatically:

- infer your record unit from a vague dataset description
- infer labels from arbitrary filenames or folder names unless your source
  bridge explicitly does so
- decide the split unit for you
- resample, synchronize, or align arbitrary multi-sensor streams at the generic
  contract layer
- turn camera, image, or non-waveform inputs directly into supported
  waveform-backed records without project-specific front-gate work
- invent temporal features for a new sensor without an explicit feature
  contract
- infer deployment constraints, parity tolerances, or persistence requirements

### Supported Waveform-Backed Workflows Versus Adapter Work

The supported workflow family in this repo is waveform-backed.

That means BitTrace expects the project to arrive at:

- canonical records with waveform references
- deterministic temporal feature payloads when using the stable temporal
  frontend
- packed frontend rows that become backend inputs

If your project can reach that boundary cleanly, you are inside the current
supported modeling surface.

Adapter work is anything needed before the project can satisfy the waveform-
backed canonical record contract.

Common adapter-work triggers are:

- raw files in a format the shipped source bridge does not decode
- non-waveform modalities
- project-specific window extraction or aggregation rules
- cross-sensor synchronization requirements
- multiclass or hierarchical label handling outside the stable binary lane

Adapter work is real project work. BitTrace does not hide it, and the stable
lane does not treat it as automatic.

## Canonical Record Contract

### Required Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `source_record_id` | string | yes | Stable identifier for the record within the dataset bundle |
| `split` | string | yes | Record split membership |
| `state_label` | string | yes | Record label at modeling time |
| `waveforms` | mapping | yes | Non-empty mapping from channel name to waveform payload reference |

### Optional Fields

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `label_metadata` | mapping | no | Extra label-side metadata that should travel with the record |
| `sampling_hz` | positive number | no | Positive sampling rate when known |
| `rpm` | positive number | no | Positive RPM when known |
| `operating_condition` | string | no | Human-readable condition identifier |
| `context_metadata` | mapping | no | Extra per-record metadata in the Python object surface |
| `lineage_metadata` | mapping | no | Provenance or staging lineage metadata |

### Identity Expectations

`source_record_id` should be:

- non-empty
- unique within a bundle
- deterministic across reruns of the same staged dataset
- stable enough that parity, golden vectors, and downstream handoff can refer
  to the same record unambiguously

`split` should be:

- non-empty
- deterministic
- stable across reruns unless the project intentionally changes the split
  contract

For the stable lane, the practical split vocabulary is:

- `train`
- `val`
- `test`

### Label Expectations

`state_label` must be non-empty.

For the supported stable CLI lane, the practical label posture is binary:

- `healthy`
- `unhealthy`

If your project needs something else, settle that in the kickoff packet and do
not assume the stable CLI lane will infer or support it automatically.

### Waveform Mapping Expectations

`waveforms` must be:

- a mapping
- non-empty
- keyed by channel name
- populated with valid waveform payload references

Each value in `waveforms` must resolve to exactly one waveform payload
reference shape.

### Generic Canonical JSON Shape

```json
{
  "source_record_id": "<stable_record_id>",
  "split": "<train_or_val_or_test>",
  "state_label": "<label_name>",
  "waveforms": {
    "<channel_name>": {
      "waveform_path": "<absolute_or_resolved_path>"
    }
  },
  "label_metadata": {
    "<label_key>": "<label_value>"
  },
  "context": {
    "sampling_hz": "<positive_number>",
    "rpm": "<positive_number>",
    "operating_condition": "<condition_name>",
    "metadata": {
      "<context_key>": "<context_value>"
    }
  },
  "lineage_metadata": {
    "<lineage_key>": "<lineage_value>"
  }
}
```

### Python Object Surface Versus Materialized JSON

The public Python object surface uses `WaveformDatasetRecord`.

One naming detail matters:

- the Python object field is `context_metadata`
- the materialized canonical JSON payload nests that content under
  `context.metadata`

Do not invent a third shape. Pick one of the supported surfaces and keep it
consistent.

## Waveform Payload Reference Contract

`WaveformPayloadRef` is the record-level pointer to the waveform payload
BitTrace will use for encoding and downstream runtime work.

Every entry under a canonical record's `waveforms` mapping must resolve to one
`WaveformPayloadRef`.

### Required Rule

Exactly one of these must be present:

- `waveform_path`
- `waveform_payload_ref`

These must never both be present on the same waveform reference.

### Fields

| Field | Required | Meaning |
| --- | --- | --- |
| `waveform_path` | conditionally required | Path to the waveform payload file |
| `waveform_payload_ref` | conditionally required | Artifact-style reference to a previously materialized waveform payload |
| `metadata` | optional | Extra JSON metadata about the waveform payload |

### Generic Shapes

Path-backed shape:

```json
{
  "waveform_path": "<path_to_waveform_payload>"
}
```

Artifact-backed shape:

```json
{
  "waveform_payload_ref": {
    "kind": "<artifact_kind>",
    "schema_version": "<artifact_schema_version>",
    "path": "<artifact_path>",
    "sha256": "<artifact_sha256>"
  },
  "metadata": {
    "<key>": "<value>"
  }
}
```

### Relationship To Canonical Records

A canonical record owns the labeling, split, and identity contract. The
waveform payload reference owns only the link to the waveform bytes or prior
waveform artifact.

In practice:

- `record.waveforms` maps channel names to waveform refs
- at least one channel must exist
- the channel names used in the record must line up with the frontend contract

For the stable temporal frontend:

- `temporal_features.channel_name` must exist in every included record

### Channel, Time-Base, And Alignment Assumptions

At this layer, BitTrace assumes:

- the referenced waveform payload is the waveform for that record
- channel naming is already settled
- any needed windowing boundary is already representable for the chosen payload
- any required time-base consistency is already handled before or inside the
  project's front gate

BitTrace does not assume:

- automatic cross-channel synchronization
- automatic resampling
- automatic alignment across multiple files
- automatic gap filling or interpolation

If those behaviors matter, they belong in staging or adapter work before the
canonical record is emitted.

### What Is Out Of Scope At This Layer

This layer does not define:

- raw dataset discovery rules
- arbitrary raw file decoding
- project-specific camera or image adaptation
- label mapping
- split assignment
- temporal feature extraction policy
- deployment behavior

`WaveformPayloadRef` is the pointer layer, not the whole ingestion pipeline.

## Legacy/Reference Temporal Feature Contract

This section documents the legacy temporal feature contract used by the
historical Paderborn reference path.

### Legacy Frontend Identity

The legacy semantic frontend identity for that reference path is:

- `temporal_threshold_36`

The current row transport format is:

- packed integer
- `packed_int_lsb0`
- carried in a packed 64-bit transport shape

That legacy reference frontend is driven by the deterministic temporal feature
vector described below. New projects should not treat it as the default API
identity.

### Extraction Steps

For the stable temporal frontend, BitTrace performs these steps:

1. resolve the waveform channel named by `temporal_features.channel_name`
2. select the current window and, when available, the immediately previous
   window using `window_size` and `window_anchor`
3. quantize waveform samples by multiplying by `sample_scale`, rounding half
   away from zero, and clamping to `[-sample_clip, sample_clip]`
4. compute the raw integer feature set from the current quantized window
5. compute selected delta features as `current_raw_feature - previous_raw_feature`
   or `current_raw_feature - 0` when no previous window exists
6. apply any per-feature right-shift scaling from
   `temporal_features.feature_scale_shifts`
7. clamp final feature values to `[clamp_min, clamp_max]`
8. derive threshold bits from the train-split feature distributions

### Exact Stable Feature Order

The current stable feature order is:

1. `mean`
2. `median`
3. `min`
4. `max`
5. `peak_to_peak`
6. `variance`
7. `rms`
8. `mean_abs_deviation`
9. `first_diff_mean`
10. `first_diff_abs_mean`
11. `max_delta`
12. `diff_sign_change_rate`
13. `slope`
14. `cumulative_delta`
15. `count_above_threshold`
16. `max_spike_above_mean`
17. `delta_rms`
18. `delta_variance`

That order is the feature vector order carried in the temporal feature payload.

### Exact Feature Meanings

The current features are computed over quantized integer samples:

- `mean`: rounded integer mean of the selected window
- `median`: approximate median from the strided sample set defined by
  `approx_median_stride`
- `min`: minimum quantized sample
- `max`: maximum quantized sample
- `peak_to_peak`: `max - min`
- `variance`: rounded integer mean of squared deviations from `mean`
- `rms`: integer square root of the rounded mean squared value
- `mean_abs_deviation`: rounded integer mean absolute deviation from `mean`
- `first_diff_mean`: rounded integer mean of first differences
- `first_diff_abs_mean`: rounded integer mean absolute first difference
- `max_delta`: maximum absolute first difference
- `diff_sign_change_rate`: signed-difference sign-change rate scaled by
  `rate_scale`
- `slope`: window slope scaled by `slope_scale`
- `cumulative_delta`: `last_sample - first_sample`
- `count_above_threshold`: sample count above the spike threshold
- `max_spike_above_mean`: `max(0, max_sample - mean)`
- `delta_rms`: current raw `rms` minus previous raw `rms`
- `delta_variance`: current raw `variance` minus previous raw `variance`

The spike threshold is:

- `mean + round(mean_abs_deviation * multiplier_numerator / multiplier_denominator)`

### Integerization, Scaling, And Clipping

The current contract uses integer math throughout the feature path.

Important points:

- sample integerization uses `sample_scale`
- sample clipping uses `sample_clip`
- `variance` and `delta_variance` are right-shift scaled by default
- right-shift scaling uses rounded integer division by `2^shift`
- final feature clipping uses `clamp_min` and `clamp_max`

The shipped stable config currently sets:

- `sample_scale = 4096`
- `sample_clip = 32767`
- `rate_scale = 1024`
- `slope_scale = 4096`
- `clamp_min = -32768`
- `clamp_max = 32767`
- `feature_scale_shifts.variance = 8`
- `feature_scale_shifts.delta_variance = 8`

No floating-point normalization, z-scoring, or learned feature standardization
exists in this deploy contract.

### Threshold Semantics

The temporal frontend converts feature values into threshold bits.

For the current stable lane:

- there are 18 feature values
- the semantic bit budget is 36
- that yields exactly 2 thresholds per feature

Thresholds are derived from the train split only:

- sort the train-split values for a feature
- choose `threshold_count` positions using
  `round(max_index * step / (threshold_count + 1))`
- for the stable 2-threshold case, this corresponds to the lower and upper
  interior train quantiles

A bit is set when:

- `feature_value >= threshold`

Bits are packed in:

- feature order first
- threshold order within each feature
- least-significant-bit-first row layout

### What Must Remain Stable For Parity

These must remain stable if you expect parity with the legacy frozen frontend:

- `channel_name`
- `window_size`
- `window_anchor`
- `sample_scale`
- `sample_clip`
- `approx_median_stride`
- `spike_rule`
- `rate_scale`
- `slope_scale`
- `selected_persistence_deltas` and their order
- `feature_scale_shifts`
- `clamp_min`
- `clamp_max`
- the exact train records used to derive thresholds
- the exact feature order
- the threshold comparison rule `>=`
- the bit packing order `packed_int_lsb0`
- the semantic frontend identity `temporal_threshold_36`

If any of those change, the parity target changes.

## Labels And Binary Mapping

### Stable Supported Label Posture

The current supported stable CLI lane is binary.

The practical stable labels are:

- `healthy`
- `unhealthy`

That binary posture is what the stable workflow and shipped persistence
profiles are aligned to in the current supported lane. A project's front gate
is still project-owned outside that lane.

### Binary Mapping In The Stable Lane

The stable source lane maps project-facing identifiers into the binary model
labels before campaign materialization.

In the shipped source profile, that mapping is configured under:

- `binary_mapping.healthy_regex`
- `binary_mapping.unhealthy_regex`

For a custom project, the same rule still applies:

- settle the raw-to-modeled mapping before campaign
- make the mapping deterministic
- ensure every included record lands in the intended modeled class

The stable binary lane also requires:

- both classes to exist in `train`
- both classes to exist in `val`
- both classes to exist in `test`

### Multiclass Posture

Current posture by category:

- directly supported in the stable CLI lane: no
- project-orchestrated outside the stable lane: possible only as custom project
  work with explicit adapters and validation
- experimental: possible as research work, but not part of the supported
  commercial lane

Do not describe the stable shipping lane as multiclass-capable unless the repo
and docs explicitly freeze and support that lane.

### Reject Behavior

The runtime can emit a `reject` flag, but that is not a third stable label.

In the frozen runtime, `reject` indicates that the runtime could not classify
because no prototypes were available for the deep decision path.

Treat `reject` as:

- runtime decision metadata
- not a training label
- not a substitute for `unknown`
- not a substitute for multiclass labeling

### How To Describe The Label Contract In The Kickoff Packet

Your kickoff packet should state:

- what the raw labels or source identifiers are
- how they map into `healthy` and `unhealthy`
- which records are excluded rather than forced into the mapping
- how split coverage is guaranteed for both classes
- whether the project is binary by design or is being reduced to binary for the
  supported lane

## Source Profile Reference

BitTrace does not treat one shipped source profile as universal API truth.
Project source profiles are project-owned.

The retained historical bearing reference profile lives under:

- `configs/legacy_paderborn_reference_source_profile.yaml`

### Edit Status Legend

- `user-editable`: normal project adaptation field
- `conditionally editable`: valid to change, but the change alters workflow
  assumptions or moves you closer to adapter or custom-project work
- `fixed`: treat as frozen only if you are intentionally reproducing that
  retained reference path

### Top-Level Sections

| Section | Status | Purpose |
| --- | --- | --- |
| `profile_name` | user-editable | Human-readable profile identity carried into artifacts |
| `data` | user-editable | Raw or staged input root resolution |
| `binary_mapping` | user-editable | Mapping from project labels or identifiers into `healthy` and `unhealthy` |
| `selection` | user-editable | Include and exclude filters before modeling |
| `deploy_constraints` | conditionally editable | Target-side and deploy-budget constraints |
| `ranking_intent` | conditionally editable | How winner selection is prioritized |
| `splits` | user-editable | Deterministic train, val, and test assignment |
| `backend` | conditionally editable | CPU or GPU execution choice for Lean and Deep stages |
| `enable_temporal_features` | fixed | Must remain enabled if you are reproducing the legacy temporal reference path |
| `temporal_features` | conditionally editable | Exact temporal feature contract for a declared front gate |
| `locked_frontend` | fixed | Frozen frontend identity when a project intentionally locks one |
| `hard_mode` | conditionally editable | Search-space and stage-search settings for the campaign |

### Section Reference

#### `profile_name`

Status:

- `user-editable`

Purpose:

- names the source profile carried through the campaign artifacts

Safe use:

- change it to reflect the project or staged dataset identity

#### `data`

Status:

- `user-editable`

Important fields:

- `raw_root`

Purpose:

- points the source bridge at the raw or staged dataset root

Safe edits:

- point `raw_root` at the staged dataset you intend to model

Boundary:

- `bittrace campaign` does not turn an arbitrary raw directory into canonical
  records by magic
- if the staged files do not match the source bridge or your own adapter path,
  fix staging first

#### `binary_mapping`

Status:

- `user-editable`

Important fields:

- `healthy_regex`
- `unhealthy_regex`

Purpose:

- converts the project-facing label space into the stable binary modeling
  contract

Safe edits:

- replace the mapping with project-appropriate rules that deterministically
  resolve records into `healthy` and `unhealthy`

Boundary:

- the stable lane is binary
- if your project cannot honestly reduce to the binary contract, you are
  outside the stable framework posture documented here

#### `selection`

Status:

- `user-editable`

Important fields:

- operating-condition filters
- include and exclude lists
- deterministic filtering

Purpose:

- narrows the candidate dataset before campaign materialization

Safe edits:

- filter to the project subset you actually intend to model

#### `deploy_constraints`

Status:

- `conditionally editable`

Important fields:

- `target`
- `max_selected_k_per_class`
- `notes`

Purpose:

- carries the deploy-facing limits that the campaign and downstream freeze are
  expected to honor

Use with care:

- changing `target` or `max_selected_k_per_class` changes the deploy contract
- these are legitimate project edits, but they are not cosmetic

#### `ranking_intent`

Status:

- `conditionally editable`

Purpose:

- defines how the stable campaign should prefer one winner over another

Use with care:

- ranking changes alter what the workflow considers a good winner
- change this only if the project acceptance criteria genuinely changed

#### `splits`

Status:

- `user-editable`

Important fields:

- split strategy
- train assignments
- val assignments
- test assignments

Purpose:

- makes split handling deterministic and reviewable

Safe edits:

- update the split policy to match the project record unit and acceptance logic

Hard rule for the stable binary lane:

- every canonical split must contain both `healthy` and `unhealthy` examples

#### `backend`

Status:

- `conditionally editable`

Important fields:

- `backend.lean.backend`
- `backend.lean.allow_backend_fallback`
- `backend.deep.backend`
- `backend.deep.allow_backend_fallback`

Purpose:

- chooses CPU or GPU execution behavior for the campaign stages

Safe edits:

- switch CPU versus GPU for the actual environment

Boundary:

- the shipped stable configs use GPU and keep fallback disabled
- changing fallback behavior changes operator expectations and can hide setup
  failures

#### `enable_temporal_features`

Status:

- `fixed`

Purpose:

- keeps the stable temporal frontend path enabled

Boundary:

- disabling this means you are no longer using the supported temporal frontend

#### `temporal_features`

Status:

- `conditionally editable`

Important fields:

- `channel_name`
- `window_size`
- `window_anchor`
- `sample_scale`
- `sample_clip`
- `approx_median_stride`
- `rate_scale`
- `slope_scale`
- `spike_rule`
- `selected_persistence_deltas`
- `feature_scale_shifts`

Purpose:

- defines the exact temporal feature contract that feeds the selected frontend

Safe edits:

- adapt the feature settings to the real waveform channel and sensor scaling
  for a new project

Boundary:

- any change here changes the frontend input contract
- parity, golden vectors, and deployment assumptions must all be interpreted
  against the new feature contract
- changing these settings is normal project work when you are defining a
  project-specific front gate

#### `locked_frontend`

Status:

- `fixed`

Important fields:

- `enabled`
- `regime_id`
- `encoding_regime`
- `temporal_features_enabled`
- `threshold_strategy`
- `bit_length`

Purpose:

- freezes the frontend identity for a locked-profile workflow

Boundary:

- changing this means you are no longer reproducing the same locked profile

#### `hard_mode`

Status:

- `conditionally editable`

Purpose:

- defines campaign search and trial settings for the source-lane stages

Use with care:

- small tuning is possible for project work
- broad changes move you away from the shipped source-lane behavior and make
  comparisons to the canonical lane less meaningful

### Run And Output Settings

Run and output settings are not top-level source-profile fields.

They are controlled by the CLI:

- `--run-id`
- `--runs-root`
- `--campaign-seed`
- `--prepare-only`

Treat those as execution settings, not source-profile contract fields.

### Most Common Safe Edits

- `data.raw_root`
- `profile_name`
- `binary_mapping`
- `selection`
- `splits`
- CPU versus GPU backend selection for the actual machine
- `temporal_features.channel_name`
- `temporal_features` sample-scaling settings that match the real sensor

### Edits That Usually Mean You Are Outside The Supported Lane

- disabling temporal features
- changing `locked_frontend.regime_id`
- changing the frontend bit length or encoding regime
- treating the stable lane as multiclass without custom project orchestration
- replacing the binary mapping contract with a non-binary posture
- changing deploy constraints without re-validating the deployment story
- using the source profile to hide missing staging or adapter work

## CLI Surface And Stable Workflow

### Public CLI

Primary executable:

- `bittrace`

Supported module fallback:

- `python -m bittrace`

Top-level help exposes:

- `campaign`
- `deployment-candidate`
- `persistence`
- `verify`
- `experimental`

Stable commands stay at the top level. Research-only workflows stay under
`bittrace experimental ...`.

### Stable Workflow Sequence

Run the supported stable workflow in this order:

1. `bittrace campaign`
2. `bittrace verify`
3. `bittrace deployment-candidate`
4. `bittrace persistence`

Do not treat experimental outputs as the supported shipping package.

### `bittrace campaign`

Purpose:

- build the canonical campaign request
- run the stable campaign stages
- produce the freeze and export artifacts that define the frozen model handoff

Actual help syntax:

```bash
bittrace campaign [-h] [--config CONFIG] --run-id RUN_ID [--runs-root RUNS_ROOT] [--campaign-seed CAMPAIGN_SEED] [--prepare-only]
```

Practical stable usage:

```bash
bittrace campaign \
  --config configs/<project_source_profile>.yaml \
  --run-id <campaign_run_id> \
  --runs-root runs \
  --campaign-seed 31
```

Required inputs:

- source profile YAML
- `--run-id`
- accessible raw or staged data

Default behavior from the code:

- no public default config; pass an explicit project source-profile path
- default runs root: `runs/`
- default campaign seed: `31`
- `--prepare-only` writes the request and stage config scaffolding without
  launching the canonical campaign

Run-root layout when using the canonical config:

- `runs/<project_source_profile>/<campaign_run_id>/`

Successful stdout includes key-value lines such as:

- `run_root=...`
- `campaign_request_json=...`
- `inventory_row_count=...`
- `lean_smoke_row_count=...`
- `campaign_result_path=...`

Key outputs:

- `bt3.campaign_request.json`
- `bt3.campaign_result.json`
- downstream freeze and export artifacts under the run root

Do not continue if:

- `--prepare-only` fails
- no usable records are resolved
- any split is missing one side of the binary contract
- the final `bt3.campaign_result.json` records a failed stage

### `bittrace verify`

Purpose:

- generate the canonical verification kit
- emit golden vectors
- emit the parity report against the frozen runtime path

Actual help syntax:

```bash
bittrace verify [--output-dir OUTPUT_DIR] run_root
```

Practical stable usage:

```bash
bittrace verify runs/<project_source_profile>/<campaign_run_id>
```

Required inputs:

- completed campaign run root containing the freeze and export artifacts

Default behavior from the code:

- required positional argument: completed campaign `run_root`
- default output dir: `<run_root>/07_parity_verification`

Successful stdout includes key-value lines such as:

- `run_root=...`
- `verification_output_dir=...`
- `stage_request_path=...`
- `verification_kit_manifest_path=...`
- `golden_vector_manifest_path=...`
- `parity_report_path=...`
- `golden_vector_count=...`
- `parity_observation_count=...`

Key outputs:

- `bt3.stage_request.json`
- `bt3.verification_kit_manifest.json`
- `bt3.golden_vector_manifest.json`
- `bt3.parity_report.json`

Do not continue if:

- the run root does not contain the expected freeze and export artifacts
- the parity report does not pass
- the parity report contains mismatches you do not understand

### `bittrace deployment-candidate`

Purpose:

- run a deployment-candidate search from an explicit project config
- emit the deployment summary
- scaffold persistence-prep artifacts

Actual help syntax:

```bash
bittrace deployment-candidate [-h] [--config CONFIG] --run-id RUN_ID [--runs-root RUNS_ROOT] [--search-seed SEARCH_SEED] [--prepare-only]
```

Practical stable usage:

```bash
bittrace deployment-candidate \
  --config configs/<project_deployment_candidate>.yaml \
  --run-id <deployment_run_id> \
  --runs-root runs \
  --search-seed 7100
```

Required inputs:

- deployment-candidate YAML
- `--run-id`
- source profile path referenced by that YAML

Default behavior from the code:

- no public default config; pass an explicit project deployment-candidate path
- default runs root: `runs/`
- `--search-seed` is optional; when omitted, the config search seed is used
- the canonical deployment config currently sets the search seed to `7100`
- `--prepare-only` validates the config and emits the plan and persistence
  scaffold without launching the search

Run-root layout when using the canonical config:

- `runs/<project_deployment_candidate>/<deployment_run_id>/`

Successful stdout includes key-value lines such as:

- `run_root=...`
- `plan_path=...`
- `source_profile_path=...`
- `frontend_regime=<declared_project_frontend>`
- `semantic_bit_length=36`
- `comparison_bundle_bit_length=64`
- `leanlean_search_seed=7100`
- `persistence_scaffold_path=...`
- `window_output_template_path=...`
- `summary_json_path=...`
- `summary_csv_path=...`
- `summary_md_path=...`

The `comparison_bundle_bit_length=64` value is expected. It is a packed bundle
detail, not a different semantic frontend regime.

Key outputs:

- `leanlean_deployment_candidate_plan.json`
- `leanlean_deployment_candidate_summary.json`
- `summary.csv`
- `summary.md`
- `persistence_prep/leanlean_persistence_tuning_prep.json`
- `persistence_prep/leanlean_window_outputs_template.json`

Do not continue if:

- `--prepare-only` fails
- the summary is missing
- the candidate no longer reflects the locked stable frontend and backend lane
- persistence is required and the persistence-prep scaffold is missing

### `bittrace persistence`

Purpose:

- replay one of the two supported persistence profiles against the deployment
  candidate outputs
- select and report the persistence behavior that will be handed to integration

Actual help syntax:

```bash
bittrace persistence [-h] [--config CONFIG] --run-id RUN_ID [--source-run-root SOURCE_RUN_ROOT] [--force-rematerialize-window-outputs]
```

Stable quiet scout usage:

```bash
bittrace persistence \
  --config configs/persistence_quiet_scout.yaml \
  --source-run-root runs/<project_deployment_candidate>/<deployment_run_id> \
  --run-id <quiet_run_id>
```

Stable aggressive usage:

```bash
bittrace persistence \
  --config configs/persistence_aggressive.yaml \
  --source-run-root runs/<project_deployment_candidate>/<deployment_run_id> \
  --run-id <aggressive_run_id>
```

Required inputs:

- persistence YAML
- `--run-id`
- `--source-run-root` unless the profile embeds one

Default behavior from the code:

- default config: `configs/persistence_quiet_scout.yaml`
- run root: `<source_run_root>/persistence_tuning/<run_id>/`
- `--force-rematerialize-window-outputs` rebuilds per-record Lean-Lean outputs
  even if they already exist

Important launch detail:

- the shipped stable persistence profiles intentionally leave
  `source_deployment_run_root` blank
- stable launch usage should therefore pass `--source-run-root` explicitly

Successful stdout includes key-value lines such as:

- `source_run_root=...`
- `run_root=...`
- `source_summary_path=...`
- `materialized_window_outputs_path=...`

Key outputs:

- `*_summary.json`
- `*_summary.csv`
- `*_summary.md`
- `*_profile.json`
- `*_examples.json`

Do not continue if:

- the deployment-candidate run root is missing
- the window-output materialization cannot be built
- the persistence summary does not reflect the application behavior you need

### Prepare-Only Boundaries

Supported prepare-only commands:

- `bittrace campaign --prepare-only`
- `bittrace deployment-candidate --prepare-only`

No prepare-only mode exists for:

- `bittrace verify`
- `bittrace persistence`

### Run Discipline

For every stable run, preserve:

- exact command
- working directory
- config path
- run root
- stdout and stderr
- key output artifact paths

Use a fresh run id every time. The CLI rejects reusing a non-empty run root.

## Artifact Reference

This section explains the small set of artifacts that matter most in the stable
v3 workflow.

### `bt3.campaign_request.json`

When produced:

- at campaign preparation time
- present even when `--prepare-only` is used

Why it matters:

- it freezes the exact campaign plan that will run

Critical fields to inspect:

- `campaign_id`
- `campaign_seed`
- `output_dir`
- `stage_sequence`
- `stage_search_policies`

How it informs the next step:

- if the request is wrong, stop before running the campaign

Placeholder shape:

```json
{
  "campaign_id": "<campaign_id>",
  "campaign_seed": "<integer_seed>",
  "output_dir": "<run_root>",
  "stage_sequence": ["<stage_1>", "<stage_2>", "<stage_3>"]
}
```

### `bt3.campaign_result.json`

When produced:

- after the campaign run finishes or fails

Why it matters:

- it tells you whether the stable campaign actually completed
- it tells you which freeze and export refs were emitted

Critical fields to inspect:

- `completed_stages`
- `failed_stage`
- `final_promoted_winner_refs`
- `freeze_export_refs`
- `verification_refs`

How it informs the next step:

- continue only if there is no meaningful stage failure and the freeze and
  export refs exist

Placeholder shape:

```json
{
  "campaign_request": {
    "campaign_id": "<campaign_id>"
  },
  "completed_stages": ["<stage_summary>"],
  "failed_stage": null,
  "freeze_export_refs": ["<artifact_ref>"]
}
```

### `bt3.parity_report.json`

When produced:

- during `bittrace verify`

Why it matters:

- this is the direct parity sign-off artifact for the frozen runtime path

Critical fields to inspect:

- `pass_fail`
- `summary`
- `exact_match_count`
- `mismatch_count`
- `unsupported_count`
- `results`

How it informs the next step:

- do not hand off deploy artifacts until parity is understood and accepted

Placeholder shape:

```json
{
  "pass_fail": "<PASS_or_FAIL>",
  "exact_match_count": "<count>",
  "mismatch_count": "<count>",
  "unsupported_count": "<count>",
  "results": [
    {
      "vector_id": "<record_id>",
      "verification_level": "<level>",
      "comparison_status": "<EXACT_MATCH_or_MISMATCH_or_UNSUPPORTED>"
    }
  ]
}
```

### `leanlean_deployment_candidate_summary.json`

When produced:

- during `bittrace deployment-candidate`

Why it matters:

- this is the main summary artifact for the supported deployment search

Critical fields to inspect:

- `source_profile_path`
- `variant`
- `summary_row`
- `deploy_constraints`
- `deploy_export`
- `persistence_prep`

How it informs the next step:

- use it to decide whether the deployment candidate is the one you will hand
  forward and whether persistence replay must run next

Placeholder shape:

```json
{
  "source_profile_path": "<source_profile_yaml>",
  "variant": {
    "variant_id": "<variant_id>",
    "artifact_path": "<artifact_path>"
  },
  "summary_row": {
    "<metric_name>": "<metric_value>"
  },
  "persistence_prep": {
    "artifacts": {
      "scaffold_path": "<path>"
    }
  }
}
```

### Persistence Outputs

When produced:

- during `bittrace persistence`

Why they matter:

- they convert raw classifier outputs into deploy-facing persistence behavior

Primary files:

- `*_summary.json`
- `*_summary.csv`
- `*_summary.md`
- `*_profile.json`
- `*_examples.json`

Critical fields to inspect in the JSON summary:

- `selection_scope`
- `selection_policy`
- `raw_classifier_metrics`
- `policy_candidates`
- `comparison_results`
- `selected_policy_path`

How they inform the next step:

- they tell integration which policy was selected and what replay evidence
  supports it

Placeholder shape:

```json
{
  "profile_name": "<persistence_profile>",
  "source_deployment_run_root": "<deployment_run_root>",
  "selection_policy": {
    "<policy_key>": "<policy_value>"
  },
  "selected_policy_path": "<selected_policy_json>"
}
```

### Minimum Artifact Set To Share When Asking For Help

Share at least:

- exact command
- exact YAML path
- run root
- `bt3.campaign_request.json`
- `bt3.campaign_result.json`
- `bt3.parity_report.json` if verification has run
- `leanlean_deployment_candidate_summary.json` if deployment-candidate has run
- the selected persistence `*_summary.json` and `*_profile.json` if
  persistence has run

## Experimental Boundary And Research Workflows

Everything in this section is intentionally outside the supported commercial
lane.

### Command Surface

- `bittrace experimental backend-comparison`
- `bittrace experimental frontend-capacity-check`
- `bittrace experimental seed-sweep`
- `bittrace experimental leanlean-max-search`
- `bittrace experimental leanlean-deep-layer-max-search`
- `bittrace experimental leanlean-ceiling-search`
- `bittrace experimental leandeep-max-search`

### Package Surface

- `bittrace.experimental.backend_architecture_comparison`
- `bittrace.experimental.frontend_capacity_check`
- `bittrace.experimental.leanlean_seed_sweep`
- `bittrace.experimental.leanlean_max_search`
- `bittrace.experimental.leanlean_ceiling_search`
- `bittrace.experimental.leandeep_max_search`

`leanlean-deep-layer-max-search` reuses
`bittrace.experimental.leanlean_max_search` with the alternate
`configs/experimental/leanlean_max_search_deep.yaml` config.

### Config Layout

- stable configs remain under `configs/`
- experimental configs live under `configs/experimental/`
- the frontend-capacity source-profile template lives under
  `configs/experimental/source_profiles/`

### Important Template Rule

Several experimental configs consume artifacts from previous runs and therefore
include `REPLACE_WITH_RUN_ID` placeholders under `runs/...`.

Before running these workflows:

1. produce the prerequisite stable or experimental run
2. replace the placeholder run id in the config with the real run directory
3. use `--prepare-only` first to confirm the graph of inputs is correct

### Stability Rule

These commands exist so in-house research stays in the canonical repo, package,
and CLI. They do not have stability guarantees for config schema, artifact
schema, report wording, or workflow semantics.

### Shared Front Gate

Experimental does not mean raw data goes straight into a model.

The current experimental lanes still rely on the same general front gate:

1. canonical records are defined with split and label discipline
2. records are bundled into a source or deep-input lineage path
3. a frontend emits packed rows
4. backend comparison or search operates on those packed rows

Most current experimental Lean-Lean and Lean-Deep searches still reuse a locked
frontend from the retained reference path or from another explicitly declared
project profile.
`frontend-capacity-check` varies frontend capacity, but it still starts from
canonical records and the same general bundle-to-frontend-to-packed-row path.

### Which Experimental Command To Use

Use experimental only after you can already run the stable workflow and state
the exact question you are trying to answer.

- `backend-comparison`: use when the question is whether Lean-Lean actually
  beats front-only and Lean-Deep under the same locked frontend
- `frontend-capacity-check`: use when the question is whether the frontend bit
  budget or encoding regime is the real bottleneck
- `seed-sweep`: use when the question is whether the deployment-candidate
  result is stable across search seeds or just lucky
- `leanlean-max-search`: use when you want a broader Lean-Lean search than the
  supported deployment-candidate lane while keeping the same locked frontend
  and deploy-relevant limits
- `leanlean-deep-layer-max-search`: use when you specifically want to know
  whether raising only the Lean-Lean layer ceiling changes the outcome
- `leanlean-ceiling-search`: use when you want to probe the practical Lean-Lean
  ceiling while still staying within deploy-constrained limits
- `leandeep-max-search`: use when you want to test whether a much larger
  Lean-Deep search can justify moving away from Lean-Lean

Command and config pairing rule:

- do not drive a deeper-layer Lean-Lean config through
  `bittrace experimental leanlean-max-search`
- if the question is specifically about raising the Lean-Lean layer ceiling,
  use `bittrace experimental leanlean-deep-layer-max-search`

### Recommended Decision Order

Use this order when exploring:

1. run stable `campaign`, `verify`, and `deployment-candidate` first
2. use `seed-sweep` if you need confidence that the stable winner is
   reproducible
3. use `frontend-capacity-check` if you suspect the frontend is limiting
   performance
4. use `backend-comparison` if you need an architecture-level comparison under
   one shared frontend
5. use the Lean-Lean or Lean-Deep max-search families only if the earlier
   checks show a justified reason to keep exploring

### When To Stop Exploring

Return to the stable lane when you need:

- a real deployment handoff package
- parity and golden-vector evidence for integrators
- persistence replay outputs for a shipping application
- release-facing or customer-facing artifacts

Experimental summaries are research evidence. They are not the supported
shipping package.

## Ownership Boundary And Deployment Handoff

BitTrace owns:

- the stable CLI workflow
- frozen artifacts
- parity and golden-vector evidence
- deployment-candidate outputs
- persistence replay outputs

Your integration team still owns:

- board support
- toolchain and flashing flow
- on-chip runtime porting
- transport and device integration
- target-side persistence implementation
- system acceptance testing

### Deployment Handoff

For a real deployment handoff, do not send only a summary markdown file.

Minimum handoff materials:

- exact repo revision or source snapshot
- exact install command used for the run
- exact stable commands that were executed
- source profile YAML
- deployment-candidate YAML
- persistence YAML if persistence is part of the application contract

Frozen artifact set:

- `bt3.campaign_request.json`
- `bt3.campaign_result.json`
- `bt3.freeze_export_manifest.json`
- `bt3.deep_anchor_artifact.json`
- `bt3.frontend_export_reference.json`

Verification and parity materials:

- `bt3.stage_request.json`
- `bt3.verification_kit_manifest.json`
- `bt3.golden_vector_manifest.json`
- `bt3.parity_report.json`

Deployment-candidate materials:

- `leanlean_deployment_candidate_plan.json`
- `leanlean_deployment_candidate_summary.json`
- `summary.csv`
- `summary.md`
- `persistence_prep/leanlean_persistence_tuning_prep.json`
- `persistence_prep/leanlean_window_outputs_template.json`

Persistence materials if applicable:

- `persistence_prep/leanlean_window_outputs.json`
- selected persistence `*_summary.json`
- selected persistence `*_summary.csv`
- selected persistence `*_summary.md`
- selected persistence `*_profile.json`
- selected persistence `*_examples.json`

BitTrace does not provide:

- a complete MCU firmware solution
- vendor IDE projects for every target
- board bring-up
- automatic on-chip deployment
- automatic target-side persistence implementation
- automatic system-level sign-off

## Troubleshooting And Help

| Symptom | Check First | Next Action |
| --- | --- | --- |
| campaign fails because `data.raw_root` is wrong | source profile `data.raw_root` | fix the path and rerun |
| campaign resolves zero usable records | `selection`, `binary_mapping`, `temporal_features`, campaign stdout such as `inventory_row_count` | fix staging or source-profile assumptions before rerunning |
| split mismatch or missing split coverage | `splits`, `selection`, kickoff packet split unit | restate the split rule in staging or the source profile |
| label mismatch | `binary_mapping` and staged dataset metadata | fix the label contract before rerunning |
| malformed canonical record | emitted record shape and `context_metadata` versus `context.metadata` | keep the Python object surface separate from the materialized bundle JSON surface |
| missing waveform refs | record `waveforms` and resolved waveform paths on disk | repair the staged waveform references before rerunning |
| unsupported modality assumption | input modality versus the waveform-backed contract | stop and define the front gate or adapter path explicitly |
| run-root collision | existing run root contents | use a fresh `--run-id` |
| verification parity failure | `bt3.parity_report.json`, `bt3.freeze_export_manifest.json`, `bt3.deep_anchor_artifact.json`, `bt3.frontend_export_reference.json` | compare the frozen artifacts and rerun from a clean source if needed |
| no deployment candidate | deployment-candidate stdout and `leanlean_deployment_candidate_plan.json` | inspect the search plan and use `--prepare-only` first if needed |
| persistence mismatch | `leanlean_persistence_tuning_prep.json`, `leanlean_window_outputs.json`, selected persistence summary and profile | verify the prep bundle and chosen persistence policy before comparing outputs |

When asking for help, send:

1. the exact command
2. full stdout and stderr
3. the exact YAML path
4. the run root
5. the key artifacts from the stage that failed

Use [`AI_ASSISTANT_GUIDE.md`](AI_ASSISTANT_GUIDE.md) if you want a ready-made
prompt structure.

## Core Terms

- staged dataset: deterministic project-owned dataset view before modeling
- canonical record: modeling-ready record unit consumed before frontend
  encoding
- waveform payload ref: record-level pointer to the waveform payload
- source profile: YAML used by `bittrace campaign` to define input resolution,
  selection, split handling, label mapping, and stable campaign behavior
- frontend: deterministic encoding layer that turns canonical-record inputs into
  packed rows
- backend: model layer that consumes packed rows after frontend encoding
- packed row: encoded integer row emitted by the frontend for backend use
- campaign: stable workflow step that prepares and runs the canonical
  freeze/export pipeline
- verify: stable workflow step that emits the verification kit, golden vectors,
  and parity report
- parity: agreement check between expected frozen outputs and actual runtime
  outputs
- deployment candidate: stable search and summary workflow driven by an
  explicit project deployment config
- persistence: replay step that turns per-record classifier outputs into a
  supported alert policy behavior
- supported lane: documented stable public workflow and commercial deployment
  path
- experimental: research-only commands under `bittrace experimental ...`
- handoff artifact: emitted file needed to inspect, validate, or integrate the
  frozen BitTrace model behavior
