# Supported Scope

This file defines the supported launch surface for the canonical BitTrace API.
BitTrace is source-available, not OSI open source.

This file defines the supported technical surface. For the fastest runnable
public path, start with [`docs/QUICKSTART.md`](docs/QUICKSTART.md). For the
canonical technical reference, start with
[`docs/HANDBOOK.md`](docs/HANDBOOK.md). For deployment and commercial-boundary
expectations, see
[`DEPLOYMENT_BOUNDARY.md`](DEPLOYMENT_BOUNDARY.md). For license rights, see
[`LICENSE.md`](LICENSE.md).

If a draft note, stale example, or older launch memo disagrees with the code,
CLI help, or this file, the repo and this file win.

For the launch-facing command cookbook, YAML mapping, and direct frozen-runtime
usage examples, start with [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md).

## Public Surface

Supported public import namespace:

- `bittrace`

Supported public CLI:

- `bittrace`

Supported module fallback:

- `python -m bittrace`

The install metadata name in `pyproject.toml` is currently
`bittrace-api-v3-source`. That is a packaging detail, not a second supported
public package name.

## Supported Stable Workflows

Top-level supported workflows:

- `bittrace campaign`
- `bittrace verify`
- `bittrace deployment-candidate`
- `bittrace persistence`

Supported stable config files:

- `configs/canonical_source_profile.yaml`
- `configs/canonical_deployment_candidate.yaml`
- `configs/persistence_quiet_scout.yaml`
- `configs/persistence_aggressive.yaml`

Supported Python version:

- Python `3.12.x`

## Supported Commercial Deployment Lane

The supported commercial deployment lane is frozen to:

- frontend: `temporal_threshold_36`
- backend: `Lean-Lean`
- quiet scout persistence: `i1 / d1 / y6 / r12 / no-latch`
- aggressive persistence: `i1 / d1 / y3 / r7 / no-latch`

Important nuance:

- `bittrace campaign` and `bittrace verify` are supported stable workflows for
  canonical freeze/export and parity evidence.
- The supported commercial deployment lane itself is the fixed
  `temporal_threshold_36` + `Lean-Lean` path surfaced by
  `bittrace deployment-candidate` and the two shipped persistence profiles.

## Experimental Surface

Experimental workflows remain under:

- `bittrace experimental ...`
- `configs/experimental/`
- `bittrace.experimental`

Current experimental command families exposed by the repo:

- `backend-comparison`
- `frontend-capacity-check`
- `seed-sweep`
- `leanlean-max-search`
- `leanlean-deep-layer-max-search`
- `leanlean-ceiling-search`
- `leandeep-max-search`

Experimental commands and configs:

- are not part of the supported commercial lane
- have no support or compatibility guarantee
- may change command semantics, config schema, artifact names, or report layout
- must be labeled experimental anywhere they appear

## Frozen Items

Do not change these casually:

- the public import namespace `bittrace`
- the public CLI name `bittrace`
- the four stable workflow names
- the supported `temporal_threshold_36` + `Lean-Lean` commercial lane
- the two named persistence profiles
- the stable versus experimental boundary

## Safe Documentation Changes

These are safe to revise without changing supported scope:

- wording and structure
- copy-paste examples
- troubleshooting guidance
- release and packaging guidance
- structured troubleshooting handoff guidance
- commercial clarity and notice placement

## Unsupported Implications To Avoid

Launch-facing docs must not imply:

- that every command in the repo is commercially supported
- that every importable module is a public stable API
- that experimental commands have compatibility guarantees
- that additional persistence policies are part of the supported lane
- that a different frontend/backend pairing becomes supported because it runs
- that the packaging metadata name is a second public package surface

## Release Check

Before release, confirm:

- launch docs use `bittrace` as the public CLI
- stable workflow names match actual help output
- stable docs use explicit config paths and run-root layouts
- experimental content is fenced off clearly
- commercial and patent wording matches `LICENSE.md` and
  `NOTICE_PATENT_PENDING.md`
