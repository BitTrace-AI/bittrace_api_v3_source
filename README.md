# BitTrace API

BitTrace is a source-available Python package and CLI for generating,
verifying, and packaging BitTrace artifacts inside the launch-supported source
lane.

BitTrace is source-available, not OSI open source.

- Evaluation use is allowed.
- Commercial or production use requires a separate written commercial license.

## Canonical Reference

Technical truth lives in [`docs/HANDBOOK.md`](docs/HANDBOOK.md).

If another repo document conflicts with the handbook, the handbook wins. The
other docs are intentionally shorter and role-specific.

## Public Surface

- import namespace: `bittrace`
- CLI: `bittrace`
- module fallback: `python -m bittrace`
- installed distribution name: `bittrace-api-v3-source`

Supported stable workflows:

- `bittrace campaign`
- `bittrace verify`
- `bittrace deployment-candidate`
- `bittrace persistence`

Experimental workflows remain under `bittrace experimental ...` and are
outside the supported commercial lane.

## Documentation

| Need | Home |
| --- | --- |
| Fastest repo-native setup and smoke validation | [`docs/QUICKSTART.md`](docs/QUICKSTART.md) |
| Canonical technical reference | [`docs/HANDBOOK.md`](docs/HANDBOOK.md) |
| Command, YAML, and direct-runtime cheat sheet | [`docs/USER_GUIDE.md`](docs/USER_GUIDE.md) |
| Supported public surface | [`SUPPORTED_SCOPE.md`](SUPPORTED_SCOPE.md) |
| Deployment ownership boundary | [`DEPLOYMENT_BOUNDARY.md`](DEPLOYMENT_BOUNDARY.md) |
| Working effectively with an AI assistant | [`docs/AI_ASSISTANT_GUIDE.md`](docs/AI_ASSISTANT_GUIDE.md) |
| Release-facing checks and packaging | [`docs/RELEASE_WORKFLOW.md`](docs/RELEASE_WORKFLOW.md) |
| Evaluation and commercial terms | [`LICENSE.md`](LICENSE.md) |
| Patent notice wording | [`NOTICE_PATENT_PENDING.md`](NOTICE_PATENT_PENDING.md) |

Start with [`docs/README_DOC_MAP.md`](docs/README_DOC_MAP.md) if you want the
reading order first.

## Boundary Reminder

Use [`SUPPORTED_SCOPE.md`](SUPPORTED_SCOPE.md) and
[`DEPLOYMENT_BOUNDARY.md`](DEPLOYMENT_BOUNDARY.md) for support and ownership
questions. Use the handbook for workflow, contract, artifact, and
troubleshooting detail.
