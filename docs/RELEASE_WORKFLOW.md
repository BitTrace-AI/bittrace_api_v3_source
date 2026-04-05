# Release Workflow

This document covers the launch-facing release workflow for documentation,
installability, CLI validation, and the bounded stable smoke run.

## 1. Use The Repo-Local Source Environment

Run from the repository root and use the repo-local `.venv_source`.

```bash
rm -rf .venv_source
python3.12 -m venv .venv_source
. .venv_source/bin/activate
python -m pip install -e ".[gpu]"
```

Use `.[gpu]` for release validation. The canonical stable configs request GPU
backends and do not allow backend fallback.

In the current repo, `.[gpu]` installs `torch>=2.10`. That dependency is used
by the current Lean and Deep GPU backend implementations for CUDA execution. It
is not the public BitTrace modeling API.

The retained temporal-threshold reference profile may still appear in bounded
release smoke or example material, but it should not be described as the
universal frontend identity of BitTrace.

## 2. Confirm The Public Surface

Run and archive the public help surface:

```bash
bittrace --help
bittrace campaign --help
bittrace verify --help
bittrace deployment-candidate --help
bittrace persistence --help
bittrace experimental --help
python -m bittrace --help
```

The stable launch surface must remain:

- `bittrace campaign`
- `bittrace verify`
- `bittrace deployment-candidate`
- `bittrace persistence`

## 3. Run The Automated CI Smoke

Run the lightweight repo-local smoke path that CI uses:

```bash
python scripts/run_ci_smoke.py
```

This path stays CPU-friendly. It validates the installed `bittrace` console
script, the `python -m bittrace` fallback, the stable help surface, the
experimental help boundary, and import/compile smoke without requiring the
private raw dataset path from the canonical source profile.

## 4. Run The Bounded Release Smoke

Standard invocation:

```bash
python scripts/run_release_smoke.py
```

Optional explicit smoke run id:

```bash
python scripts/run_release_smoke.py release_smoke_manual_01
```

The smoke runner:

- creates or recreates `.venv_source` when needed
- installs `.[gpu]`
- resolves the `bittrace` console script
- runs `py_compile` over `src/` and `scripts/`
- smoke-checks imports for `bittrace`, `bittrace.source`,
  `bittrace.experimental`, and `bittrace.v3`
- smoke-checks top-level help
- smoke-checks stable command help
- smoke-checks experimental help
- smoke-checks `bittrace experimental backend-comparison --help`
- smoke-checks `python -m bittrace --help`
- runs the stable workflow family end to end

The full release smoke remains a manual release gate because it bootstraps an
isolated `.venv_source`, installs `.[gpu]`, and runs the stable workflow end to
end.

## 5. Release-Smoke Outputs

Release-smoke outputs are written under:

- `runs/release_smoke/<smoke_run_id>/`

Inspect first:

- `release_smoke_summary.json`
- `release_smoke_summary.md`

Also inspect relevant logs:

- `editable_install.stdout.log`
- `editable_install.stderr.log`
- `cli_help.stdout.log`
- `stable_command_help.stdout.log`
- `experimental_help.stdout.log`
- `experimental_command_help.stdout.log`
- `module_help.stdout.log`
- `canonical_campaign.stdout.log`
- `verify_parity.stdout.log`
- `deployment_candidate.stdout.log`
- `quiet_persistence.stdout.log`
- `aggressive_persistence.stdout.log`

Matching `*.stderr.log` files are written beside them.

## 6. Manual Stable Command Order

If you need to run the stable path one step at a time, use this order:

```bash
bittrace campaign --config configs/<project_source_profile>.yaml --run-id <campaign_run_id> --runs-root runs --campaign-seed 31
bittrace verify runs/<project_source_profile>/<campaign_run_id>
bittrace deployment-candidate --config configs/<project_deployment_candidate>.yaml --run-id <deployment_run_id> --runs-root runs --search-seed 7100
bittrace persistence --config configs/persistence_quiet_scout.yaml --source-run-root runs/<project_deployment_candidate>/<deployment_run_id> --run-id <quiet_run_id>
bittrace persistence --config configs/persistence_aggressive.yaml --source-run-root runs/<project_deployment_candidate>/<deployment_run_id> --run-id <aggressive_run_id>
```

Use a fresh run id every time. The CLI rejects non-empty run roots.

## 7. Launch-Docs Checklist

Before release-bundle assembly, confirm the following files exist and agree with
the actual repo:

- [`docs/HANDBOOK.md`](HANDBOOK.md)
- [`docs/README_DOC_MAP.md`](README_DOC_MAP.md)
- [`README.md`](../README.md)
- [`SUPPORTED_SCOPE.md`](../SUPPORTED_SCOPE.md)
- [`DEPLOYMENT_BOUNDARY.md`](../DEPLOYMENT_BOUNDARY.md)
- [`docs/QUICKSTART.md`](QUICKSTART.md)
- [`docs/USER_GUIDE.md`](USER_GUIDE.md)
- [`docs/RELEASE_WORKFLOW.md`](RELEASE_WORKFLOW.md)
- [`docs/AI_ASSISTANT_GUIDE.md`](AI_ASSISTANT_GUIDE.md)
- [`NOTICE_PATENT_PENDING.md`](../NOTICE_PATENT_PENDING.md)
- [`LICENSE.md`](../LICENSE.md)
- [`CHANGELOG.md`](../CHANGELOG.md)
- [`KNOWN_ISSUES.md`](../KNOWN_ISSUES.md)

## 8. Commercial And Notice Checklist

Before release, confirm:

- `LICENSE.md` matches the source-available commercial posture
- evaluation use is described consistently
- commercial or production use requiring a separate written commercial license
  is described consistently
- `NOTICE_PATENT_PENDING.md` uses restrained factual wording only
- no doc implies the patent notice changes license rights

Preferred patent wording:

> Patent Pending. Certain core BitTrace symbolic packed-bit classification
> methods and systems are the subject of one or more pending U.S. patent
> applications.

## 9. Experimental Boundary Checklist

Before release, confirm:

- experimental commands stay under `bittrace experimental ...`
- experimental examples are labeled experimental
- experimental configs stay under `configs/experimental/`
- launch docs do not present experimental workflows as stable commitments

## 10. Ignored-Artifacts Check

The repo currently ignores:

- `release_artifacts/`
- `runs/`
- `.venv_source/`

That is compatible with the launch workflow, but release-bundle assembly should
still archive the actual evidence outside the git index.
