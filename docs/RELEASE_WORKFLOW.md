# Release Workflow

## Environment

- Python `3.12.x`
- Use a clean repo-local source-lane venv: `.venv_source`
- Do not reuse a previously created or shared BitTrace virtualenv; validate from a clean repo-local environment so the installed `bittrace` package comes from this checkout
- Use `bittrace` as the only executable of record

Create the clean source-lane venv from the repository root:

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

## Canonical Release Smoke

Run the bundled smoke workflow from that clean `.venv_source`:

```bash
python scripts/run_release_smoke.py
```

Optional explicit run id:

```bash
python scripts/run_release_smoke.py release_smoke_manual_01
```

If the script is launched from another interpreter, it recreates `.venv_source` and re-runs itself there before doing the editable install and CLI/import checks.
The smoke runner installs `.[gpu]` inside that isolated venv because the canonical source profile requests the GPU backend.

The smoke still executes only the stable release path. Experimental commands are limited to packaging/help smoke so the canonical repo stays unified without turning the release bundle into a research pass.

The workflow stops on the first failure and writes:

- `runs/release_smoke/<smoke_run_id>/release_smoke_summary.json`
- `runs/release_smoke/<smoke_run_id>/release_smoke_summary.md`

## Manual Command Order

If you need to run the supported path one step at a time, use this order:

```bash
bittrace campaign --config configs/canonical_source_profile.yaml --run-id <campaign_run_id> --campaign-seed 31
bittrace verify runs/canonical_source_profile/<campaign_run_id>
bittrace deployment-candidate --config configs/canonical_deployment_candidate.yaml --run-id <deployment_run_id> --search-seed 7100
bittrace persistence --config configs/persistence_quiet_scout.yaml --source-run-root runs/canonical_deployment_candidate/<deployment_run_id> --run-id <quiet_run_id>
bittrace persistence --config configs/persistence_aggressive.yaml --source-run-root runs/canonical_deployment_candidate/<deployment_run_id> --run-id <aggressive_run_id>
```

## Boundary Reminder

The source/commercial release supports one frontend lane, one backend lane, two persistence profiles, and one verify/parity handoff. The retained research workflows now ship in the same package, but only under `bittrace experimental ...` and `configs/experimental/`; they remain out of scope for the release bundle.
