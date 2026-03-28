# Release Workflow

## Environment

- Python `3.12.x`
- Install with `python -m pip install -e .`
- Use `bittrace-source` as the supported executable of record

## Canonical Release Smoke

Run the bundled smoke workflow from the repository root:

```bash
python scripts/run_release_smoke.py
```

Optional explicit run id:

```bash
python scripts/run_release_smoke.py release_smoke_manual_01
```

The workflow stops on the first failure and writes:

- `runs/release_smoke/<smoke_run_id>/release_smoke_summary.json`
- `runs/release_smoke/<smoke_run_id>/release_smoke_summary.md`

## Manual Command Order

If you need to run the supported path one step at a time, use this order:

```bash
bittrace-source campaign --config configs/canonical_source_profile.yaml --run-id <campaign_run_id> --campaign-seed 31
bittrace-source verify runs/canonical_source_profile/<campaign_run_id>
bittrace-source deployment-candidate --config configs/canonical_deployment_candidate.yaml --run-id <deployment_run_id> --search-seed 7100
bittrace-source persistence --config configs/persistence_quiet_scout.yaml --source-run-root runs/canonical_deployment_candidate/<deployment_run_id> --run-id <quiet_run_id>
bittrace-source persistence --config configs/persistence_aggressive.yaml --source-run-root runs/canonical_deployment_candidate/<deployment_run_id> --run-id <aggressive_run_id>
```

## Boundary Reminder

The source/commercial release supports one frontend lane, one backend lane, two persistence profiles, and one verify/parity handoff. Research-only workflows remain out of scope for this release path.
