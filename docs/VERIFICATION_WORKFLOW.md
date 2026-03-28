# Verification Workflow

## 1. Canonical Campaign / Freeze Export

Run the canonical six-stage path when you need source-lane freeze/export artifacts:

```bash
bittrace campaign \
  --config configs/canonical_source_profile.yaml \
  --run-id <campaign_run_id> \
  --campaign-seed 31
```

This is the path that emits the V3 freeze/export artifacts consumed by parity verification.

## 2. Canonical Deployment Candidate

Run the frozen shipping lane:

```bash
bittrace deployment-candidate \
  --config configs/canonical_deployment_candidate.yaml \
  --run-id <deployment_run_id> \
  --search-seed 7100
```

This keeps the frontend fixed to `temporal_threshold_36` and the backend fixed to `Lean-Lean`.

## 3. Supported Persistence Profiles

Replay the fixed quiet-scout profile:

```bash
bittrace persistence \
  --config configs/persistence_quiet_scout.yaml \
  --source-run-root <deployment_candidate_run_root> \
  --run-id <quiet_run_id>
```

Replay the fixed aggressive profile:

```bash
bittrace persistence \
  --config configs/persistence_aggressive.yaml \
  --source-run-root <deployment_candidate_run_root> \
  --run-id <aggressive_run_id>
```

The supported lane stops here. Do not substitute research policies, latch variants, alternate thresholds, or architecture-comparison flows into the source/commercial release path.

## 4. Golden Vectors / Parity

Verify a completed run root that already contains freeze/export artifacts:

```bash
bittrace verify <campaign_run_root>
```

Expected output: `<campaign_run_root>/07_parity_verification`

This directory is the handoff for parity and golden-vector evidence. It is the supported verification boundary for customer-side integration checks.

## 5. Release Hardening

Use a clean repo-local source-lane venv for release hardening. Do not reuse a previously created or shared BitTrace virtualenv; it can import the wrong historical `bittrace` package state.

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

Then run the bundled smoke workflow:

```bash
python scripts/run_release_smoke.py
```

The smoke runner performs:

- isolated editable-install smoke via `python -m pip install -e ".[gpu]"`
- `py_compile` over `src/` and `scripts/` via `find src scripts -type f -name '*.py' | sort`
- import smoke for `bittrace`, `bittrace.source`, `bittrace.experimental`, and `bittrace.v3`
- unified CLI `--help` smoke
- stable command help smoke
- experimental command help smoke
- module entrypoint `python -m bittrace --help` smoke
- canonical campaign
- verify/parity
- canonical deployment-candidate
- quiet persistence
- aggressive persistence

If the script is started from another interpreter, it recreates `.venv_source` and re-runs itself there before performing the smoke steps above.

## Experimental Reminder

Experimental workflows are packaged and help-smoked so the canonical repo stays unified, but they are not part of the verification boundary above. Use `bittrace experimental --help` and `docs/EXPERIMENTAL_WORKFLOWS.md` for the retained research lane.

Summary artifacts are written under `runs/release_smoke/<smoke_run_id>/` as:

- `release_smoke_summary.json`
- `release_smoke_summary.md`
