# Quickstart

This is the shortest repo-native path to validate the supported stable
BitTrace surface.

This document is Phase 0 only: environment setup plus smoke validation. It is
for toolchain validation, not model tuning, not optimization, and not final
candidate selection.

Technical detail lives in [`HANDBOOK.md`](HANDBOOK.md). If another document
conflicts with the handbook, the handbook wins.

## 1. Create A Clean Repo-Local Environment

Help-only setup:

```bash
rm -rf .venv_source
python3.12 -m venv .venv_source
. .venv_source/bin/activate
python -m pip install -e .
```

Full stable workflow setup:

```bash
rm -rf .venv_source
python3.12 -m venv .venv_source
. .venv_source/bin/activate
python -m pip install -e ".[gpu]"
```

Use `.[gpu]` for the documented stable workflow lane.

## 2. Confirm The Stable CLI Surface

```bash
bittrace --help
bittrace campaign --help
bittrace verify --help
bittrace deployment-candidate --help
bittrace persistence --help
python -m bittrace --help
```

If the four stable workflow families are missing, stop and fix the install
before continuing.

## 3. Run The Fastest Full Validation

This smoke path checks that the supported stable workflow can launch and emit
artifacts. It does not prove that you have a tuned model or a deployment-ready
candidate.

Run the bounded release smoke:

```bash
python scripts/run_release_smoke.py
```

Optional explicit smoke run id:

```bash
python scripts/run_release_smoke.py release_smoke_manual_01
```

Expected output directory:

- `runs/release_smoke/<smoke_run_id>/`

Check first:

- `release_smoke_summary.json`
- `release_smoke_summary.md`
- `*.stdout.log`
- `*.stderr.log`

## 4. What This Did And Did Not Prove

What this proved:

- the package installed correctly
- dependencies resolved
- the documented commands can run
- the repo can move data through the pipeline and emit expected artifacts

What this did not prove:

- that the first result is the best model
- that thresholds are tuned for your operating goal
- that search breadth or depth is appropriate
- that one profile is better than another for your dataset
- that persistence or referee settings are tuned where those choices matter
- that a deployment candidate has been selected from comparative evidence

Treat smoke success as "the toolchain works." Do not treat it as "the modeling
work is done."

## 5. Where To Go After Quickstart

Move into the real workflow in order:

1. Phase 1, baseline/reference run:
   produce a reproducible starter/reference result on real project data so you
   have a comparison anchor.
2. Phase 2, deliberate tuning and sweeps:
   run hyperparameter sweeps, threshold sweeps, search-breadth and
   search-depth exploration, profile comparison, and persistence or referee
   tuning where supported. Compare precision, recall, F1, and false-positive
   tradeoffs against your operating goals.
3. Phase 3, validation and candidate selection:
   choose a deployment candidate from sweep evidence, then validate
   repeatability, stability, and fit for the intended use case.

The baseline is a comparison anchor, not the finish line. Serious development
requires deliberate tuning before candidate selection.

## 6. Where To Go Next

- Full technical reference: [`HANDBOOK.md`](HANDBOOK.md)
- Command and YAML cheat sheet: [`USER_GUIDE.md`](USER_GUIDE.md)
- AI-ready help packaging: [`AI_ASSISTANT_GUIDE.md`](AI_ASSISTANT_GUIDE.md)
- Support and ownership boundaries:
  [`../SUPPORTED_SCOPE.md`](../SUPPORTED_SCOPE.md) and
  [`../DEPLOYMENT_BOUNDARY.md`](../DEPLOYMENT_BOUNDARY.md)

## 7. If You Need Help

Capture all of the following:

1. repository root
2. working directory
3. exact command
4. full stdout and stderr
5. exact config or input path
6. exact output path
