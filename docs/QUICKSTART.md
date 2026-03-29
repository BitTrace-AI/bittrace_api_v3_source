# Quickstart

This is the shortest repo-native path to validate the supported stable
BitTrace surface.

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

## 4. Where To Go Next

- Full technical reference: [`HANDBOOK.md`](HANDBOOK.md)
- Command and YAML cheat sheet: [`USER_GUIDE.md`](USER_GUIDE.md)
- AI-ready help packaging: [`AI_ASSISTANT_GUIDE.md`](AI_ASSISTANT_GUIDE.md)
- Support and ownership boundaries:
  [`../SUPPORTED_SCOPE.md`](../SUPPORTED_SCOPE.md) and
  [`../DEPLOYMENT_BOUNDARY.md`](../DEPLOYMENT_BOUNDARY.md)

## 5. If You Need Help

Capture all of the following:

1. repository root
2. working directory
3. exact command
4. full stdout and stderr
5. exact config or input path
6. exact output path
