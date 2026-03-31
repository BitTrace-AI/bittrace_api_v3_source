# Known Issues

This file tracks current limitations for the BitTrace API v3 source repository.

- The shipped canonical source profile points at a repo-external raw dataset
  root (`/mnt/d/_bearing_eval_scratch`). That path is intentionally not used by
  CI or the lightweight automated smoke checks.
- The full `scripts/run_release_smoke.py` workflow remains a manual release
  gate. It bootstraps `.venv_source`, installs `.[gpu]`, and runs the stable
  workflow end to end.
- The supported public lane is intentionally narrow. Only `bittrace campaign`,
  `bittrace verify`, `bittrace deployment-candidate`, and `bittrace
  persistence` are stable.
- Everything under `bittrace experimental ...`, `bittrace.experimental`, and
  `configs/experimental/` remains outside compatibility guarantees.
- The repo currently supports Python `3.12.x` only.
