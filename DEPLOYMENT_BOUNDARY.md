# Deployment Boundary

## BitTrace Owns In The Supported Lane

- Canonical model behavior for `temporal_threshold_36` + `Lean-Lean`
- Artifact contracts and freeze/export artifacts from the canonical campaign lane
- Golden-vector and parity verification tooling for completed canonical runs
- The two supported persistence profile definitions:
  - `quiet_scout`: `i1 / d1 / y6 / r12 / no-latch`
  - `aggressive`: `i1 / d1 / y3 / r7 / no-latch`
- Deterministic documentation, smoke workflow, and the supported top-level `bittrace` commands

## Retained But Outside The Deployment Boundary

- `bittrace.experimental`
- `bittrace experimental ...`
- `configs/experimental/`
- Lean-Deep, max-search, ceiling-search, deep-layer search, seed-sweep, frontend-capacity, and backend-comparison workflows

These workflows remain available in the canonical repo because they are still useful in-house. They are not part of the commercial/support commitment and they are not the customer deployment handoff.

## Customer Owns

- Adapter implementation on target hardware
- BSP, drivers, build, flash, and packaging
- Deployment orchestration and observability wiring
- Evidence that the target implementation matches BitTrace parity expectations
- Operational policy rollout using one of the two shipped persistence profiles without changing the policy logic

## Boundary Rule

BitTrace ships the frozen stable lane, the verify/golden-vector path, and the fixed persistence definitions. The customer ships the target-specific integration. No hidden preprocessing, target-only feature logic, experimental-lane substitution, alternate frontend/backend pairing, or reinterpretation of the stable model is part of the supported deployment boundary.
