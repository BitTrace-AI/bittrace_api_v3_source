# Deployment Boundary

## BitTrace Owns

- Canonical model behavior for `temporal_threshold_36` + `Lean-Lean`
- Artifact contracts and freeze/export artifacts from the canonical campaign lane
- Golden-vector and parity verification tooling for completed canonical runs
- The two supported persistence profile definitions:
  - `quiet_scout`: `i1 / d1 / y6 / r12 / no-latch`
  - `aggressive`: `i1 / d1 / y3 / r7 / no-latch`
- Deterministic source-lane documentation, smoke workflow, and supported CLI surface

## Customer Owns

- Adapter implementation on target hardware
- BSP, drivers, build, flash, and packaging
- Deployment orchestration and observability wiring
- Evidence that the target implementation matches BitTrace parity expectations
- Operational policy rollout using one of the two shipped persistence profiles without changing the policy logic

## Boundary Rule

BitTrace ships the frozen model, the verify/golden-vector path, and the fixed persistence definitions. The customer ships the target-specific integration. No hidden preprocessing, adaptive target-only feature logic, research-lane substitution, alternate frontend/backend pairing, or customer-side model reinterpretation is part of the supported deployment boundary.
