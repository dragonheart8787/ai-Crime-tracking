# Limitations

Status: Phase 0 (design only). This file lists known limitations of the *design*. It will grow with measured
limitations once there is code and data. Nothing here has been evaluated unless stated.

## Scope of evidence

- All primary data is synthetic. Every result is conditional on the generator and its configured regime; nothing
  established here transfers to real financial data without external validation.
- "Plausible" confounders are designed look-alikes, not calibrated to real populations.
- Calibrated probabilities are calibrated to the generator's prevalence and are meaningful only for comparing methods
  within this project.

## Named Milestone 1 simplifications

| ID | Simplification | Known bias | When it must be revisited |
|---|---|---|---|
| S1 | No concurrent scenario membership: an account is in at most one active scenario at a time | Networks never overlap in time through a shared account, so network separation is cleaner than in reality; network recall and alert compression are optimistic | Before claims about network recovery in overlapping settings |
| S2 | Single-scenario-for-life: an account is in at most one scenario instance over the whole simulation (enforced by the recruitment partition, decision 0002) | Networks are vertex-disjoint; no account changes role *across* scenarios (e.g. VICTIM_LIKE in one network, later RELAY in another) | **Must be relaxed before any experiment that tests tracking of role changes over time.** It is a Milestone 1 convenience, not a permanent property. Any such experiment's pre-registration must state that S2 is off |
| S3 | Single currency per dataset, no per-row currency column | No FX behavior, no currency-based signals | When multi-currency behavior is in scope |
| S4 | Network-level retrospective label knowledge: a network's labels become known all at once, after its terminal (or last) event | No partially known networks, no labels known during an ongoing network | When label-knowledge models are compared |

## Label knowledge (`known_at`)

- **The default detection-delay model is an uncalibrated assumption.** Latency is LogNormal with median 14 days and
  log-sigma 0.75, and 10% of networks are never detected (`p_never_known = 0.10`). These values were chosen so that a
  90-day simulation retains known positives in its training period; they are not derived from any real institution or
  verified source.
- **A sensitivity analysis over the delay distribution's parameters (median, sigma, `p_never_known`) is planned, not yet
  run.** NOT YET EVALUATED.
- Unknown positives enter training as 0 (positive-unlabeled). The resulting label noise depends on recency; label maturity
  (decision 0009) makes it controllable, but "mature" examples still contain unknown positives (never-detected networks, and
  outcomes whose network terminates later). The residual rate will be measured, not assumed.
- With a 90-day simulation and the default 48-day maturity horizon, the `mature_only` training mode keeps only about 9% of
  the training window (computed in Phase 0); the maturity ablation needs a longer simulation or a shorter horizon.

## OOD evaluation design

- OOD families' labels are withheld from the model for the whole simulation by construction. Real institutions do see labels
  of patterns they detect; this is a deliberate experimental construction to measure pattern generalization.

## Other

- DEV profile (about 2,000 to 3,000 accounts, 90 days, 1% prevalence) contains only tens of suspicious accounts; it is for
  tests and fast iteration, never for statistics.
- GPU environment (torch / CUDA / PyG on the Windows laptop) is NOT YET EVALUATED.
