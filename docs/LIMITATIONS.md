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

## Measured at Milestone 1 checkpoint 1 (DEV profile, seed 42)

- **DEV prevalence is above target.** DEV forces at least one instance of every family (`min_instances: 1`) so that
  all families are exercised; at 2,500 accounts this gives a realized prevalence of 1.84% against the 1% target
  (46 member accounts in 11 networks). DEV is not used for statistics.
- **Settlement couples untouched accounts to scenario changes.** Disabling one family leaves generation intents of
  unrelated entities identical, but 4 to 19 of about 2,490 untouched internal accounts (their direct
  counterparties) see different balances and 1 to 13 see a changed status on some event (decision 0002, measured
  table). Shared external sink balances (card network, cash) differ throughout.
- **Declines are common for some archetypes.** Outgoing declined share per archetype ranges from 1.0% (hf_merchant)
  to 6.4% (spending_surge); overall 4.8% of transactions are declined. These values come from untuned placeholder
  parameters, not calibration.
- **Scenario-created accounts have no normal activity.** When a recruitment cell has too few eligible accounts, a
  scenario opens new accounts that only carry scenario events. No DEV seed-42 instance needed this (shortfall share
  0.0), but it can happen in small profiles and would make those accounts easy to spot.

## Measured at Milestone 1 checkpoint 2 (RESEARCH profile, calibration seeds 1000 to 1004)

- **Delayed labels remove many training positives.** On seed 1000, 463 of 1,077 TRAIN ground-truth positives (43%)
  are not yet known at `T_fit` and would be 0 under the realistic training regime (R15). With the 48-day maturity
  horizon, 1,039,074 of 4,639,845 TRAIN points (22%) are mature.
- **The Milestone 1 probe target is very sparse.** "Active scenario phase at the daily snapshot" is positive for 0.03% to
  0.05% of points; label intervals are short (median 6 h HOLD to 31 h INFLOW). Probes pass the pre-registered floor, but
  some narrowly; see EXP-M1-G.
- **Per-account statics are weak here, largely because of confounders.** Businesses (about 9% of accounts) account for
  61% to 85% of the probe's top false positives. The pre-registered context features did not show a consistent gain for
  a linear probe (G3); this says nothing yet about graph models.
- **Timing heterogeneity is small.** Mean hour of outgoing activity is 12.4 to 13.5 and the weekend share 0.28 to 0.29
  for every archetype; archetypes differ in volume, amounts, channels and counterparties, not in daily rhythm.
- **Carrier-NAT IPs are very widely shared.** The busiest IP is used by 12,768 accounts (10 carrier-NAT addresses per
  region). This makes shared IP a weak signal, as intended, but is more extreme than a real carrier pool would look.
- **Memory.** A RESEARCH generation peaks at about 11.9 GB and a gate run at about 10.5 GB, close to the 15 GB limit of
  the cloud sandbox used here; runs must be sequential there.
