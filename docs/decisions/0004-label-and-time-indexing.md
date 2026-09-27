# 0004: Label and time indexing scheme, and label knowledge time

- Status: ACCEPTED (Phase 0 review), revised in the Phase 0 revision pass (`known_at` made load-bearing,
  default latency distribution, logins as a timed event source); round 2: label maturity and the
  ground-truth-only evaluation rule (details in 0009), simplification numbering S1/S2/S4
- Scope: generator outputs (`labels`, `event_labels`, `ground_truth_networks`), as-of label access,
  splits (decision 0009), targets, training-example construction

## Problem

Three things are separated:

1. **State labels**: what the synthetic ground truth says an entity is doing at time t (role, phase).
2. **Prediction targets**: future outcomes relative to a prediction point (e.g. a high-risk event in (t, t+H]).
   They look forward by definition; splits purge a gap of at least `H_max`.
3. **Label knowledge time `known_at`**: when an institution would plausibly know a label. A label may be
   *used* (as a training target, or as an input feature such as "a counterparty was previously flagged")
   at cutoff c only if `known_at <= c`. The purge gap alone does not enforce this.

## Time

- int64 **seconds** since the simulation epoch `t0`; `t0` maps to a configured calendar anchor (a Monday,
  00:00, one fixed UTC offset) used only for seasonality. No timezone-aware datetimes in canonical tables.
- Intervals are half-open `[valid_from, valid_to)`; open-ended intervals use `sim_end`.
- Total event order `(ts, event_id)` (decision 0003) across **both** event tables, `transactions` and
  `logins`. Login events are the timed source for LOGIN_WITH and USES_IP edges and for the NEW_DEVICE_LOGIN
  and NEW_IP_LOGIN event types (derived through the as-of view, not stored as flags).

## Tables

### `labels` (interval table; one row per entity x scenario x phase segment)

`entity_type, entity_id, valid_from, valid_to, risk_label, role_label, scenario_id, network_id,
family, phase, is_ood_family, known_at (nullable)`

- Before its first scenario activity an entity has no scenario row; it is NORMAL.
- Role at t is derived from the phase at t. Milestone 1 forbids membership in more than one scenario
  instance at a time (S1), and in fact in at most one instance for the whole simulation (S2, decision 0002), so
  the point-in-time role is single-valued. Role changes *within* one network (e.g. AGGREGATOR then DISTRIBUTOR) are
  still represented; role changes *across* scenarios (e.g. VICTIM_LIKE, later RELAY) are not, until S2 is relaxed.
- `known_at` is null when the network is never detected (see below).

### `event_labels` (one row per scenario event, transactions and logins)

`event_id, event_table, scenario_id, network_id, family, phase, event_type, is_high_risk, is_terminal,
is_ood_family, known_at (nullable)`

- `is_terminal` marks the terminal high-risk event per network; at most one per network. Networks truncated
  by `sim_end` have none and are **censored**.

### `ground_truth_networks`

`network_id, family, instance_idx, is_ood_family, start_ts, terminal_ts (nullable), censored, n_members,
detected, known_at (nullable), instance parameters (amount scale, duration, counterparty count, ...)`,
plus a long table `network_members(network_id, entity_type, entity_id, first_role, joined_ts)`.

### Phases

Configurable per family, default `SETUP, INFLOW, HOLD, MOVEMENT, EXIT, INACTIVE`, with **stochastic**
transitions (per-family Markov chain; probabilities and duration distributions in config).

## Label knowledge model (how `known_at` is generated)

**Decision: network-level retrospective detection.**

For each network n (its own RNG stream, decision 0002):

1. With probability `p_never_known` (default **0.10**) the network is never detected: `known_at = null` for
   all its rows.
2. Otherwise draw a latency `L_n` from the configured distribution. Default:
   **LogNormal with median 14 days and sigma 0.75** (in log-days), which gives roughly a 5th percentile of
   4 days and a 95th of 48 days.
3. Anchor `A_n` = timestamp of the terminal event, or of the last scenario event if the network is censored.
4. Detection time `D_n = A_n + L_n`. If `D_n >= sim_end`, the network is not detected within the simulation
   (`known_at = null`, `detected = false`).
5. Every `labels`, `event_labels` and `ground_truth_networks` row of n gets `known_at = D_n`. Hence for every
   labeled event `known_at - event_time >= L_n > 0`.

Why this model:

- It is coherent: an investigation completes once and labels the network's history retrospectively, so no
  label can be known before the behavior it describes, and there are no partially known networks whose
  pattern would be inconsistent.
- It is non-degenerate by construction and makes the `known_at` path matter: with a 90-day simulation and a
  training period ending around day 55 to 60, networks that finish late in the training period are *not* yet
  labeled at the training cutoff.
- The default values are **assumptions, not calibrated to any real institution** (no verified source for
  investigation latencies was consulted). Median 14 days was chosen so that a meaningful share of training-period
  networks is known at the training cutoff in a 90-day simulation; a longer median (e.g. 60 days) would leave
  almost no known positives in DEV. How many training positives are lost is NOT YET EVALUATED and will be
  reported by the Milestone 1 EDA.

**Fail-loudly rules in the config schema (decision 0005):** the latency distribution must have
median >= 1 day and log-sigma >= 0.1; `p_never_known` must be in [0, 0.5]. A constant or near-zero latency
is rejected. The only way to get zero latency is an explicit `label_knowledge.regime: oracle` flag intended
for ablations (e.g. measuring how much delayed labels cost), which is recorded in metadata and in every result
that uses it.

## Enforcement (where `known_at` is filtered)

Every consumer filters on `known_at <= cutoff` in addition to the event-time rules:

| Consumer | Rule |
|---|---|
| `AsOfView.known_labels()` (decision 0003) | returns rows with `known_at` not null and `known_at <= view cutoff`; there is no argument to relax it |
| Training-example construction (decision 0009) | a training example's *label* is computed only from label rows with `known_at <= T_fit`, where `T_fit` is the end of the training window (after the purge gap is removed) |
| Walk-forward / rolling splits (later) | the same rule per fold, with that fold's `T_fit` |
| Features that use labels (e.g. "counterparty previously flagged") | only through `known_labels()` at the feature's own cutoff t |
| Evaluation-time features | same as features: `known_at <= t` |
| Evaluation *targets* (VAL, TEST, OOD) | **always** the full ground truth via `OracleTargets`, never `known_at`-limited labels; metric functions reject any other target type; required test in 0009 |

### How "not yet known" examples are treated in training

An (entity, t) example whose oracle target is positive but whose label is not known at `T_fit` is **not a
positive training example**. It stays in the training set with target 0 (unlabeled), exactly as a real
institution would see it (a positive-unlabeled setting). It is *not* dropped on the basis of its label: deciding
that would require the oracle, which would itself be a leak. The count of such latent positives is reported as a
diagnostic from `OracleLabels` in evaluation code only.

This noise is recency-dependent, so it is made explicit and controllable through **label maturity** (decision 0009):
`training_label_mode = include_immature` (default, keeps the forced-0 labels) or `mature_only` (ablation, drops TRAIN
examples whose target window ended less than `maturity_horizon_days` before `T_fit`). Maturity depends only on
`T_fit`, `t` and `H`, never on `known_at`, so dropping immature examples is not a leak.

### Required test (Milestone 1)

Construct a tiny dataset with one network whose events are at day 10, whose terminal event is at day 20 and
whose `known_at` is day 70 (explicit latency, oracle regime disabled). With `T_fit` = day 60:

- the network's (entity, t) examples appear in no training set as positives, and `known_labels()` at any
  cutoff < day 70 does not return its rows;
- a label-derived feature for a counterparty at day 65 is 0;
- with `T_fit` = day 75 the same examples are positives and the feature at day 72 is 1;
- a sibling network with `known_at` = day 30 is a positive at `T_fit` = day 60 (control case).

Plus a property test: for random datasets and random cutoffs, no label row with `known_at > cutoff` or with null
`known_at` ever influences any training target or feature value (metamorphic: changing such rows' roles, phases or
membership leaves all training targets and features bit-identical).

## Trade-offs

- Retrospective network-level detection is one simplification among several possible knowledge models (per-event
  alerts, partial investigations). It is configurable in shape but not yet in *kind*; alternatives are future work.
- Treating unknown positives as 0 lowers training signal and adds label noise; that is the realistic setting and the
  point of the exercise. The `oracle` regime exists to measure the cost.

## Rejected

- Final-role point labels (violates `CLAUDE.md` 2.2).
- Per-row independent latencies (produces partially known networks and, for late phases, labels known before
  the behavior ends).
- Zero or constant latency as default (would make `known_at` decorative).
- Dropping not-yet-known positives from training (requires oracle knowledge).
