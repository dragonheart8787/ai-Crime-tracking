# 0004: Label and time indexing scheme

- Status: PROPOSED (awaiting review, Phase 0)
- Scope: generator outputs (`labels`, `event_labels`, `ground_truth_networks`), splits, targets

## Problem

Labels must be time-indexed so that a role or risk label at t reflects the scenario phase at t,
never the entity's final role (`CLAUDE.md` 2.2). Three different things are usually conflated
and must be separated:

1. **State labels**: what the synthetic ground truth says an entity is doing at time t
   (role, phase). Known to the generator, not to an observer.
2. **Prediction targets**: future outcomes relative to a prediction point, e.g. "entity has a
   high-risk event in (t, t+H]". They look forward by definition, so they are built by a
   dedicated target builder, and splits purge a gap of at least H.
3. **Label knowledge time**: when an institution would plausibly *know* a label (after
   investigation). Using labels at training time that were not yet known is a subtle leak that
   the spec does not currently address.

## Options

- **A. Point labels per entity** (final role). Violates the spec.
- **B. Per-event labels only.** Cannot express "dormant but already recruited" or role changes.
- **C. Interval labels + event labels + knowledge time.**

## Recommendation

**C.**

### Time

- int64 **seconds** since a simulation epoch `t0`. `t0` maps to a configured calendar anchor
  (a Monday, 00:00, single fixed UTC offset) used only for weekday / month-day seasonality.
  No timezone-aware datetimes in core tables; human-readable datetimes appear only in reports.
- Intervals are half-open `[valid_from, valid_to)`. Open-ended intervals use `sim_end`, not null.

### `labels` (interval table; one row per entity x scenario x phase segment)

`entity_type, entity_id, valid_from, valid_to, risk_label, role_label, scenario_id,
network_id, phase, known_at`

- Before an entity's first scenario activity, it has **no** scenario label (it is NORMAL), even if
  it will later be recruited. Being "pre-recruitment" is not observable behavior.
- Role at t is derived from the phase at t (e.g. an account can be AGGREGATOR during INFLOW and
  DISTRIBUTOR during MOVEMENT). An entity in two concurrent scenarios has two rows; the
  point-in-time role is multi-label, and single-label views use a configured precedence. (Open
  question for review.)
- `known_at` = `valid_from` + a sampled investigation delay (configurable distribution, possibly
  "never" for undetected networks). Default policy for training (Milestone 3+): a label may be used
  as a supervised target only if `known_at <= fit_time`. For evaluation the oracle label is used.

### `event_labels` (one row per scenario event)

`event_id, scenario_id, network_id, phase, event_type (taxonomy), is_high_risk, is_terminal`

- `is_terminal` marks the terminal high-risk event per network (e.g. the final cash-out-like
  withdrawal in an account-to-account-to-cash chain). Exactly one per network instance that reaches
  termination; networks truncated by `sim_end` have none and are **censored** (recorded in
  `ground_truth_networks`).

### `ground_truth_networks`

`network_id, scenario_family, instance_idx, split_role (train_eligible | held_out),
start_ts, terminal_ts_or_null, censored, n_accounts, member_entity_ids`
(members possibly as a separate long table for Parquet friendliness).

### Phases

An abstract, configurable phase vocabulary per family, default:
`SETUP, INFLOW, HOLD, MOVEMENT, EXIT, INACTIVE`. Phase transitions are **stochastic**
(a small per-family Markov chain with configured probabilities and duration distributions), so
the next phase is not deterministic given the current one (see Challenge 2 in the assessment).

### Targets and splits

- Horizon set `H` is configured; the longest horizon `H_max` sets the purge gap.
- Split boundaries are defined on prediction times. Training rows require `t + H_max <= train_end`
  so that no training target looks into the purge gap or validation period.

## Trade-offs

- Interval labels are more complex than per-entity labels and need their own invariants
  (non-overlap per entity x scenario, `valid_from < valid_to`, phases ordered consistently with
  event timestamps). All testable.
- `known_at` introduces a modeling assumption (investigation delay) that is not grounded in
  real data; it is configurable and can be set to zero to reproduce the "oracle labels" regime,
  making its effect measurable rather than hidden.
