# 0009: Chronological splits and the OOD scenario-family holdout pool

- Status: ACCEPTED (round-1 revision review); extended in round 2 with label maturity, the two training-label
  modes and the ground-truth-only evaluation rule
- Scope: split utility (Milestone 1), training-example construction and evaluation (Milestone 3 onward)

## Problem

Two different generalization questions must be measurable separately:

1. **Temporal generalization**: does a model trained on the past work on the future (chronological test split)?
2. **Pattern generalization**: does it detect suspicious *families it never saw labeled* (OOD families)?

The first assessment offered two options for held-out families: confine them to after the validation boundary,
or let them occur anywhere and mask them. Both were rejected in review: the first conflates the two questions
(OOD families would only ever be tested in the future), the second was underspecified.

## Decision

A subset of scenario families is designated the **OOD holdout pool** (`is_ood_family: true` per family in config).
Their instances occur **across the full simulation timeline**, exactly like other families, but they are excluded
from the training and validation example sets **by family label**, and appear only in a separate OOD evaluation pool.

## Definitions

- **Prediction point**: `(entity, t)`; cadence configured (daily snapshot in Milestone 1 probes).
- `H_max`: longest label horizon. Boundaries `train_end < val_end < sim_end` configured in days.
- **In-distribution (ID) entity**: an entity that is not a member of any OOD-family network.
- **OOD entity set** `E_ood`: all members (any role, including VICTIM_LIKE) of OOD-family networks, determined from
  `OracleLabels`. Using the oracle here is part of the *experimental protocol*, not a model input: it decides which
  rows exist in which set, never a feature value. This is the only place outside evaluation where split code reads
  the oracle, and it is covered by the import-boundary allow-list.
- **Negative reference pool** `E_ref`: a fixed fraction (default 10%) of normal-population entities, selected by stable
  hash (decision 0002), that never become scenario members and are excluded from TRAIN and VAL. It supplies clean,
  never-trained-on negatives for the OOD evaluation (see below).

## Reference pool: method and validation (A2, Milestone 1 checkpoint 2)

`E_ref` is drawn only from the recruitment partition's NONE cell (accounts no scenario can ever recruit, decision
0002), so it can never contain a scenario member. The partition is a keyed hash of the account ID and does not look
at archetypes, but a hash sample can still deviate by chance, and a skewed pool would make OOD negatives easier or
harder than the population. Two methods are implemented (`splits.ref_pool_method`):

- `none_cell_hash`: a plain keyed-hash sample of the NONE cell, 10% of normal accounts in expectation;
- `stratified_archetype`: the same hash ranking taken per archetype, so the pool matches the population's archetype
  shares exactly (archetypes come from the oracle table `person_truth`; this is protocol use, never a feature).

**Decision rule, fixed before looking (EXPERIMENTS.md, EXP-M1-G freeze item 6):** use `stratified_archetype` if any
archetype share of the hash pool differs from the population by more than 1.0 percentage point, or a chi-square
goodness-of-fit test gives p < 0.01.

**Measured on RESEARCH seed 1000** (100,000 normal accounts; `reports/a2_reference_pool_seed1000.json`):

| Archetype | Population share | Hash pool share | Difference (pp) | Stratified pool share |
|---|---|---|---|---|
| family | 13.82% | 13.46% | -0.37 | 13.82% |
| hf_merchant | 3.07% | 2.95% | -0.12 | 3.07% |
| high_volume_business | 1.05% | 0.99% | -0.05 | 1.05% |
| salary_worker | 40.15% | 41.15% | **+1.0005** | 40.15% |
| small_business | 8.08% | 7.83% | -0.25 | 8.08% |
| spending_surge | 11.90% | 11.53% | -0.36 | 11.90% |
| student | 12.00% | 12.10% | +0.10 | 12.00% |
| traveler | 9.94% | 9.99% | +0.05 | 9.94% |

Hash pool: 10,152 accounts, chi-square 6.26, p = 0.51 (no evidence of systematic skew), but the salary_worker share
is 1.00045 pp above the population, just over the 1.0 pp threshold. **By the pre-registered rule the gate run uses
`stratified_archetype`** (10,001 accounts, largest deviation 0.008 pp). The margin is negligible and the chi-square
test gives no evidence of bias; the switch follows the rule, not a judgment that the hash pool was biased. The
stratified method is the default from checkpoint 2 on.

## Example sets

| Set | Entities | Prediction times t | Targets / labels |
|---|---|---|---|
| TRAIN | ID entities, minus `E_ref` | `t <= train_end - H_max` (purge: every target window ends by `train_end`); in `mature_only` mode additionally `is_mature` (below) | computed only from labels with `known_at <= T_fit = train_end` (decision 0004) |
| VAL | ID entities, minus `E_ref` | `train_end <= t <= val_end - H_max` | oracle targets for model selection; label-derived features use `known_at <= t` |
| TEST (ID) | ID entities, minus `E_ref` | `val_end <= t <= sim_end - H_max` | oracle targets |
| OOD | `E_ood` positives and negatives from `E_ref` | all t in `[t_min, sim_end - H_max]`, reported by time block (train / val / test period) | oracle targets |

Gaps: consecutive sets are separated by at least `H_max` of prediction time, because each set's last prediction time is
`H_max` before its boundary. The config validator rejects `H_max` larger than any period.

Additional rules:

- `E_ood` entities are excluded from TRAIN and VAL **for the whole timeline**, including before recruitment and after
  the network ends. Pre-recruitment points would carry positive forward-looking targets for the OOD pattern; excluding
  the entity entirely is the fail-closed choice.
- **OOD labels are never visible as known labels**: `AsOfView.known_labels()` never returns rows with
  `is_ood_family = true`, at any cutoff. In this experimental design, an OOD family is a pattern the institution has
  never labeled. Its events stay in the world (they are transactions like any other and appear in neighbors' histories);
  only the labels are withheld.
- The OOD pool is **never** used for model selection, early stopping, threshold choice or calibration. Only VAL is.
- The ID TEST set does not contain OOD entities, so ID test metrics are not diluted by OOD positives, and OOD metrics are
  not inflated by ID positives.

## Label maturity and training-label modes

### Why

Under decision 0004, a TRAIN example whose true target is positive but whose label is not yet known at `T_fit` is
labeled 0. The probability of that happening depends on how recent the example is: an example whose outcome occurred
shortly before `T_fit` is very likely mislabeled, one far in the past much less so. This recency-dependent label noise
must be explicit and controllable, not silently absorbed.

With the default latency (LogNormal, median 14 days, sigma 0.75, `p_never_known` 0.10), computed exactly in this
session, the probability that a network's label is still unknown `d` days after its anchor event is:

| d (days) | 3 | 7 | 14 | 28 | 48 | 60 |
|---|---|---|---|---|---|---|
| latency only | 0.980 | 0.822 | 0.500 | 0.178 | 0.050 | 0.026 |
| including `p_never_known` | 0.982 | 0.840 | 0.550 | 0.260 | 0.145 | 0.124 |

(latency quantiles: p50 14.0, p90 36.6, p95 48.1, p99 80.1 days)

### Definitions

- `maturity_horizon_days` (config, default **48**, about the 95th percentile of the default latency distribution).
- For a TRAIN example `(entity, t)` with horizon `H`: `reference_time = t + H`, the end of the target window, i.e. the
  latest time the outcome being predicted could have occurred.
- `is_mature = (T_fit - reference_time) >= maturity_horizon_days`.
- `is_mature` uses only `T_fit`, `t` and `H`, all known to the modeler at fit time. It never reads `known_at` or the
  oracle, so selecting on it is not a leak.

### Two modes (both implemented in the Milestone 1 split utility)

`training_label_mode` (config):

1. **`include_immature`** (default): all TRAIN examples are kept, immature ones with their forced-0 labels where the label
   is unknown. This matches what an institution would actually train on.
2. **`mature_only`** (ablation): TRAIN examples with `is_mature = false` are dropped. Targets of the remaining examples are
   still computed from labels with `known_at <= T_fit`; maturity lowers the chance that a positive is still unknown but
   does not remove it.

Both modes produce the same VAL, TEST and OOD sets; only TRAIN differs. The `SplitIndex` records the mode,
`maturity_horizon_days`, and per-example `is_mature`, so both modes can be built from one dataset and one index.

The difference in performance between the two modes is an **experiment to report** (see `docs/EXPERIMENTS.md`,
EXP-LM, planned), not something to hide inside one dataset choice.

### Two limits of maturity, stated plainly

1. **Mature does not mean correctly labeled.** A share `p_never_known` of networks (default 10%) is never known, so even
   fully mature positives from those networks stay 0. Also, the latency is anchored at the network's *terminal* event
   (decision 0004), which can be later than `reference_time` when the predicted outcome is an intermediate high-risk event,
   so the real delay for such examples is longer than the maturity rule assumes. The residual rate of unknown positives among
   mature examples is therefore not zero; it is measured with the oracle, in evaluation code only, and reported.
2. **At a 90-day simulation the default maturity horizon leaves almost nothing to train on.** Computed for the default
   split (`train_end` = day 60, `H_max` = 7 days, maturity 48 days): training prediction times run from day 0 to 53, mature
   ones only from day 0 to 5, about 9% of the training window, and those have at most 5 days of history. For a 180-day
   simulation with `train_end` = day 120 it would be 58%; for 365 days with `train_end` = day 240, 79%.
   Consequences:
   - the config validator **raises** if `mature_only` leaves fewer than `min_mature_train_days` (default 21) days of
     training prediction times, instead of silently producing a tiny training set;
   - the maturity ablation needs a longer simulation (proposed: a `RESEARCH_LONG` profile with 180 days) or a sweep over
     `maturity_horizon_days` in {14, 28, 48}. This is open question Q-M1 in the assessment.

## Evaluation rule: ground truth only

**All metrics are computed against the full synthetic ground truth (`OracleLabels`), never against `known_at`-limited
labels.** This holds for VAL (model selection, thresholds, calibration), TEST and OOD. The designer holds the ground truth;
scoring predictions against it passes no information into the model, it only checks honestly whether the predictions were
right. `known_at` limits what the *model* may learn from and use as input, never what it is graded against.

Enforcement:

- Metric functions accept targets only as an `OracleTargets` object (a distinct type produced by the target builder from
  `OracleLabels`); passing known-label objects or raw arrays raises `TypeError`. Known-label objects are a different type
  with no conversion path.
- **Required test (Milestone 1):** a tiny dataset where a large share of TEST positives belong to networks with `known_at`
  null or after `sim_end`. A "perfect" scorer that assigns score 1 to every true positive and 0 elsewhere must obtain
  AP = 1.0 and recall = 1.0. If evaluation code substituted `known_at`-filtered labels, those positives would be counted as
  negatives with top scores and AP would drop below 1, so the test would fail. A second assertion checks that the number of
  positives reported by the evaluator equals the oracle count, not the known count.

## Reporting (2 x 2)

|  | train period times | test period times |
|---|---|---|
| **ID families** | (in-sample, diagnostics only) | ID TEST: temporal generalization |
| **OOD families** | OOD-train-block vs `E_ref` negatives: pattern generalization without time shift | OOD-test-block vs `E_ref` + ID TEST negatives: both shifts |

The `E_ref` pool is what makes the OOD-train-block comparison clean: without it, OOD positives in the training period
would be ranked against negatives the model was trained on, which inflates precision.

## Split utility interface (to implement in Milestone 1)

`build_example_index(store, oracle, split_cfg) -> SplitIndex` returning `train, val, test, ood_by_block, ref_negatives`
as arrays of `(entity_id, t)`, plus `T_fit`, `training_label_mode`, `maturity_horizon_days` and a per-TRAIN-example
`is_mature` flag. It asserts, and raises on violation:

- `E_ood` ∩ entities(TRAIN ∪ VAL) = ∅; `E_ref` ∩ entities(TRAIN ∪ VAL) = ∅; `E_ref` ∩ `E_ood` = ∅;
- time bounds per set, as above;
- every TRAIN target was computed from labels with `known_at <= T_fit`;
- deterministic output for a given dataset and config (content hash of the index recorded).

## Required tests (Milestone 1)

- OOD exclusion: an OOD-family network whose members transact with ID accounts; assert none of its members appear in
  TRAIN or VAL at any t, they appear in the OOD pool in all time blocks, and ID neighbors' features still see the
  transactions but never the labels.
- `known_labels()` never returns an OOD-family row, even at `sim_end`.
- Purge: no TRAIN target window overlaps VAL prediction times; same for VAL and TEST.
- `known_at` test from decision 0004.
- Maturity: `is_mature` computed from `T_fit`, `t`, `H` only (metamorphic: changing `known_at` values leaves `is_mature`
  unchanged); `mature_only` drops exactly the immature TRAIN examples and leaves VAL/TEST/OOD identical; the validator
  raises when the mature training window is shorter than `min_mature_train_days`.
- Ground-truth-only evaluation test above.
- Disabling the OOD flag on a family moves its entities back into TRAIN/VAL/TEST and changes nothing else in the
  generated data (the flag affects splits only, not generation).

## Trade-offs

- Excluding OOD entities and `E_ref` removes some normal-looking training data (bounded by the `E_ref` fraction and
  the OOD share of prevalence).
- With 1% prevalence and a few OOD families, the number of OOD networks in DEV will be very small; OOD metrics are only
  meaningful at RESEARCH scale (see the power note in the assessment).
- Real institutions do see labels of every pattern they detect; withholding OOD labels is a deliberate experimental
  construction and is documented as such in `LIMITATIONS.md`.
