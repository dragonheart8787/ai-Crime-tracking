# 0009: Chronological splits and the OOD scenario-family holdout pool

- Status: PROPOSED in the Phase 0 revision pass (replaces the held-out-family options in the first
  assessment, which were both rejected in review)
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

## Example sets

| Set | Entities | Prediction times t | Targets / labels |
|---|---|---|---|
| TRAIN | ID entities, minus `E_ref` | `t <= train_end - H_max` (purge: every target window ends by `train_end`) | computed only from labels with `known_at <= T_fit = train_end` (decision 0004) |
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

## Reporting (2 x 2)

|  | train period times | test period times |
|---|---|---|
| **ID families** | (in-sample, diagnostics only) | ID TEST: temporal generalization |
| **OOD families** | OOD-train-block vs `E_ref` negatives: pattern generalization without time shift | OOD-test-block vs `E_ref` + ID TEST negatives: both shifts |

The `E_ref` pool is what makes the OOD-train-block comparison clean: without it, OOD positives in the training period
would be ranked against negatives the model was trained on, which inflates precision.

## Split utility interface (to implement in Milestone 1)

`build_example_index(store, oracle, split_cfg) -> SplitIndex` returning `train, val, test, ood_by_block, ref_negatives`
as arrays of `(entity_id, t)`, plus the `T_fit` for TRAIN. It asserts, and raises on violation:

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
- Disabling the OOD flag on a family moves its entities back into TRAIN/VAL/TEST and changes nothing else in the
  generated data (the flag affects splits only, not generation).

## Trade-offs

- Excluding OOD entities and `E_ref` removes some normal-looking training data (bounded by the `E_ref` fraction and
  the OOD share of prevalence).
- With 1% prevalence and a few OOD families, the number of OOD networks in DEV will be very small; OOD metrics are only
  meaningful at RESEARCH scale (see the power note in the assessment).
- Real institutions do see labels of every pattern they detect; withholding OOD labels is a deliberate experimental
  construction and is documented as such in `LIMITATIONS.md`.
