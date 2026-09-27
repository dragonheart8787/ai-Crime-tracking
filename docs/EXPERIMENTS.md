# Experiments: pre-registrations and results

Rules (`CLAUDE.md` 2.3): every experiment is registered here **before** it is run: hypothesis, metrics,
comparison, success/failure criteria. Results are appended afterwards, including negative and null results.
The protocol is never changed to rescue a hypothesis; any change after results exist is logged as a deviation.

Standing rules for every experiment in this file:

1. **Evaluation uses the full synthetic ground truth only.** Every metric (VAL, TEST, OOD) is computed against
   `OracleLabels` via `OracleTargets`, never against `known_at`-limited labels. `known_at` restricts what the model may learn
   from and use as input, not what it is graded against (decision 0009).
2. **Training-label mode is always stated.** Every model experiment records `training_label_mode`
   (`include_immature` default, or `mature_only`) and `maturity_horizon_days`. Results from the two modes are never pooled.
3. **Simplifications are stated.** Every pre-registration lists which named simplifications (S1 to S4, `LIMITATIONS.md`)
   are in force. An experiment about role changes over time cannot be registered while S2 is in force.
4. **References to the original spec are resolved by topic name, not section number**, because the original spec's
   numbering was collapsed when `CLAUDE.md` was written and the two schemes no longer correspond.

| ID | Title | Status |
|---|---|---|
| EXP-M1-G | Milestone 1 non-triviality gates | PRE-REGISTERED DRAFT (Phase 0 revision); to be frozen before the first gate run |
| EXP-LM (planned) | Label-maturity ablation: `include_immature` vs `mature_only` | NOT REGISTERED YET (note only, below) |
| (planned) | Prevalence sweep for distribution shift | NOT REGISTERED YET (note only, below) |
| (planned) | Detection-delay sensitivity analysis | NOT REGISTERED YET (note only, below) |

---

## EXP-M1-G: Milestone 1 non-triviality gates

### Purpose

This is a **data-quality check**, not a research hypothesis. It checks that the generated data is neither trivially
separable by simple per-account statistics (ceiling gates) nor devoid of learnable signal (floor gate). Graph and temporal
context contribution is **reported, not gated**, because gating on it would tune the generator toward the project's own
hypothesis (Challenge 1 in `PHASE0_ASSESSMENT.md`).

### Data

- Generator version: the Milestone 1 candidate under test (git commit recorded per run).
- Profile: **RESEARCH** (about 100K accounts, `suspicious_prevalence = 0.01`). If RESEARCH generation takes more than
  15 minutes on the laptop, a `GATE` profile with 25K accounts and otherwise RESEARCH settings is used instead. This choice is
  made on runtime alone, before any gate result is seen, and recorded. DEV (a few thousand accounts, about 25 to 30 suspicious
  accounts) is too small for these estimates and is never used for gates.
- **Calibration seeds 1000, 1001, 1002, 1003, 1004.** The locked research seeds (42, 43, 44) are not generated until all gates
  pass and the generator is frozen.
- Splits per decision 0009: probes are fit on TRAIN and scored on VAL. OOD families and `E_ref` are excluded from both.

### Prediction points and target

- Daily snapshots at 00:00, for every internal account that is open at t and satisfies the TRAIN/VAL rules of decision 0009.
- Target `y(a, t) = 1` if the oracle label of account a at t is in a scenario phase other than `INACTIVE` (any role), else 0.
- **Probe training uses oracle targets** (label-knowledge regime `oracle` for this experiment only). This makes the probes as
  strong as possible, which is the conservative choice for the ceiling gates. It is not the regime used for model milestones.
- `pi` = empirical share of positive points in the VAL set (point prevalence; lower than the 1% account prevalence because
  accounts are positive only during their active phases). Reported per seed.

### Pre-registered single-account feature set

All features are computed through the as-of view at t (decision 0003) over a trailing window `W` in {7 days, 30 days}, from
**settled** transactions unless stated. "Transfers" means account-to-account transfers (TRANSFER_TO edges).

| # | Feature | Definition |
|---|---|---|
| F1 | `total_amount` | sum of `amount_minor` over all settled transactions (in and out) involving a in W. **(required)** |
| F2 | `total_amount_in` | sum of settled credits to a in W |
| F3 | `total_amount_out` | sum of settled debits from a in W |
| F4 | `txn_count` | number of settled transactions involving a in W. **(required)** |
| F5 | `in_degree` | number of distinct accounts that sent a transfer to a in W. **(required)** |
| F6 | `out_degree` | number of distinct accounts that received a transfer from a in W. **(required)** |
| F7 | `avg_holding_time` | amount-weighted mean over debits of a in W of (debit ts minus ts of the most recent settled credit to a before it), in hours; credits are searched up to 30 days back; if no credit is found the term is capped at 30 days; if a has no debits in W the feature equals 30 days (cap). **(required)** |
| F8 | `max_amount` | largest single settled amount involving a in W |
| F9 | `net_flow_ratio` | `abs(in - out) / (in + out)` over W; 1.0 if `in + out = 0` |
| F10 | `atm_share` | ATM withdrawal amount / total debit amount in W; 0 if no debits |
| F11 | `night_share` | share of a's transactions in W with local time in [00:00, 05:00) |
| F12 | `declined_count` | number of declined transactions of a in W |
| F13 | `distinct_devices` | distinct devices in a's successful logins in W |
| F14 | `distinct_ips` | distinct IPs in a's successful logins in W |
| F15 | `account_age_days` | t minus account opening time (window independent) |
| F16 | `balance` | balance of a at t |

F1 to F14 at two windows plus F15 and F16 gives 30 features. Each feature is scored in both directions (as is and negated),
and the better direction is kept (this favors the feature, so it is conservative for the ceiling).

### Probes (fixed hyperparameters; no tuning)

- **LR**: `log1p` on nonnegative features, standardization fit on TRAIN, L2 logistic regression, `C = 1.0`, `lbfgs`,
  `max_iter = 1000`, `class_weight = "balanced"`, on all 30 features.
- **Tree**: decision tree, `max_depth = 3`, `min_samples_leaf = 50`, `class_weight = "balanced"`, `random_state` from a named
  stream, on all 30 features.
- Library: scikit-learn (version recorded).

### Metric

PR-AUC estimated as **average precision** (`sklearn.metrics.average_precision_score`) on VAL points.

### Pre-registered criteria

A criterion "holds" if it holds for the **mean over the 5 calibration seeds and for at least 4 of the 5 seeds individually**.

| Gate | Criterion | Type |
|---|---|---|
| G1 single-feature ceiling | max over the 30 features (best direction) of AP <= **0.30** | pass/fail |
| G2a shallow-probe ceiling | AP(LR) <= **0.60** and AP(Tree) <= **0.60** | pass/fail |
| G2b signal floor | max(AP(LR), AP(Tree)) >= **3 x pi** | pass/fail |
| G3 context contribution | AP gain of LR when adding pre-registered 1-hop neighbor aggregates (mean over counterparties of F4, F5, F6, F15) and temporal-order features (burstiness, median inter-event gap, time from first credit to first debit in W), with a 95% cluster-bootstrap CI (positives clustered by network, negatives by account; 1,000 resamples) | report only |
| G4 confounder share | among the top 1% of LR-scored VAL points, the share of false positives coming from each confounder archetype | report only |

Numbers kept from the Phase 0 draft. One weakness is noted rather than changed: at a point prevalence well below 1%,
`3 x pi` is a low bar (a few percent AP). It rules out pure noise but not a nearly useless dataset. A stronger learnability
check is the oracle-phase-feature ceiling in `PHASE0_ASSESSMENT.md` section 5.3, reported alongside (report only).

### Procedure on failure

1. If a pass/fail gate fails, the **generator** is changed (never the threshold, feature list or probe settings), the
   generator version is bumped, and the iteration is logged in the table below with what was changed and why.
2. All gates are re-run on the same 5 calibration seeds.
3. After 3 failed iterations, work stops and the situation is reported for a decision (possible outcomes include accepting
   that a gate was mis-specified, which would be recorded as a protocol deviation, not silently applied).

### Iteration log

| Iteration | Generator commit | G1 | G2a | G2b | G3 (report) | G4 (report) | Change made |
|---|---|---|---|---|---|---|---|
| (none yet) | | NOT YET EVALUATED | NOT YET EVALUATED | NOT YET EVALUATED | NOT YET EVALUATED | NOT YET EVALUATED | |

---

## Note: planned prevalence sweep (not yet registered)

`suspicious_prevalence` is a config parameter (decision 0005), default 0.01 for DEV and RESEARCH. It is intended to be swept for
the **distribution shift** experiments (topic listed under Evaluation in `CLAUDE.md`: "distribution shift (amounts, pattern
frequency, populations, structures, unseen patterns)"): for example training at one prevalence and evaluating at others
(candidate values 0.002, 0.005, 0.01, 0.02). Confirmed in review: the original spec's "section 56" maps to this topic. The
sweep will be pre-registered here with hypotheses and criteria before the first model milestone that uses it (Milestone 3 at
the earliest). Nothing has been run.

## Note: EXP-LM, label-maturity ablation (planned, not yet registered)

Question: how much does recency-dependent label noise (positives not yet known at `T_fit`, labeled 0) change model quality,
compared with training only on mature examples (decision 0009)?

Design constraints already fixed:

- Same dataset, same VAL / TEST / OOD sets, same model and hyperparameter budget; only the TRAIN set differs by
  `training_label_mode`.
- Both arms are scored against the full ground truth (standing rule 1).
- Also reported: the residual share of unknown positives among mature TRAIN examples (from the oracle, evaluation code only),
  and the training set size and positive count per arm, since `mature_only` trades noise for data volume.
- An `oracle` label-knowledge arm (zero delay) may be added as an upper reference, clearly labeled as unrealistic.
- **Feasibility constraint:** with a 90-day simulation and `maturity_horizon_days = 48`, `mature_only` keeps about 9% of the
  training window (computed in Phase 0) and the config validator rejects it. The ablation therefore needs either a
  `RESEARCH_LONG` profile (180 days, 58% of the training window mature) or a sweep over `maturity_horizon_days` in
  {14, 28, 48} on the standard profile. To be decided before registration (open question Q-M1).

Earliest milestone: 3 (first trained baseline). Nothing has been run.

## Note: detection-delay sensitivity analysis (planned, not yet registered)

The default delay model (LogNormal median 14 days, sigma 0.75, `p_never_known` 0.10) is an uncalibrated assumption
(`LIMITATIONS.md`). A sensitivity analysis over median, sigma and `p_never_known` is planned for the first model milestone
that trains on delayed labels. Nothing has been run.
