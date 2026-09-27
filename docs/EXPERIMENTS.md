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
| EXP-M1-G | Milestone 1 non-triviality gates | FROZEN 2026-09-27 before the first run; **RUN: all pass/fail gates PASS** (iteration 1) |
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

### Freeze addendum (2026-09-27, written before any gate run on a RESEARCH or GATE dataset)

The draft above left some implementation details open. They are fixed here, before results exist. No threshold,
feature or probe setting was changed.

1. **Target intervals.** "Active scenario phase at t" uses the label intervals as amended by A1 (a member's labels
   start at its own first observable event in the network).
2. **Snapshot cutoff.** Snapshot day `d` has inclusive cutoff `t = d * 86400 - 1`; window `W` = days `d-W .. d-1`.
   TRAIN days are those with `t + H_max <= train_end` (days 1 to 53 in the 90-day profiles), VAL days those with
   `train_end <= t` and `t + H_max <= val_end` (days 61 to 68). Only accounts open at `t` are prediction points.
3. **Feature details.** F4 and F11 count settled transactions only; F7 uses the most recent settled credit that is
   strictly earlier in `(ts, event_id)` order; F12 counts declined transactions with the account as source. Every
   feature is computed through the temporal store; the fast grid path is verified equal to the per-row reference.
4. **Probe preprocessing.** `log1p` is applied to the features that are non-negative on TRAIN (all except
   `balance`); standardization is fit on TRAIN. Probes train on all TRAIN points (`include_immature`) with
   ground-truth targets (Q-R5).
5. **Excluded entities.** OOD-family members and the reference pool `E_ref` are excluded from TRAIN and VAL.
6. **Reference pool method (A2), decision rule fixed in advance.** Compute the archetype composition of the default
   pool (`none_cell_hash`) against all normal accounts. If any archetype's share differs by more than
   **1.0 percentage point**, or a chi-square goodness-of-fit test gives **p < 0.01**, use `stratified_archetype`
   instead. The comparison is recorded in decision 0009 either way.
7. **G3 computation.** For computational reasons G3 is computed on uniform random samples (named stream, per
   seed) of **200,000 TRAIN** and **200,000 VAL** points (all points if fewer). Both LR models (static only;
   static plus context) are fit on the same TRAIN sample and scored on the same VAL sample. Context features:
   mean over distinct internal counterparties of settled transfers in the last 30 days of their `txn_count_30d`,
   `in_degree_30d`, `out_degree_30d` and `account_age_days` at the same snapshot (0 if none), and burstiness,
   median inter-event gap and time from first credit to first later debit in the last 30 days. CI: 1,000
   cluster-bootstrap resamples, positive clusters (by network) and negative clusters (by account) resampled
   separately. Report only.
8. **G4.** Top 1% of LR-scored VAL points; share of false positives by archetype. Report only.
9. **Informational, requested in review (not a gate):** G1-style best single-feature AP and LR AP restricted to VAL
   points of the archetype cluster that looked alike on 90-day aggregates (salary_worker, traveler, family,
   spending_surge) versus the clearly distinct one (small_business, hf_merchant, student).
10. **Disclosure.** Before this freeze the gate runner was smoke-tested once on DEV seed 42 (not a gate dataset;
    DEV is never used for gates). That run showed probes close to prevalence (best single-feature AP 0.0057, LR AP
    0.0019, point prevalence 0.0012, 22 VAL positives). Nothing in this addendum was changed in response.
11. **Profile.** The Q-R4 rule is applied on the measured wall-clock time of one RESEARCH generation (A5), recorded
    in decision 0010 before the gate run.

### Results (run 2026-09-27; iteration 1)

Datasets: RESEARCH profile, calibration seeds 1000 to 1004, generator code at commit `5f9574f` plus working-tree
changes that do not affect generation. Verified afterwards: a fresh clone of commit `08707a9` regenerates seed 1004
with the identical dataset hash `43ce496f...` (551.9 s). Reference pool: `stratified_archetype` (A2 rule, decision 0009). Output: `reports/gate_exp_m1_g.json`
(per-seed files in `reports/gate/`). Each seed: about 366 s and 10.5 GB peak.

| Seed | Dataset hash | VAL points | VAL positives | pi | G1 best single-feature AP | AP LR | AP tree | floor 3 x pi | G3 delta AP [95% CI] |
|---|---|---|---|---|---|---|---|---|---|
| 1000 | `e311027f7f7a` | 714,377 | 340 | 0.000476 | 0.0023 (total_amount_in_7d) | 0.0094 | 0.0018 | 0.0014 | +0.0101 [-0.0010, +0.0504] (89 pos) |
| 1001 | `7352d7ff97e3` | 713,824 | 293 | 0.000410 | 0.0015 (total_amount_in_7d) | 0.0038 | 0.0047 | 0.0012 | -0.0049 [-0.0404, -0.0004] (76 pos) |
| 1002 | `3efb2cffc359` | 714,115 | 269 | 0.000377 | 0.0015 (total_amount_in_7d) | 0.0017 | 0.0027 | 0.0011 | -0.0001 [-0.0003, +0.0001] (72 pos) |
| 1003 | `c42d6eacb6f2` | 713,991 | 312 | 0.000437 | 0.0088 (distinct_devices_30d) | 0.0027 | 0.0015 | 0.0013 | +0.0020 [-0.0000, +0.0123] (82 pos) |
| 1004 | `43ce496f259f` | 714,328 | 195 | 0.000273 | 0.0016 (total_amount_in_7d) | 0.0018 | 0.0029 | 0.0008 | +0.0004 [-0.0001, +0.0039] (57 pos) |

**Pre-registered verdict (mean over seeds and at least 4 of 5 seeds):**

| Gate | Result | Evidence |
|---|---|---|
| G1 single-feature ceiling (<= 0.30) | **PASS** | mean 0.0031; all 5 seeds pass |
| G2a shallow-probe ceiling (<= 0.60) | **PASS** | mean LR 0.0039, mean tree 0.0027; all 5 seeds pass |
| G2b signal floor (>= 3 x pi) | **PASS** | all 5 seeds pass |
| G3 context contribution | report only | no consistent effect: 2 CIs include 0 with positive point estimates, 2 near 0, 1 significantly negative (seed 1001) |
| G4 confounder share | report only | table below |

G4: share of the top-1% LR-scored false positives by archetype (largest four):

| Seed | Composition |
|---|---|
| 1000 | small_business 0.45, high_volume_business 0.23, salary_worker 0.12, traveler 0.06 |
| 1001 | small_business 0.45, high_volume_business 0.31, salary_worker 0.11, student 0.04 |
| 1002 | high_volume_business 0.42, small_business 0.31, salary_worker 0.11, family 0.06 |
| 1003 | small_business 0.40, high_volume_business 0.21, salary_worker 0.16, family 0.08 |
| 1004 | small_business 0.53, high_volume_business 0.32, salary_worker 0.05, family 0.03 |

Informational (review request): separability within the archetype cluster that looked alike on 90-day aggregates
versus the clearly distinct cluster, on VAL points of each cluster (LR trained on all TRAIN):

| Seed | weak-cluster positives | weak: best single-feature AP | weak: LR AP | distinct-cluster positives | distinct: best single-feature AP | distinct: LR AP |
|---|---|---|---|---|---|---|
| 1000 | 256 | 0.0867 (total_amount_in_7d) | 0.0174 | 78 | 0.1057 (total_amount_in_7d) | 0.0494 |
| 1001 | 237 | 0.0714 (total_amount_in_7d) | 0.0146 | 53 | 0.0765 (max_amount_7d) | 0.0011 |
| 1002 | 196 | 0.0469 (total_amount_in_7d) | 0.0033 | 70 | 0.0052 (declined_count_7d) | 0.0038 |
| 1003 | 245 | 0.0739 (total_amount_in_7d) | 0.0064 | 61 | 0.0660 (distinct_ips_30d) | 0.0011 |
| 1004 | 143 | 0.1121 (total_amount_in_7d) | 0.0176 | 44 | 0.0106 (max_amount_7d) | 0.0004 |

### Interpretation (written after seeing the results)

1. **The gates pass, but the margin that matters is the floor, not the ceilings.** The better of the two probes
   reaches about 6 to 20 times the point prevalence (pi about 0.03% to 0.05%), far below the ceilings. The floor passes on every seed but narrowly on
   some (seed 1002: LR AP 0.0017 against a floor of 0.0011). The weakness noted at registration applies: `3 x pi` rules
   out noise, not a nearly unlearnable dataset.
2. **Confounders drive the low overall single-feature AP.** Within the salary-like cluster a single inflow feature
   reaches AP 0.047 to 0.112 on every seed (within the distinct cluster 0.005 to 0.106), but over all points the best
   single feature reaches only 0.0015 to 0.0088: the high-volume and small businesses
   (about 9% of accounts) dominate the top of any amount- or degree-based ranking and account for 61% to 85% of the top
   false positives (G4). This is the intended effect of the confounders, and it means per-account statics are not
   enough on this data.
3. **No evidence that the pre-registered context features add signal for a linear probe** (G3), with only 57 to 89 VAL
   positives in the 200,000-point samples. This is a statement about these seven hand-made context features and a
   linear model, not about graph models.
4. **The target is hard by construction.** "Active scenario phase at the daily snapshot" is positive for only 0.03% to
   0.05% of points, because label intervals are short (median 6 h HOLD to 31 h INFLOW, EDA report). This is relevant to
   how Milestone 3 defines its targets and is recorded as an open question, not changed here.

No generator iteration was triggered; the thresholds, features and probes were not changed.

### Iteration log

| Iteration | Generator commit | G1 | G2a | G2b | G3 (report) | G4 (report) | Change made |
|---|---|---|---|---|---|---|---|
| 1 | `5f9574f` (+ non-generation changes) | PASS | PASS | PASS | no consistent effect | businesses dominate top false positives | none (first run) |

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
