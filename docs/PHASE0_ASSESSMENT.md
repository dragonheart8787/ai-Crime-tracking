# Phase 0 Assessment: Predictive Financial Crime Intelligence Platform

- Date: 2026-09-26
- Session type: cloud planning session (no GPU, allowlisted network). Planning and assessment only.
- Status: **revised twice after review (revision passes on 2026-09-27); awaiting review of round 2**.
  No generator, feature, model or pipeline code exists.

Revision pass summary: stable 64-bit RNG keys and a `PYTHONHASHSEED` regression test (0002); precise
canonical content hash (0001); `known_at` made load-bearing with a non-degenerate latency model (0004);
`suspicious_prevalence` as a config parameter (0005); OOD scenario-family holdout pool redesigned (new 0009);
single-scenario membership as named simplification (now S1 and S2) with per-instance parameter diversity (0002, 0005);
gate feature set pre-registered in `docs/EXPERIMENTS.md`; `logins` event table added to schema, outputs and
ER model (section 4.5).

Round-2 summary: label maturity with two training-label modes and a ground-truth-only evaluation rule (0009, 0004,
`EXPERIMENTS.md`); simplifications renumbered S1 (no concurrent membership), S2 (single-scenario-for-life, must be
relaxed before role-transition experiments), S3 (single currency, no per-row currency column), S4 (network-level
label knowledge); `docs/LIMITATIONS.md` created; DEV fixed at about 2,000 to 3,000 accounts over 90 days; channel
enum with the login invariant scoped to `APP` and `WEB` only.

Conventions: anything not run is marked **NOT YET EVALUATED**. Statements from prior knowledge that
were not verified in this session are marked *(unverified)*. The only things actually executed in this
session are listed in section 9.

Decision records referenced below live in `docs/decisions/`:

| # | Title |
|---|---|
| 0001 | Dataframe library and storage format |
| 0002 | RNG stream design |
| 0003 | As-of view design (leakage prevention by construction) |
| 0004 | Label and time indexing scheme |
| 0005 | Generator configuration schema |
| 0006 | Rust evaluation for candidate components |
| 0007 | GPU environment target |
| 0008 | Package layout and CLI entry point |
| 0009 | Chronological splits and the OOD scenario-family holdout pool |

---

## 1. PROJECT ASSESSMENT

### 1.1 Repository state at start of session

The repository was empty: no commits, no files (`CLAUDE.md` was also absent; it was added in this
session verbatim from the text supplied in the kickoff message, so later sessions can read it).

### 1.2 What the project is, stated precisely

A research prototype that, on synthetic data, tests whether modeling transactions as a dynamic
heterogeneous temporal graph adds measurable value over (i) per-account tabular features and
(ii) per-account event sequences, for four task families: entity risk, behavioral role, suspicious
network recovery, and next high-risk event (type, target, time).

### 1.3 Honest assessment

Strengths of the spec:

- The temporal-integrity rules (as-of view, chronological splits, purge gap, time-indexed labels) are
  the right priorities and are stricter than most published work in this area.
- Mandatory baselines and pre-registration protect against the most common failure mode (a complex
  model "winning" against weak or no baselines).
- Synthetic data gives access to things real data never provides: exact ground-truth roles, phases,
  network membership, and the generator's own conditional event distribution.

Central weakness: **every conclusion is conditional on the generator.** If graph structure is what
separates suspicious from normal in the generator, graph models will win, and that result says more
about the generator than about the world. This does not make the project pointless, but it changes
what can be claimed. The research design in sections 3 to 5 is built around that fact (difficulty
sweeps, Bayes ceilings, conditional claims, optional external sanity checks on public data).

Scope: 12 milestones with ~6 temporal-GNN families, heterogeneous GNNs, multi-task weighting studies,
explainability, streaming and a dashboard is several person-months at minimum. Section 6 proposes where
to cut without losing the core research question.

### 1.4 What would count as a meaningful result

- A positive result: on a pre-registered set of generator regimes, temporal graph models give earlier
  warnings (lead time at a fixed alert budget) than tabular and sequence-only baselines, with confidence
  intervals over network instances and over generator seeds, and the gain shrinks gracefully as the
  graph signal is weakened.
- A negative result (equally publishable internally): tabular features with well-designed temporal and
  1-hop aggregates match graph models in most regimes. This is plausible and must be reported if found.

---

## 2. PROPOSED SYSTEM ARCHITECTURE

### 2.1 Layers and data flow

```
configs/ (Pydantic-validated YAML, 0005)
   |
   v
simulation/  --(named RNG streams, 0002)-->  canonical tables (Parquet, Arrow schemas, 0001)
   |                                          transactions, logins, persons, accounts, devices,
   |                                          ips, atms, merchants, relations, labels,
   |                                          event_labels, ground_truth_networks,
   |                                          network_members, metadata
   v
temporal/  TemporalStore (sealed) --as_of(t)--> AsOfView      (0003)
                                  --point_in_time(ids, ts)--> PITQuery primitives
           splits: chronological, purged by H_max, known_at <= T_fit,
                   OOD family pool + reference negatives (0004, 0009)
   |                                   |
   v                                   v
validation/ (invariants, stats,     features/ (M2)  graph/ (M4)  sampler (M5+)
 triviality probes, EDA)                 |              |            |
                                         v              v            v
labels/oracle (targets, eval only) --> models/ (M3+) --> evaluation/ --> alerts / explain / dashboard (later)
```

Key architectural properties:

1. **Single source of truth**: the canonical tables. Everything downstream is derived and
   reproducible from config + seed + code version.
2. **One door to the data**: features, graph construction, samplers and splits read only through the
   `TemporalStore` API (0003). Ground-truth labels sit behind a separate `OracleLabels` object that
   only target builders and evaluation import; enforced by an import-boundary test.
3. **Events are not only transactions.** Logins (account, device, IP, ts, outcome) are a separate event
   table sharing the global order `(ts, event_id)`. It is the timed source of LOGIN_WITH and USES_IP edges
   and of `NEW_DEVICE_LOGIN` / `NEW_IP_LOGIN` (derived through the as-of view). Schema in section 4.5.
4. **Relations are temporal**: OWNS_ACCOUNT, CONTROLS, USES_IP, etc. carry `valid_from` / `valid_to`.
   An account opened on day 40 is invisible on day 39.
5. **Money is conserved explicitly**: external source/sink accounts (employers, billers, the outside
   world) make every inflow and outflow a transfer between accounts, so conservation is checkable.

### 2.2 Generator internals (planned for M1, described not built)

Two-stage design:

1. **Intent generation** (vectorized, per named RNG stream): each normal person's archetype produces
   intended events (salary, rent, bills, card payments, ATM withdrawals, family transfers, logins);
   each scenario instance first draws its own instance parameters from the family's distributions, then
   recruits members from its own recruitment cell (0002), then produces intended events per its
   stochastic phase grammar; confounder populations produce their own intents. The number of active
   instances follows from `suspicious_prevalence` (0005).
2. **Settlement pass** (sequential, global order `(ts, event_id)`): applies each intent to a ledger,
   enforces each account's overdraft limit, marks failures as `status=DECLINED` (which is realistic
   and informative), writes `balance_before/after`.

The settlement pass couples accounts that transact with each other, which is why "unrelated
entities unchanged" is defined precisely in 0002 as entities not touched by the scenario.

### 2.3 Deviations from the suggested layout

- Named package under `src/fcip/` instead of `src` as the package (0008).
- Added `temporal/` and `validation/` sub-packages.
- No empty placeholder packages for future milestones.
- No MLflow, Hydra, FastAPI, Streamlit, torch in Milestone 1 dependencies.

---

## 3. RESEARCH RISKS

| # | Risk | Why it matters | Mitigation |
|---|---|---|---|
| R1 | **Generator circularity**: the models learn the generator's script | Results do not transfer; "graph beats tabular" is true by construction | Difficulty knobs swept, not fixed; report performance curves over regimes; Bayes ceilings; claims phrased conditionally; optional external check on public data (M3+) |
| R2 | **Temporal leakage** (future events, balances, labels, neighbors) | Inflated metrics, the primary methodological risk | As-of store (0003), metamorphic future-perturbation test, import-boundary test, purge gap, `known_at` labels |
| R3 | **Label-knowledge leakage**: training on labels an institution would not yet know | Optimistic early-warning results | `known_at` enforced in `known_labels()`, training-example construction and splits; non-degenerate latency by default; explicit test (0004, 0009); oracle regime only as a labeled ablation |
| R4 | **Trivial separability** (a single feature or amount band gives it away) | Everything looks good, nothing is learned | Pre-registered non-triviality gates with both a ceiling and a floor (sections 5, 7) |
| R5 | **Unlearnable data** (gates pushed too far, signal destroyed) | Null results caused by the generator, not the models | Floor gate; Bayes / oracle-feature ceiling showing signal exists |
| R6 | **Garden of forking paths on the generator** (tuning the generator until later models look good) | Hidden researcher degrees of freedom | Calibration seeds separate from the locked research seed; generator frozen and versioned before any model milestone; any later generator change is a new dataset version with its own entry in `EXPERIMENTS.md` |
| R7 | **Deterministic scenario scripts** make next-event prediction trivial | Next-event accuracy measures the script | Stochastic phase grammars; compute the generator's true conditional next-event distribution as the Bayes-optimal reference |
| R8 | **Transductive memorization**: time split, but the same entities and networks appear in train and test | Models memorize IDs / embeddings | Report separately for networks that started before vs after the train cutoff; OOD family pool and reference negatives (0009); no ID embeddings in baselines |
| R9 | **Evaluation variance**: few suspicious networks per split | Confidence intervals wider than differences between models | Cluster bootstrap over network instances; multiple generator seeds for final claims; enough instances in RESEARCH profile (power check before M3) |
| R10 | **Dual-use drift** (generator knobs becoming evasion tools) | Violates 2.1 | Schema has shape-only parameters; no detector-in-the-loop; review checklist in `SECURITY_AND_ETHICS.md` |
| R11 | **Environment** (Blackwell GPU + Windows + PyG extensions) | Blocks M5+ | Torch not needed until M5; verification plan in 0007; in-house sampler removes compiled-extension dependency |
| R12 | **Scope creep** | Project never reaches the core comparison | Roadmap cuts in section 6 |
| R13 | **Simplification S1/S2 bias** (no concurrent membership; single scenario for life) | Networks are vertex-disjoint, which makes network detection easier than in reality; no cross-scenario role transitions | Named simplifications (4.6, `LIMITATIONS.md`); results labeled with the simplifications in force; S2 relaxed before any role-transition experiment |
| R14 | **Low positive counts at 1% prevalence** | DEV has only about 25 to 30 suspicious accounts; OOD families even fewer | DEV used for tests only; gates and all metrics at RESEARCH (or GATE) scale; power check before Milestone 3 |
| R15 | **PU label noise from delayed labels** | Not-yet-known positives enter training as 0, more often near `T_fit` | Label maturity and two training-label modes (0009); latent-positive and residual-noise counts reported; EXP-LM |
| R16 | **Maturity ablation infeasible at 90 days** | With the default 48-day horizon only about 9% of the training window is mature (computed) | Validator rejects it; ablation on a 180-day profile or a horizon sweep (Q-M1) |

---

## 4. DATA STRATEGY

### 4.1 Primary: in-house synthetic generator

Rationale: only a generator provides time-indexed roles and phases, network membership, terminal
events and controllable difficulty, all needed by the research question.

**Entities**: persons, accounts, devices, IPs, ATMs, merchants, plus explicit external accounts
(employers, billers, "outside world") for conservation of funds.

**Event tables**: `transactions` (transfer, card/merchant payment, ATM withdrawal, deposit, salary,
bill; channel, region, device, IP, status, balance before/after) and `logins`
(account, device, IP, ts, outcome). Beneficiary age, frequency and interval are *derived features*
computed through the as-of view, not stored columns, to avoid storing values that could encode the
future.

**Currency**: one synthetic currency per dataset (`SYN`, 100 minor units per major), declared once in config and
metadata; **no per-row `currency` column** (simplification S3, decision 0001).

**Location**: coarse region codes, not coordinates.

**Normal archetypes** (each with its own amount, timing, counterparty and channel distributions):
salary worker, student, small business, high-frequency merchant, traveler, family transfers,
temporary spending surge, legitimate high-volume business; cross-cutting recurring bills, rent,
savings, ordinary ATM use.

**Suspicious scenario families** (abstract detection research patterns): fan-in, fan-out,
pass-through, burst, dormant activation, multi-hop, cycle, structuring-like, shared-device / shared-IP
coordination, account-to-account-to-cash. Each family has a stochastic phase grammar (0004) and
shape parameters only (0005).

**Confounders (designed to create hard negatives)**:

- legitimate high-volume businesses with high fan-in and fan-out and many counterparties;
- payroll distributors (one-to-many at month end, looks like fan-out);
- households, public Wi-Fi, corporate NAT and mobile carrier NAT producing **legitimate shared IPs**
  (so a shared IP alone is never informative, as required by 2.1);
- shared family devices;
- legitimately dormant accounts that reactivate (seasonal workers, returning travelers);
- temporary spending surges (weddings, moves) that look like bursts;
- rent collectors and small landlords (regular fan-in);
- travelers with ATM use in unusual regions.

Additionally, a configurable fraction of scenario participants keeps its normal background activity,
and scenario amounts are drawn from distributions overlapping normal amounts.

**Prevalence**: `suspicious_prevalence` (share of internal accounts that are members of any scenario
instance over the simulation) is a config parameter, default 0.01 in DEV and RESEARCH, intended to be swept
later (0005, `EXPERIMENTS.md`).

**Instance diversity**: every instance draws its own amount scale, timing offsets, phase durations,
counterparty count, holding time and background-activity retention from its family's distributions; drawn
values are stored in `ground_truth_networks`, and validation fails if a family's instances are near-identical
(0005).

**OOD holdout pool** (0009): a configured subset of families (`is_ood_family: true`) occurs across the full
timeline like any other family, but its members are excluded from TRAIN and VAL for the whole timeline, its
labels are never visible as known labels, and it is evaluated only in a separate OOD pool against a fixed
reference pool of never-trained-on normal accounts. Temporal and pattern generalization are thereby
measured separately (2 x 2 table in 0009).

### 4.2 Profiles

| Profile | Accounts | Days | Transactions | Purpose |
|---|---|---|---|---|
| DEV | about 2,000 to 3,000 (decided in review) | 90 | ~100K to 150K (estimate) | tests, CI, fast iteration on CPU; never gates or statistics |
| RESEARCH | ~100K | 90 | order of 10^7 (estimate, NOT YET EVALUATED) | main experiments |
| LARGE | only with memory estimate | | | not planned before M12 |

All profiles use `suspicious_prevalence = 0.01` by default. At DEV size that is roughly 25 to 30 suspicious
accounts, enough for tests and the end-to-end integration run, not for any statistic (R14). The 90-day duration was
preferred over 10,000 accounts for 30 days because it exercises archetype heterogeneity and temporal features. This
deviates from the "about 10K accounts" DEV figure in `CLAUDE.md` section 10, with the reviewer's approval.

Memory estimate for RESEARCH (back-of-envelope, NOT YET EVALUATED): ~25 columns, ~8 bytes average
per value, 10^7 rows gives ~2 GB for transactions in memory, well within 64 GB.

### 4.3 Seeds and dataset versioning

- **Calibration seeds** (e.g. 1000 to 1099) are used while developing the generator and running the
  non-triviality gates.
- The **locked research seed** (42, plus 43 and 44 for seed-variance runs) is generated only once
  the generator version is frozen and the gates passed on calibration seeds.
- Dataset identity = canonical content hash (0001). Metadata stores full resolved config, seed,
  generator version (git commit), package versions and per-table hashes.

### 4.4 Secondary: public datasets (not used before Milestone 3, all NOT YET VERIFIED)

Candidates named in the spec: IBM AML synthetic transactions (AMLSim-derived), PaySim, Elliptic.
Before any use, a data card will record license, schema, label definition, timestamp granularity and
realism, whether synthetic or real, and leakage risks. Only role: external sanity check that the
feature pipeline and baselines behave plausibly on data not produced by this generator. No citation
details are given here because none were verified in this session.

### 4.5 Output tables and entity-relationship model

As built at Milestone 1 checkpoint 1; `src/fcip/common/schemas.py` is authoritative and decision 0010 lists the
differences from the Phase 0 plan.

Parquet files, one per table (CSV via explicit export: same names with `.csv`). All times are int64 seconds
since the simulation epoch; all money is int64 minor units (0001). Primary keys are the canonical sort keys
used by the content hash.

| Table | Primary key | Columns (planned) |
|---|---|---|
| `persons` | `person_id` | region, household_id, created_at (archetype moved to the oracle table `person_truth`, 0010) |
| `person_truth` (oracle) | `person_id` | archetype, origin (normal / scenario) |
| `accounts` | `account_id` | account_kind (internal / external), external_role (employer / biller / landlord / external_world / card_network / cash), owner_person_id (null for external), opened_at, closed_at (nullable, unused in M1), overdraft_limit_minor (null = unlimited, external only), initial_balance_minor, region |
| `devices` | `device_id` | device_kind, first_seen_at |
| `ips` | `ip_id` | ip_context (household / public_wifi / corporate_nat / mobile_cgnat / residential_single), region |
| `atms` | `atm_id` | region |
| `merchants` | `merchant_id` | merchant_category, region |
| `relations` | `(relation_type, src_id, dst_id, valid_from)` | relation_type in {OWNS_ACCOUNT, CONTROLS, USES_IP}, valid_to |
| `transactions` | `event_id` | ts, txn_type, channel (APP / WEB / ATM / POS / SCHEDULED / INBOUND_EXTERNAL), src_account_id, dst_account_id (never null: card payments credit the `card_network` account, ATM withdrawals the `cash` account, 0010), merchant_id (nullable), atm_id (nullable), amount_minor, region, device_id (nullable), ip_id (nullable), login_event_id (nullable), status (SETTLED / DECLINED), src_balance_before_minor, src_balance_after_minor, dst_balance_before_minor, dst_balance_after_minor |
| **`logins`** | `event_id` | ts, account_id, device_id, ip_id, channel (app / web), outcome (SUCCESS / FAILURE) |
| `labels` | `(entity_type, entity_id, scenario_id, valid_from)` | see 0004 |
| `event_labels` | `(event_table, event_id)` | see 0004 |
| `ground_truth_networks` | `network_id` | see 0004; also scenario_id, end_ts, n_created_accounts, n_rounds and the drawn instance parameters (amount scale, fan degree, hops, holding seconds, retention and forward fraction in permille) |
| `network_members` | `(network_id, entity_type, entity_id)` | first_role, joined_ts |
| `metadata.json` | | resolved config and its hash, seed, generator commit, package versions, per-table content hashes, dataset hash, per-file SHA-256, realized prevalence, recruitment shortfalls |

Entity-relationship model (edge types from `CLAUDE.md` 5, with their timed source):

| Edge | Source table | Time |
|---|---|---|
| OWNS_ACCOUNT (Person -> Account) | `relations` | `valid_from`, `valid_to` |
| CONTROLS (Person -> Device) | `relations` | `valid_from`, `valid_to` |
| USES_IP (Device -> IP) | `relations` (assignment) and `logins` (observed use) | interval; login ts |
| LOGIN_WITH (Account -> Device) | **`logins`** | login ts |
| TRANSFER_TO (Account -> Account) | `transactions` (txn_type transfer) | ts |
| WITHDRAW_AT (Account -> ATM) | `transactions` (txn_type atm_withdrawal) | ts |
| PAYS_MERCHANT (Account -> Merchant) | `transactions` (txn_type card / merchant payment) | ts |

Invariants that tie the tables together (tested in Milestone 1): every transaction with channel `APP` or `WEB`
references a successful login of the same account, device and IP within the configured session window before it;
this applies to digital channels **only**: `ATM`, `POS`, `SCHEDULED` and `INBOUND_EXTERNAL` transactions have no
login step in this model and must have null `device_id`, `ip_id` and `login_event_id` (checked both ways; channel
table in 0005); a login's
device is controlled by the account owner at that time (except in configured shared-device scenarios and
confounders); `NEW_DEVICE_LOGIN` / `NEW_IP_LOGIN` are derived from `logins` via the as-of view and never stored.

### 4.6 Named Milestone 1 simplifications

Full table with biases and revisit conditions in `docs/LIMITATIONS.md`.

- **S1: no concurrent scenario membership.** An account is in at most one active scenario at a time. *Bias:*
  networks never overlap through a shared account at the same time; network recall and alert compression are
  optimistic.
- **S2: single-scenario-for-life** (accepted in the round-1 review). The recruitment partition (0002) makes an
  account a member of at most one instance over the whole simulation, which is what keeps scenarios isolated for the
  reproducibility requirement. *Bias:* networks are vertex-disjoint and no account changes role across scenarios
  (e.g. VICTIM_LIKE, later RELAY). **S2 must be relaxed before any experiment that tests tracking of role changes over
  time**, which is part of the core research question; `EXPERIMENTS.md` forbids registering such an experiment while
  S2 is in force, and the roadmap lists the relaxation as a prerequisite (section 6). Relaxing it needs a
  coupling-aware isolation test.
- **S3: single currency**, no per-row currency column (0001).
- **S4: network-level retrospective label knowledge** (0004): no partially known networks.

---

## 5. EVALUATION STRATEGY

### 5.1 Prediction points and splits

- Predictions are made at **prediction points** `(entity, t)`: a fixed cadence (e.g. daily snapshot)
  plus event-triggered points (after each observed event, for next-event tasks).
- Chronological split on prediction time, e.g. days 1 to 60 / 61 to 75 / 76 to 90, with a purge gap
  >= `H_max` between consecutive sets (each set's last prediction time is `H_max` before its boundary),
  full rules in 0009.
- Training targets are built only from labels with `known_at <= T_fit` (0004); label-derived features at
  any t use only labels with `known_at <= t`.
- Neighborhoods, features and labels at `t` come only from the as-of view (0003).
- OOD families are evaluated only in the OOD pool, against the reference negative pool (0009).
- ID test metrics are reported separately for (a) networks active across the train cutoff and (b) networks
  that started after it (R8).

### 5.2 Primary and secondary metrics

- Risk: PR-AUC (primary), Precision@K and Recall@K at fixed alert budgets, ROC-AUC, Brier, ECE.
- **Early-warning lead time**, made measurable (Challenge 7): at a fixed alert budget
  (alerts per 1,000 accounts per day), for each network that reaches a terminal event, lead time =
  terminal_ts minus first alert on any member; networks never alerted get lead time 0 and count as
  misses. Reported as a curve over budgets, plus the median at a pre-registered budget.
- Role: macro F1, per-class P/R, confusion matrix, evaluated on time-indexed roles.
- Network recovery: network recall (share of member accounts in the same predicted cluster),
  share of network found before the terminal event, alert compression ratio.
- Next event / target / time: top-k, macro F1, cross entropy; MRR, Hits@k; bucket accuracy, C-index
  with censoring.

### 5.3 Reference points unique to synthetic data

- **Bayes ceiling for next event**: the generator knows the true conditional distribution of the
  next phase transition; its cross entropy is a lower bound that no model can beat. Report models as
  a fraction of the gap between a marginal-frequency baseline and this bound.
- **Oracle-feature ceiling for risk**: a model given ground-truth phase features, to show the task is
  learnable in principle.
- **Difficulty sweep**: a small, pre-registered grid of generator regimes (confounder strength,
  share of scenario accounts with normal background activity, amount overlap). Claims are made per
  regime, not on one dataset.

### 5.4 Statistics

- Cluster bootstrap over network instances (events within a network are correlated) for 95% CIs.
- 3 generator seeds for final comparisons; report mean and spread across seeds.
- Paired comparisons between models on the same prediction points.
- A pre-registered minimum effect size to call a model better (proposed: non-overlapping CIs *and*
  consistent sign across all 3 seeds).

### 5.5 Milestone 1 non-triviality gates

Pre-registered (as a draft to be frozen before the first gate run) in `docs/EXPERIMENTS.md`, experiment
**EXP-M1-G**: exact 30-feature single-account set (including total transaction amount, transaction count,
in-degree, out-degree and average holding time, each with a precise definition), fixed probe
hyperparameters, calibration seeds 1000 to 1004, RESEARCH (or GATE) profile, and the criteria:
single-feature AP <= 0.30; shallow-probe AP <= 0.60 and >= 3 x point prevalence; context contribution and
confounder share reported, not gated. Numbers unchanged from the first draft; the weakness of the `3 x pi`
floor at low prevalence is noted there.

---

## 6. IMPLEMENTATION ROADMAP

Milestone order follows `CLAUDE.md` 4, with proposed adjustments marked **(proposal)**.

| M | Deliverable | Exit criterion (abridged) | Notes |
|---|---|---|---|
| 1 | Synthetic generator + validation | DoD in kickoff, section 7 here | No models except probes |
| 2 | EDA + feature pipeline on `PITQuery` | Every feature passes the metamorphic leakage test | Feature registry; Python motif oracle (small scale) |
| 3 | Tabular baselines (LR, RF, LightGBM or XGBoost) | Pre-registered results with CIs over seeds | MLflow (or a simpler run log) introduced here **(proposal)** |
| 4 | Transaction graph construction (typed, temporal) | As-of snapshot tests | numpy/scipy/NetworkX, no torch yet |
| 5 | GraphSAGE baseline + in-house as-of sampler | Environment verified (0007) | First torch dependency |
| 6 | Sequence-only next-event model (GRU, then Transformer) | Compared against Bayes ceiling | |
| 7 | Temporal graph model | **(proposal)** start with one memory-based model (TGN, which PyG provides a module for *(unverified for the pinned version)*) and one simple non-memory baseline; others only if justified | Library review document first |
| 8 | Multi-task model | Only if each single-task head beats its baseline **(proposal)**; weighting comparison limited to fixed vs uncertainty-based **(proposal)** | |

**Prerequisite recorded in review:** before any experiment that tests whether the system tracks an entity's role
changing over time (at the latest the role-inference work in Milestones 7 and 8), simplification S2 must be relaxed in the
generator. That is a generator change, so it produces a new dataset version with its own `EXPERIMENTS.md` entry.
| 9 | Explainability + calibration | | |
| 10 | Streaming simulation | Needs a latency target first (Challenge 13) | Rust decision (c) here |
| 11 | Dashboard | | |
| 12 | Comparisons and ablations | | Many were already run per milestone |

---

## 7. FIRST MILESTONE (scope description only; not built)

**Milestone 1: Synthetic Financial Environment + Data Validation.** Nothing is implemented in this
session.

In scope:

1. Pydantic config schema and YAML profiles DEV / RESEARCH (0005).
2. Named RNG streams and namespaced IDs (0002).
3. Entities and temporal relations: persons, accounts, devices, IPs, ATMs, merchants, external
   accounts; OWNS_ACCOUNT, CONTROLS, USES_IP, LOGIN_WITH with validity intervals.
4. Normal archetypes and confounders (section 4.1).
5. Scenario families with stochastic phase grammars, `enabled`, `weight`, `max_instances` and
   `is_ood_family` per family; per-instance parameters drawn from family distributions; instance count
   derived from `suspicious_prevalence`; recruitment partition enforcing S1 and S2 (0002, 0005).
6. Settlement pass with overdraft rule, declined status and balances.
7. Outputs (schema in 4.5): `transactions`, **`logins`**, `accounts`, `persons`, `devices`, `ips`, `atms`,
   `merchants`, `relations`, `labels`, `event_labels`, `ground_truth_networks`, `network_members`,
   `metadata.json` (Parquet primary, CSV on request). Label tables carry `known_at` drawn from the
   non-degenerate latency model (0004).
8. `TemporalStore` / `AsOfView` / `PITQuery` minimal versions (enough for splits, probes and leakage
   tests).
9. Split utility (0009): chronological TRAIN / VAL / TEST with purge gap, `known_at <= T_fit` for
   training targets, OOD family pool and reference negative pool, per-example `is_mature`, and **both**
   training-label modes (`include_immature`, `mature_only`) implemented and tested.
10a. Evaluation target type `OracleTargets` and metric-sanity functions that accept only it (ground-truth-only
   evaluation rule, 0009). Only the metric plumbing needed for the probes and tests; no model evaluation.
10. Validation: invariants, heterogeneity statistics, non-triviality probes, EDA report generated from
    computed numbers only.
11. CLI `simulate`, `validate`, `eda`.
12. Docs: README reproduction commands, `DATA.md`, `EXPERIMENTS.md` (freeze EXP-M1-G before the first
    gate run, then log iterations), `ARCHITECTURE.md`, `LIMITATIONS.md` (exists; S1 to S4 recorded there).

Invariants to test (definitions):

- Amounts > 0 (sign carried by direction); timestamps within `[0, sim_end)`; order key unique.
- Per account, in order: `balance_before(k) == balance_after(k-1)`; `balance_after == balance_before
  +/- amount` for settled events; declined events do not move money; balance >= `-overdraft_limit`.
- **Conservation**: sum of all balances (internal + external) is constant over time; checked at every
  day boundary and at the end.
- Causality inside scenarios: each forwarded amount has a preceding inflow on that account (strict
  timestamp order, minimum delay >= 1 s).
- Relations: an event referencing an account, device or IP occurs within the validity interval of the
  relation that connects them.
- Labels: interval invariants from 0004; every scenario event has an `event_labels` row; at most one
  terminal event per network.
- Logins (digital channels only): every `APP` / `WEB` transaction references a successful login of the same account,
  device and IP within the session window before it; `ATM`, `POS`, `SCHEDULED`, `INBOUND_EXTERNAL` transactions have
  null device, IP and login references.
- Currency: no `currency` column in any table; currency declared once in metadata.
- Maturity and evaluation: `is_mature` independent of `known_at`; `mature_only` drops exactly the immature TRAIN
  examples; validator rejects too-short mature windows; evaluation with a perfect scorer reaches AP = 1 on a dataset
  where most positives are never known (fails if `known_at`-filtered labels are substituted).
- Reproducibility: same seed gives same content hash, also across fresh subprocesses with different
  `PYTHONHASHSEED`; different seeds differ; disabling a family leaves other instances and untouched
  entities identical by `event_id` (exact guarantee in 0002); content hash equal across polars and pyarrow
  write paths (0001).
- Prevalence: realized prevalence within tolerance of `suspicious_prevalence`; no recruitment shortfall above
  the configured maximum; instance-parameter diversity above the configured floor.
- Label knowledge: latency distribution non-degenerate (config validator); the `known_at` exclusion test
  (0004); OOD exclusion tests (0009).
- Leakage: adversarial and metamorphic tests from 0003.

Out of scope: any trained model beyond the probes, feature pipeline beyond probe features, graph
construction, torch.

---

## 8. EXACT FILES I PLAN TO CREATE OR MODIFY (when implementation starts, after approval)

Created in the first Phase 0 session: `CLAUDE.md`, `README.md`, `.gitignore`,
`docs/PHASE0_ASSESSMENT.md`, `docs/SECURITY_AND_ETHICS.md`, `docs/decisions/0001` to `0008`.
Created or rewritten in the revision pass: `docs/EXPERIMENTS.md` (new), `docs/decisions/0009` (new),
`0001`, `0002`, `0004`, `0005` (rewritten), `0003`, `0006`, `0007`, `0008` (status / small alignment edits),
this document.

Planned for Milestone 1 (subject to change after review; each is a small module):

```
pyproject.toml                      # uv project, deps: numpy, pyarrow, polars, pydantic, pyyaml,
                                    # scikit-learn (probes only); dev: pytest, hypothesis, ruff, mypy
uv.lock
.python-version                     # 3.11
configs/generator/base.yaml
configs/generator/dev.yaml
configs/generator/research.yaml

src/fcip/__init__.py
src/fcip/cli.py                     # simulate | validate | eda
src/fcip/config/generator.py        # Pydantic models (0005)
src/fcip/config/loader.py           # YAML load + overlay merge + resolved-config hash
src/fcip/common/errors.py           # FutureAccessError, InvariantViolation, ConfigError
src/fcip/common/rng.py              # stable_key (blake2b-64), named streams (0002)
src/fcip/common/ids.py              # namespaced int64 ids
src/fcip/common/timebase.py         # epoch, calendar anchor, interval helpers
src/fcip/common/hashing.py          # canonical content hash (0001)
src/fcip/common/schemas.py          # Arrow schemas for every table
src/fcip/common/io.py               # validated Parquet read/write, CSV export
src/fcip/simulation/population.py   # persons, accounts, external accounts
src/fcip/simulation/infrastructure.py  # devices, IPs (incl. shared contexts), ATMs, merchants
src/fcip/simulation/relations.py    # temporal relations
src/fcip/simulation/archetypes/{__init__,base,salary_worker,student,small_business,
                                hf_merchant,traveler,family,spending_surge,
                                high_volume_business}.py
src/fcip/simulation/confounders.py
src/fcip/simulation/recruitment.py  # rendezvous recruitment partition (0002, S1 and S2)
src/fcip/simulation/scenarios/{__init__,base,phases,fan_in,fan_out,pass_through,burst,
                               dormant_activation,multi_hop,cycle,structuring_like,
                               shared_infrastructure,account_to_cash}.py
src/fcip/simulation/logins.py
src/fcip/simulation/settlement.py   # ledger pass (0006 a')
src/fcip/simulation/labels.py       # labels, event_labels, ground_truth_networks, network_members
src/fcip/simulation/label_knowledge.py  # network-level known_at model (0004)
src/fcip/simulation/generator.py    # orchestration only
src/fcip/simulation/metadata.py
src/fcip/temporal/store.py          # TemporalStore (0003)
src/fcip/temporal/asof.py           # AsOfView, FullHistoryView (EDA only)
src/fcip/temporal/pit.py            # PITQuery primitives
src/fcip/temporal/splits.py         # build_example_index: TRAIN/VAL/TEST, OOD pool, E_ref (0009)
src/fcip/labels/oracle.py           # OracleLabels, target builder
src/fcip/validation/invariants.py
src/fcip/validation/stats.py        # heterogeneity statistics
src/fcip/validation/probes.py       # non-triviality probes (only modeling allowed in M1)
src/fcip/validation/eda.py          # writes reports/eda_milestone1.md

tests/conftest.py                   # tiny-profile fixtures
tests/unit/test_config.py
tests/unit/test_rng.py              # stable keys, collision registry, no built-in hash()
tests/unit/test_recruitment.py      # partition properties
tests/unit/test_ids.py
tests/unit/test_hashing.py          # incl. polars vs pyarrow write paths give equal content hash
tests/unit/test_schemas.py
tests/unit/test_settlement.py       # conservation, overdraft, declined
tests/unit/test_scenarios.py        # per-family shape and causality
tests/unit/test_labels.py           # interval invariants, phase-dependent roles
tests/unit/test_asof.py             # API semantics
tests/unit/test_pit.py              # Hypothesis vs brute-force oracle
tests/unit/test_splits.py           # purge, OOD exclusion, E_ref disjointness
tests/unit/test_label_knowledge.py  # latency non-degenerate, validators
tests/unit/test_logins.py           # login/transaction session invariant
tests/unit/test_metrics_sanity.py   # PR-AUC etc. on known cases
tests/leakage/test_adversarial_asof.py
tests/leakage/test_future_perturbation.py
tests/leakage/test_import_boundaries.py
tests/leakage/test_known_at.py      # delayed-label exclusion case (0004)
tests/leakage/test_ood_labels_hidden.py
tests/unit/test_maturity.py         # is_mature, both training-label modes, validator
tests/unit/test_eval_ground_truth.py  # evaluation never uses known_at-limited labels
tests/reproducibility/test_seed_determinism.py
tests/reproducibility/test_pythonhashseed.py   # fresh subprocesses, different PYTHONHASHSEED
tests/reproducibility/test_prevalence.py
tests/reproducibility/test_scenario_isolation.py
tests/integration/test_dev_end_to_end.py

docs/ARCHITECTURE.md
docs/DATA.md
docs/EXPERIMENTS.md                 # exists (draft EXP-M1-G); frozen before first gate run
docs/LIMITATIONS.md                 # exists since round 2; kept up to date
docs/ROADMAP.md
docs/ENVIRONMENT.md                 # local verification results (filled on the laptop)
reports/eda_milestone1.md           # generated
README.md                           # reproduction commands, only verified ones
```

---

## 9. What was actually executed in this session

All exploratory, in a scratch virtualenv outside the repository; nothing here is deliverable code.

1. Repository inspection: empty repository, no commits.
2. PyPI metadata queries for `torch` and `torch-geometric` (results in 0007).
3. `download.pytorch.org` and `data.pyg.org`: **HTTP 403** from the sandbox proxy; not inspected.
4. Parquet determinism and `SeedSequence` properties (numpy 2.4.6, pyarrow 25.0.1, polars 1.44.2,
   duckdb 1.5.5):
   ```
   pyarrow True polars True duckdb True
   pyarrow vs polars same bytes: False
   spawn child stable across spawn counts: True
   explicit spawn_key == spawn()[7]: True
   ```

Revision pass (same scratch virtualenv):

5. 64-bit blake2b spawn keys under three `PYTHONHASHSEED` values (0, 1, 12345): identical streams, while
   built-in `hash("scenario")` differed each time (output in 0002).
6. Prototype of `fcip-content-hash-v1` on a 50K-row table written by pyarrow (zstd) and by polars
   (snappy, shuffled rows):
   ```
   file bytes equal: False
   polars read-back types: ['int64', 'int64', 'int64', 'large_string']
   content hash equal: True
   one-value change detected: True
   ```

Round-2 revision pass (same scratch virtualenv):

7. Exact latency quantiles and label-unknown probabilities for the default model, and the size of the mature
   training window for several simulation lengths (tables in 0009):
   ```
   latency quantiles (days): p50 14.0 p90 36.6 p95 48.1 p99 80.1
   d=48: latency-only 0.050   incl. p_never_known 0.145
   sim 90d, train_end 60, H_max 7d, maturity 48d: training t in [0,53], mature-only t in [0,5] -> 9% of training window
   sim 180d, train_end 120, H_max 7d, maturity 48d: training t in [0,113], mature-only t in [0,65] -> 58% of training window
   sim 365d, train_end 240, H_max 7d, maturity 48d: training t in [0,233], mature-only t in [0,185] -> 79% of training window
   ```

Everything else in this document is design, and every performance figure is an estimate that is
NOT YET EVALUATED.

---

## 10. Rust evaluation (summary; full record in 0006)

| Component | Verdict |
|---|---|
| (a) Synthetic event generator | Python only; re-evaluate if RESEARCH generation > 15 min or > 24 GB |
| (a') Settlement pass (found in Phase 0) | Python oracle, then Rust (or numba) port if profiled as the bottleneck |
| (b) As-of temporal edge index / neighbor sampler | Python oracle, then Rust port (expected at M5 to M7): per-seed cutoffs in the training loop, leakage boundary, avoids Windows compiled PyG extensions |
| (c) Streaming incremental updater | Python oracle; decision deferred until a latency target is defined |
| (d) Temporal motif counting | Python oracle; Rust port only if motif features prove useful in ablations |

No component is recommended as "Rust from the start": there is no profiling evidence yet, the Windows
PyO3/maturin build is NOT YET EVALUATED on the laptop, and a Rust generator would break seed-level
reproducibility with the numpy RNG streams.

---

## 11. Environment plan (summary; full record in 0007)

**NOT YET EVALUATED — requires verification on the local GPU machine before Milestone 3+.**

- Primary target: Python 3.11, `torch==2.9.1+cu128` (PyTorch cu128 index), `torch_geometric==2.7.0`
  (pure Python), no `pyg-lib` / `torch-scatter` / `torch-sparse` / `torch-cluster`.
- Alternative if the laptop driver supports CUDA 13: newest torch with a cu130 Windows wheel (2.14.0
  listed on PyPI at the time of writing) and newest PyG (2.8.0.post1 on PyPI).
- Verified in this session from PyPI metadata only: Linux torch 2.9.1 pins CUDA 12.8 runtime packages;
  2.12.1 and 2.14.0 pin CUDA 13.0. Windows CUDA wheel availability could not be checked (403).
- Torch is not needed until Milestone 5, so this does not block Milestones 1 to 4.

---

## 12. Decision records (summary)

| Topic | Recommendation | Main rejected alternative |
|---|---|---|
| Dataframe / storage (0001) | polars + explicit Arrow schemas + Parquet; money as int64 minor units; dataset identity = `fcip-content-hash-v1` (cast to schema, sort by full PK, fixed byte encoding) | pandas primary; file-byte hashing; Arrow IPC bytes |
| RNG streams (0002) | `SeedSequence(seed, spawn_key=(blake2b-64 keys of name path))`, per-entity and per-instance streams, namespaced IDs, rendezvous recruitment partition, `PYTHONHASHSEED` regression test | built-in `hash()`; positional `spawn(n)`; global counters; sequential recruitment |
| As-of view (0003) | sealed `TemporalStore`, frozen single-cutoff `AsOfView` that raises on future reads, vetted `PITQuery` primitives, separate `OracleLabels`, metamorphic + adversarial + import-boundary tests | filtered dataframes by convention; SQL views |
| Labels / time (0004) | int64 seconds, half-open intervals, phase-derived roles, `known_at` from network-level LogNormal(median 14 d, sigma 0.75) latency with `p_never_known` 0.10, enforced everywhere labels are used | final-role labels; per-row latencies; zero-latency default; dropping unknown positives |
| Generator config (0005) | Pydantic v2 (`extra=forbid`, frozen) + YAML overlays, typed distributions, shape-only scenario params, `suspicious_prevalence`, per-instance parameter distributions, `is_ood_family` | Hydra now |
| Language policy (0006) | see section 10 | Rust from the start |
| Environment (0007) | see section 11 | assuming latest torch works with a CUDA 12.8 driver |
| Layout / CLI (0008) | `src/fcip`, `python -m fcip.cli simulate --profile dev --seed 42` | `python -m src.cli` |
| Splits / OOD (0009) | OOD families across the whole timeline, excluded from TRAIN/VAL by family, labels never known, evaluated in an OOD pool against a reference negative pool; label maturity with `include_immature` (default) and `mature_only` modes; evaluation always against full ground truth | OOD confined to after validation; unspecified masking; evaluating against known labels |

---

## 13. Challenges to the spec

Status after the first review is marked in brackets: [RESOLVED] means a decision was taken and folded
into the records; [OPEN] means no decision yet.

1. **Circularity of synthetic evidence (methodologically weak as stated).** "Graph and temporal context
   must matter" is a generator design goal, so demonstrating that graph models win on this generator is
   partly circular. *Proposal*: (a) treat generator regimes as an experimental factor and report curves
   over a pre-registered difficulty grid; (b) phrase all conclusions conditionally on the regime;
   (c) keep G3 (context contribution) as report-only, never a gate; (d) add an external sanity check on
   one public dataset after verifying its card (M3+).

2. **Next-event prediction on scripted scenarios is unmeasurable in a meaningful way** if phases follow
   a fixed script: a model would score near-perfectly by learning the script. *Proposal*: stochastic
   phase grammars and reporting against the generator's own Bayes ceiling (section 5.3).

3. **DEV profile is internally inconsistent.** 10K accounts with 100K transactions over 90 days is
   ~10 transactions per account, too sparse for a salary worker archetype (salary, rent, bills, card
   payments are already dozens per month), so archetypes cannot be heterogeneous and scenarios drown.
   *Proposal*: DEV = ~2K to 3K accounts, 90 days, ~100K to 150K transactions (or 10K accounts over 30
   days). Final numbers after measuring the per-archetype rates (Q2). [RESOLVED: about 2,000 to 3,000 accounts over
   90 days]

4. **Missing event table.** The event taxonomy has NEW_DEVICE_LOGIN, NEW_IP_LOGIN and the edge list has
   LOGIN_WITH and USES_IP, but the M1 output list has no login/session table, so those edges would have
   no timestamps and those events could not exist. *Proposal*: add `logins` and a `relations` table
   with validity intervals. [RESOLVED: `logins` added to schema, outputs and ER model, section 4.5]

5. **Label knowledge time is missing.** Phase-derived labels at t are ground truth, but an institution
   learns them later, if at all. Training on labels not yet known at fit time is a leak that the purge
   gap does not cover. *Proposal*: `known_at` (0004) plus a training-label policy, with a zero-delay
   setting to recover the oracle regime and measure the difference. [RESOLVED: 0004 rewritten; `known_at`
   enforced in splits, training targets and label-derived features; non-degenerate default latency]

6. **"Byte-identical data" is the wrong reproducibility criterion.** Verified in this session: Parquet
   bytes differ between writers (pyarrow vs polars) for identical content, and file footers embed the
   writer version. *Proposal*: canonical content hash as dataset identity; file hashes recorded and
   asserted stable only within one pinned environment. [RESOLVED: algorithm specified in 0001, with a
   cross-writer test]

7. **"Early-warning lead time" as headline metric is undefined** until an alert budget and a treatment
   of never-alerted and censored networks are fixed; any lead time can be achieved with enough alerts.
   *Proposal*: definition in 5.2 (lead time at fixed alerts per 1,000 accounts per day, misses count as
   zero, curve over budgets).

8. **"False alerts per analyst" is unmeasurable** without an analyst capacity model. *Proposal*: replace
   with false alerts per 1,000 accounts per day at a fixed budget, optionally converted with a stated
   assumed capacity.

9. **Non-triviality gates need a floor, not only a ceiling,** otherwise a generator emitting noise
   passes. Iterating the generator until gates pass is also a garden-of-forking-paths risk.
   *Proposal*: G2b floor gate; calibration seeds disjoint from research seeds; generator frozen before
   any model milestone. [RESOLVED: floor kept; exact feature set and
   procedure pre-registered in `EXPERIMENTS.md`]

10. **Held-out scenario families are ambiguous in a chronological split.** If a held-out family occurs
    in the train period and its accounts are labeled NORMAL for training, that is label noise; if it
    is removed from the world, the world differs between experiments. *Proposal* (Q3): schedule held-out
    instances to start after the validation boundary, and mask any overlap from training loss.
    [RESOLVED differently: both options rejected in review; OOD pool design in 0009]

11. **Role taxonomy overlaps and changes over time.** CASH_OUT_RISK (role) and ATM_WITHDRAWAL (event)
    overlap; one account can be AGGREGATOR then DISTRIBUTOR within a network; an account can be in two
    scenarios. *Proposal*: phase-dependent multi-label roles with a configured precedence for
    single-label views (Q4). [RESOLVED for M1: no concurrent membership (S1) and single scenario for life (S2), so roles are single-valued at
    any t; the CASH_OUT_RISK vs ATM_WITHDRAWAL overlap remains a naming issue to settle in `DATA.md`]

12. **Scope is too large for the research value of some items.** Six temporal-GNN families, three
    multi-task weighting schemes and homogeneous-vs-heterogeneous comparisons on synthetic data will cost
    months and yield claims conditional on the generator. *Proposal*: one memory-based temporal GNN plus
    one simple non-memory baseline at M7; multi-task only after single-task heads beat baselines; fixed vs
    uncertainty weighting only.

13. **Streaming Rust evaluation is unmeasurable** without a latency target. *Proposal*: define a target
    (for example p99 per-event update latency and event rate) before M10.

14. **Calibration claims on synthetic data are internal only.** Probabilities are calibrated to the
    generator's prevalence and do not transfer. *Proposal*: state this in LIMITATIONS; use calibration
    for relative comparison of methods only.

15. **"Previously unknown connected accounts found" needs a defined partial-knowledge setup** (which
    accounts are "known" at t). *Proposal*: define an investigation-expansion task: given seed accounts
    with `known_at <= t`, rank other accounts; evaluate with recall@K of true network members.

16. **Tooling timing.** Hydra and MLflow add complexity with no benefit in M1. *Proposal*: Pydantic + YAML
    now (0005); experiment tracking from M3.

17. **Per-component Rust decision records** for every new module would produce paperwork without
    information. *Proposal*: a per-milestone language-verdict table in the review packet, with 0006 updated
    only when a verdict changes.

18. **"Realistic" confounders cannot be verified** without real data. *Proposal*: call them "plausible",
    document each confounder's intended look-alike, and measure its effect (G4) rather than asserting
    realism.

19. **`python -m src.cli`** uses `src` as a package name (0008). *Proposal*: `src/fcip`.

---

## 14. Open questions for the reviewer

Resolved in the first review: former Q3 (OOD pool, 0009), Q4 (single membership), Q5 (gate numbers kept), Q6
(`known_at`), Q7 (`fcip`), Q8 (`suspicious_prevalence` = 0.01, configurable).

Resolved in the round-1 revision review: Q-R1 (for-life partition accepted, recorded as S2 with a relaxation
requirement), Q-R2 (PU reading confirmed; maturity added), Q-R3 (delay defaults accepted; limitation and planned
sensitivity analysis recorded in `LIMITATIONS.md`), Q-R4 (gate profile by runtime), Q-R5 (gate probes on ground
truth), Q-R6 (distribution shift lives under Evaluation; spec references resolved by topic from now on), Q2 (DEV
about 2,000 to 3,000 accounts over 90 days).

Still open:

- **Q1** Environment: `nvidia-smi` output on the laptop. Deferred by the reviewer to before Milestones 3 to 5; does not
  block Milestone 1.

New in round 2:

- **Q-M1** Maturity ablation feasibility. With the requested default (`maturity_horizon_days` = 48) and a 90-day
  simulation, `mature_only` keeps only prediction times from day 0 to 5 (about 9% of the training window, with at most
  5 days of history), and the validator will reject it. Options: (a) run EXP-LM on a `RESEARCH_LONG` profile of 180 days
  (58% mature), keeping 48 days; (b) keep 90 days and sweep `maturity_horizon_days` in {14, 28, 48}, accepting that 14
  and 28 leave more noise (residual unknown rates about 0.55 and 0.26 at those distances from the anchor); (c) both.
  Recommendation: (c), with (a) as the primary arm. This affects Milestone 3 onward, not the Milestone 1 build, where both
  modes and the validator are implemented regardless.
- **Q-M2** Maturity reference time. I defined `reference_time = t + H` (end of the target window). An alternative is `t`
  itself, which is simpler but more conservative (drops more data). Accept `t + H`?
