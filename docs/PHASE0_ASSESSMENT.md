# Phase 0 Assessment: Predictive Financial Crime Intelligence Platform

- Date: 2026-09-26
- Session type: cloud planning session (no GPU, allowlisted network). Planning and assessment only.
- Status: **awaiting review**. No generator, feature, model or pipeline code exists.

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
   |                                          event_labels, ground_truth_networks, metadata
   v
temporal/  TemporalStore (sealed) --as_of(t)--> AsOfView      (0003)
                                  --point_in_time(ids, ts)--> PITQuery primitives
           splits: chronological, purged by H_max (0004)
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
3. **Events are not only transactions.** Logins (account, device, IP, ts) are a separate event table.
   Without it, `NEW_DEVICE_LOGIN` / `NEW_IP_LOGIN` and the Device/IP edges have no timestamps
   (Challenge 4).
4. **Relations are temporal**: OWNS_ACCOUNT, CONTROLS, USES_IP, etc. carry `valid_from` / `valid_to`.
   An account opened on day 40 is invisible on day 39.
5. **Money is conserved explicitly**: external source/sink accounts (employers, billers, the outside
   world) make every inflow and outflow a transfer between accounts, so conservation is checkable.

### 2.2 Generator internals (planned for M1, described not built)

Two-stage design:

1. **Intent generation** (vectorized, per named RNG stream): each normal person's archetype produces
   intended events (salary, rent, bills, card payments, ATM withdrawals, family transfers, logins);
   each scenario instance produces intended events per its stochastic phase grammar; confounder
   populations produce their own intents.
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
| R3 | **Label-knowledge leakage**: training on labels an institution would not yet know | Optimistic early-warning results | `known_at` column and training policy (0004); report oracle-label vs delayed-label regimes |
| R4 | **Trivial separability** (a single feature or amount band gives it away) | Everything looks good, nothing is learned | Pre-registered non-triviality gates with both a ceiling and a floor (sections 5, 7) |
| R5 | **Unlearnable data** (gates pushed too far, signal destroyed) | Null results caused by the generator, not the models | Floor gate; Bayes / oracle-feature ceiling showing signal exists |
| R6 | **Garden of forking paths on the generator** (tuning the generator until later models look good) | Hidden researcher degrees of freedom | Calibration seeds separate from the locked research seed; generator frozen and versioned before any model milestone; any later generator change is a new dataset version with its own entry in `EXPERIMENTS.md` |
| R7 | **Deterministic scenario scripts** make next-event prediction trivial | Next-event accuracy measures the script | Stochastic phase grammars; compute the generator's true conditional next-event distribution as the Bayes-optimal reference |
| R8 | **Transductive memorization**: time split, but the same entities and networks appear in train and test | Models memorize IDs / embeddings | Report separately for networks that started before vs after the train cutoff; held-out families; no ID embeddings in baselines |
| R9 | **Evaluation variance**: few suspicious networks per split | Confidence intervals wider than differences between models | Cluster bootstrap over network instances; multiple generator seeds for final claims; enough instances in RESEARCH profile (power check before M3) |
| R10 | **Dual-use drift** (generator knobs becoming evasion tools) | Violates 2.1 | Schema has shape-only parameters; no detector-in-the-loop; review checklist in `SECURITY_AND_ETHICS.md` |
| R11 | **Environment** (Blackwell GPU + Windows + PyG extensions) | Blocks M5+ | Torch not needed until M5; verification plan in 0007; in-house sampler removes compiled-extension dependency |
| R12 | **Scope creep** | Project never reaches the core comparison | Roadmap cuts in section 6 |

---

## 4. DATA STRATEGY

### 4.1 Primary: in-house synthetic generator

Rationale: only a generator provides time-indexed roles and phases, network membership, terminal
events and controllable difficulty, all needed by the research question.

**Entities**: persons, accounts, devices, IPs, ATMs, merchants, plus explicit external accounts
(employers, billers, "outside world") for conservation of funds.

**Event tables**: `transactions` (transfer, card/merchant payment, ATM withdrawal, deposit, salary,
bill; channel, region, currency, device, IP, status, balance before/after) and `logins`
(account, device, IP, ts, outcome). Beneficiary age, frequency and interval are *derived features*
computed through the as-of view, not stored columns, to avoid storing values that could encode the
future.

**Currency**: single currency in M1 (multi-currency adds FX conservation complexity with little
research value now).

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

**Held-out families**: families flagged `split_role: held_out` are excluded from training and used for
out-of-distribution evaluation. Placement of their instances in time is an open question (Q3).

### 4.2 Profiles (sizes to be confirmed, see Challenge 3)

| Profile | Accounts | Days | Transactions | Purpose |
|---|---|---|---|---|
| DEV | 2K to 3K (proposed) | 90 | ~100K to 150K | tests, CI, fast iteration on CPU |
| RESEARCH | ~100K | 90 | order of 10^7 (estimate, NOT YET EVALUATED) | main experiments |
| LARGE | only with memory estimate | | | not planned before M12 |

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

---

## 5. EVALUATION STRATEGY

### 5.1 Prediction points and splits

- Predictions are made at **prediction points** `(entity, t)`: a fixed cadence (e.g. daily snapshot)
  plus event-triggered points (after each observed event, for next-event tasks).
- Chronological split on prediction time, e.g. days 1 to 60 / 61 to 75 / 76 to 90, with purge gap
  >= `H_max` before validation and before test; training rows require `t + H_max <= train_end`.
- Neighborhoods, features and labels at `t` come only from the as-of view (0003).
- Metrics are reported separately for (a) networks active across the train cutoff and (b) networks
  that started after it (R8), and for (c) held-out families.

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

### 5.5 Milestone 1 non-triviality gates (draft; to be finalized and pre-registered in `EXPERIMENTS.md` before the final dataset)

Target for the probes: "account is in an active scenario phase at snapshot t" (and, secondarily,
"has a high-risk event within H"). Features: ~20 simple per-account trailing-window aggregates
computed through the as-of view. Probes are fit on calibration-seed train periods and scored on
their validation periods.

| Gate | Proposed criterion (draft) |
|---|---|
| G1 single feature | best single-feature PR-AUC (either sign) <= **0.30** at a prevalence of roughly 1% |
| G2a shallow probe ceiling | logistic regression and depth-3 tree on per-account statics: PR-AUC <= **0.60** |
| G2b signal floor | the same probe: PR-AUC >= **3 x prevalence** (otherwise the data may be unlearnable) |
| G3 context contribution | report PR-AUC gain from adding 1-hop neighbor and temporal-order aggregates, with bootstrap CI. **Report only, not a gate** (making it a gate would tune the generator to favor the hypothesis) |
| G4 confounder presence | among the probe's top-K false positives, the share coming from confounder archetypes is reported (report only) |

The numbers above are placeholders for discussion; the user sets them (Q5).

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
5. Scenario families with stochastic phase grammars, `enabled` and `split_role` per family.
6. Settlement pass with overdraft rule, declined status and balances.
7. Outputs: `transactions`, `logins`, `accounts`, `persons`, `devices`, `ips`, `atms`, `merchants`,
   `relations`, `labels`, `event_labels`, `ground_truth_networks`, `metadata.json` (Parquet primary,
   CSV on request).
8. `TemporalStore` / `AsOfView` / `PITQuery` minimal versions (enough for splits, probes and leakage
   tests).
9. Chronological split utility with purge gap.
10. Validation: invariants, heterogeneity statistics, non-triviality probes, EDA report generated from
    computed numbers only.
11. CLI `simulate`, `validate`, `eda`.
12. Docs: README reproduction commands, `DATA.md`, `EXPERIMENTS.md` (gate pre-registration),
    `ARCHITECTURE.md`, `LIMITATIONS.md`.

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
- Reproducibility: same seed gives same content hash; different seeds differ; toggling a scenario
  leaves untouched entities' events identical by `event_id`.
- Leakage: adversarial and metamorphic tests from 0003.

Out of scope: any trained model beyond the probes, feature pipeline beyond probe features, graph
construction, torch.

---

## 8. EXACT FILES I PLAN TO CREATE OR MODIFY (when implementation starts, after approval)

Created in this Phase 0 session: `CLAUDE.md`, `README.md`, `.gitignore`,
`docs/PHASE0_ASSESSMENT.md`, `docs/SECURITY_AND_ETHICS.md`, `docs/decisions/0001` to `0008`.

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
src/fcip/common/rng.py              # named streams (0002)
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
src/fcip/simulation/scenarios/{__init__,base,phases,fan_in,fan_out,pass_through,burst,
                               dormant_activation,multi_hop,cycle,structuring_like,
                               shared_infrastructure,account_to_cash}.py
src/fcip/simulation/logins.py
src/fcip/simulation/settlement.py   # ledger pass (0006 a')
src/fcip/simulation/labels.py       # labels, event_labels, ground_truth_networks
src/fcip/simulation/generator.py    # orchestration only
src/fcip/simulation/metadata.py
src/fcip/temporal/store.py          # TemporalStore (0003)
src/fcip/temporal/asof.py           # AsOfView, FullHistoryView (EDA only)
src/fcip/temporal/pit.py            # PITQuery primitives
src/fcip/temporal/splits.py         # chronological purged splits
src/fcip/labels/oracle.py           # OracleLabels, target builder
src/fcip/validation/invariants.py
src/fcip/validation/stats.py        # heterogeneity statistics
src/fcip/validation/probes.py       # non-triviality probes (only modeling allowed in M1)
src/fcip/validation/eda.py          # writes reports/eda_milestone1.md

tests/conftest.py                   # tiny-profile fixtures
tests/unit/test_config.py
tests/unit/test_rng.py
tests/unit/test_ids.py
tests/unit/test_hashing.py
tests/unit/test_schemas.py
tests/unit/test_settlement.py       # conservation, overdraft, declined
tests/unit/test_scenarios.py        # per-family shape and causality
tests/unit/test_labels.py           # interval invariants, phase-dependent roles
tests/unit/test_asof.py             # API semantics
tests/unit/test_pit.py              # Hypothesis vs brute-force oracle
tests/unit/test_splits.py
tests/unit/test_metrics_sanity.py   # PR-AUC etc. on known cases
tests/leakage/test_adversarial_asof.py
tests/leakage/test_future_perturbation.py
tests/leakage/test_import_boundaries.py
tests/reproducibility/test_seed_determinism.py
tests/reproducibility/test_scenario_isolation.py
tests/integration/test_dev_end_to_end.py

docs/ARCHITECTURE.md
docs/DATA.md
docs/EXPERIMENTS.md                 # M1 gate pre-registration first
docs/LIMITATIONS.md
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
| Dataframe / storage (0001) | polars + explicit Arrow schemas + Parquet; dataset identity = canonical content hash | pandas primary; file-byte hashing (verified writer-dependent) |
| RNG streams (0002) | `SeedSequence(seed, spawn_key=stable_hash(name path))`, per-entity and per-scenario-instance streams, namespaced IDs | positional `spawn(n)`; global counters |
| As-of view (0003) | sealed `TemporalStore`, frozen single-cutoff `AsOfView` that raises on future reads, vetted `PITQuery` primitives, separate `OracleLabels`, metamorphic + adversarial + import-boundary tests | filtered dataframes by convention; SQL views |
| Labels / time (0004) | int64 seconds from epoch, half-open intervals, interval `labels` with phase-derived roles and `known_at`, `event_labels` with terminal flag, censoring | final-role point labels |
| Generator config (0005) | Pydantic v2 (`extra=forbid`, frozen) + YAML overlays, typed distributions, shape-only scenario params | Hydra now |
| Language policy (0006) | see section 10 | Rust from the start |
| Environment (0007) | see section 11 | assuming latest torch works with a CUDA 12.8 driver |
| Layout / CLI (0008) | `src/fcip`, `python -m fcip.cli simulate --profile dev --seed 42` | `python -m src.cli` |

---

## 13. Challenges to the spec

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
   days). Final numbers after measuring the per-archetype rates (Q2).

4. **Missing event table.** The event taxonomy has NEW_DEVICE_LOGIN, NEW_IP_LOGIN and the edge list has
   LOGIN_WITH and USES_IP, but the M1 output list has no login/session table, so those edges would have
   no timestamps and those events could not exist. *Proposal*: add `logins` and a `relations` table
   with validity intervals.

5. **Label knowledge time is missing.** Phase-derived labels at t are ground truth, but an institution
   learns them later, if at all. Training on labels not yet known at fit time is a leak that the purge
   gap does not cover. *Proposal*: `known_at` (0004) plus a training-label policy, with a zero-delay
   setting to recover the oracle regime and measure the difference.

6. **"Byte-identical data" is the wrong reproducibility criterion.** Verified in this session: Parquet
   bytes differ between writers (pyarrow vs polars) for identical content, and file footers embed the
   writer version. *Proposal*: canonical content hash as dataset identity; file hashes recorded and
   asserted stable only within one pinned environment.

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
   any model milestone.

10. **Held-out scenario families are ambiguous in a chronological split.** If a held-out family occurs
    in the train period and its accounts are labeled NORMAL for training, that is label noise; if it
    is removed from the world, the world differs between experiments. *Proposal* (Q3): schedule held-out
    instances to start after the validation boundary, and mask any overlap from training loss.

11. **Role taxonomy overlaps and changes over time.** CASH_OUT_RISK (role) and ATM_WITHDRAWAL (event)
    overlap; one account can be AGGREGATOR then DISTRIBUTOR within a network; an account can be in two
    scenarios. *Proposal*: phase-dependent multi-label roles with a configured precedence for
    single-label views (Q4).

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

- **Q1** Environment: what does `nvidia-smi` report on the laptop (driver version, "CUDA Version")? This
  decides cu128 (torch 2.9.1) versus cu130 (latest torch).
- **Q2** DEV profile size: accept ~2K to 3K accounts over 90 days, or 10K accounts over 30 days?
- **Q3** Held-out families: schedule them after the validation boundary (recommended), or allow them
  anywhere and mask?
- **Q4** Concurrent scenarios / roles: allow an account in more than one scenario (multi-label roles), or
  forbid overlap in M1 for simplicity?
- **Q5** Gate thresholds: accept the draft values in 5.5 as a starting point for pre-registration, or set
  your own?
- **Q6** Investigation delay (`known_at`): include in M1 output (recommended, cheap) even though it is
  only used from M3?
- **Q7** Package name `fcip`: acceptable?
- **Q8** Prevalence target: roughly 1% of accounts involved in scenarios over 90 days, or another value?
