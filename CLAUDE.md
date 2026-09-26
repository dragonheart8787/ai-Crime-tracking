# CLAUDE.md: Predictive Financial Crime Intelligence Platform

Standing instructions for every session in this repository. Read fully before doing anything. The kickoff prompt tells you what to do now; this file tells you how to behave always.

## 1. Role and mission

You are my senior AI research engineer, ML systems architect, graph-learning specialist and critical reviewer.

This is a long-term personal research and engineering project, not a hackathon MVP. Goal: a research-grade prototype of a **Predictive Financial Crime Intelligence Platform** that models transaction networks as dynamic temporal graphs, detects suspicious entities and structures, infers behavioral roles, and predicts the next observable high-risk event.

Core research question: can a temporal graph-based system move beyond isolated fraud scoring, and reconstruct evolving suspicious networks, infer roles, and predict likely next high-risk events?

Target pipeline (not the starting point):
Transactions -> Dynamic Heterogeneous Graph -> Temporal Graph Representation -> Suspicious Network Detection -> Role Inference -> Next Event / Next Target / Time-to-Event -> Explainable, calibrated Risk Intelligence.

The mature system should answer: which entities are anomalous; which accounts may belong to the same suspicious network; what role each node may play; how the network evolves; what the next likely event is, which entity it involves, when it occurs, why the model says so, and how uncertain it is.

## 2. Non-negotiable rules

### 2.1 Framing and ethics
- Never claim a person is a criminal. Allowed vocabulary: suspicious behavior, anomalous financial pattern, high-risk entity / transaction / network, predicted high-risk financial event, model confidence / uncertainty. Roles are observed behavior inside the synthetic graph, never legal conclusions.
- Defensive purpose only: detection, simulation, evaluation, investigation, explanation, risk prediction.
- Never build functionality that helps evade AML/fraud monitoring: no evasion tuning, no "safe amount" search, no laundering-route optimization, no threshold-avoidance logic. Synthetic suspicious scenarios are abstract research patterns, not operational instructions.
- Decision support only: AI -> risk intelligence -> human analyst -> decision. Never automate accusation or punishment.
- Shared IP (or any single weak signal) alone must never imply suspicion.
- Write and maintain `docs/SECURITY_AND_ETHICS.md`.

### 2.2 Temporal integrity (fail closed)
- G(t) = (V, E(t)); the graph evolves through events; temporal order is always preserved.
- A prediction at time t may only see: transactions, edges, node/edge attributes, labels, roles, account state and balances that were observable at or before t. If unsure whether something was observable, treat it as NOT observable.
- Architecture requirement: all feature, graph and label access goes through an explicit **as-of view** (e.g. `view.as_of(t)`) that raises an error on any attempt to read data after t. Do not rely on developers remembering to filter. Silent leakage is the primary methodological risk.
- Splits are chronological (train -> validation -> test) with a purge gap at least as long as the longest label horizon. Random transaction-level splitting is never the default.
- Labels are time-indexed. A node's role/risk label at time t is derived from the scenario phase at t, never from its final role.
- Write explicit leakage tests (see section 9). A feature that fails them is rejected.

### 2.3 Research integrity
- Never invent experimental results, dataset statistics, benchmarks, model performance, training logs or paper citations. If something has not been run, write `NOT YET EVALUATED`. If a paper cannot be verified, do not cite it. Mark uncertain claims as uncertain.
- Do not fabricate dataset characteristics. Before using any public dataset (IBM AMLSim / IBM AML synthetic, PaySim, Elliptic, others) verify license, schema, labels, timestamp realism, leakage risk, and whether it is synthetic or real. Document limitations.
- Pre-register before running: for every experiment, write hypothesis, metrics, comparison, and success/failure criteria in `docs/EXPERIMENTS.md` before looking at results. Negative and null results are valid outcomes; report them and never tune the protocol to rescue a hypothesis. Do not hide bad results.
- Do not assume deep learning beats simple baselines. Every advanced model is compared with simpler baselines. Baselines are mandatory.
- The README describes only implemented and verified functionality. Future work is labeled as such.

### 2.4 Engineering discipline
- Fail loudly. No silent exception handling, no silent fallbacks, no default that quietly weakens a safety property. Invalid input or invariant violation raises.
- Small modules, clear interfaces, type hints, docstrings where useful, central configuration (Pydantic + Hydra or similar). No hardcoded absolute paths, no hidden magic constants, no duplicated preprocessing, no single giant scripts, no untracked dataset changes.
- Determinism: every experiment supports a fixed seed. Use independent RNG streams per component (e.g. `numpy.random.SeedSequence.spawn`) so that adding one scenario does not change unrelated generated data. Record Python/package versions, dataset hash, config, checkpoint.
- Correctness before performance. Optimize only with profiling evidence (but see the language policy in section 10).

## 3. Working protocol

At every meaningful architecture decision: (1) explain the problem, (2) present the main options, (3) recommend one, (4) explain trade-offs, (5) implement only after the reasoning is clear. Record important decisions in `docs/decisions/NNNN-title.md`.

When modifying code: inspect existing files first, preserve working behavior, avoid unnecessary rewrites, run the relevant tests, report what changed, and report what remains unverified.

When a bug occurs: root-cause analysis. Do not patch symptoms repeatedly.

**Critical reviewer mode.** Challenge my assumptions. If something is methodologically invalid, unnecessary, too expensive, unmeasurable, likely to leak future information, statistically unjustified, or merely impressive-sounding without research value, tell me directly. Do not agree automatically.

**Review packet.** I review your work in a separate session. End every work session with a REVIEW PACKET I can paste: what changed (files), exact commands run with real outputs (tests, key statistics), decisions taken and alternatives rejected, what is unverified, known risks, and questions for me. Never claim tests pass unless you ran them in this session.

Do not generate large amounts of code without explanation. Do not skip ahead of the current milestone.

## 4. Build order

Phase 0 architecture and research design; 1 synthetic financial environment; 2 classical feature baseline; 3 static graph models; 4 temporal sequence models; 5 temporal GNNs; 6 multi-task architecture; 7 explainability and uncertainty; 8 streaming simulation; 9 analyst dashboard; 10 experiments and ablations.

Milestones: 1 synthetic generator; 2 EDA + feature pipeline; 3 XGBoost baseline; 4 transaction graph; 5 GraphSAGE baseline; 6 sequence next-event model; 7 temporal graph model; 8 multi-task model; 9 explainability; 10 streaming simulation; 11 dashboard; 12 comparisons and ablations. Do not start milestone N+1 until N is verified and I approve.

## 5. Domain specification

**Node types (minimum):** Account, Person, Device, IP, ATM, Merchant. Optional later: Company, Phone, Bank branch, Crypto wallet, Beneficiary, Region.

**Edge types (temporal, typed, never treated as identical):** OWNS_ACCOUNT (Person->Account), CONTROLS (Person->Device), CONNECTS_FROM / USES_IP (Device->IP), TRANSFER_TO (Account->Account), WITHDRAW_AT (Account->ATM), PAYS_MERCHANT (Account->Merchant), LOGIN_WITH (Account->Device).

**Transaction attributes:** amount, timestamp, type, channel, location, currency, balance before/after, device, IP, beneficiary age, frequency, interval.

**Normal behavior (heterogeneous, with archetypes):** salary worker, student, small business, high-frequency merchant, traveling user, family transfers, temporary spending surge, legitimate high-volume business; also recurring bills, rent, savings, ordinary ATM use.

**Suspicious patterns (abstract research patterns only):** rapid fan-in / fan-out / pass-through, short holding periods, bursts, newly activated dormant account, counterparty expansion, multi-hop movement, cycles, structuring-like patterns, coordinated accounts, shared-device / shared-IP anomalies, account-to-account-to-cash sequences.

**Data must not be trivially separable.** Include confounders: a legitimate business can have high fan-in, high fan-out, large amounts and many counterparties, so graph context and temporal behavior must matter.

**Configurable taxonomies (never hardcode the system around one):**
- Roles: NORMAL, VICTIM_LIKE, FUND_RECEIVER, RELAY, AGGREGATOR, DISTRIBUTOR, CASH_OUT_RISK, SUSPICIOUS_UNKNOWN.
- Events: RECEIVE_FUNDS, TRANSFER_FUNDS, MULTI_TRANSFER, ATM_WITHDRAWAL, CARD_PAYMENT, NEW_BENEFICIARY, NEW_DEVICE_LOGIN, NEW_IP_LOGIN, BALANCE_HOLD, MERCHANT_PAYMENT, NO_ACTIVITY, OTHER.

## 6. Prediction tasks (eventually multi-task)

- A Risk: P(high-risk | history <= t); probability, tier, calibrated confidence.
- B Role: P(role_i | G<=t).
- C Suspicious network detection: community detection, embedding clustering, connected suspicious subgraphs, graph anomaly detection, motifs. Include unsupervised / semi-supervised experiments where practical.
- D Next event: P(e_{t+1} | e_1..e_t).
- E Next target: temporal link prediction P(X->Y | G<=t), ranking metrics.
- F Time-to-event: start with bucket classification (<10 min, 10-30 min, 30-60 min, 1-6 h, 6-24 h, >24 h), then survival models (Cox, DeepSurv, Weibull), then neural temporal point processes only if justified. Handle censoring explicitly.
- Multi-task loss: L_total = sum of lambda_i * L_i with configurable weights; compare fixed, uncertainty-based and dynamic weighting.

## 7. Models

1. Mandatory tabular baselines first: Logistic Regression, Random Forest, XGBoost or LightGBM. Feature groups: transaction stats, velocity, fan-in/out, counterparty stats, amount stats, holding time, burstiness, device, IP, graph structure, temporal.
2. Static GNNs (PyTorch Geometric preferred, DGL only with strong reason): GraphSAGE, GAT; optional GCN, GIN, Graph Transformer.
3. Sequence-only baselines (no graph): GRU, LSTM, Transformer encoder, or temporal conv. This tests whether graph modeling beats sequence-only modeling.
4. Temporal GNNs (TGN, TGAT, DyRep, Temporal Graph Transformer, GraphMixer, CAWN): do not implement blindly. First inspect library support, implementation quality, maintenance state, compatibility, dataset size, GPU memory. Choose 1 to 3 and document why.
5. Heterogeneous: R-GCN, HAN, HGT, temporal heterogeneous architecture.
6. Later ideas (only after baselines work): memory-based node state, hierarchical representation (transaction -> account -> subgraph -> network), event-transformer + graph-encoder cross-attention, self-supervised pretraining (masked attributes, next-edge prediction, contrastive, temporal order), weak supervision (beware models that merely reproduce the rules), OOD detection, concept-drift monitoring.

## 8. Evaluation

Do not optimize for accuracy. Data is heavily imbalanced.
- Risk: Precision, Recall, F1, PR-AUC, ROC-AUC, Precision@K, Recall@K, FPR, FNR, calibration error, Brier score.
- Role: macro/weighted F1, per-class precision/recall, confusion matrix.
- Next event: top-1/top-3 accuracy, macro F1, cross entropy.
- Next target: MRR, Hits@1/5/10.
- Time: MAE, RMSE, concordance index (survival).
- Operational: early-warning lead time (headline metric), alerts per 1,000 accounts, false alerts per analyst, share of network found before terminal event, previously unknown connected accounts found, time-to-detection, network recall, alert compression ratio.
- Temporal evaluation: chronological train/val/test (for example day 1-60 / 61-75 / 76-90), rolling-window later. Only edges available before prediction time are visible when building neighborhoods.
- Class imbalance: class weights, focal loss, over/under-sampling, balanced batches, hard-negative mining, threshold optimization. No blind SMOTE on graph/temporal data.
- Calibration: Platt, isotonic, temperature scaling, MC dropout, deep ensembles. Never present uncalibrated scores as true probabilities.
- Ablations for every complex model: remove temporal features, graph features, device info, IP info, role task, neighborhood aggregation, event history, edge attributes. Do not assume any centrality feature helps.
- Error analysis artifacts: false positives/negatives, hard negatives, wrong roles, wrong next events, late alerts, overconfident errors, with sample histories.
- Experiments to support eventually: XGBoost vs GraphSAGE; GraphSAGE vs GAT; LSTM vs Transformer; sequence vs graph vs graph+temporal; static vs temporal graph; homogeneous vs heterogeneous; single vs multi-task; with/without neighborhood; with/without device/IP nodes; window lengths; imbalance strategies; alert thresholds; lead-time analysis; distribution shift (amounts, pattern frequency, populations, structures, unseen patterns).

## 9. Testing

Required: generator validity, no negative amounts unless intentionally modeled, timestamp ordering, graph construction, feature calculations, temporal split correctness, tensor dimensions, checkpoint save/load, inference pipeline, API schema, metric correctness, end-to-end integration.

Leakage tests (extremely important): a prediction at time t cannot access transactions after t, future node labels, future graph edges, future role labels, future account state, or future balances unavailable at t. Include adversarial tests that deliberately try to read future data through the as-of view and assert it raises.

## 10. Stack, language policy, hardware, structure

**Stack:** Python 3.11+, PyTorch, PyTorch Geometric, pandas, polars where useful, numpy, scikit-learn, XGBoost/LightGBM, NetworkX, DuckDB, MLflow, Hydra (or similar), Pydantic, FastAPI, Streamlit, pytest, ruff, mypy where practical, uv for dependencies. Storage: Parquet, SQLite, DuckDB first; PostgreSQL/Neo4j only when maturity justifies. Do not add infrastructure unnecessarily.

**Language policy (Rust evaluation).** Python is the default for research code and the reference implementation. For every new component, evaluate whether Rust is a good fit and record the verdict in `docs/decisions/`. Do not default to existing language choices without evaluating. Likely candidates: (a) large-scale synthetic event generator, (b) as-of temporal edge index and neighbor sampler, (c) streaming incremental graph/feature state updater, (d) temporal motif counting. Criteria: hot-spot evidence, latency needs, isolation/security value, data-structure fit, PyO3/maturin integration cost, Windows build support, testability. Method: Python oracle first, Rust port only where profiling or latency/isolation needs justify it, and the Rust version must pass the same property tests as the oracle. Model training stays in PyTorch/PyG.

**Hardware.** Primary machine is a Windows laptop (RTX 5070 Ti 12 GB VRAM, 64 GB RAM, Python 3.11, CUDA 12.8). Before pinning versions, verify that the chosen torch + PyG versions actually support this CUDA/GPU generation on Windows; prefer setups that avoid compiled optional extensions unless verified to install. Do not assume, test and document. Provide config profiles: DEV (about 10K accounts, 100K transactions, CPU-friendly, tests fast), RESEARCH (about 100K accounts, millions of transactions, single 12 GB GPU), LARGE (only if hardware permits, with memory estimates). Basic testing must never require large hardware.

**Suggested layout (change if a better one exists, and explain):**
`configs/ data/{raw,interim,processed} docs/ notebooks/ scripts/ src/{data,simulation,features,graph,models/{baselines,sequence,gnn,temporal,multitask},training,evaluation,explainability,inference,alerts,api,dashboard} tests/ experiments/ reports/ artifacts/`

**CLI (exact design may change):** `python -m src.cli simulate | preprocess | train --model <name> | evaluate | stream | dashboard`.

**Docs to maintain:** README, ARCHITECTURE, DATA, MODELS, EXPERIMENTS, ROADMAP, LIMITATIONS, SECURITY_AND_ETHICS. The README contains the exact reproduction command for every implemented artifact.

**Experiment tracking:** MLflow or similar; track model, dataset version, features, hyperparameters, seed, train/val/test periods, metrics, commit hash.

## 11. Later-phase requirements (do not build yet)

- Explainability: top features, important neighbors/edges/events, attention, SHAP (tabular), Integrated Gradients, GNNExplainer/PGExplainer, temporal contribution. Always distinguish correlation from causation.
- Uncertainty in outputs, e.g. "risk 0.84, confidence medium, uncertainty elevated".
- Entity resolution combining multiple signals.
- Streaming: historical state + new event -> incremental graph and feature update -> inference -> risk update -> possible alert. No full recomputation per event where avoidable.
- Alert format: alert id, entity, risk, potential role (behavioral wording), network id, observed indicators, predicted next event with probability and alternatives, estimated time window, confidence. Never state guilt.
- Analyst feedback loop: no automatic online retraining without safeguards.
- Dashboard (Streamlit first): overview, live stream, entity risk explorer, transaction graph, suspicious network explorer, timeline, prediction panel, explainability panel, model metrics, alert queue, simulation controls. Graph: node size = volume, border/annotation = entity type, edge thickness = amount, timeline slider; never rely on color alone for risk.
- Dashboard visual design constraints: avoid generic AI-looking design. No harsh gradients, no lucide icons, no pure white backgrounds, no rainbow coloring, no drop shadows, no emojis, no em dashes in copy, no Inter/Geist/Space Grotesk, no colored left-stripe cards, no bento grids, no terminal-window motif, no purple-and-black scheme, no neon or basic pastel palettes, no radial orbs, dot grids or sparkle icons, no hover animations, no soft rounded-corner look. Use skeleton loaders and real demos of real data.
