# Predictive Financial Crime Intelligence Platform

Defensive research prototype for fraud / AML detection on **synthetic** transaction data:
temporal graph modeling, suspicious-network detection, behavioral role inference and
next-high-risk-event prediction, as decision support for human analysts.

## Status

**Milestone 1 (synthetic financial environment and data validation) is implemented.** Implemented and tested:

- configurable synthetic generator (profiles `tiny` for tests, `dev`, `research`), deterministic per seed, with named,
  independent RNG streams and a canonical content hash (`fcip-content-hash-v1`);
- entities and temporal relations, transactions and logins; eight normal archetypes with confounders; ten abstract
  suspicious-pattern families (two designated as OOD holdout families); settlement with exact conservation of funds
  in a single synthetic currency;
- time-indexed ground truth: phase-dependent role intervals starting at each member's first own event, event labels,
  networks, members, network-level `known_at`;
- a sealed temporal store with an as-of view that refuses future reads, point-in-time reference features and a fast
  daily-snapshot grid verified against them;
- chronological splits with purge gap, OOD holdout pool, archetype-stratified reference negatives, label maturity with
  both training modes, and a two-tier label accessor (evaluation only ever against full ground truth);
- leakage tests (adversarial future reads, future perturbation, import boundaries, `known_at`), invariant checks,
  the pre-registered non-triviality gate run and an EDA report.

**No trained model of any kind exists** apart from the logistic-regression and depth-3 tree probes inside the gate.
Milestone 2 has not started.

## Reproduce from a clean environment

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/). Datasets are written to `data/processed/` (not
committed).

```bash
uv sync                                                          # environment from uv.lock

# DEV dataset (about 12 s, 0.5 GB)
uv run python -m fcip.cli simulate --profile dev --seed 42
uv run python -m fcip.cli validate --data data/processed/dev_seed42
uv run python -m fcip.cli sanity   --data data/processed/dev_seed42

# RESEARCH calibration datasets for the gate (about 9 min and 12 GB peak memory each; run one at a time)
for s in 1000 1001 1002 1003 1004; do
  uv run python -m fcip.cli simulate --profile research --seed $s
done

# Pre-registered EXP-M1-G gate (one process per seed, about 6 min and 10.5 GB each), then the combined verdict
for s in 1000 1001 1002 1003 1004; do
  uv run python -m fcip.cli gate --data data/processed/research_seed$s \
      --ref-pool-method stratified_archetype --out reports/gate/gate_seed$s.json
done
uv run python -m fcip.cli gate-combine --inputs reports/gate/gate_seed100{0,1,2,3,4}.json --out reports/gate_exp_m1_g.json

# EDA report
uv run python -m fcip.cli eda --data data/processed/research_seed1000 --out reports/eda_milestone1.md
```

Expected dataset hashes (identity is the content hash, not file bytes; decision 0001):

| Dataset | Dataset hash |
|---|---|
| `dev`, seed 42 | `72e97efacef3448f1a29be8d0c1e755a2d3f051fbd136e3c134043e00bb64fef` |
| `research`, seed 1000 | `e311027f7f7a82c9a6d36324cf846221a6ad74429b5961634acdeae818a0e054` |
| `research`, seed 1001 | `7352d7ff97e360a0427c88310b004a6e341e5c35239a6a9fc467c4fc58536df1` |
| `research`, seed 1002 | `3efb2cffc359fef8545bb7d91f0e7820db0fbad73046aa1ec811c2f8fcf05548` |
| `research`, seed 1003 | `c42d6eacb6f2b383bedaf00db0b1cc08779edc0718e8d5edefa1baa892f9931d` |
| `research`, seed 1004 | `43ce496f259f810f1049f87b9119d35aaa737e39991d7be73ed00115062163cd` |

`simulate` prints the dataset hash; `validate` recomputes it from the Parquet files and re-runs every invariant check.
Add `--csv` to `simulate` (or run `export-csv --data <dir>`) for CSV copies. `metadata.json` records the resolved
config, seed, generator commit, package versions, per-table hashes and file SHA-256. Tables `person_truth`, `labels`,
`event_labels`, `ground_truth_networks` and `network_members` are **oracle** tables and must never be model inputs.

## Tests

```bash
uv run pytest                  # full suite, including the DEV end-to-end run and subprocess reproducibility tests
uv run pytest -m "not slow"
uv run ruff check src tests && uv run ruff format --check src tests && uv run mypy src
```

## Documentation

- Architecture and enforced boundaries: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
- Data tables and profiles: [`docs/DATA.md`](docs/DATA.md)
- Experiment pre-registrations and results: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- Known limitations: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- Decision records: [`docs/decisions/`](docs/decisions/)
- Assessment and plan: [`docs/PHASE0_ASSESSMENT.md`](docs/PHASE0_ASSESSMENT.md)
- Security and ethics: [`docs/SECURITY_AND_ETHICS.md`](docs/SECURITY_AND_ETHICS.md)
- Standing project instructions: [`CLAUDE.md`](CLAUDE.md)

## Framing

Outputs of this project describe anomalous or high-risk *financial patterns* in synthetic data.
They never state or imply that any person is a criminal. Suspicious scenario families are abstract detection
research patterns; the generator has no parameter that targets or reads any detector. See
`docs/SECURITY_AND_ETHICS.md`.
