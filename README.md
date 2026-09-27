# Predictive Financial Crime Intelligence Platform

Defensive research prototype for fraud / AML detection on **synthetic** transaction data:
temporal graph modeling, suspicious-network detection, behavioral role inference and
next-high-risk-event prediction, as decision support for human analysts.

## Status

Milestone 1 (synthetic financial environment), **checkpoint 1 of 2**. Implemented and tested:

- configurable synthetic generator (profiles `tiny` for tests, `dev`, and `research`, which is defined but has
  **not been run yet**), deterministic per seed, with named, independent RNG streams;
- entities and relations (persons, accounts, devices, IPs, ATMs, merchants, external accounts), transactions
  and logins;
- eight normal archetypes, confounders, and ten abstract suspicious-pattern families (two designated as OOD holdout
  families by default);
- settlement with overdraft rules and exact conservation of funds (single synthetic currency `SYN`, int64 minor units);
- time-indexed ground truth (phase-dependent roles, event labels, networks, members, `known_at`);
- canonical content hash (`fcip-content-hash-v1`), invariant checks, and a data-shape sanity summary.

**Not implemented yet** (Milestone 1 checkpoint 2): as-of view, chronological splits with purge gap, OOD pool,
label-maturity training modes, ground-truth-only evaluation plumbing, leakage tests, the pre-registered
non-triviality gate run on RESEARCH, and the EDA report. No model of any kind exists.

## Reproduce the DEV dataset from a clean environment

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                                        # creates .venv from uv.lock
uv run python -m fcip.cli simulate --profile dev --seed 42     # writes data/processed/dev_seed42/
uv run python -m fcip.cli validate --data data/processed/dev_seed42
uv run python -m fcip.cli sanity   --data data/processed/dev_seed42
```

`simulate` prints the dataset hash. With the pinned environment, DEV seed 42 must reproduce
`918b0d3adb174f6068523b184df773410d6ac66ed09fb5975d15f9747bd3a563` (verified at checkpoint 1; identity is the
content hash, not file bytes, see `docs/decisions/0001`). `validate` recomputes the hash from the Parquet files and
re-runs every invariant check. Add `--csv` to `simulate` (or run `export-csv --data <dir>`) for CSV copies.

Output: one Parquet file per table plus `metadata.json` (resolved config and hash, seed, generator commit,
package versions, per-table and dataset hashes, file SHA-256, realized prevalence, activation report).
Tables `person_truth`, `labels`, `event_labels`, `ground_truth_networks` and `network_members` are **oracle**
(ground truth) tables and must never be used as model inputs.

## Tests

```bash
uv run pytest            # full suite, including the DEV end-to-end run and subprocess reproducibility tests
uv run pytest -m "not slow"
uv run ruff check src tests && uv run mypy src
```

## Documentation

- Assessment and plan: [`docs/PHASE0_ASSESSMENT.md`](docs/PHASE0_ASSESSMENT.md)
- Decision records: [`docs/decisions/`](docs/decisions/)
- Experiment pre-registrations: [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)
- Known limitations: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md)
- Security and ethics: [`docs/SECURITY_AND_ETHICS.md`](docs/SECURITY_AND_ETHICS.md)
- Standing project instructions: [`CLAUDE.md`](CLAUDE.md)

## Framing

Outputs of this project describe anomalous or high-risk *financial patterns* in synthetic data.
They never state or imply that any person is a criminal. Suspicious scenario families are abstract detection
research patterns; the generator has no parameter that targets or reads any detector. See
`docs/SECURITY_AND_ETHICS.md`.
