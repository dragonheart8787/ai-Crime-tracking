# Architecture (as built at Milestone 1)

Only implemented components are listed. Design rationale lives in `docs/decisions/`.

```
configs/generator/*.yaml ──> fcip.config (Pydantic, 0005) ──> fcip.simulation.generator
                                                               │  population, recruitment partition (0002),
                                                               │  scenario plans, normal behavior, suppression,
                                                               │  settlement (0006), labels + known_at (0004)
                                                               ▼
                             data/processed/<profile>_seed<seed>/*.parquet + metadata.json
                                   │ (fcip.common.schemas / hashing / io, 0001)
             ┌─────────────────────┼──────────────────────────────┐
             ▼                     ▼                              ▼
  fcip.temporal.store       fcip.labels.oracle             fcip.validation
  TemporalStore (sealed)    OracleLabels (ground truth)    invariants, sanity, eda, gate
   ├─ AsOfView (as_of(t))    ├─ OracleTargets (eval only)
   ├─ PITQuery (reference)   └─ TrainingTargets (known_at <= T_fit)
   └─ SnapshotGrid (fast)
             │                     │
             ▼                     ▼
  fcip.features (store API only)   fcip.evaluation.metrics (accepts OracleTargets only)
             └───────────┬─────────┘
                         ▼
             fcip.temporal.splits (0009): TRAIN / VAL / TEST with purge gap, OOD pool,
             reference negatives, label maturity and both training modes
```

## Boundaries (enforced by tests)

| Rule | Where | Test |
|---|---|---|
| No read after the cutoff: requests past `t` raise `FutureAccessError`, never clipped | `AsOfView`, `PITQuery`, `SnapshotGrid` | `tests/leakage/test_adversarial_asof.py` |
| Features at `t` unchanged when anything after `t` changes | grid, reference, every view accessor | `tests/leakage/test_future_perturbation.py` |
| Fast grid equals the per-row reference | `SnapshotGrid` vs `PITQuery` | `tests/leakage/test_grid_matches_reference.py` |
| Feature and temporal code never import the oracle, the generator or validation code, never name oracle tables | `fcip.features`, `fcip.temporal` (except `splits.py`, protocol) | `tests/leakage/test_import_boundaries.py` |
| No private access to other objects outside `fcip.temporal` | all other packages | same |
| Labels usable only once known | `known_labels()`, `training_targets()` | `tests/leakage/test_known_at.py` |
| Metrics only against full ground truth | `fcip.evaluation.metrics` | `tests/unit/test_eval_ground_truth.py` |
| Generated data satisfy every invariant | generator (runs all checks), `fcip.cli validate` | `tests/unit/test_invariants.py` |

## Package map

| Package | Contents |
|---|---|
| `fcip.common` | errors, taxonomy, stable RNG streams, namespaced IDs, time base, schemas, content hash, Parquet I/O |
| `fcip.config` | typed distributions, generator config schema, YAML loader with profile overlays |
| `fcip.simulation` | population, recruitment partition, behavior engine, scenario families, logins, settlement, labels, generator |
| `fcip.temporal` | temporal store, as-of view, point-in-time reference features, snapshot grid, splits |
| `fcip.labels` | oracle labels and target types |
| `fcip.features` | probe features (single-account and G3 context), label-derived features |
| `fcip.evaluation` | metrics that accept `OracleTargets` only |
| `fcip.validation` | invariants, sanity summary, EDA report, EXP-M1-G gate runner |
