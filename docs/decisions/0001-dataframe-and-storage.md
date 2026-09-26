# 0001: Dataframe library and storage format

- Status: PROPOSED (awaiting review, Phase 0)
- Scope: Milestone 1 onward (generator output, as-of store, EDA, features)

## Problem

The generator must emit several related tables (transactions, logins, entities, relations,
labels) that are:

1. reproducible: same config + seed gives the same dataset identity (hash);
2. large enough for the RESEARCH profile (order of 10^7 events) on a 64 GB laptop;
3. strongly typed (no silent dtype drift such as int to float on nulls, or timezone-naive vs
   aware timestamps);
4. cheap to scan by time range and by entity for the as-of store (decision 0003).

We need one primary in-memory representation, one on-disk format, and a definition of
"dataset hash" that is not fragile.

## Options

| Option | In-memory | On-disk | Notes |
|---|---|---|---|
| A. pandas + Parquet | pandas | Parquet via pyarrow | Familiar; nullable-int and datetime semantics are a common source of silent dtype changes; slower group-bys at 10^7 rows. |
| B. polars + Parquet (pyarrow schemas) | polars (Arrow memory) | Parquet | Strict typing, fast, lazy scans; explicit Arrow schemas pin dtypes. |
| C. DuckDB as primary store | DuckDB tables | `.duckdb` file or Parquet | Excellent SQL, but a DB file is a less transparent artifact and SQL makes it easy to write non-as-of queries. |
| D. numpy structured arrays + custom binary | numpy | `.npy` | Fastest and simplest for kernels, but no schema, poor interop, reinvents Parquet. |

## Recommendation

Option B, with clear roles:

- **numpy** inside generator kernels (vectorized sampling per RNG stream).
- **pyarrow** owns the *schemas*: one explicit `pa.schema` per table in a single module;
  every table is validated against it on write and on read (fail loudly on mismatch).
- **polars** is the dataframe layer for validation, EDA and (later) feature pipelines.
- **Parquet** (written by pyarrow, zstd, fixed settings) is the on-disk format.
  CSV export is an explicit CLI option, never the source of truth.
- **DuckDB** is optional, for ad hoc analysis only; it is not on the path that features or
  labels flow through, because arbitrary SQL bypasses the as-of view.
- **pandas** is not a Milestone 1 dependency. It enters only at library boundaries that need it.

### Dataset hash: content hash, not file bytes

Exploratory check in this session (scratch venv; numpy 2.4.6, pyarrow 25.0.1, polars 1.44.2,
duckdb 1.5.5, a 100K-row table):

```
pyarrow True polars True duckdb True          # each writer is byte-stable across 2 writes
pyarrow vs polars same bytes: False           # but different writers give different bytes
```

Parquet bytes depend on the writer and its version (the footer records `created_by`, page
layout differs). A "same seed gives byte-identical files" requirement would therefore be
coupled to library versions rather than to the data. Proposal:

- **Dataset identity = canonical content hash**: for each table, sort by primary key, then
  hash (schema string, column name, dtype, little-endian values buffer, validity bitmap) per
  column in schema order with SHA-256; combine table hashes in a fixed table order.
- **File SHA-256** of each Parquet file is also recorded in metadata, and the tests assert
  that it is stable across two runs *in the same environment* (this is still a useful check).

## Trade-offs

- polars API changes between minor versions more often than pandas: pin in `uv.lock`.
- Maintaining explicit Arrow schemas is extra code but it is the main defence against
  silent dtype drift.
- A custom content hash must itself be tested (permutation invariance under the PK sort,
  sensitivity to a single changed value, stability across two processes).

## Rejected

- A (pandas primary): weaker typing guarantees for no gain at this stage.
- C (DuckDB primary): convenient SQL is exactly what makes leakage easy.
- D (raw numpy files): no schema, no interop.
