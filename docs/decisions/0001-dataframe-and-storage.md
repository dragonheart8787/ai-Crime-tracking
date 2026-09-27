# 0001: Dataframe library, storage format and canonical content hash

- Status: ACCEPTED (Phase 0 review), revised in the Phase 0 revision pass (content hash made precise;
  money stored as integer minor units)
- Scope: Milestone 1 onward (generator output, as-of store, EDA, features)

## Problem

The generator emits several related tables (transactions, logins, entities, relations, labels) that
must be:

1. reproducible: same config + seed gives the same dataset identity;
2. large enough for the RESEARCH profile (order of 10^7 events) on a 64 GB laptop;
3. strongly typed (no silent dtype drift such as int to float on nulls, `string` vs `large_string`);
4. cheap to scan by time range and by entity for the as-of store (decision 0003).

## Options

| Option | In-memory | On-disk | Notes |
|---|---|---|---|
| A. pandas + Parquet | pandas | Parquet via pyarrow | Nullable-int and datetime semantics are a common source of silent dtype changes. |
| B. polars + Parquet (explicit Arrow schemas) | polars (Arrow memory) | Parquet | Strict typing, fast; explicit schemas pin dtypes. |
| C. DuckDB as primary store | DuckDB tables | `.duckdb` or Parquet | Convenient SQL makes non-as-of queries easy to write. |
| D. numpy structured arrays + custom binary | numpy | `.npy` | No schema, poor interop. |

## Decision

Option B:

- **numpy** inside generator kernels.
- **pyarrow** owns the *schemas*: one explicit `pa.schema` per table in one module
  (`fcip/common/schemas.py`), with column order, dtype and nullability fixed. Every table is cast and
  validated against it on write and on read; a mismatch that cannot be cast losslessly raises.
- **polars** for validation, EDA and later feature pipelines.
- **Parquet** (written by pyarrow, zstd, fixed writer settings) on disk. CSV is an explicit export
  option, never the source of truth.
- **DuckDB** only for ad hoc analysis, never on the feature/label path.
- **pandas** is not a Milestone 1 dependency.

### Money as integer minor units

All monetary columns (`amount_minor`, `balance_before_minor`, `balance_after_minor`,
`overdraft_limit_minor`) are **int64 minor units** (e.g. cents) in **one currency per dataset**
(simplification S3). There is **no per-row `currency` column** in any Milestone 1 table: a column that looked
supported but could not be converted correctly would be worse than none. The currency is declared once in config
and `metadata.json` as `{code: "SYN", minor_units_per_major: 100}` (a synthetic unit, so no real-currency amounts are
implied). The schema registry rejects any `currency` column. Multi-currency support (per-currency minor-unit exponent,
FX events, conservation per currency) is future work. Rationale:
conservation of funds becomes an exact integer identity instead of a float tolerance, and the content
hash never has to canonicalize floats. Float columns are allowed in derived feature tables later, not
in canonical generator tables. NaN is forbidden in canonical tables.

## Canonical content hash (dataset identity)

Parquet file bytes depend on the writer and its version. Checked in this session with a 50K-row table
(pyarrow 25.0.1 zstd vs polars 1.44.2 snappy, rows shuffled for the polars path):

```
file bytes equal: False
polars read-back types: ['int64', 'int64', 'int64', 'large_string']
content hash equal: True
one-value change detected: True
```

The read-back type difference (`large_string` vs `string`) is why the cast in step 1 below is required.

**Algorithm `fcip-content-hash-v1`** (SHA-256 throughout):

1. **Cast** the table to its registered schema: select columns in schema order, cast each to the schema
   dtype (e.g. `large_string` to `string`, dictionary to plain). Unknown extra columns or failed casts raise.
2. **Allowed canonical dtypes**: `int8/16/32/64`, `uint8/16/32/64`, `bool`, `string` (UTF-8). Anything
   else raises. (Timestamps are int64 seconds, decision 0004; money is int64, above.)
3. **Sort** rows by the table's full primary key (registered per table, e.g. `event_id` for
   transactions, `(entity_type, entity_id, scenario_id, valid_from)` for labels), ascending. Assert the
   primary key is unique; duplicates raise.
4. `combine_chunks()` so the chunking of the input never matters.
5. **Byte stream** fed to the hash, in order:
   - the ASCII tag `fcip-content-hash-v1\0` and the table name followed by `\0`;
   - the schema fingerprint: for each field in order, `"{name}:{arrow_type}:{nullable as 0/1};"`;
   - the row count as uint64 little-endian;
   - for each column in schema order:
     - validity as one uint8 (0 or 1) per row (not a packed bitmap, so buffer offsets and padding never matter);
     - values: integers and bools as fixed-width little-endian after filling nulls with 0; strings as, per
       row, a uint64 little-endian byte length followed by the UTF-8 bytes (null encodes as length 0,
       disambiguated by the validity byte).
6. **Dataset hash** = SHA-256 over `fcip-dataset-hash-v1\0` followed by, for each table in
   lexicographic table-name order, `name\0` and the 32-byte table digest.

The version tag lets the algorithm change later without silently comparing incompatible hashes.

**Also recorded** in `metadata.json`: per-file SHA-256 of each Parquet file. Tests assert these are
stable across two runs in the same pinned environment, but they are not the dataset identity.

### Required tests (Milestone 1)

- Same logical table written via polars and via pyarrow (different compression, shuffled rows, different
  chunking) gives different file bytes and **identical content hashes**.
- A single changed value, a changed null, a swapped column order in the *input* (must still hash the same
  after the cast), and a dtype change that is not losslessly castable (must raise).
- Duplicate primary key raises.
- Hash identical in two fresh subprocesses (ties in with the `PYTHONHASHSEED` test in 0002).

## Trade-offs

- Pure-Python per-string hashing is slow for very large string columns. Canonical tables keep strings to
  low-cardinality categorical columns, so this is expected to be acceptable; RESEARCH-scale hashing time is
  NOT YET EVALUATED. A vectorized variant (offsets + data buffers after normalization to `large_string`) can
  replace it later if it produces the same bytes, verified by a test.
- polars' API changes more often than pandas': pinned in `uv.lock`.

## Rejected

- pandas primary, DuckDB primary, raw numpy files (reasons above).
- File-byte hashing as dataset identity (shown above to depend on the writer).
- Arrow IPC bytes as the canonical representation: IPC framing, alignment and metadata are
  library-version dependent, so they would reintroduce the same fragility.
