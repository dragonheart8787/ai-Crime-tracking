# 0003: As-of view design (leakage prevention by construction)

- Status: ACCEPTED (Phase 0 review); label-access rules aligned with the revised 0004 and new 0009
- Scope: every read of events, entity state, relations and labels by features, graph builders,
  samplers, splits and models, from Milestone 1 onward

## Problem

Silent temporal leakage is the primary methodological risk (`CLAUDE.md` 2.2). The design must
make the leak-free path the *only convenient* path, and make the leaky path fail loudly.
Python cannot make anything truly private, so "by construction" here means: a narrow public
API that cannot express a future read, immutable returned data, import boundaries enforced by
tests, and metamorphic tests that detect leaks empirically.

## Definitions

- **Time**: int64 seconds since the simulation epoch (decision 0004).
- **Total order**: `(ts, event_id)`. A **cutoff** is a pair `(t, event_id_or_None)`:
  `None` means "everything with `ts <= t`"; an event ID means "up to and including this event",
  which is what next-event prediction after an observed event needs. Ties at the same second are
  thereby resolved deterministically.
- **Observable at t**: events with order key <= cutoff; entity attributes whose `valid_from <= t`;
  relations whose `valid_from <= t` (and not yet ended, or with end visible only if
  `valid_to <= t`); account balance = `balance_after` of the last settled event <= cutoff;
  labels whose **`known_at` is not null and `known_at <= t`** (not merely `valid_from <= t`, see decision 0004),
  excluding OOD-family label rows, which are never visible as known labels (decision 0009).

## Options

- **A. Convention**: pass a filtered dataframe; developers remember to filter. Rejected by the spec.
- **B. View object wrapping dataframes**: `store.as_of(t)` returns an object exposing filtered
  frames. Easy, but a returned frame can be joined with an unfiltered one obtained elsewhere.
- **C. Sealed store + as-of view + vetted point-in-time primitives.**
  - `TemporalStore` loads the tables, sorts by the total order, builds per-entity time-sorted
    index arrays (CSR style). It exposes **no** method returning raw tables. Its only public
    methods are `as_of(cutoff) -> AsOfView` and `point_in_time(entity_ids, cutoffs) -> PITQuery`.
  - `AsOfView` is frozen and bound to one cutoff; there is no `advance()` or setter.
    Methods: `events(start=None, end=None)`, `events_for(entity, window)`, `balance(account)`,
    `relations(kind)`, `entity_attrs(kind)`, `known_labels()`. Any requested `end` or window that
    extends past the cutoff **raises `FutureAccessError`**; it is never clipped silently.
  - Returned arrays are read-only (`writeable=False` numpy views, immutable Arrow/polars data).
  - `PITQuery` is the scalable path for training data, where every (entity, t) row has its own
    cutoff: a small set of vetted aggregation primitives (count, sum, min, max, distinct count,
    last value, time since last, over trailing windows) implemented with `searchsorted` on the
    per-entity sorted arrays. Features are compositions of these primitives; there is no
    free-form access to the underlying arrays.
  - Ground-truth labels live in a separate `OracleLabels` object that is **not** reachable from
    `TemporalStore`. Only the target builder and evaluation modules may import it.
- **D. Database with row-level time security** (e.g. DuckDB views parameterized by t). Strong
  isolation, but point-in-time queries with per-row cutoffs become awkward and slow in SQL,
  and SQL access elsewhere is easy to abuse.

## Recommendation

**C.** Enforcement layers:

1. **API**: future reads raise; no raw-table accessor; views immutable and single-cutoff.
2. **Import boundary test**: an AST-based test fails if any module under `features/`, `graph/`,
   `models/` imports `labels.oracle` or any underscore-prefixed attribute of the store.
3. **Adversarial tests** (required by `CLAUDE.md` 9): deliberately request events or logins after t,
   balances after t, relations created after t, windows crossing t, and a cutoff tie at the same
   second; each must raise `FutureAccessError`. Labels with `known_at > t`, null `known_at`, or an OOD
   family must be absent from `known_labels()` (there is no parameter that could request them).
4. **Metamorphic "future perturbation" test** (the strongest check): compute every feature at
   cutoffs t; then delete, shuffle, rescale or inject events strictly after t (and labels known
   after t, plus label rows with `known_at > t` or null), rebuild the store, recompute; every feature value at t must be bit-identical. Any
   feature that fails is rejected. This test is generic and runs over the whole feature registry.
5. **Brute-force oracle test** for `PITQuery` primitives: compare against a naive
   filter-then-aggregate implementation on small random data (property-based, Hypothesis).
6. The later temporal neighbor sampler (Milestone 5+) is built *on* the store's index, so graph
   neighborhoods inherit the same cutoff semantics.

## Trade-offs

- More upfront code than B, and some legitimate analyses (EDA over the full period) need a
  separate, clearly named `FullHistoryView` restricted to validation/EDA modules and forbidden to
  feature code by the import test.
- Performance: per-row cutoffs via `searchsorted` are O(log n) per row per primitive, vectorized.
  Adequate for DEV; RESEARCH performance NOT YET EVALUATED.
- Python privacy is convention; the tests are what make it binding.
