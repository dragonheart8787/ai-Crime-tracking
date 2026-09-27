# 0002: RNG stream design, stable keys and scenario recruitment isolation

- Status: ACCEPTED (Phase 0 review), revised in the Phase 0 revision pass (64-bit stable keys, explicit
  `PYTHONHASHSEED` regression test, recruitment partition for single-scenario membership)
- Scope: generator (Milestone 1), later any stochastic component (sampling, splits, training)

## Problem

Requirements (`CLAUDE.md` 2.4, Milestone 1 definition of done):

1. Same seed gives the same dataset (same content hash, decision 0001), across processes and machines.
2. Different seeds give different data.
3. **Adding or disabling one scenario does not change unrelated entities' data.**
4. (New, Phase 0 revision) An account belongs to at most one suspicious scenario instance in Milestone 1,
   which introduces coupling between scenarios that must not break requirement 3.

Requirement 1 fails if any key is derived with Python's built-in `hash()`: string hashing is salted per
process unless `PYTHONHASHSEED` is fixed. Requirement 3 fails with positional stream derivation, a shared
generator, global ID counters, or conflict resolution between scenarios competing for the same accounts.

## Options (streams)

- A. One global `Generator`: violates 3.
- B. Positional `SeedSequence(seed).spawn(k)`: inserting a component shifts later children.
- C. **Named, hierarchical spawn keys** derived from a stable hash of name paths.
- D. Counter-based RNG (Philox) keyed by (entity, purpose): same independence, more plumbing.

## Decision

**C.**

### Stable keys

```python
def stable_key(name: str) -> int:
    return int.from_bytes(hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest(), "big")
```

- 64-bit, big-endian, UTF-8. Python's built-in `hash()` is **never** used for anything that influences
  generated data, IDs, ordering or hashing. A ruff/AST check in the test suite fails on any `hash(` call
  inside `src/fcip/simulation`, `src/fcip/common` and `src/fcip/temporal`.
- A stream is `np.random.default_rng(SeedSequence(seed, spawn_key=(stable_key(c1), stable_key(c2), ..., int_id, ...)))`.
  Checked in this session: `SeedSequence` accepts 64-bit `spawn_key` elements, and the stream is identical
  under `PYTHONHASHSEED=0`, `1` and `12345` while the built-in `hash("scenario")` differs in each:
  ```
  True [7410424052719531763, 9071762512890867597, 1563158158060767637] -2113018908918104440
  True [7410424052719531763, 9071762512890867597, 1563158158060767637] -4342373628281728472
  True [7410424052719531763, 9071762512890867597, 1563158158060767637] 2842664297308961649
  ```
  (columns: key > 2^32, first three draws, built-in `hash("scenario")`)
- A registry of all stream name components is checked for 64-bit collisions in a unit test.

### Stream paths

| Component | Path |
|---|---|
| population sizes and attributes | `("population", <table>)` |
| normal behavior of one person | `("normal", <archetype>, person_id)` |
| confounder entity | `("confounder", <kind>, entity_id)` |
| scenario instance | `("scenario", <family>, instance_idx)` (parameters, members, events) |
| label latency of one network | `("label_latency", <family>, instance_idx)` |
| recruitment partition | vectorized integer hash, see below (no RNG stream) |

### IDs and ordering

- Namespaced IDs: normal-population IDs from the population stage; scenario-created entities in a scenario
  namespace `(family_code, instance_idx, local_idx)` packed into int64; event IDs `(stream_code, local_counter)`
  packed into int64. No global counters.
- Global order is `(ts, event_id)`; row position is derived and may change when events are added.

### Scenario recruitment partition (single-scenario membership without coupling)

Forbidding overlap naively (instances pick members in sequence and skip taken accounts) makes each
instance's membership depend on every earlier instance, so disabling one family would change the members of
others. Instead, eligibility is decided by a **static partition that does not depend on which scenarios are
enabled**:

1. The code-level family registry lists every family in the taxonomy with a per-profile `max_instances`.
   The set of **recruitment cells** is `{(family, i) : i < max_instances(family)}` over the *whole registry*,
   whether or not a family is enabled.
2. Each normal-population account is assigned to exactly one cell by **weighted rendezvous hashing**:
   `cell(a) = argmax_c  -w_c / ln(u(a, c))`, with `u(a, c)` a uniform in (0, 1) from a fixed 64-bit integer
   mixing function (splitmix64) of `stable_key(f"recruit/{seed}")`, `account_id` and `stable_key(cell name)`,
   vectorized in numpy. Weights `w_c` come from the registry (not from the enabled set), plus one large
   `NONE` cell, so most accounts are never eligible for any scenario.
3. An instance `(family, i)` selects its members only from its own cell, using its own RNG stream.
4. Consequences:
   - an account is in at most one scenario instance **over the whole simulation** (stricter than "one active
     scenario at a time"; see Q-R1 in the assessment);
   - disabling a family leaves every other instance's members, parameters and events unchanged, because no
     cell is reassigned;
   - raising `max_instances` for a family moves accounts only *into* the new cells (rendezvous property), so
     other instances are unaffected except for accounts that moved;
   - if a cell lacks enough eligible accounts (e.g. dormant accounts for dormant activation), the instance
     opens scenario-namespace accounts for the shortfall. This is recorded per instance in metadata and
     reported in validation; validation fails if the shortfall share exceeds a configured maximum. It is not
     a silent fallback.

### Exact isolation guarantee to test

For a baseline config and a variant with one family disabled (all else equal, `suspicious_prevalence` in the
same mode):

- every instance of every other family with an index present in both runs is identical (members, parameters,
  events, labels by `event_id`);
- every entity not touched by any instance that differs between the runs has identical normal events
  (by `event_id`, all columns except derived row position).

Balances of touched accounts may differ (settlement pass). When prevalence is held fixed by renormalizing quotas
over enabled families, disabling one family activates *additional* instances of others; those are new instances,
covered by the second bullet.

## Required tests (Milestone 1)

- **`PYTHONHASHSEED` regression**: run the DEV-tiny generator in two fresh subprocesses with
  `PYTHONHASHSEED=0` and `PYTHONHASHSEED=4242` (and a third with it unset), assert identical dataset content
  hash and identical per-table hashes.
- Same seed twice gives the same hash; seeds 42 and 43 give different hashes.
- Isolation test as defined above (family disabled, and one instance added).
- Registry collision test; AST test forbidding built-in `hash(` in generator code.
- Rendezvous partition: every account in exactly one cell; disabling a family changes no assignment.

## Trade-offs

- Per-entity streams mean many small `Generator` objects (a few microseconds each; well under a second at
  10^5 persons, NOT YET EVALUATED).
- The static partition fixes each cell's candidate pool size to roughly `N * w_c / sum(w)`. Small profiles
  may have tiny cells; the shortfall mechanism covers this, and its rate is reported.
- A future Rust generator port would not reproduce numpy PCG64 streams bit-exactly; its equivalence test
  would be statistical plus invariant-based. Open risk, unchanged.

## Rejected

- Built-in `hash()`, 32-bit keys (unnecessary collision risk), positional spawning, global counters.
- Sequential "skip already taken" recruitment (couples instances).
- Global deterministic conflict resolution between instances (still couples instances).
