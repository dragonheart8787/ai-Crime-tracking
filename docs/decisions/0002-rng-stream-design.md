# 0002: RNG stream design

- Status: PROPOSED (awaiting review, Phase 0)
- Scope: generator (Milestone 1), later any stochastic component (sampling, splits, training)

## Problem

Requirements from `CLAUDE.md` 2.4 and the Milestone 1 definition of done:

1. Same seed gives the same dataset.
2. Different seeds give different data.
3. **Adding or disabling one scenario does not change unrelated entities' data.**

Requirement 3 is the hard one. It fails if streams are derived by *position*
(`root.spawn(n)[i]`) and a component is inserted before others, if one shared generator is
consumed by several components, or if entity IDs or event IDs are allocated from a global
counter that a new scenario advances.

## Options

- **A. One global `Generator`.** Simple; violates requirement 3 immediately.
- **B. Positional spawn** (`SeedSequence(seed).spawn(k)`, component i gets child i). Children
  are stable if the *order and count* of earlier components are fixed; inserting a component
  shifts later children.
- **C. Named, hierarchical spawn keys.** Each stream is `SeedSequence(seed, spawn_key=path)`
  where `path` is a tuple of stable 32-bit integers derived from a *name path*, e.g.
  `("population", "accounts")`, `("archetype", "salary_worker", <person_id>)`,
  `("scenario", "fan_in", <instance_idx>)`. The key depends only on the name, never on order.
- **D. Counter-based RNG (Philox) keyed by (entity, purpose).** Same independence, more
  plumbing; numpy supports Philox bit generators.

## Recommendation

**C**, with these rules:

1. `stable_key(name) = int.from_bytes(blake2b(name.encode(), digest_size=4).digest(), "little")`.
   Never Python's built-in `hash()` (salted per process). A test fails the build on any
   collision among registered stream names.
2. A stream path is a tuple: component, sub-component, then integer IDs. Per-entity streams
   for normal behavior: `("normal", archetype, person_id)`. Per-instance streams for scenarios:
   `("scenario", family, instance_idx)`. Scenario *selection* of participating accounts uses
   the scenario's own stream, never the population's.
3. Exploratory check in this session confirmed the two properties this relies on
   (numpy 2.4.6):
   ```
   spawn child stable across spawn counts: True    # spawn(3)[1] == spawn(10)[1]
   explicit spawn_key == spawn()[7]: True          # SeedSequence(42, spawn_key=(7,)) == spawn(8)[7]
   ```
4. **IDs are namespaced, not global counters.** Normal-population entities get IDs from the
   population stage only. Scenario-created entities (for example a newly opened account used by
   a scenario) get IDs in a scenario namespace, e.g. `(family_code, instance_idx, local_idx)`
   packed into int64. Event IDs are `(stream_code, local_counter)` packed into int64, so adding
   a scenario does not renumber unrelated events.
5. **Global ordering** is `(timestamp, event_id)`. Sequence position is *derived* and is allowed
   to change when events are added; identity is `event_id`, not row number.
6. **What "unrelated" means** (to be tested exactly): entities that the added or removed
   scenario does not touch (not recruited, not a counterparty of any scenario event, not sharing
   a device or IP assigned by the scenario). For those entities, the set of their normal events
   (keyed by `event_id`, all columns except derived sequence position) must be identical.
   Balances of *touched* entities legitimately change (see decision 0004 and the settlement
   note in `PHASE0_ASSESSMENT.md`).

## Trade-offs

- Per-entity streams mean many small `Generator` objects. Construction costs a few
  microseconds each; at 10^5 persons that is well under a second. Vectorized sampling is still
  possible within one entity's stream, and per-archetype batch streams are an allowed
  optimization only if the independence test still passes.
- 32-bit name hashes can collide; guarded by the registry test.
- D (Philox) is a reasonable later alternative if a Rust port must reproduce the exact same
  streams: PCG64 via SeedSequence would have to be re-implemented bit-exactly in Rust. If a Rust
  generator port ever happens, the equivalence test is statistical plus invariant-based, not
  bit-exact, unless we switch to a counter-based RNG available in both languages. Recorded as an
  open risk, not solved now.
