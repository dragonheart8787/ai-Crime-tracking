# 0006: Rust evaluation for candidate components

- Status: PROPOSED (awaiting review, Phase 0)
- Scope: the four candidates in `CLAUDE.md` section 10, plus one found during Phase 0
  (the balance settlement pass)

No profiling has been done: there is no code yet. Every verdict below is a **plan conditional
on later profiling evidence**, and every performance statement is an expectation, NOT YET EVALUATED.

## Criteria (from `CLAUDE.md` 10)

Hot-spot evidence, latency needs, isolation/security value, data-structure fit,
PyO3/maturin integration cost, Windows build support, testability.

Cross-cutting cost of *any* Rust component on the primary machine: a Rust toolchain plus the MSVC
build tools on Windows, maturin builds in CI, and a second implementation to keep equivalent.
Windows build of a PyO3 extension is NOT YET EVALUATED on the target laptop. This fixed cost is the
main reason not to start any component in Rust before evidence exists.

## Verdicts

| Component | Verdict | Earliest milestone the question becomes real |
|---|---|---|
| (a) Synthetic event generator | **Python only** (vectorized numpy); revisit only if RESEARCH generation exceeds a budget | 1 |
| (a') Balance settlement / ledger pass inside the generator | **Python oracle, then Rust port if profiled as the bottleneck** | 1 |
| (b) As-of temporal edge index and neighbor sampler | **Python oracle, then Rust port (expected)** | 5 to 7 |
| (c) Streaming incremental graph / feature updater | **Python oracle; Rust decision deferred until a latency target exists** | 10 |
| (d) Temporal motif counting | **Python oracle; Rust port only if motif features earn their place in ablations** | 2 to 3 |
| none | Rust from the start | n/a |

### (a) Generator: Python only

- Data-structure fit: most of generation is embarrassingly parallel per entity or per scenario
  instance (sample counts, inter-arrival times, amounts), which vectorizes well in numpy.
- Hot spot: unknown. The expectation is that per-entity generation is not the bottleneck.
- Reproducibility: the RNG design (decision 0002) uses numpy `SeedSequence` + PCG64. A Rust port
  would not reproduce the same streams without re-implementing them bit-exactly, which would break
  seed-level reproducibility across implementations.
- Budget proposal to trigger re-evaluation: RESEARCH profile generation wall time > 15 minutes
  or peak RAM > 24 GB on the laptop.

### (a') Settlement pass: Python oracle, then Rust port if profiled

- This is a *sequential* loop in global time order: apply each event to the ledger, check the
  overdraft rule, mark declined events, write `balance_before/after`. It does not vectorize.
- Expected cost: a few microseconds per event in pure Python, so order of a minute for 10^7 events.
  Probably acceptable; if not, the kernel is small, pure (arrays in, arrays out), easy to property-test
  against the oracle, and a good first PyO3 candidate. numba is a cheaper alternative worth comparing
  before Rust.

### (b) As-of neighbor sampler: Python oracle, then Rust port (expected)

- Hot spot (expected): multi-hop sampling where every seed node has its own cutoff t, inside the
  training loop, repeated every epoch. Python loops over nodes are slow; numpy vectorizes one hop at
  a time but multi-hop with per-seed cutoffs and fan-out limits is awkward.
- Isolation value: moderate. The sampler is part of the leakage boundary; a compiled kernel with a
  narrow API is harder to bypass casually than Python internals.
- Dependency value: PyG's own temporal sampling relies on compiled extensions (pyg-lib or
  torch-sparse) whose Windows wheel availability for the chosen torch version is NOT YET EVALUATED
  (see decision 0007). An in-house sampler removes that dependency.
- Test plan: the same property tests (no neighbor with `ts > cutoff`, distribution of sampled
  neighbors matches the oracle, deterministic under seed) run against both implementations.

### (c) Streaming updater: Python oracle; decision deferred

- The deciding criterion is latency, and there is no latency requirement in the spec yet. Without a
  target (e.g. p99 per-event update < X ms at Y events/s) the Rust question is unmeasurable. See
  Challenge 13 in the assessment.

### (d) Temporal motif counting: Python oracle; Rust port only if the features help

- Exact counting of time-constrained motifs is combinatorial and sequential, a good fit for Rust in
  principle. But the research value of motif features is itself unproven here; `CLAUDE.md` 8 says not
  to assume any structural feature helps. First show value in the Milestone 2/3 ablations with a
  Python implementation on DEV-scale data, then port if it is both useful and slow.

## Review cadence

Rather than one decision record per module, each milestone's review packet includes a short
"language verdict" table for the components it adds, and this record is updated when a verdict
changes.
