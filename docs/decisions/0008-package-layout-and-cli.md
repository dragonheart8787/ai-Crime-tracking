# 0008: Package layout and CLI entry point

- Status: PROPOSED (awaiting review, Phase 0)

## Problem

`CLAUDE.md` suggests `src/{data,simulation,...}` and `python -m src.cli`. Using `src` itself as the
import package works, but it is the known anti-pattern that the "src layout" exists to avoid: the
package name is meaningless, tests can import the working tree instead of the installed package,
and it collides with every other project doing the same.

## Options

- A. `src/` as the package, `python -m src.cli`.
- B. src layout with a named package: `src/fcip/...` (Financial Crime Intelligence Platform),
  installed in editable mode by uv; CLI `uv run fcip simulate ...` or `python -m fcip.cli simulate ...`.
- C. Flat layout `fcip/` at repo root.

## Recommendation

**B.** Proposed command for Milestone 1:

```
uv run python -m fcip.cli simulate --profile dev --seed 42 --out data/processed/dev_seed42
```

(with a `fcip` console script as a convenience alias). Sub-packages mirror the suggested layout,
adding `temporal/` (as-of store, splits) and `validation/` (invariants, stats, probes), and deferring
empty packages (`models/`, `api/`, `dashboard/` ...) until their milestone: no empty placeholder
modules.

## Trade-offs

Slightly longer import paths; standard tooling (uv, pytest, mypy) supports it natively.
