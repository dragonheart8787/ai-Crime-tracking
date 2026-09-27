"""Static recruitment partition (decision 0002; simplifications S1 and S2).

Every normal-population account is assigned to exactly one *recruitment cell* by weighted rendezvous
hashing over the whole family registry ``{(family, i) : i < max_instances(family)}`` plus a large ``NONE``
cell. A scenario instance may recruit members only from its own cell. Consequences:

* an account can belong to at most one scenario instance over the whole simulation (S2, hence also S1);
* the assignment does not depend on which families are enabled, so disabling a family never changes
  any other instance's candidate pool.

The partition uses a keyed integer hash (splitmix64), not an RNG stream, and never Python's ``hash()``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fcip.common.rng import keyed_uniform, stable_key
from fcip.common.taxonomy import Family
from fcip.config.generator import GeneratorConfig

NONE_CELL = -1
_CHUNK = 20000


@dataclass(frozen=True)
class Partition:
    cells: tuple[tuple[Family, int], ...]  # cell index -> (family, instance_idx)
    cell_of_account: dict[int, int]  # account_id -> cell index, or NONE_CELL
    by_cell: dict[int, tuple[int, ...]]  # cell index (or NONE_CELL) -> sorted account ids

    def candidates(self, family: Family, instance_idx: int) -> tuple[int, ...]:
        return self.by_cell.get(self.cells.index((family, instance_idx)), ())

    def none_cell_accounts(self) -> tuple[int, ...]:
        return self.by_cell.get(NONE_CELL, ())


def build_partition(cfg: GeneratorConfig, account_ids: list[int]) -> Partition:
    cells: list[tuple[Family, int]] = []
    for fam in Family:  # registry order; independent of the enabled flag
        cells.extend((fam, i) for i in range(cfg.scenarios[fam].max_instances))
    n_cells = len(cells)
    share = cfg.recruitment.eligible_share
    w_cell = 1.0
    w_none = n_cells * w_cell * (1.0 - share) / share
    weights = np.asarray([w_cell] * n_cells + [w_none], dtype=np.float64)
    cell_keys = [stable_key(f"recruit-cell/{f.value}/{i}") for f, i in cells] + [
        stable_key("recruit-cell/NONE")
    ]
    seed_key = stable_key(f"recruit/{cfg.simulation.seed}")

    acc = np.asarray(account_ids, dtype=np.int64)
    assignment = np.empty(len(acc), dtype=np.int64)
    for lo in range(0, len(acc), _CHUNK):
        chunk = acc[lo : lo + _CHUNK]
        best = np.full(len(chunk), -np.inf)
        best_idx = np.full(len(chunk), NONE_CELL, dtype=np.int64)
        for c, (key, w) in enumerate(zip(cell_keys, weights, strict=True)):
            u = keyed_uniform(seed_key ^ key, chunk)
            score = -w / np.log(u)
            better = score > best
            best = np.where(better, score, best)
            best_idx = np.where(better, c if c < n_cells else NONE_CELL, best_idx)
        assignment[lo : lo + _CHUNK] = best_idx
    cell_of = dict(zip(acc.tolist(), assignment.tolist(), strict=True))
    by_cell: dict[int, list[int]] = {}
    for a, c in cell_of.items():
        by_cell.setdefault(c, []).append(a)
    return Partition(
        cells=tuple(cells), cell_of_account=cell_of, by_cell={c: tuple(sorted(v)) for c, v in by_cell.items()}
    )
