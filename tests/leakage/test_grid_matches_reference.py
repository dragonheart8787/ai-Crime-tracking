"""The fast daily-snapshot grid reproduces the per-row reference features exactly (property test over random
accounts and days)."""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st

from fcip.temporal.grid import snapshot_cutoff
from fcip.temporal.pit import feature_names
from fcip.temporal.store import TemporalStore

_CACHE: dict = {}


def _grid(tiny_ds) -> tuple[TemporalStore, pl.DataFrame]:
    if "g" not in _CACHE:
        s = TemporalStore(tiny_ds.tables, tiny_ds.metadata["sim_end"], tiny_ds.tables["labels"])
        g = s.snapshot_grid().features(list(range(1, tiny_ds.metadata["sim_end"] // 86400 + 1)))
        acc = pl.from_arrow(tiny_ds.tables["accounts"]).select("account_id", "opened_at")
        g = g.join(acc, on="account_id").filter(pl.col("opened_at") <= pl.col("day") * 86400 - 1)
        _CACHE["g"] = (s, g)
    return _CACHE["g"]


@settings(max_examples=25, deadline=None)
@given(st.integers(0, 10_000))
def test_grid_equals_reference(tiny_ds, seed: int) -> None:
    store, grid = _grid(tiny_ds)
    rows = grid.sample(40, seed=seed)
    ref = store.point_in_time(
        rows["account_id"].to_numpy(), np.asarray([snapshot_cutoff(d) for d in rows["day"]])
    ).features((7, 30))
    for name in feature_names((7, 30)):
        np.testing.assert_allclose(rows[name].to_numpy(), ref[name], rtol=1e-12, atol=1e-9, err_msg=name)
