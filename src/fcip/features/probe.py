"""Features for the Milestone 1 non-triviality probes (EXP-M1-G). Store API only; no ground truth.

* single-account features F1 to F16: :func:`snapshot_features` (fast grid, identical to the reference);
* G3 context features (report only): :func:`context_features` - 1-hop neighbor aggregates over distinct
  internal counterparties of settled transfers in the trailing 30 days (mean of their ``txn_count_30d``,
  ``in_degree_30d``, ``out_degree_30d``, ``account_age_days`` at the same snapshot; 0 when there are none)
  and temporal-order features (burstiness, median inter-event gap, time from first credit to first later
  debit, 30-day window).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fcip.temporal.grid import snapshot_cutoff
from fcip.temporal.pit import feature_names
from fcip.temporal.store import TemporalStore

WINDOWS = (7, 30)
NEIGHBOR_SOURCE = ("txn_count_30d", "in_degree_30d", "out_degree_30d", "account_age_days")
CONTEXT_WINDOW_DAYS = 30


def single_account_feature_names() -> list[str]:
    return feature_names(WINDOWS)


def snapshot_features(store: TemporalStore, days: list[int]) -> pl.DataFrame:
    return store.snapshot_grid().features(sorted(set(days)), WINDOWS)


def context_features(store: TemporalStore, points: pl.DataFrame, grid: pl.DataFrame) -> pl.DataFrame:
    """``points``: account_id, day. ``grid``: snapshot features covering every open account on those days."""
    acc = points["account_id"].to_numpy()
    cut = np.asarray([snapshot_cutoff(int(d)) for d in points["day"]], dtype=np.int64)
    q = store.point_in_time(acc, cut)
    order = q.temporal_order_features(CONTEXT_WINDOW_DAYS)
    ids = store.account_ids
    pairs_row, pairs_cp = [], []
    for k in range(len(acc)):
        cps = q.internal_transfer_counterparties(k, CONTEXT_WINDOW_DAYS)
        pairs_row.append(np.full(len(cps), k))
        pairs_cp.append(ids[cps])
    pr = pl.DataFrame(
        {
            "row": np.concatenate(pairs_row) if pairs_row else np.zeros(0, np.int64),
            "cp": np.concatenate(pairs_cp) if pairs_cp else np.zeros(0, np.int64),
        }
    )
    pr = pr.with_columns(pl.Series("day", points["day"].to_numpy()[pr["row"].to_numpy()]))
    nb = pr.join(
        grid.select("account_id", "day", *NEIGHBOR_SOURCE),
        left_on=["cp", "day"],
        right_on=["account_id", "day"],
        how="inner",
    )
    agg = nb.group_by("row").agg([pl.col(c).mean().alias(f"nbr_mean_{c}") for c in NEIGHBOR_SOURCE])
    base = (
        pl.DataFrame({"row": np.arange(len(acc))}).join(agg, on="row", how="left").sort("row").fill_null(0.0)
    )
    out = base.drop("row")
    for name, values in order.items():
        out = out.with_columns(pl.Series(name, values))
    return out
