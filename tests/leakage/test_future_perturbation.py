"""Metamorphic leakage test (decision 0003): change everything after the cutoff (delete, rescale, inject
events; rewrite labels known later), rebuild the store, and require every feature at the cutoff to be
bit-identical. Covers the fast grid, the reference PIT primitives and every as-of view accessor."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyarrow as pa

from fcip.common.schemas import conform
from fcip.features.label_features import flagged_counterparties
from fcip.temporal.grid import snapshot_cutoff
from fcip.temporal.store import TemporalStore

DAY = 86400
CUT_DAY = 20


def _perturb(tables: dict[str, pa.Table], t: int, seed: int) -> dict[str, pa.Table]:
    rng = np.random.default_rng(seed)
    out = dict(tables)
    tx = pl.from_arrow(tables["transactions"])
    past, fut = tx.filter(pl.col("ts") <= t), tx.filter(pl.col("ts") > t)
    fut = fut.filter(pl.Series(rng.random(fut.height) < 0.5))  # delete half
    fut = fut.with_columns(
        (pl.col("amount_minor") * 3).alias("amount_minor"),  # rescale
        pl.lit(12345).cast(pl.Int64).alias("src_balance_after_minor"),
        pl.lit(-777).cast(pl.Int64).alias("dst_balance_after_minor"),
    )
    inject = past.sample(50, seed=seed).with_columns(
        (pl.col("event_id") + (1 << 58)).alias("event_id"),
        (t + 1 + pl.int_range(pl.len())).cast(pl.Int64).alias("ts"),
        (pl.col("amount_minor") * 100).alias("amount_minor"),
    )
    out["transactions"] = conform("transactions", pl.concat([past, fut, inject]).to_arrow())
    lg = pl.from_arrow(tables["logins"])
    lg_f = lg.filter(pl.col("ts") > t).with_columns(
        (pl.col("device_id") + 1).alias("device_id"), (pl.col("ip_id") + 1).alias("ip_id")
    )
    out["logins"] = conform("logins", pl.concat([lg.filter(pl.col("ts") <= t), lg_f]).to_arrow())
    lab = pl.from_arrow(tables["labels"])
    later = pl.col("known_at").is_null() | (pl.col("known_at") > t)
    lab = lab.with_columns(
        pl.when(later).then(pl.lit("RELAY")).otherwise(pl.col("role_label")).alias("role_label"),
        pl.when(later).then(pl.lit(0).cast(pl.Int8)).otherwise(pl.col("risk_label")).alias("risk_label"),
    )
    out["labels"] = conform("labels", lab.to_arrow())
    acc = pl.from_arrow(tables["accounts"])
    acc = acc.with_columns(
        pl.when(pl.col("opened_at") > t)
        .then(pl.col("initial_balance_minor") + 999)
        .otherwise(pl.col("initial_balance_minor"))
        .alias("initial_balance_minor")
    )
    out["accounts"] = conform("accounts", acc.to_arrow())
    return out


def _everything_at(store: TemporalStore, t: int, accounts: np.ndarray, days: list[int]) -> dict:
    grid = store.snapshot_grid().features(days)
    opened = pl.DataFrame({"account_id": store.account_ids})
    rows = grid.filter(pl.col("account_id").is_in(accounts.tolist())).sort("account_id", "day")
    view_accounts = store.as_of(t).accounts().select("account_id", "opened_at")
    rows = rows.join(view_accounts, on="account_id").filter(pl.col("opened_at") <= pl.col("day") * DAY - 1)
    del opened
    ref = store.point_in_time(
        rows["account_id"].to_numpy(), np.asarray([snapshot_cutoff(d) for d in rows["day"]])
    ).features((7, 30))
    ref_ctx = store.point_in_time(
        rows["account_id"].to_numpy(), np.asarray([snapshot_cutoff(d) for d in rows["day"]])
    ).temporal_order_features(30)
    v = store.as_of(t)
    return {
        "grid": grid.sort("account_id", "day"),
        "ref": {k: v_.tolist() for k, v_ in {**ref, **ref_ctx}.items()},
        "tx": v.transactions().sort("event_id"),
        "logins": v.logins().sort("event_id"),
        "known": v.known_labels().sort(["entity_id", "valid_from"]),
        "accounts": v.accounts().sort("account_id"),
        "relations": v.relations().sort(["relation_type", "src_id", "dst_id"]),
        "balances": [v.balance(int(a)) for a in accounts],
        "flagged": [flagged_counterparties(v, int(a), 30 * DAY) for a in accounts],
    }


def test_features_at_cutoff_ignore_everything_after_it(tiny_ds) -> None:
    t = CUT_DAY * DAY - 1
    sim_end = tiny_ds.metadata["sim_end"]
    base = TemporalStore(tiny_ds.tables, sim_end, tiny_ds.tables["labels"])
    acc = pl.from_arrow(tiny_ds.tables["accounts"])
    open_internal = acc.filter((pl.col("account_kind") == "internal") & (pl.col("opened_at") <= t))
    accounts = open_internal["account_id"].sample(60, seed=3).to_numpy()
    members = pl.from_arrow(tiny_ds.tables["network_members"])["entity_id"].to_numpy()
    accounts = np.unique(
        np.concatenate([accounts, members[np.isin(members, open_internal["account_id"].to_numpy())]])
    )
    days = [5, 12, CUT_DAY]
    before = _everything_at(base, t, accounts, days)
    for seed in (0, 1):
        pert = _perturb(tiny_ds.tables, t, seed)
        after = _everything_at(TemporalStore(pert, sim_end, pert["labels"]), t, accounts, days)
        assert before["grid"].equals(after["grid"])
        assert before["ref"] == after["ref"]
        for key in ("tx", "logins", "known", "accounts", "relations"):
            assert before[key].equals(after[key]), key
        assert before["balances"] == after["balances"]
        assert before["flagged"] == after["flagged"]
    # sanity: the perturbation really changed the future
    assert (
        not base.snapshot_grid()
        .features([44])
        .equals(TemporalStore(pert, sim_end, pert["labels"]).snapshot_grid().features([44]))
    )
