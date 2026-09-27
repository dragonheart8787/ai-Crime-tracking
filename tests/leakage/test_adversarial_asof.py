"""Deliberate attempts to read the future through the as-of view must raise (CLAUDE.md section 9)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fcip.common.errors import FutureAccessError
from fcip.temporal.store import TemporalStore

DAY = 86400


@pytest.fixture(scope="module")
def store(tiny_ds) -> TemporalStore:
    return TemporalStore(tiny_ds.tables, tiny_ds.metadata["sim_end"], tiny_ds.tables["labels"])


@pytest.fixture(scope="module")
def t() -> int:
    return 20 * DAY + 12345


def test_reading_transactions_or_logins_after_cutoff_raises(store, t) -> None:
    v = store.as_of(t)
    for fn in (v.transactions, v.logins):
        with pytest.raises(FutureAccessError):
            fn(end=t + 1)
        with pytest.raises(FutureAccessError):
            fn(start=t + 1)
        with pytest.raises(FutureAccessError):
            fn(start=t - DAY, end=t + DAY)  # a window crossing the cutoff is refused, never clipped
    assert v.transactions()["ts"].max() <= t and v.logins()["ts"].max() <= t


def test_future_balance_and_future_accounts_raise(store, tiny_ds, t) -> None:
    acc = pl.from_arrow(tiny_ds.tables["accounts"])
    later = acc.filter(pl.col("opened_at") > t)
    assert later.height, "fixture needs an account opened after the cutoff"
    a = int(later["account_id"][0])
    with pytest.raises(FutureAccessError):
        store.as_of(t).balance(a)
    assert a not in store.as_of(t).accounts()["account_id"].to_list()
    with pytest.raises(FutureAccessError):
        store.point_in_time(np.asarray([a]), np.asarray([t]))


def test_balance_is_the_last_balance_at_or_before_cutoff(store, tiny_ds, t) -> None:
    tx = pl.from_arrow(tiny_ds.tables["transactions"])
    a = int(tx.filter(pl.col("ts") > t)["src_account_id"][0])
    rows = pl.concat(
        [
            tx.filter(pl.col("src_account_id") == a).select(
                "ts", "event_id", pl.col("src_balance_after_minor").alias("b")
            ),
            tx.filter(pl.col("dst_account_id") == a).select(
                "ts", "event_id", pl.col("dst_balance_after_minor").alias("b")
            ),
        ]
    ).sort("ts", "event_id")
    before = rows.filter(pl.col("ts") <= t)
    assert store.as_of(t).balance(a) == int(before["b"][-1])
    assert rows.filter(pl.col("ts") > t).height > 0


def test_relations_hide_future_end_times_and_future_relations(store, t) -> None:
    r = store.as_of(t).relations()
    assert (r["valid_from"] <= t).all()
    assert r["valid_to"].is_null().all()  # every relation ends at sim_end, which is in the future


def test_known_labels_never_include_future_unknown_or_ood_rows(store, tiny_ds) -> None:
    lab = pl.from_arrow(tiny_ds.tables["labels"])
    for day in (5, 20, 44):
        k = store.as_of(day * DAY).known_labels()
        assert (k["known_at"] <= day * DAY).all()
        assert not k["is_ood_family"].any()
    end = store.as_of(tiny_ds.metadata["sim_end"] - 1).known_labels()
    assert end.height == lab.filter(pl.col("known_at").is_not_null() & ~pl.col("is_ood_family")).height
    assert lab.filter(pl.col("is_ood_family")).height > 0


def test_same_second_ties_follow_the_event_order(store, tiny_ds) -> None:
    tx = pl.from_arrow(tiny_ds.tables["transactions"])
    dup = tx.group_by("ts").agg(pl.col("event_id").sort()).filter(pl.col("event_id").list.len() > 1)
    assert dup.height, "fixture needs two events in the same second"
    ts, (e1, e2) = int(dup["ts"][0]), dup["event_id"][0].to_list()[:2]
    seen = store.as_of(ts, event_id=e1).transactions()["event_id"].to_list()
    assert e1 in seen and e2 not in seen
    assert e2 in store.as_of(ts).transactions()["event_id"].to_list()


def test_view_is_immutable_and_arrays_read_only(store, t) -> None:
    v = store.as_of(t)
    with pytest.raises(AttributeError):
        v._cutoff = None  # type: ignore[misc]
    with pytest.raises(ValueError):
        store._s["amount"][0] = 1
    with pytest.raises(ValueError):
        store.as_of(store.sim_end)


def test_grid_and_pit_refuse_out_of_range_cutoffs(store) -> None:
    with pytest.raises(ValueError):
        store.snapshot_grid().features([store.sim_end // DAY + 1])
    with pytest.raises(ValueError):
        store.point_in_time(store.account_ids[:1], np.asarray([store.sim_end]))
