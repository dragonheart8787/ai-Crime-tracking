"""Sealed temporal store and as-of views (decision 0003).

The store loads the observable (non-oracle) tables, sorts them by the total order ``(ts, event_id)`` and
builds account-centric indexes. It exposes **no** raw tables. The only ways in are:

* ``store.as_of(t)`` -> :class:`AsOfView`, frozen to one cutoff; any request that reaches past the cutoff
  raises :class:`FutureAccessError` (never clipped silently);
* ``store.point_in_time(account_ids, cutoffs)`` -> reference per-row feature primitives (``fcip.temporal.pit``);
* ``store.snapshot_grid(...)`` -> fast daily-snapshot features (``fcip.temporal.grid``), validated against the
  reference primitives by tests.

Label rows enter the store only if they can ever be known (non-null ``known_at``) and do not belong to an OOD
family (decision 0009); everything else is dropped at load time, so the store never holds ground truth that
a model could not legitimately see. Ground truth for evaluation lives in ``fcip.labels.oracle`` only.

Cutoff semantics: a cutoff is ``(t, event_id)``. With ``event_id=None`` every event with ``ts <= t`` is
visible; with an event id, events with ``ts < t`` plus events at ``ts == t`` up to and including that id.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa

from fcip.common.errors import FutureAccessError
from fcip.common.frames import to_df
from fcip.common.io import read_columns, read_table
from fcip.common.taxonomy import TxnType

OBSERVABLE_TABLES = (
    "transactions",
    "logins",
    "accounts",
    "relations",
    "devices",
    "ips",
    "atms",
    "merchants",
    "persons",
)
KEY_SHIFT = 1 << 34  # ts < 2**34 seconds; account index in the high bits of the side-index key
NIGHT_END_HOUR = 5


@dataclass(frozen=True)
class Cutoff:
    t: int
    event_id: int | None = None


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


_T_INDEX_COLS = [
    "event_id",
    "ts",
    "txn_type",
    "src_account_id",
    "dst_account_id",
    "amount_minor",
    "status",
    "src_balance_after_minor",
    "dst_balance_after_minor",
]
_L_INDEX_COLS = ["event_id", "ts", "account_id", "device_id", "ip_id", "outcome"]


def _sorted_by_ts_eid(df: pl.DataFrame) -> pl.DataFrame:
    """Tables are written in (ts, event_id) order; sort only if that is not the case (avoids a full copy)."""
    ts, eid = df["ts"].to_numpy(), df["event_id"].to_numpy()
    if len(ts) < 2:
        return df
    ok = (ts[1:] > ts[:-1]) | ((ts[1:] == ts[:-1]) & (eid[1:] > eid[:-1]))
    return df if bool(ok.all()) else df.sort("ts", "event_id")


class TemporalStore:
    """Holds observable data only; see module docstring.

    ``views=False`` builds only the compact indexes used by point-in-time queries and the snapshot grid (much less
    memory); :meth:`as_of` then raises, because the frames behind the view accessors were not kept.
    """

    def __init__(
        self, tables: dict[str, pa.Table], sim_end: int, labels: pa.Table | None = None, views: bool = True
    ) -> None:
        need = OBSERVABLE_TABLES if views else ("transactions", "logins", "accounts")
        missing = [n for n in need if n not in tables]
        if missing:
            raise ValueError(f"TemporalStore needs tables {missing}")
        self._sim_end = int(sim_end)
        self._views = views
        acc = to_df(tables["accounts"]).sort("account_id")
        self._acc_ids = _ro(acc["account_id"].to_numpy())
        self._acc_opened = _ro(acc["opened_at"].to_numpy())
        self._acc_initial = _ro(acc["initial_balance_minor"].to_numpy())
        self._acc_internal = _ro((acc["account_kind"] == "internal").to_numpy())
        self._accounts = acc

        t = _sorted_by_ts_eid(to_df(tables["transactions"]))
        self._t_ts = _ro(t["ts"].to_numpy())
        self._t_eid = _ro(t["event_id"].to_numpy())
        lg = _sorted_by_ts_eid(to_df(tables["logins"]))
        self._l_ts = _ro(lg["ts"].to_numpy())
        self._l_eid = _ro(lg["event_id"].to_numpy())
        if views:
            self._transactions = t
            self._logins = lg
            self._relations = to_df(tables["relations"])

        # ---- account-centric side index: one row per (transaction, side), sorted by (account, ts, event_id).
        # The order is computed first, then each column is gathered on its own to keep temporaries small.
        n = t.height
        src_idx = self.account_index(t["src_account_id"].to_numpy())
        dst_idx = self.account_index(t["dst_account_id"].to_numpy())
        acct = np.concatenate([src_idx, dst_idx])
        order = np.lexsort(
            (np.concatenate([self._t_eid, self._t_eid]), np.concatenate([self._t_ts, self._t_ts]), acct)
        )

        def both(a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray:
            return _ro(np.concatenate([a, a if b is None else b])[order])

        ttype = t["txn_type"]
        self._s = {
            "acct": _ro(acct[order]),
            "ts": both(self._t_ts),
            "eid": both(self._t_eid),
            "amount": both(t["amount_minor"].to_numpy()),
            "out": _ro(order < n),
            "cp": both(dst_idx, src_idx),
            "settled": both((t["status"] == "SETTLED").to_numpy()),
            "transfer": both((ttype == TxnType.TRANSFER.value).to_numpy()),
            "atm": both((ttype == TxnType.ATM_WITHDRAWAL.value).to_numpy()),
            "night": both(((self._t_ts % 86400) // 3600) < NIGHT_END_HOUR),
            "after": both(t["src_balance_after_minor"].to_numpy(), t["dst_balance_after_minor"].to_numpy()),
        }
        del acct, src_idx, dst_idx, ttype
        self._s_key = _ro(self._s["acct"] * KEY_SHIFT + self._s["ts"])
        self._s_ptr = _ro(np.searchsorted(self._s["acct"], np.arange(len(self._acc_ids) + 1)))

        l_acct = self.account_index(lg["account_id"].to_numpy())
        lorder = np.lexsort((self._l_eid, self._l_ts, l_acct))
        self._ls = {
            "acct": _ro(l_acct[lorder]),
            "ts": _ro(self._l_ts[lorder]),
            "device": _ro(lg["device_id"].to_numpy()[lorder]),
            "ip": _ro(lg["ip_id"].to_numpy()[lorder]),
            "success": _ro((lg["outcome"] == "SUCCESS").to_numpy()[lorder]),
        }
        self._ls_key = _ro(self._ls["acct"] * KEY_SHIFT + self._ls["ts"])

        # ---- labels: only rows that can ever be known and are not OOD-family rows (decisions 0004, 0009)
        if labels is None:
            self._known = pl.DataFrame(schema={"entity_id": pl.Int64, "known_at": pl.Int64})
        else:
            lab = to_df(labels)
            self._known = lab.filter(pl.col("known_at").is_not_null() & ~pl.col("is_ood_family")).sort(
                "known_at"
            )

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def from_dir(cls, data_dir: Path, sim_end: int, views: bool = True) -> TemporalStore:
        if views:
            tables = {n: read_table(n, data_dir) for n in OBSERVABLE_TABLES}
        else:
            tables = {
                "transactions": read_columns("transactions", data_dir, _T_INDEX_COLS),
                "logins": read_columns("logins", data_dir, _L_INDEX_COLS),
                "accounts": read_table("accounts", data_dir),
            }
        store = cls(tables, sim_end, labels=read_table("labels", data_dir), views=views)
        del tables
        pa.default_memory_pool().release_unused()  # return Arrow's freed buffers to the OS
        return store

    # ------------------------------------------------------------------ public API
    @property
    def sim_end(self) -> int:
        return self._sim_end

    @property
    def account_ids(self) -> np.ndarray:
        return self._acc_ids

    def account_index(self, account_ids: np.ndarray) -> np.ndarray:
        ids = np.asarray(account_ids, dtype=np.int64)
        idx = np.searchsorted(self._acc_ids, ids)
        idx = np.clip(idx, 0, len(self._acc_ids) - 1)
        if len(ids) and not np.array_equal(self._acc_ids[idx], ids):
            raise KeyError("unknown account id")
        return idx.astype(np.int64)

    def as_of(self, t: int, event_id: int | None = None) -> AsOfView:
        if not self._views:
            raise RuntimeError("this store was built with views=False; as-of views are not available")
        if not 0 <= t < self._sim_end:
            raise ValueError(f"cutoff {t} outside [0, sim_end)")
        return AsOfView(self, Cutoff(int(t), event_id))

    def point_in_time(self, account_ids: np.ndarray, cutoffs: np.ndarray):  # -> PITQuery
        from fcip.temporal.pit import PITQuery

        return PITQuery(self, np.asarray(account_ids, np.int64), np.asarray(cutoffs, np.int64))

    def snapshot_grid(self):  # -> SnapshotGrid
        from fcip.temporal.grid import SnapshotGrid

        return SnapshotGrid(self)


class AsOfView:
    """Everything observable at one cutoff. Frozen: there is no way to move the cutoff."""

    __slots__ = ("_store", "_cutoff")
    _store: TemporalStore
    _cutoff: Cutoff

    def __init__(self, store: TemporalStore, cutoff: Cutoff) -> None:
        object.__setattr__(self, "_store", store)
        object.__setattr__(self, "_cutoff", cutoff)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("AsOfView is immutable")

    @property
    def cutoff(self) -> Cutoff:
        return self._cutoff

    # ---- internal: visible prefix of a (ts, event_id)-sorted array
    def _visible(self, ts: np.ndarray, eid: np.ndarray) -> int:
        c = self._cutoff
        if c.event_id is None:
            return int(np.searchsorted(ts, c.t, side="right"))
        lo = int(np.searchsorted(ts, c.t, side="left"))
        hi = int(np.searchsorted(ts, c.t, side="right"))
        return lo + int(np.searchsorted(eid[lo:hi], c.event_id, side="right"))

    def _check_window(self, start: int | None, end: int | None) -> tuple[int, int]:
        t = self._cutoff.t
        if end is not None and end > t:
            raise FutureAccessError(f"requested end {end} is after the cutoff {t}")
        if start is not None and start > t:
            raise FutureAccessError(f"requested start {start} is after the cutoff {t}")
        if start is not None and end is not None and start > end:
            raise ValueError("start > end")
        return (-(2**62) if start is None else start), (t if end is None else end)

    def transactions(self, start: int | None = None, end: int | None = None) -> pl.DataFrame:
        s, e = self._check_window(start, end)
        st = self._store
        hi = self._visible(st._t_ts, st._t_eid)
        return st._transactions.slice(0, hi).filter((pl.col("ts") >= s) & (pl.col("ts") <= e))

    def logins(self, start: int | None = None, end: int | None = None) -> pl.DataFrame:
        s, e = self._check_window(start, end)
        st = self._store
        hi = self._visible(st._l_ts, st._l_eid)
        return st._logins.slice(0, hi).filter((pl.col("ts") >= s) & (pl.col("ts") <= e))

    def accounts(self) -> pl.DataFrame:
        """Accounts opened at or before the cutoff (an account that does not exist yet is not observable)."""
        return self._store._accounts.filter(pl.col("opened_at") <= self._cutoff.t)

    def balance(self, account_id: int) -> int:
        st = self._store
        a = int(st.account_index(np.asarray([account_id]))[0])
        if st._acc_opened[a] > self._cutoff.t:
            raise FutureAccessError(f"account {account_id} opens after the cutoff")
        rows = self.transactions()
        src = rows.filter(pl.col("src_account_id") == account_id).select(
            "ts", "event_id", pl.col("src_balance_after_minor").alias("b")
        )
        dst = rows.filter(pl.col("dst_account_id") == account_id).select(
            "ts", "event_id", pl.col("dst_balance_after_minor").alias("b")
        )
        both = pl.concat([src, dst]).sort("ts", "event_id")
        return int(both["b"][-1]) if both.height else int(st._acc_initial[a])

    def relations(self, relation_type: str | None = None) -> pl.DataFrame:
        """Relations that exist at the cutoff; an end time after the cutoff is not observable and is null."""
        t = self._cutoff.t
        r = self._store._relations.filter(pl.col("valid_from") <= t)
        if relation_type is not None:
            r = r.filter(pl.col("relation_type") == relation_type)
        return r.with_columns(
            pl.when(pl.col("valid_to") <= t).then(pl.col("valid_to")).otherwise(None).alias("valid_to")
        )

    def known_labels(self) -> pl.DataFrame:
        """Label rows known at the cutoff (``known_at <= t``); never OOD-family rows, never unknown rows."""
        return self._store._known.filter(pl.col("known_at") <= self._cutoff.t)
