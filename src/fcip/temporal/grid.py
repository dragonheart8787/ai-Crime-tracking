"""Daily-snapshot features for every account at once (fast path of ``fcip.temporal.pit``).

Snapshot day ``d`` means the inclusive cutoff ``t = d * 86400 - 1`` (everything strictly before midnight
starting day ``d``). A window of ``W`` days is then exactly simulation days ``d-W .. d-1``. All aggregates are
built from per-(account, day) buckets and combined with prefix sums, rolling maxima and difference arrays, so a
snapshot only ever reads buckets of days ``< d``. ``tests/leakage`` checks this against the per-row reference
and with a future-perturbation test.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from fcip.temporal.pit import DAY, HOLD_CAP_S, STATIC_FEATURES, WINDOW_FEATURES
from fcip.temporal.store import KEY_SHIFT

if TYPE_CHECKING:
    from fcip.temporal.store import TemporalStore


def snapshot_cutoff(day: int) -> int:
    return day * DAY - 1


class SnapshotGrid:
    def __init__(self, store: TemporalStore) -> None:
        self._st = store
        self._n_acc = len(store.account_ids)
        self._n_days = store.sim_end // DAY
        s = store._s
        self._day = (s["ts"] // DAY).astype(np.int64)
        self._flat = s["acct"] * self._n_days + self._day

    # ---------------------------------------------------------------- bucket helpers
    def _daily_sum(self, mask: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
        w = mask.astype(np.float64) if values is None else np.where(mask, values, 0).astype(np.float64)
        d = np.bincount(self._flat, weights=w, minlength=self._n_acc * self._n_days)
        return d.reshape(self._n_acc, self._n_days)

    @staticmethod
    def _prefix(daily: np.ndarray) -> np.ndarray:
        c = np.zeros((daily.shape[0], daily.shape[1] + 1))
        np.cumsum(daily, axis=1, out=c[:, 1:])
        return c

    @staticmethod
    def _window(prefix: np.ndarray, d: int, w: int) -> np.ndarray:
        return prefix[:, d] - prefix[:, max(d - w, 0)]

    def _rolling_distinct(self, acct: np.ndarray, key: np.ndarray, day: np.ndarray, w: int) -> np.ndarray:
        """``out[a, d]`` = number of distinct ``key`` values of account ``a`` on days ``d-w .. d-1``."""
        n_days = self._n_days
        diff = np.zeros((self._n_acc, n_days + 2), dtype=np.int64)
        if len(acct):
            order = np.lexsort((day, key, acct))
            a, k, x = acct[order], key[order], day[order]
            keep = np.ones(len(a), bool)
            keep[1:] = (a[1:] != a[:-1]) | (k[1:] != k[:-1]) | (x[1:] != x[:-1])
            a, k, x = a[keep], k[keep], x[keep]
            same_next = np.zeros(len(a), bool)
            same_next[:-1] = (a[1:] == a[:-1]) & (k[1:] == k[:-1])
            y = np.where(same_next, np.roll(x, -1), np.iinfo(np.int64).max // 4)
            start = x + 1
            end_excl = np.minimum(np.minimum(y, x + w) + 1, n_days + 1)
            np.add.at(diff, (a, start), 1)
            np.add.at(diff, (a, end_excl), -1)
        return np.cumsum(diff, axis=1)

    # ---------------------------------------------------------------- features
    def features(
        self, days: list[int], windows: tuple[int, ...] = (7, 30), internal_only: bool = True
    ) -> pl.DataFrame:
        st, s = self._st, self._st._s
        if any(d < 1 or d > self._n_days for d in days):
            raise ValueError("snapshot days must lie in 1..n_days")
        settled, out = s["settled"], s["out"]
        sin, sout = settled & ~out, settled & out
        amt = s["amount"]
        p_in = self._prefix(self._daily_sum(sin, amt))
        p_out = self._prefix(self._daily_sum(sout, amt))
        p_cnt = self._prefix(self._daily_sum(settled))
        p_night = self._prefix(self._daily_sum(settled & s["night"]))
        p_atm = self._prefix(self._daily_sum(sout & s["atm"], amt))
        p_decl = self._prefix(self._daily_sum(~settled & out))
        # holding time per settled debit: time since the most recent earlier settled credit (capped)
        k = s["acct"] * KEY_SHIFT
        cand = np.where(sin, k + s["ts"], k - 1)
        cm = np.maximum.accumulate(cand)
        # the running maximum is taken over all earlier rows; a credit strictly earlier in order is required,
        # so shift by one row within the same account
        prev = np.empty_like(cm)
        prev[0] = -1
        prev[1:] = cm[:-1]
        has_prev = prev >= k
        gap = s["ts"] - (prev - k)
        hold = np.where(has_prev & (gap <= HOLD_CAP_S), gap, HOLD_CAP_S).astype(np.float64)
        p_hold_num = self._prefix(self._daily_sum(sout, amt.astype(np.float64) * hold))
        # daily max of settled amounts, for rolling maxima
        dmax_flat = np.zeros(self._n_acc * self._n_days)
        np.maximum.at(dmax_flat, self._flat[settled], amt[settled].astype(np.float64))
        dmax = dmax_flat.reshape(self._n_acc, self._n_days)
        # balance: last side row position per (account, day), carried forward
        lastpos = np.full(self._n_acc * self._n_days, -1, dtype=np.int64)
        np.maximum.at(lastpos, self._flat, np.arange(len(self._flat), dtype=np.int64))
        lastpos = np.maximum.accumulate(lastpos.reshape(self._n_acc, self._n_days), axis=1)
        tr = s["transfer"]
        din = {
            w: self._rolling_distinct(s["acct"][sin & tr], s["cp"][sin & tr], self._day[sin & tr], w)
            for w in windows
        }
        dout = {
            w: self._rolling_distinct(s["acct"][sout & tr], s["cp"][sout & tr], self._day[sout & tr], w)
            for w in windows
        }
        ls = st._ls
        ok = ls["success"]
        lday = ls["ts"][ok] // DAY
        ddev = {w: self._rolling_distinct(ls["acct"][ok], ls["device"][ok], lday, w) for w in windows}
        dip = {w: self._rolling_distinct(ls["acct"][ok], ls["ip"][ok], lday, w) for w in windows}

        all_rows = np.flatnonzero(st._acc_internal) if internal_only else np.arange(self._n_acc)
        frames = []
        for d in days:
            t = snapshot_cutoff(d)
            # only accounts that exist at the snapshot: an account opened later is not observable at t
            rows = all_rows[st._acc_opened[all_rows] <= t]
            cols: dict[str, np.ndarray] = {"account_id": st.account_ids[rows], "day": np.full(len(rows), d)}
            for w in windows:
                a_in = self._window(p_in, d, w)[rows]
                a_out = self._window(p_out, d, w)[rows]
                cnt = self._window(p_cnt, d, w)[rows]
                tot = a_in + a_out
                num = self._window(p_hold_num, d, w)[rows]
                cols[f"total_amount_{w}d"] = tot
                cols[f"total_amount_in_{w}d"] = a_in
                cols[f"total_amount_out_{w}d"] = a_out
                cols[f"txn_count_{w}d"] = cnt
                cols[f"in_degree_{w}d"] = din[w][rows, d].astype(np.float64)
                cols[f"out_degree_{w}d"] = dout[w][rows, d].astype(np.float64)
                with np.errstate(invalid="ignore", divide="ignore"):
                    cols[f"avg_holding_time_{w}d"] = (
                        np.where(a_out > 0, num / np.where(a_out > 0, a_out, 1), HOLD_CAP_S) / 3600.0
                    )
                    cols[f"net_flow_ratio_{w}d"] = np.where(
                        tot > 0, np.abs(a_in - a_out) / np.where(tot > 0, tot, 1), 1.0
                    )
                    cols[f"atm_share_{w}d"] = np.where(
                        a_out > 0, self._window(p_atm, d, w)[rows] / np.where(a_out > 0, a_out, 1), 0.0
                    )
                    cols[f"night_share_{w}d"] = np.where(
                        cnt > 0, self._window(p_night, d, w)[rows] / np.where(cnt > 0, cnt, 1), 0.0
                    )
                cols[f"max_amount_{w}d"] = dmax[rows, max(d - w, 0) : d].max(axis=1)
                cols[f"declined_count_{w}d"] = self._window(p_decl, d, w)[rows]
                cols[f"distinct_devices_{w}d"] = ddev[w][rows, d].astype(np.float64)
                cols[f"distinct_ips_{w}d"] = dip[w][rows, d].astype(np.float64)
            cols["account_age_days"] = (t - st._acc_opened[rows]) / DAY
            pos = lastpos[rows, d - 1]
            cols["balance"] = np.where(
                pos >= 0, s["after"][np.maximum(pos, 0)], st._acc_initial[rows]
            ).astype(np.float64)
            frames.append(pl.DataFrame(cols))
        names = (
            ["account_id", "day"]
            + [f"{f}_{w}d" for w in windows for f in WINDOW_FEATURES]
            + list(STATIC_FEATURES)
        )
        return pl.concat(frames).select(names)
