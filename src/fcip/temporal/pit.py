"""Point-in-time reference feature primitives (decision 0003).

Each query row is ``(account, t)`` with its own inclusive cutoff ``t``; a trailing window of ``W`` days is
``(t - W*86400, t]``. Everything is read from the store's account-side index by ``searchsorted`` up to the
row's own cutoff, so a row can never see an event after its ``t``.

This module is the *reference* implementation: per-row, obviously correct, slow. The fast daily-snapshot
path (``fcip.temporal.grid``) must reproduce it exactly; a property test compares them.

Pre-registered single-account features (EXP-M1-G, ``docs/EXPERIMENTS.md``): F1 to F14 per window, F15, F16.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from fcip.common.errors import FutureAccessError
from fcip.temporal.store import KEY_SHIFT

if TYPE_CHECKING:
    from fcip.temporal.store import TemporalStore

DAY = 86400
HOLD_CAP_S = 30 * DAY
WINDOW_FEATURES = (
    "total_amount",
    "total_amount_in",
    "total_amount_out",
    "txn_count",
    "in_degree",
    "out_degree",
    "avg_holding_time",
    "max_amount",
    "net_flow_ratio",
    "atm_share",
    "night_share",
    "declined_count",
    "distinct_devices",
    "distinct_ips",
)
STATIC_FEATURES = ("account_age_days", "balance")


def feature_names(windows: tuple[int, ...]) -> list[str]:
    return [f"{f}_{w}d" for w in windows for f in WINDOW_FEATURES] + list(STATIC_FEATURES)


def last_credit_ts(store: TemporalStore, lo: int, i: int) -> int | None:
    """Most recent settled credit strictly earlier in order than side row ``i`` (rows ``lo..i-1``)."""
    s = store._s
    for j in range(i - 1, lo - 1, -1):
        if s["settled"][j] and not s["out"][j]:
            return int(s["ts"][j])
    return None


class PITQuery:
    def __init__(self, store: TemporalStore, account_ids: np.ndarray, cutoffs: np.ndarray) -> None:
        if len(account_ids) != len(cutoffs):
            raise ValueError("account_ids and cutoffs differ in length")
        if (cutoffs >= store.sim_end).any() or (cutoffs < 0).any():
            raise ValueError("cutoff outside [0, sim_end)")
        self._store = store
        self._acct = store.account_index(account_ids)
        self._t = cutoffs
        opened = store._acc_opened[self._acct]
        if (opened > cutoffs).any():
            raise FutureAccessError("a query row refers to an account that opens after its cutoff")

    def _bounds(self, k: int, window_days: int | None) -> tuple[int, int, int]:
        st = self._store
        a, t = int(self._acct[k]), int(self._t[k])
        base = a * KEY_SHIFT
        hi = int(np.searchsorted(st._s_key, base + t, side="right"))
        start = int(st._s_ptr[a])
        lo = (
            start
            if window_days is None
            else int(np.searchsorted(st._s_key, base + t - window_days * DAY, side="right"))
        )
        return start, lo, hi

    def window_features(self, window_days: int) -> dict[str, np.ndarray]:
        st = self._store
        s, ls = st._s, st._ls
        n = len(self._t)
        out = {f"{f}_{window_days}d": np.zeros(n) for f in WINDOW_FEATURES}
        W = window_days
        for k in range(n):
            start, lo, hi = self._bounds(k, W)
            sl = slice(lo, hi)
            settled, is_out = s["settled"][sl], s["out"][sl]
            amt = s["amount"][sl]
            sin = settled & ~is_out
            sout = settled & is_out
            a_in, a_out = float(amt[sin].sum()), float(amt[sout].sum())
            r = out
            r[f"total_amount_{W}d"][k] = a_in + a_out
            r[f"total_amount_in_{W}d"][k] = a_in
            r[f"total_amount_out_{W}d"][k] = a_out
            r[f"txn_count_{W}d"][k] = float(settled.sum())
            tr = s["transfer"][sl]
            r[f"in_degree_{W}d"][k] = len(np.unique(s["cp"][sl][sin & tr]))
            r[f"out_degree_{W}d"][k] = len(np.unique(s["cp"][sl][sout & tr]))
            num = den = 0.0
            for i in np.flatnonzero(sout) + lo:
                c = last_credit_ts(st, start, int(i))
                hold = HOLD_CAP_S if c is None or s["ts"][i] - c > HOLD_CAP_S else int(s["ts"][i] - c)
                num += float(s["amount"][i]) * hold
                den += float(s["amount"][i])
            r[f"avg_holding_time_{W}d"][k] = (num / den if den > 0 else HOLD_CAP_S) / 3600.0
            r[f"max_amount_{W}d"][k] = float(amt[settled].max()) if settled.any() else 0.0
            tot = a_in + a_out
            r[f"net_flow_ratio_{W}d"][k] = abs(a_in - a_out) / tot if tot > 0 else 1.0
            r[f"atm_share_{W}d"][k] = float(amt[sout & s["atm"][sl]].sum()) / a_out if a_out > 0 else 0.0
            cnt = settled.sum()
            r[f"night_share_{W}d"][k] = float((settled & s["night"][sl]).sum()) / cnt if cnt else 0.0
            r[f"declined_count_{W}d"][k] = float((~settled & is_out).sum())
            a, t = int(self._acct[k]), int(self._t[k])
            base = a * KEY_SHIFT
            lhi = int(np.searchsorted(st._ls_key, base + t, side="right"))
            llo = int(np.searchsorted(st._ls_key, base + t - W * DAY, side="right"))
            ok = ls["success"][llo:lhi]
            r[f"distinct_devices_{W}d"][k] = len(np.unique(ls["device"][llo:lhi][ok]))
            r[f"distinct_ips_{W}d"][k] = len(np.unique(ls["ip"][llo:lhi][ok]))
        return out

    def static_features(self) -> dict[str, np.ndarray]:
        st = self._store
        n = len(self._t)
        age = (self._t - st._acc_opened[self._acct]) / DAY
        bal = np.zeros(n)
        for k in range(n):
            start, _lo, hi = self._bounds(k, None)
            bal[k] = float(st._s["after"][hi - 1]) if hi > start else float(st._acc_initial[self._acct[k]])
        return {"account_age_days": age.astype(np.float64), "balance": bal}

    def features(self, windows: tuple[int, ...] = (7, 30)) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for w in windows:
            out.update(self.window_features(w))
        out.update(self.static_features())
        return out

    # ---------------------------------------------------------------- G3 context primitives (report only)
    def internal_transfer_counterparties(self, k: int, window_days: int) -> np.ndarray:
        """Distinct internal counterparty account indices of settled transfers in the row's window."""
        st = self._store
        _start, lo, hi = self._bounds(k, window_days)
        s = st._s
        m = s["settled"][lo:hi] & s["transfer"][lo:hi]
        cps = np.unique(s["cp"][lo:hi][m])
        return cps[st._acc_internal[cps]]

    def temporal_order_features(self, window_days: int) -> dict[str, np.ndarray]:
        """Burstiness and median gap of settled events, and time from first credit to first later debit, in the
        window (hours). Fewer than 3 events: burstiness 0; fewer than 2: median gap = window length."""
        st = self._store
        s = st._s
        n = len(self._t)
        W_h = window_days * 24.0
        burst, med, c2d = np.zeros(n), np.full(n, W_h), np.full(n, W_h)
        for k in range(n):
            _start, lo, hi = self._bounds(k, window_days)
            sett = s["settled"][lo:hi]
            ts = s["ts"][lo:hi][sett]
            if len(ts) >= 2:
                gaps = np.diff(ts) / 3600.0
                med[k] = float(np.median(gaps))
                if len(ts) >= 3:
                    mu, sd = float(gaps.mean()), float(gaps.std())
                    burst[k] = (sd - mu) / (sd + mu) if sd + mu > 0 else 0.0
            outs = s["out"][lo:hi][sett]
            if (~outs).any():
                first_c = ts[~outs][0]
                later = ts[outs & (ts > first_c)]
                if len(later):
                    c2d[k] = (later[0] - first_c) / 3600.0
        return {
            f"burstiness_{window_days}d": burst,
            f"median_gap_h_{window_days}d": med,
            f"credit_to_debit_h_{window_days}d": c2d,
        }
