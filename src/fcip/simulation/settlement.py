"""Settlement pass (decision 0006 a'): apply intents to a ledger in global order ``(ts, event_id)``.

Rules:
* an internal source account may not go below ``-overdraft_limit``; otherwise the event is DECLINED;
* external accounts (employers, billers, the outside world, card network, cash) have no limit;
* an event whose funding dependency (``depends_on``) was not SETTLED is DECLINED (scenario causality);
* DECLINED events move no money; balances before/after are recorded for both sides of every event.

Money is conserved exactly: every settled event debits one account and credits another by the same int64
amount, so the sum of all balances (internal and external) never changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fcip.common.errors import InvariantViolation

SETTLED_CODE = 0
DECLINED_CODE = 1
STATUS_VOCAB: tuple[str, ...] = ("SETTLED", "DECLINED")
_CHUNK = 500_000


@dataclass
class Settled:
    order: np.ndarray  # permutation of the input rows into global order
    status: np.ndarray  # int8 codes into STATUS_VOCAB, input row order
    src_before: np.ndarray
    src_after: np.ndarray
    dst_before: np.ndarray
    dst_after: np.ndarray
    final_balance: dict[int, int]

    def status_str(self) -> list[str]:
        return [STATUS_VOCAB[c] for c in self.status.tolist()]


def settle(
    event_id: np.ndarray | list[int],
    ts: np.ndarray | list[int],
    src: np.ndarray | list[int],
    dst: np.ndarray | list[int],
    amount: np.ndarray | list[int],
    depends_on: np.ndarray | list[int],
    initial_balance: dict[int, int],
    overdraft_limit: dict[int, int | None],
) -> Settled:
    eid_a = np.asarray(event_id, dtype=np.int64)
    n = len(eid_a)
    if len(np.unique(eid_a)) != n:
        raise InvariantViolation("duplicate transaction event ids before settlement")
    order = np.lexsort((eid_a, np.asarray(ts, dtype=np.int64)))
    dep_a = np.asarray(depends_on, dtype=np.int64)
    needed = set(dep_a[dep_a != -1].tolist())  # only these events' outcomes must be remembered
    src_a, dst_a, amt_a = np.asarray(src, np.int64), np.asarray(dst, np.int64), np.asarray(amount, np.int64)
    bal = dict(initial_balance)
    status = np.zeros(n, dtype=np.int8)
    sb, sa, db, da = (np.zeros(n, dtype=np.int64) for _ in range(4))
    outcome: dict[int, bool] = {}
    for lo in range(0, n, _CHUNK):  # chunked so that only _CHUNK Python ints per column are alive at once
        idx = order[lo : lo + _CHUNK]
        rows = zip(
            idx.tolist(),
            eid_a[idx].tolist(),
            src_a[idx].tolist(),
            dst_a[idx].tolist(),
            amt_a[idx].tolist(),
            dep_a[idx].tolist(),
            strict=True,
        )
        c_sb, c_sa, c_db, c_da, c_st = [], [], [], [], []
        for _i, e, s, d, a, dp in rows:
            if s not in bal or d not in bal:
                raise InvariantViolation(f"event {e} references an unknown account")
            ok = True
            if dp != -1:
                if dp not in outcome:
                    raise InvariantViolation(f"event {e} depends on {dp}, which is not earlier in order")
                ok = outcome[dp]
            before_s, before_d = bal[s], bal[d]
            limit = overdraft_limit[s]
            if ok and limit is not None and before_s - a < -limit:
                ok = False
            if ok:
                bal[s] = before_s - a
                bal[d] = before_d + a
            if e in needed:
                outcome[e] = ok
            c_st.append(0 if ok else DECLINED_CODE)
            c_sb.append(before_s)
            c_sa.append(bal[s])
            c_db.append(before_d)
            c_da.append(bal[d])
        status[idx] = c_st
        sb[idx], sa[idx], db[idx], da[idx] = c_sb, c_sa, c_db, c_da
    return Settled(order, status, sb, sa, db, da, bal)
