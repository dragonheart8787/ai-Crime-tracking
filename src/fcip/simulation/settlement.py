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
from fcip.common.taxonomy import Status


@dataclass
class Settled:
    order: np.ndarray  # permutation of the input rows into global order
    status: np.ndarray  # object array of Status values, input row order
    src_before: np.ndarray
    src_after: np.ndarray
    dst_before: np.ndarray
    dst_after: np.ndarray
    final_balance: dict[int, int]


def settle(
    event_id: list[int],
    ts: list[int],
    src: list[int],
    dst: list[int],
    amount: list[int],
    depends_on: list[int],
    initial_balance: dict[int, int],
    overdraft_limit: dict[int, int | None],
) -> Settled:
    n = len(event_id)
    eid = np.asarray(event_id, dtype=np.int64)
    tsa = np.asarray(ts, dtype=np.int64)
    if len(np.unique(eid)) != n:
        raise InvariantViolation("duplicate transaction event ids before settlement")
    order = np.lexsort((eid, tsa))
    bal = dict(initial_balance)
    status: list[str | None] = [None] * n
    sb = [0] * n
    sa = [0] * n
    db = [0] * n
    da = [0] * n
    settled_ids: dict[int, bool] = {}
    settled_v, declined_v = Status.SETTLED.value, Status.DECLINED.value
    for i in order.tolist():
        s, d, a, dep = src[i], dst[i], amount[i], depends_on[i]
        if s not in bal or d not in bal:
            raise InvariantViolation(f"event {event_id[i]} references an unknown account")
        ok = True
        if dep != -1:
            if dep not in settled_ids:
                raise InvariantViolation(
                    f"event {event_id[i]} depends on {dep}, which is not earlier in order"
                )
            ok = settled_ids[dep]
        before_s, before_d = bal[s], bal[d]
        limit = overdraft_limit[s]
        if ok and limit is not None and before_s - a < -limit:
            ok = False
        if ok:
            bal[s] = before_s - a
            bal[d] = before_d + a
        settled_ids[event_id[i]] = ok
        status[i] = settled_v if ok else declined_v
        sb[i], sa[i], db[i], da[i] = before_s, bal[s], before_d, bal[d]
    return Settled(
        order=order,
        status=np.asarray(status, dtype=object),
        src_before=np.asarray(sb, dtype=np.int64),
        src_after=np.asarray(sa, dtype=np.int64),
        dst_before=np.asarray(db, dtype=np.int64),
        dst_after=np.asarray(da, dtype=np.int64),
        final_balance=bal,
    )
