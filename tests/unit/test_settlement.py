"""Ledger rules: overdraft, declined events move no money, dependency declines, exact conservation."""

from __future__ import annotations

import pytest

from fcip.common.errors import InvariantViolation
from fcip.simulation.settlement import settle

A, B, EXT = 1, 2, 99


def test_overdraft_limit_declines_and_declined_moves_no_money() -> None:
    s = settle(
        [10, 11, 12],
        [100, 200, 300],
        [A, A, A],
        [B, B, B],
        [60, 60, 30],
        [-1, -1, -1],
        {A: 100, B: 0, EXT: 0},
        {A: 0, B: 0, EXT: None},
    )
    assert s.status.tolist() == ["SETTLED", "DECLINED", "SETTLED"]
    assert s.src_before.tolist() == [100, 40, 40] and s.src_after.tolist() == [40, 40, 10]
    assert s.final_balance == {A: 10, B: 90, EXT: 0}
    assert sum(s.final_balance.values()) == 100


def test_external_accounts_have_no_limit() -> None:
    s = settle([1], [5], [EXT], [A], [10**9], [-1], {A: 0, EXT: 0}, {A: 0, EXT: None})
    assert s.status.tolist() == ["SETTLED"] and s.final_balance[EXT] == -(10**9)


def test_failed_dependency_declines_dependent_event() -> None:
    # 1: A -> B declined (insufficient funds); 2: B -> EXT depends on 1 -> declined although B has funds.
    s = settle(
        [1, 2], [10, 20], [A, B], [B, EXT], [500, 5], [-1, 1], {A: 0, B: 100, EXT: 0}, {A: 0, B: 0, EXT: None}
    )
    assert s.status.tolist() == ["DECLINED", "DECLINED"]


def test_order_is_ts_then_event_id_and_dependency_must_precede() -> None:
    s = settle([2, 1], [10, 10], [A, A], [B, B], [80, 80], [-1, -1], {A: 100, B: 0}, {A: 0, B: 0})
    assert s.status.tolist() == ["DECLINED", "SETTLED"]  # event 1 settles first (same ts, lower id)
    with pytest.raises(InvariantViolation):
        settle([1, 2], [10, 20], [A, A], [B, B], [1, 1], [2, -1], {A: 10, B: 0}, {A: 0, B: 0})
    with pytest.raises(InvariantViolation):
        settle([1, 1], [10, 20], [A, A], [B, B], [1, 1], [-1, -1], {A: 10, B: 0}, {A: 0, B: 0})
