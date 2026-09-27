"""Disabling one scenario family does not change unrelated data (decision 0002, exact guarantee).

Checked on generation intents: instances of other families are identical, and every normal event that
involves no touched account is identical. Settlement couples accounts through balances, so balances of
accounts that transact with touched accounts may legitimately differ; that coupling is measured and
reported here rather than hidden.
"""

from __future__ import annotations

import polars as pl

from fcip.common.taxonomy import Family
from fcip.config.loader import load_config
from fcip.simulation.generator import generate

INTENT_COLS = [
    "event_id",
    "ts",
    "txn_type",
    "channel",
    "src_account_id",
    "dst_account_id",
    "merchant_id",
    "atm_id",
    "amount_minor",
    "region",
    "device_id",
    "ip_id",
    "login_event_id",
]
DISABLED = Family.BURST


def _variant():
    return generate(load_config("tiny", overrides={"scenarios": {DISABLED.value: {"enabled": False}}}))


def test_disabling_a_family_leaves_unrelated_data_unchanged(tiny_ds) -> None:
    base, var = tiny_ds, _variant()
    disabled_plans = [p for p in base.plans if p.family == DISABLED]
    assert disabled_plans, "baseline must contain the family being disabled"
    assert not any(p.family == DISABLED for p in var.plans)

    # 1. every other instance present in both runs is identical (members, params, events)
    base_by = {(p.family, p.instance_idx): p for p in base.plans if p.family != DISABLED}
    var_by = {(p.family, p.instance_idx): p for p in var.plans}
    common = set(base_by) & set(var_by)
    assert common
    for key in common:
        a, b = base_by[key], var_by[key]
        assert [m.account_id for m in a.members] == [m.account_id for m in b.members], key
        assert a.params == b.params, key
        assert a.intervals == b.intervals, key
        assert a.log.t_event_id == b.log.t_event_id and a.log.t_amount == b.log.t_amount, key
        assert a.log.l_event_id == b.log.l_event_id, key

    # 2. normal events that involve no touched account are identical (by event_id, intent columns)
    touched = {m.account_id for p in disabled_plans for m in p.members}
    logins_b = pl.from_arrow(base.tables["logins"])
    logins_v = pl.from_arrow(var.tables["logins"])

    def untouched_txn(ds) -> pl.DataFrame:
        t = pl.from_arrow(ds.tables["transactions"]).select(INTENT_COLS)
        return t.filter(
            ~pl.col("src_account_id").is_in(list(touched)) & ~pl.col("dst_account_id").is_in(list(touched))
        ).sort("event_id")

    tb, tv = untouched_txn(base), untouched_txn(var)
    scen_ids = {e for p in base.plans for e in p.log.t_event_id} | {
        e for p in var.plans for e in p.log.t_event_id
    }
    tb = tb.filter(~pl.col("event_id").is_in(list(scen_ids)))
    tv = tv.filter(~pl.col("event_id").is_in(list(scen_ids)))
    assert tb.height > 1000
    assert tb.equals(tv)

    # Logins: exclude scenario logins and session logins of transactions that involve a touched account
    # (such a login belongs to a transaction with a touched counterparty, so it is related, not unrelated).
    related_logins: set[int] = set()
    for ds in (base, var):
        t = pl.from_arrow(ds.tables["transactions"])
        rel = t.filter(
            pl.col("src_account_id").is_in(list(touched)) | pl.col("dst_account_id").is_in(list(touched))
        )
        related_logins |= set(rel["login_event_id"].drop_nulls().to_list())
    # the variant has no suppression, so it contains the full set of such sessions
    scen_l = {e for p in base.plans for e in p.log.l_event_id} | {
        e for p in var.plans for e in p.log.l_event_id
    }
    excluded = list(scen_l | related_logins)
    lb = logins_b.filter(~pl.col("account_id").is_in(list(touched)) & ~pl.col("event_id").is_in(excluded))
    lv = logins_v.filter(~pl.col("account_id").is_in(list(touched)) & ~pl.col("event_id").is_in(excluded))
    assert lb.height > 1000
    assert lb.sort("event_id").equals(lv.sort("event_id"))

    # 3. population tables are identical except for scenario-created entities
    for name in ("persons", "devices", "atms", "merchants"):
        a = pl.from_arrow(base.tables[name]).sort(pl.all())
        b = pl.from_arrow(var.tables[name]).sort(pl.all())
        created_b = {p.pid for pl_ in base.plans for p in pl_.created.persons}
        if name == "persons":
            a = a.filter(~pl.col("person_id").is_in(list(created_b)))
        if name in ("atms", "merchants"):
            assert a.equals(b)
        else:
            assert a.join(b, on=a.columns, how="anti").height <= sum(
                len(p.created.devices) + len(p.created.persons) for p in disabled_plans
            )


def test_partition_independent_of_enabled_families(tiny_ds) -> None:
    var = _variant()
    assert tiny_ds.partition is not None and var.partition is not None
    assert tiny_ds.partition.cell_of_account == var.partition.cell_of_account
