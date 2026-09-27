"""A1: a member's label intervals start at its own first observable event in the network, not at the
network phase boundary (Milestone 1 checkpoint 2 amendment)."""

from __future__ import annotations

import polars as pl

from fcip.simulation.labels import member_intervals


def _first_own_event(ds) -> pl.DataFrame:
    """Recomputed independently from the output tables (event_labels joined to transactions and logins)."""
    ev = pl.from_arrow(ds.tables["event_labels"]).select("event_table", "event_id", "network_id")
    t = pl.from_arrow(ds.tables["transactions"]).select("event_id", "ts", "src_account_id", "dst_account_id")
    lg = pl.from_arrow(ds.tables["logins"]).select("event_id", "ts", "account_id")
    tx = ev.filter(pl.col("event_table") == "transactions").join(t, on="event_id")
    sides = pl.concat(
        [
            tx.select("network_id", "ts", pl.col("src_account_id").alias("entity_id")),
            tx.select("network_id", "ts", pl.col("dst_account_id").alias("entity_id")),
            ev.filter(pl.col("event_table") == "logins")
            .join(lg, on="event_id")
            .select("network_id", "ts", pl.col("account_id").alias("entity_id")),
        ]
    )
    return sides.group_by("network_id", "entity_id").agg(pl.col("ts").min().alias("first_ts"))


def test_labels_start_at_first_own_event(tiny_ds) -> None:
    lab = pl.from_arrow(tiny_ds.tables["labels"])
    first = _first_own_event(tiny_ds)
    j = lab.join(first, on=["network_id", "entity_id"], how="left")
    assert j["first_ts"].null_count() == 0, "every labelled account has an own event in its network"
    assert (j["valid_from"] >= j["first_ts"]).all()
    earliest = j.group_by("network_id", "entity_id").agg(
        pl.col("valid_from").min(), pl.col("first_ts").first()
    )
    assert (earliest["valid_from"] == earliest["first_ts"]).all()
    mem = pl.from_arrow(tiny_ds.tables["network_members"]).join(
        first, on=["network_id", "entity_id"], how="inner"
    )
    assert (mem["joined_ts"] == mem["first_ts"]).all()


def test_rule_changes_rows_relative_to_phase_start(tiny_ds) -> None:
    """Under the old rule the first row of a member started at its phase start. The new rule must move
    some starts later and never earlier, and drop only rows that end before the first own event."""
    sim_end = tiny_ds.metadata["sim_end"]
    moved = dropped = kept = 0
    for plan in tiny_ds.plans:
        new = member_intervals(plan, sim_end)
        for m in plan.members:
            old = [
                (k, plan.intervals[k][1], min(plan.intervals[k][2], sim_end))
                for k in m.roles
                if plan.intervals[k][1] < sim_end
            ]
            new_by_k = {k: (s, e) for k, _ph, _r, s, e in new[m.account_id]}
            for k, s, e in old:
                if k not in new_by_k:
                    dropped += 1
                    continue
                kept += 1
                ns, ne = new_by_k[k]
                assert ns >= s and ne == e
                moved += ns > s
    assert moved > 0, "the rule must actually change some interval starts"
    assert kept > 0
