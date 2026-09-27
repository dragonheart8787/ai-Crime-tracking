"""Every invariant passes on generated data and is shown to *fail* on a deliberately corrupted copy
(otherwise a check that can never fire would pass vacuously)."""

from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest

from fcip.common import schemas
from fcip.common.errors import InvariantViolation, SchemaError
from fcip.validation import invariants as inv


def _replace(ds_tables: dict, name: str, df: pl.DataFrame) -> dict:
    out = dict(ds_tables)
    out[name] = df.to_arrow().cast(ds_tables[name].schema)
    return out


def _df(ds, name: str) -> pl.DataFrame:
    return pl.from_arrow(ds.tables[name])


def test_all_checks_pass_on_generated_data(tiny_ds, check_kwargs) -> None:
    assert inv.check_all(tiny_ds.tables, **check_kwargs) == list(inv.CHECKS)
    assert tiny_ds.metadata["invariant_checks_passed"] == list(inv.CHECKS)


# ---------------------------------------------------------------- timestamps
def test_timestamp_out_of_range_detected(tiny_ds, check_kwargs) -> None:
    t = _df(tiny_ds, "transactions")
    bad = t.with_columns(
        pl.when(pl.int_range(pl.len()) == t.height - 1)
        .then(check_kwargs["sim_end"])
        .otherwise(pl.col("ts"))
        .alias("ts")
    )
    with pytest.raises(InvariantViolation, match="outside"):
        inv.check_timestamps(_replace(tiny_ds.tables, "transactions", bad), **check_kwargs)


def test_unordered_rows_detected(tiny_ds, check_kwargs) -> None:
    lg = _df(tiny_ds, "logins")
    with pytest.raises(InvariantViolation, match="global order"):
        inv.check_timestamps(_replace(tiny_ds.tables, "logins", lg.reverse()), **check_kwargs)


def test_logins_precede_their_transactions(tiny_ds) -> None:
    t = _df(tiny_ds, "transactions").filter(pl.col("login_event_id").is_not_null())
    j = t.join(_df(tiny_ds, "logins"), left_on="login_event_id", right_on="event_id", suffix="_l")
    assert j.height == t.height and (j["ts_l"] <= j["ts"]).all()


# ---------------------------------------------------------------- balances and conservation
def test_amounts_positive_and_balances_consistent(tiny_ds) -> None:
    t = _df(tiny_ds, "transactions")
    assert (t["amount_minor"] > 0).all()
    assert t["status"].is_in(["SETTLED", "DECLINED"]).all()
    declined = t.filter(pl.col("status") == "DECLINED")
    assert declined.height > 0, "fixture should exercise the declined path"
    assert (declined["src_balance_after_minor"] == declined["src_balance_before_minor"]).all()


def test_balance_tampering_detected(tiny_ds, check_kwargs) -> None:
    t = _df(tiny_ds, "transactions")
    bad = t.with_columns(
        pl.when(pl.int_range(pl.len()) == 100)
        .then(pl.col("dst_balance_after_minor") + 1)
        .otherwise(pl.col("dst_balance_after_minor"))
        .alias("dst_balance_after_minor")
    )
    with pytest.raises(InvariantViolation, match="balances"):
        inv.check_balances(_replace(tiny_ds.tables, "transactions", bad), **check_kwargs)


def test_overdraft_violation_detected(tiny_ds, check_kwargs) -> None:
    a = _df(tiny_ds, "accounts")
    t = _df(tiny_ds, "transactions")
    # shrink the overdraft limit of an account that goes below zero
    neg = t.filter(pl.col("src_balance_after_minor") < 0)
    assert neg.height, "fixture should contain an overdrawn internal account"
    acc = neg["src_account_id"][0]
    bad = a.with_columns(
        pl.when(pl.col("account_id") == acc)
        .then(0)
        .otherwise(pl.col("overdraft_limit_minor"))
        .alias("overdraft_limit_minor")
    )
    with pytest.raises(InvariantViolation, match="overdraft"):
        inv.check_balances(_replace(tiny_ds.tables, "accounts", bad), **check_kwargs)


def test_conservation_holds_and_violation_detected(tiny_ds, check_kwargs) -> None:
    inv.check_conservation(tiny_ds.tables, **check_kwargs)
    t = _df(tiny_ds, "transactions")
    # a settled event whose destination credit is inflated creates money
    idx = int(t.with_row_index().filter(pl.col("status") == "SETTLED")["index"][500])
    bad = t.with_columns(
        pl.when(pl.int_range(pl.len()) >= idx)
        .then(pl.col("dst_balance_after_minor") + 7)
        .otherwise(pl.col("dst_balance_after_minor"))
        .alias("dst_balance_after_minor")
    )
    with pytest.raises(InvariantViolation):
        inv.check_conservation(_replace(tiny_ds.tables, "transactions", bad), **check_kwargs)


def test_total_money_is_constant(tiny_ds) -> None:
    acc = _df(tiny_ds, "accounts")
    t = _df(tiny_ds, "transactions").with_row_index("pos")
    last = (
        pl.concat(
            [
                t.select(
                    "pos", pl.col("src_account_id").alias("a"), pl.col("src_balance_after_minor").alias("b")
                ),
                t.select(
                    "pos", pl.col("dst_account_id").alias("a"), pl.col("dst_balance_after_minor").alias("b")
                ),
            ]
        )
        .sort("pos")
        .group_by("a")
        .agg(pl.col("b").last())
    )
    final = (
        acc.join(last, left_on="account_id", right_on="a", how="left")
        .select(pl.coalesce("b", "initial_balance_minor").sum())
        .item()
    )
    assert final == acc["initial_balance_minor"].sum()


# ---------------------------------------------------------------- S1 - S4
def test_s1_overlap_detected(tiny_ds, check_kwargs) -> None:
    lab = _df(tiny_ds, "labels")
    row = lab.head(1)
    fake = row.with_columns(
        pl.lit("other:0").alias("scenario_id"), (pl.col("valid_from") + 1).alias("valid_from")
    )
    with pytest.raises(InvariantViolation, match=r"\[S[12]\]"):
        inv.check_labels(_replace(tiny_ds.tables, "labels", pl.concat([lab, fake])), **check_kwargs)


def test_s1_no_simultaneous_membership(tiny_ds) -> None:
    lab = _df(tiny_ds, "labels").sort("entity_id", "valid_from")
    prev = lab.with_columns(
        pl.col("valid_to").shift(1).over("entity_id").alias("p"),
        pl.col("scenario_id").shift(1).over("entity_id").alias("ps"),
    )
    assert prev.filter(pl.col("p").is_not_null() & (pl.col("valid_from") < pl.col("p"))).height == 0


def test_s2_single_scenario_for_life(tiny_ds, check_kwargs) -> None:
    mem = _df(tiny_ds, "network_members")
    assert (
        mem.group_by("entity_id").agg(pl.col("network_id").n_unique()).filter(pl.col("network_id") > 1).height
        == 0
    )
    lab = _df(tiny_ds, "labels")
    assert lab.group_by("entity_id").agg(pl.col("family").n_unique()).filter(pl.col("family") > 1).height == 0
    other_net = mem.filter(pl.col("network_id") != mem["network_id"][0])["network_id"][0]
    dup = mem.head(1).with_columns(pl.lit(other_net, dtype=pl.Int64).alias("network_id"))
    with pytest.raises(InvariantViolation, match=r"\[S2\]"):
        inv.check_labels(_replace(tiny_ds.tables, "network_members", pl.concat([mem, dup])), **check_kwargs)


def test_s2_partition_assigns_each_account_to_one_cell(tiny_ds) -> None:
    part = tiny_ds.partition
    assert part is not None
    assert sum(len(v) for v in part.by_cell.values()) == len(part.cell_of_account)
    members = [m.account_id for p in tiny_ds.plans for m in p.members if not m.is_created]
    assert len(members) == len(set(members))
    for p in tiny_ds.plans:
        cell = part.cells.index((p.family, p.instance_idx))
        assert all(part.cell_of_account[m.account_id] == cell for m in p.members if not m.is_created)


def test_s3_schema_rejects_currency_column() -> None:
    with pytest.raises(SchemaError, match="currency"):
        schemas.register(
            "bad_table",
            [pa.field("id", pa.int64(), nullable=False), pa.field("currency", pa.string())],
            ("id",),
        )
    t = pa.table({"atm_id": [1], "region": [0], "currency": ["SYN"]})
    with pytest.raises(SchemaError, match="currency"):
        schemas.conform("atms", t)
    assert all("currency" not in spec.schema.names for spec in schemas.REGISTRY.values())


def test_s3_currency_declared_once_in_metadata(tiny_ds) -> None:
    assert tiny_ds.metadata["currency"] == {"code": "SYN", "minor_units_per_major": 100}


def test_s4_known_at_uniform_per_network(tiny_ds, check_kwargs) -> None:
    lab, ev, nets = (
        _df(tiny_ds, "labels"),
        _df(tiny_ds, "event_labels"),
        _df(tiny_ds, "ground_truth_networks"),
    )
    for net in nets.iter_rows(named=True):
        vals = set(lab.filter(pl.col("network_id") == net["network_id"])["known_at"].to_list())
        vals |= set(ev.filter(pl.col("network_id") == net["network_id"])["known_at"].to_list())
        assert vals <= {net["known_at"]}, net["network_id"]
    # members of one network share known_at across all their rows
    j = lab.join(
        _df(tiny_ds, "network_members"),
        left_on=["network_id", "entity_id"],
        right_on=["network_id", "entity_id"],
    )
    assert (
        j.group_by("network_id").agg(pl.col("known_at").n_unique()).filter(pl.col("known_at") > 1).height == 0
    )
    detected = lab.filter(pl.col("known_at").is_not_null())
    assert detected.height, "fixture should contain a detected network"
    first = detected.head(1)
    bad = pl.concat(
        [
            lab.join(first, on=lab.columns, how="anti"),
            first.with_columns((pl.col("known_at") + 1).alias("known_at")),
        ]
    )
    with pytest.raises(InvariantViolation, match=r"\[S4\]"):
        inv.check_labels(_replace(tiny_ds.tables, "labels", bad), **check_kwargs)


def test_s4_known_at_strictly_after_anchor(tiny_ds) -> None:
    """known_at = anchor + latency with latency > 0 (decision 0004): never at or before the terminal event."""
    nets = _df(tiny_ds, "ground_truth_networks")
    det = nets.filter(pl.col("detected"))
    assert det.height > 0
    assert (det["known_at"] > det["start_ts"]).all()
    term = det.filter(pl.col("terminal_ts").is_not_null())
    assert ((term["known_at"] - term["terminal_ts"]) >= 3600).all()
    assert nets.filter(~pl.col("detected"))["known_at"].is_null().all()


# ---------------------------------------------------------------- login invariant (both directions)
def test_login_invariant_detects_missing_login(tiny_ds, check_kwargs) -> None:
    t = _df(tiny_ds, "transactions")
    i = int(t.with_row_index().filter(pl.col("channel") == "APP")["index"][0])
    bad = t.with_columns(
        pl.when(pl.int_range(pl.len()) == i)
        .then(None)
        .otherwise(pl.col("login_event_id"))
        .alias("login_event_id")
    )
    with pytest.raises(InvariantViolation, match="login"):
        inv.check_login_invariant(_replace(tiny_ds.tables, "transactions", bad), **check_kwargs)


def test_login_invariant_detects_wrong_device_or_late_login(tiny_ds, check_kwargs) -> None:
    lg = _df(tiny_ds, "logins")
    t = _df(tiny_ds, "transactions")
    lid = t.filter(pl.col("channel") == "WEB")["login_event_id"][0]
    moved = lg.with_columns(
        pl.when(pl.col("event_id") == lid)
        .then(pl.col("device_id") + 1)
        .otherwise(pl.col("device_id"))
        .alias("device_id")
    )
    with pytest.raises(InvariantViolation, match="login"):
        inv.check_login_invariant(_replace(tiny_ds.tables, "logins", moved), **check_kwargs)
    early = lg.with_columns(
        pl.when(pl.col("event_id") == lid).then(pl.col("ts") - 10_000).otherwise(pl.col("ts")).alias("ts")
    )
    with pytest.raises(InvariantViolation, match="login"):
        inv.check_login_invariant(_replace(tiny_ds.tables, "logins", early), **check_kwargs)


@pytest.mark.parametrize("channel", ["ATM", "POS", "SCHEDULED", "INBOUND_EXTERNAL"])
def test_non_digital_channels_have_no_login_reference(tiny_ds, check_kwargs, channel: str) -> None:
    t = _df(tiny_ds, "transactions")
    nd = t.filter(pl.col("channel") == channel)
    assert nd.height > 0
    assert nd.select(
        pl.col("device_id").is_null().all(),
        pl.col("ip_id").is_null().all(),
        pl.col("login_event_id").is_null().all(),
    ).row(0) == (True, True, True)
    i = int(t.with_row_index().filter(pl.col("channel") == channel)["index"][0])
    some_login = _df(tiny_ds, "logins")["event_id"][0]
    bad = t.with_columns(
        pl.when(pl.int_range(pl.len()) == i)
        .then(some_login)
        .otherwise(pl.col("login_event_id"))
        .alias("login_event_id")
    )
    with pytest.raises(InvariantViolation, match="non-digital"):
        inv.check_login_invariant(_replace(tiny_ds.tables, "transactions", bad), **check_kwargs)


def test_all_digital_transactions_have_matching_successful_login(tiny_ds) -> None:
    t = _df(tiny_ds, "transactions").filter(pl.col("channel").is_in(["APP", "WEB"]))
    j = t.join(_df(tiny_ds, "logins"), left_on="login_event_id", right_on="event_id", suffix="_l")
    assert j.height == t.height
    assert (j["outcome"] == "SUCCESS").all()
    assert (j["account_id"] == j["src_account_id"]).all()
    assert (j["device_id_l"] == j["device_id"]).all() and (j["ip_id_l"] == j["ip_id"]).all()


# ---------------------------------------------------------------- scenario causality and labels
def test_causality_violation_detected(tiny_ds, check_kwargs) -> None:
    ev = _df(tiny_ds, "event_labels")
    t = _df(tiny_ds, "transactions")
    outs = ev.filter((pl.col("event_table") == "transactions") & pl.col("phase").is_in(["MOVEMENT", "EXIT"]))
    target = outs.join(t.filter(pl.col("status") == "SETTLED"), on="event_id")["event_id"][0]
    bad = t.with_columns(pl.when(pl.col("event_id") == target).then(0).otherwise(pl.col("ts")).alias("ts"))
    with pytest.raises(InvariantViolation, match="causality"):
        inv.check_scenario_causality(_replace(tiny_ds.tables, "transactions", bad), **check_kwargs)


def test_roles_depend_on_phase(tiny_ds) -> None:
    lab = _df(tiny_ds, "labels")
    per = lab.group_by("entity_id").agg(pl.col("role_label").n_unique().alias("n"))
    assert per.filter(pl.col("n") > 1).height > 0, "some members must change role across phases"
    assert set(lab["phase"].unique()) <= {"SETUP", "INFLOW", "HOLD", "MOVEMENT", "EXIT"}


def test_instance_diversity_detects_template_instances(tiny_ds, check_kwargs) -> None:
    nets = _df(tiny_ds, "ground_truth_networks")
    fam = nets["family"][0]
    clones = pl.concat([nets.filter(pl.col("family") == fam).head(1)] * 3).with_columns(
        (pl.col("network_id") + pl.int_range(pl.len()) * 1_000_003).alias("network_id")
    )
    with pytest.raises(InvariantViolation, match="diversity"):
        inv.check_instance_diversity(
            _replace(tiny_ds.tables, "ground_truth_networks", clones), min_cv=0.1, min_instances=3
        )
