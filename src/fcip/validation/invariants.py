"""Dataset invariants (PHASE0_ASSESSMENT section 7). Every check raises ``InvariantViolation`` on failure.

These run on the canonical tables only, so they validate what is actually written, not generator internals.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
import pyarrow as pa

from fcip.common.errors import InvariantViolation
from fcip.common.schemas import FORBIDDEN_COLUMNS, TABLE_NAMES
from fcip.common.taxonomy import ALLOWED_CHANNELS, DIGITAL_CHANNELS, Channel, Phase, Status, TxnType
from fcip.common.timebase import SECONDS_PER_DAY

Tables = dict[str, pa.Table]


def _df(tables: Tables, name: str) -> pl.DataFrame:
    return pl.from_arrow(tables[name])  # type: ignore[return-value]


def _fail(check: str, msg: str, sample: pl.DataFrame | None = None) -> None:
    extra = "" if sample is None else f"\n{sample.head(5)}"
    raise InvariantViolation(f"[{check}] {msg}{extra}")


def check_schema_no_currency(tables: Tables, **_: object) -> None:
    for name in TABLE_NAMES:
        bad = FORBIDDEN_COLUMNS.intersection(tables[name].column_names)
        if bad:
            _fail("S3", f"table {name} has forbidden column(s) {sorted(bad)}")


def check_timestamps(tables: Tables, sim_end: int, **_: object) -> None:
    for name in ("transactions", "logins"):
        df = _df(tables, name)
        bad = df.filter((pl.col("ts") < 0) | (pl.col("ts") >= sim_end))
        if bad.height:
            _fail("timestamps", f"{name}: {bad.height} events outside [0, sim_end)", bad)
        if df["event_id"].n_unique() != df.height:
            _fail("timestamps", f"{name}: duplicate event_id")
        key = df.select("ts", "event_id")
        if not key.equals(key.sort("ts", "event_id")):
            _fail("timestamps", f"{name}: rows not in global order (ts, event_id)")
    t = _df(tables, "transactions")["event_id"]
    lg = _df(tables, "logins")["event_id"]
    if t.is_in(lg.implode()).any():
        _fail("timestamps", "event_id shared between transactions and logins")


def check_amounts_and_channels(tables: Tables, **_: object) -> None:
    t = _df(tables, "transactions")
    if (t["amount_minor"] <= 0).any():
        _fail("amounts", "non-positive amount", t.filter(pl.col("amount_minor") <= 0))
    if (t["src_account_id"] == t["dst_account_id"]).any():
        _fail("amounts", "self-transfer")
    for ttype, chans in ALLOWED_CHANNELS.items():
        bad = t.filter(
            (pl.col("txn_type") == ttype.value) & ~pl.col("channel").is_in([c.value for c in chans])
        )
        if bad.height:
            _fail("channels", f"{ttype} with disallowed channel", bad)
    if not t["status"].is_in([s.value for s in Status]).all():
        _fail("channels", "unknown status")
    need = {TxnType.CARD_PAYMENT.value: "merchant_id", TxnType.ATM_WITHDRAWAL.value: "atm_id"}
    for tname, col in need.items():
        if t.filter((pl.col("txn_type") == tname) & pl.col(col).is_null()).height:
            _fail("channels", f"{tname} without {col}")


def check_login_invariant(tables: Tables, session_window: int, **_: object) -> None:
    """APP/WEB transactions follow a successful login of the same account, device and IP within the session
    window; all other channels have no device, IP or login reference (both directions)."""
    t = _df(tables, "transactions")
    lg = _df(tables, "logins").rename(
        {c: f"l_{c}" for c in ("ts", "account_id", "device_id", "ip_id", "channel", "outcome", "event_id")}
    )
    digital = [c.value for c in DIGITAL_CHANNELS]
    d = t.filter(pl.col("channel").is_in(digital))
    nd = t.filter(~pl.col("channel").is_in(digital))
    bad = nd.filter(
        pl.col("device_id").is_not_null()
        | pl.col("ip_id").is_not_null()
        | pl.col("login_event_id").is_not_null()
    )
    if bad.height:
        _fail("login", f"{bad.height} non-digital transactions carry device/IP/login references", bad)
    if d.filter(
        pl.col("login_event_id").is_null() | pl.col("device_id").is_null() | pl.col("ip_id").is_null()
    ).height:
        _fail("login", "digital transaction without login/device/IP reference")
    j = d.join(lg, left_on="login_event_id", right_on="l_event_id", how="left")
    bad = j.filter(
        pl.col("l_ts").is_null()
        | (pl.col("l_outcome") != "SUCCESS")
        | (pl.col("l_account_id") != pl.col("src_account_id"))
        | (pl.col("l_device_id") != pl.col("device_id"))
        | (pl.col("l_ip_id") != pl.col("ip_id"))
        | (pl.col("l_channel") != pl.col("channel"))
        | (pl.col("l_ts") > pl.col("ts"))
        | (pl.col("ts") - pl.col("l_ts") > session_window)
    )
    if bad.height:
        _fail(
            "login", f"{bad.height} digital transactions without a matching preceding successful login", bad
        )


def check_accounts_open(tables: Tables, **_: object) -> None:
    t = _df(tables, "transactions")
    a = _df(tables, "accounts").select("account_id", "opened_at")
    for col in ("src_account_id", "dst_account_id"):
        j = t.join(a, left_on=col, right_on="account_id", how="left")
        if j["opened_at"].null_count():
            _fail("accounts", f"{col} references unknown account")
        bad = j.filter(pl.col("ts") < pl.col("opened_at"))
        if bad.height:
            _fail("accounts", f"{bad.height} events before {col} was opened", bad)


def _ledger(tables: Tables) -> pl.DataFrame:
    """One row per (event, side) in global order with recorded before/after balances and the settled delta."""
    t = (
        _df(tables, "transactions")
        .select(
            "event_id",
            "ts",
            "status",
            "amount_minor",
            "src_account_id",
            "dst_account_id",
            "src_balance_before_minor",
            "src_balance_after_minor",
            "dst_balance_before_minor",
            "dst_balance_after_minor",
        )
        .with_row_index("pos")
    )
    settled = pl.col("status") == Status.SETTLED.value
    src = t.select(
        "pos",
        "ts",
        pl.col("src_account_id").alias("account_id"),
        pl.col("src_balance_before_minor").alias("before"),
        pl.col("src_balance_after_minor").alias("after"),
        pl.when(settled).then(-pl.col("amount_minor")).otherwise(0).alias("delta"),
    )
    dst = t.select(
        "pos",
        "ts",
        pl.col("dst_account_id").alias("account_id"),
        pl.col("dst_balance_before_minor").alias("before"),
        pl.col("dst_balance_after_minor").alias("after"),
        pl.when(settled).then(pl.col("amount_minor")).otherwise(0).alias("delta"),
    )
    del t
    return pl.concat([src, dst]).sort("account_id", "pos")


def check_balances(tables: Tables, **_: object) -> None:
    """Per-account chain consistency, settled/declined arithmetic, overdraft limits, and an independent
    recomputation of every balance from initial balances plus settled amounts."""
    acc = _df(tables, "accounts").select(
        "account_id", "initial_balance_minor", "overdraft_limit_minor", "account_kind"
    )
    led = _ledger(tables).join(acc, on="account_id", how="left")
    bad = led.filter(pl.col("after") != pl.col("before") + pl.col("delta"))
    if bad.height:
        _fail(
            "balances",
            f"{bad.height} rows where after != before +/- amount (declined must not move money)",
            bad,
        )
    led = led.with_columns(
        (pl.col("initial_balance_minor") + pl.col("delta").cum_sum().over("account_id")).alias(
            "recomputed_after"
        ),
        pl.col("after").shift(1).over("account_id").alias("prev_after"),
    )
    bad = led.filter(pl.col("recomputed_after") != pl.col("after"))
    if bad.height:
        _fail("balances", f"{bad.height} recorded balances differ from initial + settled deltas", bad)
    first = led.filter(pl.col("prev_after").is_null())
    if first.filter(pl.col("before") != pl.col("initial_balance_minor")).height:
        _fail("balances", "first event's balance_before differs from the account's initial balance")
    chain = led.filter(pl.col("prev_after").is_not_null() & (pl.col("before") != pl.col("prev_after")))
    if chain.height:
        _fail("balances", f"{chain.height} breaks in per-account balance chains", chain)
    over = led.filter(
        (pl.col("account_kind") == "internal") & (pl.col("after") < -pl.col("overdraft_limit_minor"))
    )
    if over.height:
        _fail("balances", f"{over.height} internal balances below -overdraft_limit", over)


def check_conservation(tables: Tables, sim_end: int, **_: object) -> None:
    """Sum of all recorded balances (internal + external) equals the initial total at every day boundary
    and at the end.

    Uses only recorded balance columns: the total at a boundary is the sum over accounts of the last
    recorded ``after`` before the boundary (initial balance if none). Equivalently, with
    ``step = after - previous recorded after`` per account (previous = initial balance for the first row),
    the running sum of ``step`` over all rows before each boundary must be zero.
    """
    acc = _df(tables, "accounts").select("account_id", "initial_balance_minor")
    led = _ledger(tables).select("account_id", "pos", "ts", "after").join(acc, on="account_id", how="left")
    led = led.with_columns(
        (
            pl.col("after")
            - pl.col("after").shift(1).over("account_id").fill_null(pl.col("initial_balance_minor"))
        ).alias("step"),
        (pl.col("ts") // SECONDS_PER_DAY).alias("day"),
    )
    per_day = led.group_by("day").agg(pl.col("step").sum()).sort("day")
    days = pl.DataFrame({"day": list(range(sim_end // SECONDS_PER_DAY + 1))}, schema={"day": pl.Int64})
    running = days.join(per_day.with_columns(pl.col("day").cast(pl.Int64)), on="day", how="left").select(
        "day", pl.col("step").fill_null(0).cum_sum().alias("drift")
    )
    bad = running.filter(pl.col("drift") != 0)
    if bad.height:
        _fail("conservation", f"total balance drifts from the initial total by day {bad['day'][0]}", bad)


def check_labels(tables: Tables, high_risk_phases: list[str], **_: object) -> None:
    lab = _df(tables, "labels")
    ev = _df(tables, "event_labels")
    nets = _df(tables, "ground_truth_networks")
    mem = _df(tables, "network_members")
    if lab.filter(pl.col("valid_from") >= pl.col("valid_to")).height:
        _fail("labels", "empty or inverted label interval")
    # S2: an account is a member of at most one network over the whole simulation.
    dup = (
        mem.group_by("entity_type", "entity_id")
        .agg(pl.col("network_id").n_unique().alias("n"))
        .filter(pl.col("n") > 1)
    )
    if dup.height:
        _fail("S2", f"{dup.height} accounts belong to more than one network", dup)
    per_acc = lab.group_by("entity_type", "entity_id").agg(
        pl.col("family").n_unique().alias("nf"), pl.col("scenario_id").n_unique().alias("ns")
    )
    if per_acc.filter((pl.col("nf") > 1) | (pl.col("ns") > 1)).height:
        _fail("S2", "an account's scenario/family assignment changes over the simulation")
    # S1: no overlapping label intervals for one account across different scenarios (and within one).
    s = lab.sort("entity_type", "entity_id", "valid_from").with_columns(
        pl.col("valid_to").shift(1).over("entity_type", "entity_id").alias("prev_to")
    )
    if s.filter(pl.col("prev_to").is_not_null() & (pl.col("valid_from") < pl.col("prev_to"))).height:
        _fail("S1", "overlapping label intervals for one account")
    # S4: known_at identical for every row of a network (labels, event_labels, networks).
    ka = pl.concat(
        [
            lab.select("network_id", "known_at"),
            ev.select("network_id", "known_at"),
            nets.select("network_id", "known_at"),
        ]
    )
    nu = ka.group_by("network_id").agg(pl.col("known_at").unique().len().alias("n")).filter(pl.col("n") > 1)
    if nu.height:
        _fail("S4", f"{nu.height} networks with non-uniform known_at", nu)
    if nets.filter(pl.col("detected") != pl.col("known_at").is_not_null()).height:
        _fail("S4", "detected flag inconsistent with known_at")
    # known_at never precedes the behavior it labels.
    j = lab.join(nets.select("network_id", "start_ts"), on="network_id")
    if j.filter(pl.col("known_at").is_not_null() & (pl.col("known_at") < pl.col("valid_from"))).height:
        _fail("labels", "label known before its interval starts")
    # event_labels complete for scenario events and terminal events well formed.
    t = _df(tables, "transactions").select("event_id", "ts", "status", "src_account_id", "dst_account_id")
    evt = ev.filter(pl.col("event_table") == "transactions").join(t, on="event_id", how="left")
    if evt["ts"].null_count():
        _fail("labels", "event_labels reference missing transactions")
    evl = ev.filter(pl.col("event_table") == "logins").join(
        _df(tables, "logins").select("event_id"), on="event_id", how="anti"
    )
    if evl.height:
        _fail("labels", "event_labels reference missing logins")
    term = evt.filter(pl.col("is_terminal"))
    if term.group_by("network_id").len().filter(pl.col("len") > 1).height:
        _fail("labels", "more than one terminal event in a network")
    if term.filter((pl.col("status") != Status.SETTLED.value) | (pl.col("phase") != Phase.EXIT.value)).height:
        _fail("labels", "terminal event not a settled EXIT event")
    if ev.filter(
        pl.col("is_high_risk")
        != pl.col("phase").is_in(high_risk_phases) & (pl.col("event_table") == "transactions")
    ).height:
        _fail("labels", "is_high_risk inconsistent with high_risk_phases")
    tn = nets.join(term.select("network_id", pl.col("ts").alias("tts")), on="network_id", how="left")
    if (
        tn.filter(pl.col("terminal_ts").is_not_null() != pl.col("tts").is_not_null()).height
        or tn.filter(pl.col("terminal_ts") != pl.col("tts")).height
    ):
        _fail("labels", "ground_truth_networks.terminal_ts disagrees with event_labels")


def check_scenario_causality(tables: Tables, **_: object) -> None:
    """Every settled MOVEMENT/EXIT outflow of a scenario account is preceded by a settled scenario inflow
    to it."""
    ev = _df(tables, "event_labels").filter(pl.col("event_table") == "transactions")
    t = _df(tables, "transactions").select("event_id", "ts", "status", "src_account_id", "dst_account_id")
    se = ev.join(t, on="event_id").filter(pl.col("status") == Status.SETTLED.value)
    outs = se.filter(pl.col("phase").is_in([Phase.MOVEMENT.value, Phase.EXIT.value]))
    ins = se.select("network_id", pl.col("dst_account_id").alias("acc"), pl.col("ts").alias("in_ts"))
    j = outs.join(ins, left_on=["network_id", "src_account_id"], right_on=["network_id", "acc"], how="left")
    ok = j.group_by("event_id").agg((pl.col("in_ts") < pl.col("ts")).any().alias("ok"))
    bad = ok.filter(~pl.col("ok").fill_null(False))
    if bad.height:
        _fail("causality", f"{bad.height} scenario outflows without an earlier settled inflow", bad)


def check_relations(tables: Tables, **_: object) -> None:
    rel = _df(tables, "relations")
    acc = _df(tables, "accounts").filter(pl.col("account_kind") == "internal")
    own = rel.filter(pl.col("relation_type") == "OWNS_ACCOUNT")
    j = acc.join(own, left_on=["owner_person_id", "account_id"], right_on=["src_id", "dst_id"], how="anti")
    if j.height:
        _fail("relations", f"{j.height} internal accounts without OWNS_ACCOUNT relation", j)
    lg = _df(tables, "logins").select("device_id")
    dev = _df(tables, "devices").select("device_id")
    if lg.join(dev, on="device_id", how="anti").height:
        _fail("relations", "login references unknown device")
    if (
        _df(tables, "transactions")
        .filter(pl.col("channel") == Channel.POS.value)
        .join(_df(tables, "merchants"), on="merchant_id", how="anti")
        .height
    ):
        _fail("relations", "card payment references unknown merchant")


def check_instance_diversity(
    tables: Tables, min_cv: float = 0.1, min_instances: int = 3, **_: object
) -> None:
    """Decision 0005: repeated instances of a family must not be near-identical templates. For every family with
    at least ``min_instances`` instances, the coefficient of variation of the drawn amount scale must reach
    ``min_cv``."""
    nets = _df(tables, "ground_truth_networks")
    stats = nets.group_by("family").agg(
        pl.len().alias("n"),
        pl.col("amount_scale_minor").cast(pl.Float64).std().alias("sd"),
        pl.col("amount_scale_minor").cast(pl.Float64).mean().alias("mu"),
    )
    bad = stats.filter((pl.col("n") >= min_instances) & (pl.col("sd") / pl.col("mu") < min_cv))
    if bad.height:
        _fail("diversity", f"{bad.height} families with near-identical instances", bad)


CHECKS: dict[str, Callable[..., None]] = {
    "schema_no_currency": check_schema_no_currency,
    "timestamps": check_timestamps,
    "amounts_and_channels": check_amounts_and_channels,
    "login_invariant": check_login_invariant,
    "accounts_open": check_accounts_open,
    "balances": check_balances,
    "conservation": check_conservation,
    "labels_S1_S2_S4": check_labels,
    "scenario_causality": check_scenario_causality,
    "relations": check_relations,
    "instance_diversity": check_instance_diversity,
}


def check_all(
    tables: Tables,
    *,
    sim_end: int,
    session_window: int,
    high_risk_phases: list[str],
    min_cv: float = 0.1,
    min_instances: int = 3,
) -> list[str]:
    ran = []
    for name, fn in CHECKS.items():
        fn(
            tables,
            sim_end=sim_end,
            session_window=session_window,
            high_risk_phases=high_risk_phases,
            min_cv=min_cv,
            min_instances=min_instances,
        )
        ran.append(name)
    return ran
