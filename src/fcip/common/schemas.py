"""Explicit Arrow schemas for every canonical table (decision 0001, schema in PHASE0_ASSESSMENT 4.5).

Column order, dtype and nullability are fixed here. Money is int64 minor units in one currency per
dataset; there is no per-row currency column anywhere (simplification S3), and the registry refuses one.
Tables marked ``oracle`` hold synthetic ground truth and must never be read by feature code.
"""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa

from fcip.common.errors import SchemaError

FORBIDDEN_COLUMNS: frozenset[str] = frozenset({"currency"})

ALLOWED_TYPES: tuple[pa.DataType, ...] = (
    pa.int8(),
    pa.int16(),
    pa.int32(),
    pa.int64(),
    pa.uint8(),
    pa.uint16(),
    pa.uint32(),
    pa.uint64(),
    pa.bool_(),
    pa.string(),
)


@dataclass(frozen=True)
class TableSpec:
    name: str
    schema: pa.Schema
    primary_key: tuple[str, ...]
    oracle: bool


def _f(name: str, typ: pa.DataType, nullable: bool = False) -> pa.Field:
    return pa.field(name, typ, nullable=nullable)


REGISTRY: dict[str, TableSpec] = {}


def register(
    name: str, fields: list[pa.Field], primary_key: tuple[str, ...], oracle: bool = False
) -> TableSpec:
    """Register a table schema. Refuses forbidden columns, disallowed dtypes and bad primary keys."""
    names = [f.name for f in fields]
    if len(set(names)) != len(names):
        raise SchemaError(f"{name}: duplicate column names")
    bad = FORBIDDEN_COLUMNS.intersection(names)
    if bad:
        raise SchemaError(f"{name}: forbidden column(s) {sorted(bad)} (single-currency dataset, S3)")
    for f in fields:
        if f.type not in ALLOWED_TYPES:
            raise SchemaError(f"{name}.{f.name}: dtype {f.type} not allowed in canonical tables")
    for k in primary_key:
        if k not in names:
            raise SchemaError(f"{name}: primary key column {k} missing")
        if next(f for f in fields if f.name == k).nullable:
            raise SchemaError(f"{name}: primary key column {k} must be non-nullable")
    if name in REGISTRY:
        raise SchemaError(f"table {name} registered twice")
    spec = TableSpec(name, pa.schema(fields), primary_key, oracle)
    REGISTRY[name] = spec
    return spec


i8, i16, i32, i64, s, b = pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.string(), pa.bool_()

register(
    "persons",
    [_f("person_id", i64), _f("region", i16), _f("household_id", i64), _f("created_at", i64)],
    ("person_id",),
)
register(
    "accounts",
    [
        _f("account_id", i64),
        _f("account_kind", s),
        _f("external_role", s, True),
        _f("owner_person_id", i64, True),
        _f("opened_at", i64),
        _f("closed_at", i64, True),
        _f("overdraft_limit_minor", i64, True),
        _f("initial_balance_minor", i64),
        _f("region", i16, True),
    ],
    ("account_id",),
)
register("devices", [_f("device_id", i64), _f("device_kind", s), _f("first_seen_at", i64)], ("device_id",))
register("ips", [_f("ip_id", i64), _f("ip_context", s), _f("region", i16)], ("ip_id",))
register("atms", [_f("atm_id", i64), _f("region", i16)], ("atm_id",))
register(
    "merchants", [_f("merchant_id", i64), _f("merchant_category", s), _f("region", i16)], ("merchant_id",)
)
register(
    "relations",
    [
        _f("relation_type", s),
        _f("src_id", i64),
        _f("dst_id", i64),
        _f("valid_from", i64),
        _f("valid_to", i64),
    ],
    ("relation_type", "src_id", "dst_id", "valid_from"),
)
register(
    "transactions",
    [
        _f("event_id", i64),
        _f("ts", i64),
        _f("txn_type", s),
        _f("channel", s),
        _f("src_account_id", i64),
        _f("dst_account_id", i64),
        _f("merchant_id", i64, True),
        _f("atm_id", i64, True),
        _f("amount_minor", i64),
        _f("region", i16),
        _f("device_id", i64, True),
        _f("ip_id", i64, True),
        _f("login_event_id", i64, True),
        _f("status", s),
        _f("src_balance_before_minor", i64),
        _f("src_balance_after_minor", i64),
        _f("dst_balance_before_minor", i64),
        _f("dst_balance_after_minor", i64),
    ],
    ("event_id",),
)
register(
    "logins",
    [
        _f("event_id", i64),
        _f("ts", i64),
        _f("account_id", i64),
        _f("device_id", i64),
        _f("ip_id", i64),
        _f("channel", s),
        _f("outcome", s),
    ],
    ("event_id",),
)

# ---- oracle (ground-truth) tables ----
register(
    "person_truth", [_f("person_id", i64), _f("archetype", s), _f("origin", s)], ("person_id",), oracle=True
)
register(
    "labels",
    [
        _f("entity_type", s),
        _f("entity_id", i64),
        _f("valid_from", i64),
        _f("valid_to", i64),
        _f("risk_label", i8),
        _f("role_label", s),
        _f("scenario_id", s),
        _f("network_id", i64),
        _f("family", s),
        _f("phase", s),
        _f("is_ood_family", b),
        _f("known_at", i64, True),
    ],
    ("entity_type", "entity_id", "scenario_id", "valid_from"),
    oracle=True,
)
register(
    "event_labels",
    [
        _f("event_table", s),
        _f("event_id", i64),
        _f("scenario_id", s),
        _f("network_id", i64),
        _f("family", s),
        _f("phase", s),
        _f("event_type", s),
        _f("is_high_risk", b),
        _f("is_terminal", b),
        _f("is_ood_family", b),
        _f("known_at", i64, True),
    ],
    ("event_table", "event_id"),
    oracle=True,
)
register(
    "ground_truth_networks",
    [
        _f("network_id", i64),
        _f("scenario_id", s),
        _f("family", s),
        _f("instance_idx", i32),
        _f("is_ood_family", b),
        _f("start_ts", i64),
        _f("end_ts", i64),
        _f("terminal_ts", i64, True),
        _f("censored", b),
        _f("n_members", i32),
        _f("n_created_accounts", i32),
        _f("n_rounds", i32),
        _f("detected", b),
        _f("known_at", i64, True),
        _f("amount_scale_minor", i64),
        _f("fan_degree", i32),
        _f("hops", i32),
        _f("holding_seconds", i64),
        _f("retention_permille", i32),
        _f("forward_permille", i32),
    ],
    ("network_id",),
    oracle=True,
)
register(
    "network_members",
    [
        _f("network_id", i64),
        _f("entity_type", s),
        _f("entity_id", i64),
        _f("first_role", s),
        _f("joined_ts", i64),
        _f("is_created", b),
    ],
    ("network_id", "entity_type", "entity_id"),
    oracle=True,
)

TABLE_NAMES: tuple[str, ...] = tuple(sorted(REGISTRY))


def spec(name: str) -> TableSpec:
    if name not in REGISTRY:
        raise SchemaError(f"unknown table {name}")
    return REGISTRY[name]


def conform(name: str, table: pa.Table) -> pa.Table:
    """Cast ``table`` to its registered schema. Extra, missing or forbidden columns and lossy casts raise."""
    sp = spec(name)
    cols = set(table.column_names)
    forbidden = FORBIDDEN_COLUMNS.intersection(cols)
    if forbidden:
        raise SchemaError(f"{name}: forbidden column(s) {sorted(forbidden)} (single-currency dataset, S3)")
    extra = cols - set(sp.schema.names)
    missing = set(sp.schema.names) - cols
    if extra or missing:
        raise SchemaError(f"{name}: extra columns {sorted(extra)}, missing columns {sorted(missing)}")
    arrays = []
    for f in sp.schema:
        col = table.column(f.name)
        try:
            col = col.cast(f.type, safe=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError) as exc:
            raise SchemaError(
                f"{name}.{f.name}: cannot cast {table.schema.field(f.name).type} to {f.type}"
            ) from exc
        if not f.nullable and col.null_count:
            raise SchemaError(f"{name}.{f.name}: {col.null_count} nulls in non-nullable column")
        arrays.append(col)
    return pa.Table.from_arrays(arrays, schema=sp.schema)
