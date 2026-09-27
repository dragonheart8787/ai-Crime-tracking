"""fcip-content-hash-v1 (decision 0001)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fcip.common.errors import SchemaError
from fcip.common.hashing import _string_bytes, _string_bytes_reference, table_hash


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_same_content_via_polars_and_pyarrow_writers(tiny_ds, tmp_path: Path) -> None:
    """Same logical table written by two libraries (different compression, shuffled rows, different
    chunking): file bytes differ, content hashes are identical."""
    for name in ("transactions", "logins", "labels", "accounts"):
        table = tiny_ds.tables[name]
        a_path, b_path = tmp_path / f"{name}_pa.parquet", tmp_path / f"{name}_pl.parquet"
        pq.write_table(table, a_path, compression="zstd")
        perm = np.random.default_rng(0).permutation(table.num_rows)
        pl.from_arrow(table.take(perm)).write_parquet(b_path, compression="snappy", row_group_size=997)
        assert _sha(a_path) != _sha(b_path)
        read_pa = pq.read_table(a_path)
        read_pl = pl.read_parquet(b_path).to_arrow()  # polars reads strings back as large_string
        assert table_hash(name, read_pa) == table_hash(name, read_pl) == table_hash(name, table)


def _small() -> pa.Table:
    return pa.table(
        {
            "event_id": [3, 1, 2],
            "ts": [30, 10, 20],
            "account_id": [7, 7, 8],
            "device_id": [1, 1, 2],
            "ip_id": [5, 5, 6],
            "channel": ["APP", "WEB", "APP"],
            "outcome": ["SUCCESS"] * 3,
        }
    )


def test_row_order_chunking_and_column_order_do_not_matter() -> None:
    t = _small()
    h = table_hash("logins", t)
    assert table_hash("logins", t.take([2, 0, 1])) == h
    assert table_hash("logins", pa.concat_tables([t.slice(0, 1), t.slice(1)])) == h
    assert table_hash("logins", t.select(list(reversed(t.column_names)))) == h


def test_single_value_and_null_changes_are_detected() -> None:
    t = _small()
    h = table_hash("logins", t)
    changed = t.set_column(1, "ts", pa.array([30, 10, 21]))
    assert table_hash("logins", changed) != h
    tx = pa.table({"person_id": [1, 2], "region": [0, 1], "household_id": [0, 0], "created_at": [0, 5]})
    tx_null = pa.table(
        {
            "account_id": [1],
            "account_kind": ["external"],
            "external_role": [None],
            "owner_person_id": [None],
            "opened_at": [0],
            "closed_at": [None],
            "overdraft_limit_minor": [None],
            "initial_balance_minor": [0],
            "region": [None],
        }
    )
    tx_nonnull = tx_null.set_column(2, "external_role", pa.array([""]))
    assert table_hash("persons", tx) != table_hash(
        "persons", tx.set_column(3, "created_at", pa.array([0, 6]))
    )
    assert table_hash("accounts", tx_null) != table_hash("accounts", tx_nonnull)


def test_duplicate_primary_key_and_lossy_cast_raise() -> None:
    t = _small()
    with pytest.raises(SchemaError, match="duplicate primary key"):
        table_hash("logins", pa.concat_tables([t, t.slice(0, 1)]))
    with pytest.raises(SchemaError):
        table_hash("logins", t.set_column(1, "ts", pa.array([1.5, 2.0, 3.0])))
    with pytest.raises(SchemaError):
        table_hash("logins", t.append_column("extra", pa.array([1, 2, 3])))


@settings(max_examples=200, deadline=None)
@given(st.lists(st.one_of(st.none(), st.text(max_size=12)), max_size=40), st.integers(0, 5))
def test_vectorized_string_encoding_matches_reference(values: list[str | None], offset: int) -> None:
    arr = pa.array(values, type=pa.string())
    sl = arr.slice(min(offset, len(arr)))
    assert _string_bytes(arr) == _string_bytes_reference(arr)
    assert _string_bytes(sl) == _string_bytes_reference(sl)
    assert _string_bytes(sl.cast(pa.large_string())) == _string_bytes_reference(sl)
