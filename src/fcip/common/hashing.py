"""Canonical content hash ``fcip-content-hash-v1`` (decision 0001).

Dataset identity depends only on logical content: the table is cast to its registered schema, sorted by
its full primary key, and serialized with a fixed byte encoding. Writer library, compression, row order
and chunking do not change the hash.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from fcip.common.errors import SchemaError
from fcip.common.schemas import conform, spec

TABLE_TAG = b"fcip-content-hash-v1\0"
_INT_DTYPES: dict[pa.DataType, str] = {
    pa.int8(): "<i1",
    pa.int16(): "<i2",
    pa.int32(): "<i4",
    pa.int64(): "<i8",
    pa.uint8(): "<u1",
    pa.uint16(): "<u2",
    pa.uint32(): "<u4",
    pa.uint64(): "<u8",
}
DATASET_TAG = b"fcip-dataset-hash-v1\0"


def canonicalize(name: str, table: pa.Table) -> pa.Table:
    """Cast to schema, sort by full primary key, assert PK uniqueness, combine chunks."""
    t = conform(name, table)
    pk = spec(name).primary_key
    t = t.sort_by([(k, "ascending") for k in pk]).combine_chunks()
    if t.num_rows > 1:
        same = None
        for k in pk:
            col = t.column(k)
            eq = pc.equal(col.slice(1), col.slice(0, t.num_rows - 1))
            same = eq if same is None else pc.and_(same, eq)
        if pc.any(same).as_py():
            raise SchemaError(f"{name}: duplicate primary key {pk}")
    return t


def _string_bytes(col: pa.ChunkedArray | pa.Array) -> bytes:
    """Per row: uint64 LE byte length, then UTF-8 bytes. Nulls encode as length 0 (validity disambiguates)."""
    arr = pc.fill_null(col, "").cast(pa.large_string())
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    n = len(arr)
    if n == 0:
        return b""
    _, off_buf, data_buf = arr.buffers()
    offsets = np.frombuffer(off_buf, dtype=np.int64, count=n + 1 + arr.offset)[arr.offset :]
    start = int(offsets[0])
    lens = np.diff(offsets).astype(np.int64)
    total = int(lens.sum())
    data = (
        np.frombuffer(data_buf, dtype=np.uint8, count=start + total)[start:]
        if total
        else np.zeros(0, dtype=np.uint8)
    )
    out = np.empty(8 * n + total, dtype=np.uint8)
    rel = offsets[:-1] - start
    len_pos = 8 * np.arange(n, dtype=np.int64) + rel
    out[len_pos[:, None] + np.arange(8)] = lens.astype("<u8").view(np.uint8).reshape(n, 8)
    if total:
        row_of_byte = np.repeat(np.arange(n, dtype=np.int64), lens)
        out[8 * (row_of_byte + 1) + np.arange(total, dtype=np.int64)] = data
    return out.tobytes()


def _string_bytes_reference(col: pa.ChunkedArray | pa.Array) -> bytes:
    """Slow, obviously-correct reference for ``_string_bytes`` (used by tests)."""
    parts = []
    for v in col.to_pylist():
        raw = b"" if v is None else v.encode("utf-8")
        parts.append(struct.pack("<Q", len(raw)))
        parts.append(raw)
    return b"".join(parts)


def _column_bytes(field: pa.Field, col: pa.ChunkedArray) -> tuple[bytes, bytes]:
    validity = np.asarray(pc.is_valid(col).to_numpy(zero_copy_only=False), dtype=np.uint8).tobytes()
    typ = field.type
    if pa.types.is_integer(typ):
        np_dtype = np.dtype(_INT_DTYPES[typ])
        values = np.asarray(pc.fill_null(col, 0).to_numpy(zero_copy_only=False)).astype(np_dtype).tobytes()
    elif pa.types.is_boolean(typ):
        values = np.asarray(pc.fill_null(col, False).to_numpy(zero_copy_only=False), dtype=np.uint8).tobytes()
    elif pa.types.is_string(typ):
        values = _string_bytes(col)
    else:
        raise SchemaError(f"dtype {typ} not allowed in canonical hash")
    return validity, values


def table_digest(name: str, table: pa.Table) -> bytes:
    t = canonicalize(name, table)
    h = hashlib.sha256()
    h.update(TABLE_TAG)
    h.update(name.encode("utf-8") + b"\0")
    for f in t.schema:
        h.update(f"{f.name}:{f.type}:{int(f.nullable)};".encode())
    h.update(struct.pack("<Q", t.num_rows))
    for f in t.schema:
        validity, values = _column_bytes(f, t.column(f.name))
        h.update(validity)
        h.update(values)
    return h.digest()


def table_hash(name: str, table: pa.Table) -> str:
    return table_digest(name, table).hex()


def dataset_hash(table_digests: dict[str, bytes]) -> str:
    h = hashlib.sha256()
    h.update(DATASET_TAG)
    for name in sorted(table_digests):
        h.update(name.encode("utf-8") + b"\0")
        h.update(table_digests[name])
    return h.hexdigest()
