"""Validated Parquet read/write. Parquet (pyarrow, zstd) is the source of truth; CSV is an explicit export."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from fcip.common.schemas import conform

PARQUET_COMPRESSION = "zstd"


def write_table(name: str, table: pa.Table, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    pq.write_table(conform(name, table), path, compression=PARQUET_COMPRESSION)
    return path


def read_table(name: str, data_dir: Path) -> pa.Table:
    path = data_dir / f"{name}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return conform(name, pq.read_table(path))


def export_csv(name: str, data_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"
    pacsv.write_csv(read_table(name, data_dir), path)
    return path


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
