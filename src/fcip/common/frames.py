"""Typed Arrow-to-polars conversion (``pl.from_arrow`` is typed as returning ``DataFrame | Series``)."""

from __future__ import annotations

import polars as pl
import pyarrow as pa


def to_df(table: pa.Table) -> pl.DataFrame:
    out = pl.from_arrow(table)
    if not isinstance(out, pl.DataFrame):
        raise TypeError("expected a table, got a single column")
    return out
