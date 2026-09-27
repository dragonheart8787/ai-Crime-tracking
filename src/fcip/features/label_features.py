"""Label-derived features. They read labels only through ``AsOfView.known_labels()``, so a label can influence
a feature at cutoff ``t`` only if it was known at ``t`` (decision 0004)."""

from __future__ import annotations

import polars as pl

from fcip.temporal.store import AsOfView


def flagged_counterparties(view: AsOfView, account_id: int, window_s: int) -> int:
    """Distinct counterparties of ``account_id`` in the trailing window whose account carries a known
    high-risk label at the view's cutoff."""
    t = view.cutoff.t
    tx = view.transactions(start=max(0, t - window_s + 1), end=t)
    cps = pl.concat(
        [
            tx.filter(pl.col("src_account_id") == account_id).select(pl.col("dst_account_id").alias("cp")),
            tx.filter(pl.col("dst_account_id") == account_id).select(pl.col("src_account_id").alias("cp")),
        ]
    ).unique()
    flagged = (
        view.known_labels().filter(pl.col("risk_label") == 1).select(pl.col("entity_id").alias("cp")).unique()
    )
    return cps.join(flagged, on="cp").height
