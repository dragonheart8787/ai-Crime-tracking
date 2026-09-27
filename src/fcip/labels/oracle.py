"""Ground-truth access (decisions 0004, 0009). Two tiers, two types:

* :class:`OracleTargets` - full synthetic ground truth, for **evaluation only** (VAL, TEST, OOD). Metric
  functions accept nothing else.
* :class:`TrainingTargets` - targets built only from label rows whose network ``known_at <= fit_time``
  (never-known and later-known positives are 0, the realistic positive-unlabeled setting).

This module reads oracle tables. Feature code must never import it (enforced by
``tests/leakage/test_import_boundaries.py``); split construction may, as experimental protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa

from fcip.common.frames import to_df
from fcip.common.io import read_table

_TOKEN = object()


def _ro(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class OracleTargets:
    """Ground-truth targets. Only :class:`OracleLabels` can construct them."""

    kind: str
    account_ids: np.ndarray
    cutoffs: np.ndarray
    y: np.ndarray
    _token: object = field(repr=False, compare=False, default=None)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise TypeError("OracleTargets can only be created by OracleLabels")

    @property
    def n_positive(self) -> int:
        return int(self.y.sum())


@dataclass(frozen=True)
class TrainingTargets:
    """Targets as known at ``fit_time``. Deliberately not accepted by evaluation metrics."""

    kind: str
    account_ids: np.ndarray
    cutoffs: np.ndarray
    y: np.ndarray
    fit_time: int


class OracleLabels:
    def __init__(
        self,
        labels: pa.Table,
        event_labels: pa.Table,
        transactions: pa.Table,
        networks: pa.Table,
        members: pa.Table,
    ) -> None:
        self._labels = to_df(labels)
        self._networks = to_df(networks)
        self._members = to_df(members)
        ev = to_df(event_labels).filter(pl.col("event_table") == "transactions")
        tx = to_df(transactions).select("event_id", "ts", "src_account_id")
        self._high_risk = (
            ev.filter(pl.col("is_high_risk"))
            .join(tx, on="event_id")
            .select("src_account_id", "ts", "known_at", "is_ood_family", "network_id")
        )

    @classmethod
    def from_dir(cls, data_dir: Path) -> OracleLabels:
        return cls(
            read_table("labels", data_dir),
            read_table("event_labels", data_dir),
            read_table("transactions", data_dir),
            read_table("ground_truth_networks", data_dir),
            read_table("network_members", data_dir),
        )

    # ---------------------------------------------------------------- protocol helpers (splits)
    def ood_entities(self) -> set[int]:
        ood_nets = self._networks.filter(pl.col("is_ood_family"))["network_id"]
        return set(
            self._members.filter(pl.col("network_id").is_in(ood_nets.implode()))["entity_id"].to_list()
        )

    def member_entities(self) -> set[int]:
        return set(self._members["entity_id"].to_list())

    # ---------------------------------------------------------------- target construction
    @staticmethod
    def _active(labels: pl.DataFrame, account_ids: np.ndarray, cutoffs: np.ndarray) -> np.ndarray:
        q = pl.DataFrame({"row": np.arange(len(account_ids)), "entity_id": account_ids, "t": cutoffs})
        hit = q.join(labels.select("entity_id", "valid_from", "valid_to"), on="entity_id").filter(
            (pl.col("valid_from") <= pl.col("t")) & (pl.col("t") < pl.col("valid_to"))
        )
        y = np.zeros(len(account_ids), dtype=bool)
        y[hit["row"].unique().to_numpy()] = True
        return y

    @staticmethod
    def _event_within(
        ev: pl.DataFrame, account_ids: np.ndarray, cutoffs: np.ndarray, horizon_s: int
    ) -> np.ndarray:
        q = pl.DataFrame({"row": np.arange(len(account_ids)), "src_account_id": account_ids, "t": cutoffs})
        hit = q.join(ev.select("src_account_id", "ts"), on="src_account_id").filter(
            (pl.col("ts") > pl.col("t")) & (pl.col("ts") <= pl.col("t") + horizon_s)
        )
        y = np.zeros(len(account_ids), dtype=bool)
        y[hit["row"].unique().to_numpy()] = True
        return y

    def targets(
        self, kind: str, account_ids: np.ndarray, cutoffs: np.ndarray, horizon_s: int | None = None
    ) -> OracleTargets:
        """Full ground truth. ``kind``: ``active_phase`` (account in a labelled scenario interval at t) or
        ``high_risk_within`` (account is the source of a high-risk scenario event in ``(t, t + horizon]``)."""
        a, c = np.asarray(account_ids, np.int64), np.asarray(cutoffs, np.int64)
        y = self._compute(kind, self._labels, self._high_risk, a, c, horizon_s)
        return OracleTargets(kind, _ro(a), _ro(c), _ro(y), _TOKEN)

    def training_targets(
        self,
        kind: str,
        account_ids: np.ndarray,
        cutoffs: np.ndarray,
        fit_time: int,
        horizon_s: int | None = None,
    ) -> TrainingTargets:
        """Targets from label rows known at ``fit_time`` only (never OOD-family rows)."""
        lab = self._labels.filter(
            pl.col("known_at").is_not_null() & (pl.col("known_at") <= fit_time) & ~pl.col("is_ood_family")
        )
        ev = self._high_risk.filter(
            pl.col("known_at").is_not_null() & (pl.col("known_at") <= fit_time) & ~pl.col("is_ood_family")
        )
        a, c = np.asarray(account_ids, np.int64), np.asarray(cutoffs, np.int64)
        return TrainingTargets(
            kind, _ro(a), _ro(c), _ro(self._compute(kind, lab, ev, a, c, horizon_s)), fit_time
        )

    def _compute(
        self,
        kind: str,
        lab: pl.DataFrame,
        ev: pl.DataFrame,
        a: np.ndarray,
        c: np.ndarray,
        horizon_s: int | None,
    ) -> np.ndarray:
        if kind == "active_phase":
            return self._active(lab, a, c)
        if kind == "high_risk_within":
            if horizon_s is None or horizon_s <= 0:
                raise ValueError("high_risk_within needs a positive horizon")
            return self._event_within(ev, a, c, horizon_s)
        raise ValueError(f"unknown target kind {kind}")
