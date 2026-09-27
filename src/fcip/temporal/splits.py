"""Chronological splits, OOD holdout pool, reference negatives and label maturity (decision 0009).

Prediction points are daily snapshots ``(account, day)`` with inclusive cutoff ``t = day * 86400 - 1`` for
internal accounts open at ``t``. With ``H_max`` the longest label horizon:

* TRAIN: ``t + H_max <= train_end``; ``T_fit = train_end``; targets from labels known at ``T_fit``;
* VAL:   ``train_end <= t`` and ``t + H_max <= val_end``;
* TEST:  ``val_end <= t`` and ``t + H_max <= sim_end``;
* OOD pool: points of OOD-family members and of the reference pool ``E_ref`` over the whole timeline,
  tagged by time block; both groups are excluded from TRAIN, VAL and TEST.

This module reads the oracle for *protocol* decisions only (which entity is in which set), never for features.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from fcip.common import ids
from fcip.common.errors import InvariantViolation
from fcip.common.rng import keyed_uniform, stable_key
from fcip.config.generator import GeneratorConfig
from fcip.labels.oracle import OracleLabels
from fcip.simulation.recruitment import NONE_CELL, build_partition
from fcip.temporal.grid import snapshot_cutoff
from fcip.temporal.store import TemporalStore

DAY = 86400
TRAINING_MODES = ("include_immature", "mature_only")


@dataclass(frozen=True)
class SplitIndex:
    train_all: pl.DataFrame  # account_id, day, t, is_mature
    val: pl.DataFrame
    test: pl.DataFrame
    ood: pl.DataFrame  # account_id, day, t, block, group ("ood_member" | "ref_negative")
    t_fit: int
    horizon_s: int
    maturity_horizon_s: int
    e_ood: frozenset[int]
    e_ref: frozenset[int]
    ref_pool_method: str
    index_hash: str

    def train(self, mode: str) -> pl.DataFrame:
        if mode == "include_immature":
            return self.train_all
        if mode == "mature_only":
            return self.train_all.filter(pl.col("is_mature"))
        raise ValueError(f"unknown training mode {mode}; expected one of {TRAINING_MODES}")


def normal_account_ids(store: TemporalStore) -> np.ndarray:
    a = store.account_ids
    return a[np.asarray([ids.kind_of(int(x)) == ids.ACCOUNT_INTERNAL for x in a])]


def reference_pool(
    cfg: GeneratorConfig, normal_accounts: np.ndarray, archetype_of: dict[int, str] | None, method: str
) -> frozenset[int]:
    """``E_ref``: a fixed ``ref_negative_fraction`` of normal accounts, drawn only from the recruitment
    partition's NONE cell (accounts no scenario can ever recruit), selected by a keyed hash.

    ``none_cell_hash``: a plain hash sample of the NONE cell. ``stratified_archetype``: the same hash ranking,
    but taken per archetype so the pool matches the population's archetype shares exactly (A2)."""
    part = build_partition(cfg, normal_accounts.tolist())
    none = np.asarray(sorted(a for a in normal_accounts.tolist() if part.cell_of_account[a] == NONE_CELL))
    frac = cfg.splits.ref_negative_fraction
    u = keyed_uniform(stable_key(f"ref-pool/{cfg.simulation.seed}"), none)
    if method == "none_cell_hash":
        take = u < frac * len(normal_accounts) / len(none)
        return frozenset(none[take].tolist())
    if method == "stratified_archetype":
        if archetype_of is None:
            raise ValueError("stratified_archetype needs the archetype of every normal account")
        arch_all = np.asarray([archetype_of[a] for a in normal_accounts.tolist()])
        arch_none = np.asarray([archetype_of[a] for a in none.tolist()])
        chosen: list[int] = []
        for a in np.unique(arch_all):
            n_target = int(round(frac * (arch_all == a).sum()))
            cand = np.flatnonzero(arch_none == a)
            if n_target > len(cand):
                raise InvariantViolation(
                    f"not enough NONE-cell accounts of archetype {a} for the reference pool"
                )
            chosen += none[cand[np.argsort(u[cand], kind="stable")[:n_target]]].tolist()
        return frozenset(chosen)
    raise ValueError(f"unknown reference pool method {method}")


def _points(store: TemporalStore, days: range) -> pl.DataFrame:
    accs = normal_and_created = store.account_ids[store._acc_internal]  # internal accounts only
    opened = store._acc_opened[store.account_index(normal_and_created)]
    frames = []
    for d in days:
        t = snapshot_cutoff(d)
        m = opened <= t
        frames.append(
            pl.DataFrame(
                {
                    "account_id": accs[m],
                    "day": np.full(m.sum(), d, np.int64),
                    "t": np.full(m.sum(), t, np.int64),
                }
            )
        )
    return (
        pl.concat(frames)
        if frames
        else pl.DataFrame(schema={"account_id": pl.Int64, "day": pl.Int64, "t": pl.Int64})
    )


def _day_range(lo_t: int, hi_t: int) -> range:
    """Days whose cutoff t = d*DAY - 1 satisfies lo_t <= t <= hi_t."""
    first = -(-(lo_t + 1) // DAY)
    last = (hi_t + 1) // DAY
    return range(max(first, 1), last + 1)


def build_example_index(
    store: TemporalStore,
    oracle: OracleLabels,
    cfg: GeneratorConfig,
    horizon_s: int,
    archetype_of: dict[int, str] | None = None,
) -> SplitIndex:
    sp, tl = cfg.splits, cfg.training_labels
    h_max = int(round(sp.h_max_days * DAY))
    if horizon_s > h_max:
        raise ValueError("target horizon exceeds H_max; the purge gap would be too short")
    train_end, val_end, sim_end = int(sp.train_end_day * DAY), int(sp.val_end_day * DAY), store.sim_end
    maturity = int(round(tl.maturity_horizon_days * DAY))

    e_ood = frozenset(oracle.ood_entities())
    e_ref = reference_pool(cfg, normal_account_ids(store), archetype_of, sp.ref_pool_method)
    if e_ref & oracle.member_entities():
        raise InvariantViolation("reference pool contains a scenario member")
    excluded = list(e_ood | e_ref)

    def id_set(lo_t: int, hi_t: int) -> pl.DataFrame:
        return _points(store, _day_range(lo_t, hi_t)).filter(~pl.col("account_id").is_in(excluded))

    train = id_set(0, train_end - h_max).with_columns(
        ((train_end - (pl.col("t") + horizon_s)) >= maturity).alias("is_mature")
    )
    val = id_set(train_end, val_end - h_max)
    test = id_set(val_end, sim_end - h_max)
    blocks = [
        ("train", 0, train_end - h_max),
        ("val", train_end, val_end - h_max),
        ("test", val_end, sim_end - h_max),
    ]
    pool = list(e_ood | e_ref)
    ood = pl.concat(
        [
            _points(store, _day_range(lo, hi))
            .filter(pl.col("account_id").is_in(pool))
            .with_columns(pl.lit(b).alias("block"))
            for b, lo, hi in blocks
        ]
    ).with_columns(
        pl.when(pl.col("account_id").is_in(list(e_ood)))
        .then(pl.lit("ood_member"))
        .otherwise(pl.lit("ref_negative"))
        .alias("group")
    )

    # ---- invariants (raise on violation)
    for name, df in (("train", train), ("val", val), ("test", test)):
        if df["account_id"].is_in(excluded).any():
            raise InvariantViolation(f"{name} contains OOD or reference-pool accounts")
    if train.height and (int(train["t"].to_numpy().max()) + h_max > train_end):
        raise InvariantViolation("a TRAIN target window reaches past train_end")
    if val.height and (
        int(val["t"].to_numpy().min()) < train_end or int(val["t"].to_numpy().max()) + h_max > val_end
    ):
        raise InvariantViolation("VAL outside its bounds")
    if test.height and (
        int(test["t"].to_numpy().min()) < val_end or int(test["t"].to_numpy().max()) + h_max > sim_end
    ):
        raise InvariantViolation("TEST outside its bounds")
    h = hashlib.sha256()
    for df in (train, val, test, ood):
        h.update(df.sort(df.columns).write_csv().encode())
    return SplitIndex(
        train, val, test, ood, train_end, horizon_s, maturity, e_ood, e_ref, sp.ref_pool_method, h.hexdigest()
    )


def ref_pool_composition(
    normal_accounts: np.ndarray, pool: frozenset[int], archetype_of: dict[int, str]
) -> pl.DataFrame:
    """Archetype shares of the reference pool versus all normal accounts (A2)."""
    all_df = pl.DataFrame(
        {"account_id": normal_accounts, "archetype": [archetype_of[a] for a in normal_accounts.tolist()]}
    )
    pop = (
        all_df.group_by("archetype")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("population_share"))
    )
    ref = (
        all_df.filter(pl.col("account_id").is_in(list(pool)))
        .group_by("archetype")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum()).alias("pool_share"))
        .rename({"len": "pool_n"})
    )
    return (
        pop.rename({"len": "population_n"})
        .join(ref, on="archetype", how="left")
        .fill_null(0)
        .with_columns(((pl.col("pool_share") - pl.col("population_share")) * 100).alias("diff_pp"))
        .sort("archetype")
    )
