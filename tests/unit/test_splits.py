"""Split utility (decision 0009): purge, OOD pool, reference negatives, maturity modes."""

from __future__ import annotations

import polars as pl
import pytest

from fcip.common.errors import ConfigError
from fcip.config.loader import load_config
from fcip.labels.oracle import OracleLabels
from fcip.temporal.splits import build_example_index, normal_account_ids, ref_pool_composition, reference_pool
from fcip.temporal.store import TemporalStore

DAY = 86400


@pytest.fixture(scope="module")
def parts(tiny_ds, tiny_cfg, tiny_archetypes):
    T = tiny_ds.tables
    store = TemporalStore(T, tiny_ds.metadata["sim_end"], T["labels"])
    oracle = OracleLabels(
        T["labels"], T["event_labels"], T["transactions"], T["ground_truth_networks"], T["network_members"]
    )
    return store, oracle, tiny_cfg, tiny_archetypes


def _index(parts, horizon_s=0, cfg=None):
    store, oracle, c, arche = parts
    return build_example_index(store, oracle, cfg or c, horizon_s, archetype_of=arche)


def test_purge_gap_and_bounds(parts) -> None:
    idx = _index(parts, horizon_s=7 * DAY)
    cfg = parts[2]
    h = int(cfg.splits.h_max_days * DAY)
    tr_end, va_end = int(cfg.splits.train_end_day * DAY), int(cfg.splits.val_end_day * DAY)
    assert idx.train_all["t"].max() + h <= tr_end
    assert idx.val["t"].min() >= tr_end and idx.val["t"].max() + h <= va_end
    assert idx.test["t"].min() >= va_end and idx.test["t"].max() + h <= parts[0].sim_end
    # consecutive sets are separated by at least H_max of prediction time
    assert idx.val["t"].min() - idx.train_all["t"].max() >= h
    assert idx.test["t"].min() - idx.val["t"].max() >= h
    assert idx.t_fit == tr_end


def test_ood_members_and_reference_pool_never_in_train_val_test(parts, tiny_ds) -> None:
    idx = _index(parts)
    assert idx.e_ood, "tiny must contain OOD-family members"
    excluded = list(idx.e_ood | idx.e_ref)
    for df in (idx.train_all, idx.val, idx.test):
        assert not df["account_id"].is_in(excluded).any()
    assert set(idx.ood.filter(pl.col("group") == "ood_member")["account_id"].unique().to_list()) <= idx.e_ood
    assert set(idx.ood["block"].unique().to_list()) == {"train", "val", "test"}
    # OOD members appear across the full timeline, not only after the validation boundary
    ood_days = idx.ood.filter(pl.col("group") == "ood_member")["day"]
    assert ood_days.min() < parts[2].splits.train_end_day
    members = set(pl.from_arrow(tiny_ds.tables["network_members"])["entity_id"].to_list())
    assert not (idx.e_ref & members)
    assert not (idx.e_ref & idx.e_ood)


def test_ood_member_transactions_remain_visible_to_neighbors(parts, tiny_ds) -> None:
    store, _, _, _ = parts
    idx = _index(parts)
    tx = store.as_of(store.sim_end - 1).transactions()
    touching = tx.filter(
        pl.col("src_account_id").is_in(list(idx.e_ood)) | pl.col("dst_account_id").is_in(list(idx.e_ood))
    )
    assert touching.height > 0  # the world still contains them; only their labels are withheld


def test_maturity_modes(parts) -> None:
    store, oracle, cfg, arche = parts
    idx = _index(parts, horizon_s=DAY)
    tr = idx.train_all
    expected = (idx.t_fit - (tr["t"] + DAY)) >= idx.maturity_horizon_s
    assert (tr["is_mature"] == expected).all()
    inc, mat = idx.train("include_immature"), idx.train("mature_only")
    assert inc.height == tr.height and mat.height == int(expected.sum())
    assert mat.join(tr.filter(~pl.col("is_mature")), on=["account_id", "day"]).height == 0
    with pytest.raises(ValueError):
        idx.train("oracle")
    # is_mature does not depend on known_at: rebuild with all known_at removed
    T = dict(parts[0].__dict__) if False else None
    del T
    lab = oracle._labels.with_columns(pl.lit(None).cast(pl.Int64).alias("known_at"))  # noqa: SLF001 (test only)
    oracle2 = type(oracle).__new__(type(oracle))
    oracle2.__dict__.update(oracle.__dict__)
    oracle2._labels = lab  # noqa: SLF001
    idx2 = build_example_index(store, oracle2, cfg, DAY, archetype_of=arche)
    assert idx2.train_all["is_mature"].equals(tr["is_mature"])
    assert idx2.val.equals(idx.val) and idx2.test.equals(idx.test)


def test_mature_only_rejected_when_window_too_short() -> None:
    with pytest.raises(ConfigError, match="mature_only"):
        load_config("dev", overrides={"training_labels": {"mode": "mature_only"}})


def test_training_targets_use_known_labels_only(parts) -> None:
    _, oracle, _, _ = parts
    idx = _index(parts)
    tr = idx.train_all
    truth = oracle.targets("active_phase", tr["account_id"].to_numpy(), tr["t"].to_numpy())
    known = oracle.training_targets(
        "active_phase", tr["account_id"].to_numpy(), tr["t"].to_numpy(), idx.t_fit
    )
    assert (known.y <= truth.y).all()  # knowledge can only remove positives
    # with an earlier fit time, networks detected on days 21 to 23 are not yet known: positives disappear
    early = oracle.training_targets("active_phase", tr["account_id"].to_numpy(), tr["t"].to_numpy(), 20 * DAY)
    assert (early.y <= known.y).all() and early.y.sum() < truth.y.sum()


def test_reference_pool_methods(parts) -> None:
    store, oracle, cfg, _ = parts
    normal = normal_account_ids(store)
    import pyarrow.parquet  # noqa: F401  (ensure pyarrow loaded)

    hashed = reference_pool(cfg, normal, None, "none_cell_hash")
    assert abs(len(hashed) - cfg.splits.ref_negative_fraction * len(normal)) < 0.05 * len(normal)
    arche = {int(a): "x" if a % 2 else "y" for a in normal}
    strat = reference_pool(cfg, normal, arche, "stratified_archetype")
    comp = ref_pool_composition(normal, strat, arche)
    assert comp["diff_pp"].abs().max() < 1.0
    assert reference_pool(cfg, normal, None, "none_cell_hash") == hashed  # deterministic


def test_index_is_deterministic(parts) -> None:
    assert _index(parts).index_hash == _index(parts).index_hash
