"""Ground-truth-only evaluation rule (decision 0009): metrics are computed against full synthetic ground truth,
never against known_at-limited labels, and refuse any other target type."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fcip.evaluation.metrics import average_precision, positives, recall_at_k
from fcip.labels.oracle import OracleLabels, OracleTargets
from fcip.temporal.splits import build_example_index
from fcip.temporal.store import TemporalStore


def test_perfect_scorer_gets_ap_one_against_full_ground_truth(tiny_ds, tiny_cfg) -> None:
    T = tiny_ds.tables
    store = TemporalStore(T, tiny_ds.metadata["sim_end"], T["labels"])
    lab = pl.from_arrow(T["labels"])
    # make one detected network never known: known_at-filtered labels would then lose its positives
    detected = pl.from_arrow(T["ground_truth_networks"]).filter(pl.col("detected") & ~pl.col("is_ood_family"))
    never = detected["network_id"].sort().to_list()[:1]
    lab = lab.with_columns(
        pl.when(pl.col("network_id").is_in(never)).then(None).otherwise(pl.col("known_at")).alias("known_at")
    )
    oracle = OracleLabels(
        lab.to_arrow(), T["event_labels"], T["transactions"], T["ground_truth_networks"], T["network_members"]
    )
    idx = build_example_index(store, oracle, tiny_cfg, 0)
    pts = pl.concat(
        [
            idx.train_all.select("account_id", "t"),
            idx.val.select("account_id", "t"),
            idx.test.select("account_id", "t"),
        ]
    )
    truth = oracle.targets("active_phase", pts["account_id"].to_numpy(), pts["t"].to_numpy())
    known = oracle.training_targets(
        "active_phase", pts["account_id"].to_numpy(), pts["t"].to_numpy(), tiny_ds.metadata["sim_end"]
    )
    assert truth.n_positive > int(known.y.sum()) > 0
    perfect = truth.y.astype(float)
    assert average_precision(truth, perfect) == 1.0
    assert recall_at_k(truth, perfect, truth.n_positive) == 1.0
    assert positives(truth) == truth.n_positive  # the oracle count, not the known count
    # had evaluation substituted known labels, the same perfect scorer would not reach AP = 1
    from sklearn.metrics import average_precision_score

    assert average_precision_score(known.y, perfect) < 1.0


def test_metrics_refuse_non_oracle_targets(tiny_ds) -> None:
    y = np.asarray([True, False])
    with pytest.raises(TypeError):
        average_precision(y, np.asarray([0.9, 0.1]))  # type: ignore[arg-type]
    T = tiny_ds.tables
    oracle = OracleLabels(
        T["labels"], T["event_labels"], T["transactions"], T["ground_truth_networks"], T["network_members"]
    )
    tt = oracle.training_targets("active_phase", np.asarray([1]), np.asarray([0]), 0)
    with pytest.raises(TypeError):
        average_precision(tt, np.asarray([0.5]))  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        OracleTargets("active_phase", np.asarray([1]), np.asarray([0]), np.asarray([True]))
