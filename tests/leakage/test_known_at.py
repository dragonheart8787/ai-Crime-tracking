"""Decision 0004 required test: a label becomes usable only once it is known."""

from __future__ import annotations

import pyarrow as pa
import pytest
from tests.handmade import DAY, SIM_DAYS, A, B, C, tables

from fcip.features.label_features import flagged_counterparties
from fcip.labels.oracle import OracleLabels
from fcip.temporal.store import TemporalStore


@pytest.fixture(scope="module")
def world() -> tuple[TemporalStore, OracleLabels]:
    t = tables()
    store = TemporalStore(t, SIM_DAYS * DAY, t["labels"])
    oracle = OracleLabels(
        t["labels"], t["event_labels"], t["transactions"], t["ground_truth_networks"], t["network_members"]
    )
    return store, oracle


def test_not_yet_known_positive_is_not_a_training_positive(world) -> None:
    _, oracle = world
    accs, cut = [A, A, C], [12 * DAY, 18 * DAY, 6 * DAY]  # A active days 10-21, C active days 5-8
    truth = oracle.targets("active_phase", accs, cut)
    assert truth.y.tolist() == [True, True, True]
    at60 = oracle.training_targets("active_phase", accs, cut, fit_time=60 * DAY)
    assert at60.y.tolist() == [False, False, True]  # A known only at day 70; control C known at day 30
    at75 = oracle.training_targets("active_phase", accs, cut, fit_time=75 * DAY)
    assert at75.y.tolist() == [True, True, True]
    ev60 = oracle.training_targets(
        "high_risk_within", [A, C], [15 * DAY, 6 * DAY], 60 * DAY, horizon_s=7 * DAY
    )
    assert ev60.y.tolist() == [False, True]
    ev75 = oracle.training_targets(
        "high_risk_within", [A, C], [15 * DAY, 6 * DAY], 75 * DAY, horizon_s=7 * DAY
    )
    assert ev75.y.tolist() == [True, True]


def test_known_labels_view_respects_known_at(world) -> None:
    store, _ = world
    for day in (20, 40, 69):
        assert A not in store.as_of(day * DAY).known_labels()["entity_id"].to_list()
    assert A in store.as_of(70 * DAY).known_labels()["entity_id"].to_list()
    assert C in store.as_of(30 * DAY).known_labels()["entity_id"].to_list()
    assert C not in store.as_of(30 * DAY - 1).known_labels()["entity_id"].to_list()


def test_label_derived_feature_changes_only_when_known(world) -> None:
    store, _ = world
    window = 90 * DAY
    assert flagged_counterparties(store.as_of(65 * DAY), B, window) == 0  # A's label known only at day 70
    assert flagged_counterparties(store.as_of(72 * DAY), B, window) == 1


def test_never_known_and_ood_rows_are_not_in_the_store() -> None:
    t = tables()
    lab = t["labels"].to_pydict()
    lab["known_at"] = [None, 30 * DAY]  # A never known
    lab["is_ood_family"] = [False, True]  # C's family is OOD
    store = TemporalStore(t, SIM_DAYS * DAY, pa.table(lab, schema=t["labels"].schema))
    assert store.as_of(SIM_DAYS * DAY - 1).known_labels().height == 0
