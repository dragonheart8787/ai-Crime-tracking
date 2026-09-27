"""The EXP-M1-G aggregation rule: a criterion holds for the mean over seeds AND for at least n-1 of n seeds."""

from __future__ import annotations

from fcip.validation.gate import evaluate


def _r(g1: float, lr: float, tree: float, pi: float) -> dict:
    return {"G1": {"best_ap": g1}, "G2": {"ap_lr": lr, "ap_tree": tree, "floor": 3 * pi}}


def test_all_pass() -> None:
    c = evaluate([_r(0.1, 0.3, 0.2, 0.01)] * 5)
    assert c["G1_single_feature_ceiling"]["pass"] and c["G2a_probe_ceiling"]["pass"] and c["G2b_signal_floor"]["pass"]


def test_one_bad_seed_is_tolerated_two_are_not() -> None:
    one = [_r(0.1, 0.3, 0.2, 0.01)] * 4 + [_r(0.35, 0.3, 0.2, 0.01)]
    assert evaluate(one)["G1_single_feature_ceiling"]["pass"]
    two = [_r(0.1, 0.3, 0.2, 0.01)] * 3 + [_r(0.35, 0.3, 0.2, 0.01)] * 2
    assert not evaluate(two)["G1_single_feature_ceiling"]["pass"]


def test_mean_must_also_hold() -> None:
    res = [_r(0.1, 0.3, 0.2, 0.01)] * 4 + [_r(1.0, 0.3, 0.2, 0.01)]   # 4 of 5 pass, but the mean is 0.28 <= 0.30
    assert evaluate(res)["G1_single_feature_ceiling"]["pass"]
    res = [_r(0.29, 0.3, 0.2, 0.01)] * 4 + [_r(1.0, 0.3, 0.2, 0.01)]  # mean 0.432 > 0.30
    assert not evaluate(res)["G1_single_feature_ceiling"]["pass"]


def test_floor_fails_when_probes_are_at_prevalence() -> None:
    res = [_r(0.005, 0.012, 0.011, 0.01)] * 5         # both probes below 3 x pi = 0.03
    assert not evaluate(res)["G2b_signal_floor"]["pass"]
    assert evaluate(res)["G2a_probe_ceiling"]["pass"]
