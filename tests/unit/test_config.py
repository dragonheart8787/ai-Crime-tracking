"""Config schema fails loudly (decision 0005)."""

from __future__ import annotations

import pytest

from fcip.common.errors import ConfigError
from fcip.config.loader import load_config


def test_profiles_load() -> None:
    for profile in ("tiny", "dev", "research"):
        cfg = load_config(profile)
        assert cfg.profile == profile
        assert len(cfg.config_hash()) == 64
    dev = load_config("dev")
    assert 2000 <= dev.population.n_persons <= 3000 and dev.simulation.days == 90
    assert dev.suspicious_prevalence == 0.01
    assert dev.currency.code == "SYN" and dev.currency.minor_units_per_major == 100


@pytest.mark.parametrize(
    "overrides",
    [
        {"unknown_key": 1},
        {"simulation": {"calendar_anchor": "2025-01-07"}},  # not a Monday
        {"label_knowledge": {"latency_days": {"kind": "LogNormal", "median": 0.5, "sigma": 0.75}}},
        {"label_knowledge": {"latency_days": {"kind": "LogNormal", "median": 14, "sigma": 0.0}}},
        {"label_knowledge": {"latency_days": {"kind": "Constant", "value": 14}}},  # constant latency rejected
        {"label_knowledge": {"regime": "instant"}},
        {"training_labels": {"mode": "mature_only"}},  # 90-day sim: mature window too short
        {"population": {"archetype_mix": {"salary_worker": 0.5}}},  # does not sum to 1
        {"splits": {"train_end_day": 80, "val_end_day": 75}},
        {"infrastructure": {"merchant_category_mix": {"grocery": 1.0}, "session_window_seconds": 100}},
        {"high_risk_phases": ["MOVEMENT"]},
    ],
)
def test_invalid_configs_raise(overrides: dict) -> None:
    with pytest.raises(ConfigError):
        load_config("dev", overrides=overrides)


def test_oracle_regime_is_explicit() -> None:
    cfg = load_config(
        "dev",
        overrides={
            "label_knowledge": {
                "regime": "oracle",
                "latency_days": {"kind": "LogNormal", "median": 0.01, "sigma": 0.0},
            }
        },
    )
    assert cfg.label_knowledge.regime == "oracle"


def test_mature_only_accepted_when_window_is_long_enough() -> None:
    cfg = load_config(
        "dev",
        overrides={
            "simulation": {"days": 180},
            "splits": {"train_end_day": 120, "val_end_day": 150},
            "training_labels": {"mode": "mature_only"},
        },
    )
    assert cfg.training_labels.mode == "mature_only"


def test_every_family_must_be_listed() -> None:
    from fcip.config.generator import GeneratorConfig
    from fcip.config.loader import resolve_mapping

    m = resolve_mapping("dev")
    m["scenarios"].pop("cycle")
    with pytest.raises(ValueError, match="every family"):
        GeneratorConfig.model_validate(m)
