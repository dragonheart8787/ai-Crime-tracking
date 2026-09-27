"""Shared fixtures. The TINY profile (400 persons, 45 days) keeps generator tests fast."""

from __future__ import annotations

import pytest

from fcip.config.generator import GeneratorConfig
from fcip.config.loader import load_config
from fcip.simulation.generator import Dataset, generate


@pytest.fixture(scope="session")
def tiny_cfg() -> GeneratorConfig:
    return load_config("tiny")


@pytest.fixture(scope="session")
def tiny_ds(tiny_cfg: GeneratorConfig) -> Dataset:
    return generate(tiny_cfg)


@pytest.fixture(scope="session")
def check_kwargs(tiny_cfg: GeneratorConfig, tiny_ds: Dataset) -> dict:
    return {
        "sim_end": tiny_ds.metadata["sim_end"],
        "session_window": tiny_cfg.infrastructure.session_window_seconds,
        "high_risk_phases": [p.value for p in tiny_cfg.high_risk_phases],
    }
