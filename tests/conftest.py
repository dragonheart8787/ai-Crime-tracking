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


@pytest.fixture(scope="session")
def tiny_archetypes(tiny_ds: Dataset) -> dict[int, str]:
    """Archetype of every internal account (oracle; split protocol use in tests only)."""
    import polars as pl

    acc = pl.from_arrow(tiny_ds.tables["accounts"]).filter(pl.col("account_kind") == "internal")
    pt = pl.from_arrow(tiny_ds.tables["person_truth"])
    j = acc.join(pt, left_on="owner_person_id", right_on="person_id")
    return dict(zip(j["account_id"].to_list(), j["archetype"].to_list(), strict=True))
