"""Typed distributions (decision 0005). Never free-form strings; each validates its own parameters."""

from __future__ import annotations

import math
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Dist(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class LogNormal(_Dist):
    kind: Literal["LogNormal"] = "LogNormal"
    median: float = Field(gt=0)
    sigma: float = Field(ge=0)

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        return rng.lognormal(math.log(self.median), self.sigma, size)


class Gamma(_Dist):
    kind: Literal["Gamma"] = "Gamma"
    shape: float = Field(gt=0)
    scale: float = Field(gt=0)

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        return rng.gamma(self.shape, self.scale, size)


class Uniform(_Dist):
    kind: Literal["Uniform"] = "Uniform"
    low: float
    high: float

    @model_validator(mode="after")
    def _check(self) -> Uniform:
        if not self.high > self.low:
            raise ValueError(f"Uniform requires high > low, got [{self.low}, {self.high}]")
        return self

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        return rng.uniform(self.low, self.high, size)


class ShiftedPoisson(_Dist):
    kind: Literal["ShiftedPoisson"] = "ShiftedPoisson"
    shift: int = Field(ge=0)
    lam: float = Field(ge=0)

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | int:
        return self.shift + rng.poisson(self.lam, size)


class Beta(_Dist):
    kind: Literal["Beta"] = "Beta"
    a: float = Field(gt=0)
    b: float = Field(gt=0)

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        return rng.beta(self.a, self.b, size)


class Constant(_Dist):
    kind: Literal["Constant"] = "Constant"
    value: float

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        return self.value if size is None else np.full(size, self.value, dtype=np.float64)


class Empirical(_Dist):
    kind: Literal["Empirical"] = "Empirical"
    values: list[float] = Field(min_length=1)
    probs: list[float] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Empirical:
        if len(self.values) != len(self.probs):
            raise ValueError("Empirical values and probs must have the same length")
        if any(p < 0 for p in self.probs) or abs(sum(self.probs) - 1.0) > 1e-9:
            raise ValueError(f"Empirical probs must be >= 0 and sum to 1, got {sum(self.probs)}")
        return self

    def sample(self, rng: np.random.Generator, size: int | None = None) -> np.ndarray | float:
        return rng.choice(np.asarray(self.values, dtype=np.float64), size=size, p=np.asarray(self.probs))


Dist = Annotated[
    LogNormal | Gamma | Uniform | ShiftedPoisson | Beta | Constant | Empirical, Field(discriminator="kind")
]


def sample_int(dist: _Dist, rng: np.random.Generator, minimum: int = 0) -> int:
    """One integer draw, rounded and clipped below at ``minimum``."""
    return max(minimum, int(round(float(dist.sample(rng)))))  # type: ignore[attr-defined]
