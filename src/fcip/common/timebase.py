"""Simulation time: int64 seconds since the simulation epoch t0 (decision 0004).

t0 maps to a calendar anchor (a Monday, 00:00) used only for weekday and day-of-month seasonality.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np

SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400


@dataclass(frozen=True)
class TimeBase:
    days: int
    anchor: dt.date

    def __post_init__(self) -> None:
        if self.anchor.weekday() != 0:
            raise ValueError(f"calendar anchor must be a Monday, got {self.anchor} ({self.anchor:%A})")
        if self.days < 1:
            raise ValueError("days must be >= 1")

    @property
    def sim_end(self) -> int:
        return self.days * SECONDS_PER_DAY

    def day_dates(self) -> list[dt.date]:
        return [self.anchor + dt.timedelta(days=i) for i in range(self.days)]

    def monthly_day_indices(self, day_of_month: int) -> np.ndarray:
        """Indices of simulation days whose calendar day equals ``day_of_month`` (clipped to month length)."""
        if not 1 <= day_of_month <= 31:
            raise ValueError(f"day_of_month out of range: {day_of_month}")
        out = []
        for i, d in enumerate(self.day_dates()):
            next_month = (d.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
            last = (next_month - dt.timedelta(days=1)).day
            if d.day == min(day_of_month, last):
                out.append(i)
        return np.asarray(out, dtype=np.int64)

    @staticmethod
    def weekday(ts: np.ndarray | int) -> np.ndarray:
        """0 = Monday."""
        return (np.asarray(ts) // SECONDS_PER_DAY) % 7

    @staticmethod
    def hour_of_day(ts: np.ndarray | int) -> np.ndarray:
        return (np.asarray(ts) % SECONDS_PER_DAY) // SECONDS_PER_HOUR
