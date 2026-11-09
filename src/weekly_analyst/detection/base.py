"""Detector contract: one named score per key, from history, the current week and the metrics."""

from abc import ABC, abstractmethod
from collections.abc import Hashable

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricsFile

Key = Hashable


class Detector(ABC):
    """Base for the blend detectors. A subclass owns one name and one scoring rule."""

    name: str

    @abstractmethod
    def score(
        self, history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile, settings: Settings
    ) -> dict[Key, float]:
        """Score per key. Higher means more unusual."""
