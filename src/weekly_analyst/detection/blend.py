"""Rank blend of the four detectors into ranked candidate items, and the blend-weight file (spec 6.6)."""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from weekly_analyst.core.config import Settings
from weekly_analyst.core.items import Item
from weekly_analyst.data.metrics_file import MetricsFile
from weekly_analyst.detection.base import Detector
from weekly_analyst.detection.cusum import CusumDetector
from weekly_analyst.detection.drift import DriftDetector
from weekly_analyst.detection.validation import ValidationDetector
from weekly_analyst.detection.zscore import ZScoreDetector
from weekly_analyst.core.seeds import HOLDOUT_SEEDS

DETECTORS = ("zscore", "drift_auc", "cusum", "validation")


def default_detectors() -> list[Detector]:
    """The four detectors, one per name in DETECTORS."""
    return [ZScoreDetector(), DriftDetector(), CusumDetector(), ValidationDetector()]


def _blend_weights(weights: dict[str, float] | None) -> dict[str, float]:
    resolved = dict.fromkeys(DETECTORS, 1.0)
    if weights is not None:
        unknown = set(weights) - set(DETECTORS)
        if unknown:
            raise ValueError(f"unknown detector weight(s): {sorted(unknown)}")
        resolved.update(weights)
    if sum(resolved.values()) <= 0.0:
        raise ValueError("detector weights must have a positive sum")
    return resolved


class RankBlend:
    """Rank-averages the detectors with their weights. Metrics that validation halts are not scored."""

    def __init__(self, detectors: Sequence[Detector], weights: dict[str, float] | None = None) -> None:
        self._detectors = {d.name: d for d in detectors}
        self._weights = weights

    def score_items(
        self,
        history: pl.DataFrame,
        current: pl.DataFrame,
        metrics: MetricsFile,
        settings: Settings,
    ) -> list[Item]:
        """Candidate items for the current week, ranked by the rank-averaged detector blend.

        Metrics that validation halts (a missing or retyped column) are not scored. Items are returned
        highest blended score first; only the order carries meaning.
        """
        blend = _blend_weights(self._weights)
        validation, halted = self._detectors["validation"].scan(history, current, metrics)
        live = metrics.model_copy(
            update={"metrics": [m for m in metrics.metrics if m.name not in halted]}
        )
        zscores = self._detectors["zscore"].score(history, current, live, settings)
        keys = sorted(zscores)
        if not keys:
            return []
        drift = self._detectors["drift_auc"].score(history, current, live, settings)
        cusum = self._detectors["cusum"].score(history, current, live, settings)
        raw = {
            "zscore": [zscores[k] for k in keys],
            "drift_auc": [drift[k[0]] for k in keys],
            "cusum": [cusum[k[0]] for k in keys],
            "validation": [validation[k[0]] for k in keys],
        }
        total = sum(blend.values())
        blended = np.zeros(len(keys))
        for name, values in raw.items():
            blended += blend[name] * rankdata(values) / total
        items = [
            Item(
                metric=k[0],
                segment=k[1],
                period=k[2],
                score=float(score),
                detail={name: raw[name][i] for name in raw},
            )
            for i, (k, score) in enumerate(zip(keys, blended))
        ]
        return sorted(items, key=lambda item: (-item.score, item.metric, item.segment, item.period))


def save_weights(path: Path, weights: dict[str, float], dev_seeds: Iterable[int]) -> None:
    """Store blend weights with the dev seeds they were fitted on. Refuses holdout seeds."""
    seeds = list(dev_seeds)
    if any(seed in HOLDOUT_SEEDS for seed in seeds):
        raise ValueError("dev_seeds include holdout seeds; fit blend weights on the dev split only")
    payload = {"weights": weights, "dev_seeds": seeds}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_weights(path: Path) -> dict[str, float]:
    return json.loads(path.read_text(encoding="utf-8"))["weights"]
