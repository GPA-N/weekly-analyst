"""Drift detector: cross-validated AUC of current rows against baseline rows (spec 6.6 b)."""

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricsFile
from weekly_analyst.detection.base import Detector, Key

DRIFT_FOLDS = 5
_LABEL = "__drift_label"


def _drift_encode(combined: pl.DataFrame, columns: list[str]) -> np.ndarray:
    """Numeric matrix for the classifier. Text and date columns become dense ranks."""
    exprs = []
    for col in columns:
        dtype = combined.schema[col]
        if dtype.is_numeric() or dtype == pl.Boolean:
            exprs.append(pl.col(col).cast(pl.Float64))
        else:
            exprs.append(pl.col(col).cast(pl.Utf8).rank("dense").cast(pl.Float64))
    return combined.select(exprs).to_numpy()


def drift_auc(
    baseline: pl.DataFrame, current: pl.DataFrame, columns: list[str], seed: int = 0
) -> float:
    """Cross-validated AUC of a classifier telling current rows from baseline rows (spec 6.6 b).

    Near 0.5 means no drift. Rows are sorted into a canonical order first, so the fold split
    and the score do not depend on row order.
    """
    combined = pl.concat(
        [
            baseline.select(columns).with_columns(pl.lit(0).alias(_LABEL)),
            current.select(columns).with_columns(pl.lit(1).alias(_LABEL)),
        ]
    ).sort([*columns, _LABEL], nulls_last=True)
    # Identical feature rows share a fold, so a duplicate's label cannot leak into training.
    # Groups are dense ranks of each row's values, so a monotone rescale of a column keeps them.
    ranks = combined.select([pl.col(col).rank("dense").fill_null(0) for col in columns])
    _, groups = np.unique(ranks.to_numpy(), axis=0, return_inverse=True)
    n_splits = min(DRIFT_FOLDS, len(np.unique(groups)), baseline.height, current.height)
    if n_splits < 2:
        return 0.5
    x = _drift_encode(combined, columns)
    y = combined.get_column(_LABEL).to_numpy()
    folds = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    model = HistGradientBoostingClassifier(random_state=seed)
    scores = cross_val_score(model, x, y, cv=folds, groups=groups, scoring="roc_auc")
    return float(np.mean(scores))


def _drift_scores(
    history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile
) -> dict[str, float]:
    """Metric-level drift strength: distance of the drift AUC from 0.5."""
    return {
        m.name: abs(drift_auc(history, current, [m.column, *m.segment_columns]) - 0.5)
        for m in metrics.metrics
    }


class DriftDetector(Detector):
    """Owns the drift strength per metric: |AUC - 0.5| of current rows against baseline rows."""

    name = "drift_auc"

    def score(
        self, history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile, settings: Settings
    ) -> dict[Key, float]:
        return _drift_scores(history, current, metrics)
