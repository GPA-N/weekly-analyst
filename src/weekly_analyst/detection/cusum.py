"""Two-sided CUSUM per metric over the all-segment weekly series (spec 6.6 c)."""

import math

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.data.metrics_file import MetricsFile
from weekly_analyst.detection.base import Detector, Key
from weekly_analyst.detection.zscore import _segment_period_values


def cusum_score(series: list[float], k: float, h: float) -> float:
    """Two-sided CUSUM on the standardized series, as max statistic / h (>= 1.0 means alarm)."""
    values = [x for x in series if math.isfinite(x)]
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))
    if sd <= 0.0:
        return 0.0
    high = low = best = 0.0
    for x in values:
        z = (x - mean) / sd
        high = max(0.0, high + z - k)
        low = max(0.0, low - z - k)
        best = max(best, high, low)
    return best / h


def _cusum_scores(
    history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile, settings: Settings
) -> dict[str, float]:
    """Metric-level CUSUM over each metric's all-segment value by ISO week, history then current."""
    scores: dict[str, float] = {}
    for m in metrics.metrics:
        overall = m.model_copy(update={"segment_columns": []})
        gate = settings.min_rows_per_segment
        past = _segment_period_values(history, overall, metrics.date_column, gate)
        now = _segment_period_values(current, overall, metrics.date_column, gate)
        by_period = {period: value for (_, period), value in {**past, **now}.items()}
        series = [math.nan if by_period[p] is None else by_period[p] for p in sorted(by_period)]
        scores[m.name] = cusum_score(series, settings.cusum_k, settings.cusum_h)
    return scores


class CusumDetector(Detector):
    """Owns the CUSUM alarm per metric (score is the max statistic over h)."""

    name = "cusum"

    def score(
        self, history: pl.DataFrame, current: pl.DataFrame, metrics: MetricsFile, settings: Settings
    ) -> dict[Key, float]:
        return _cusum_scores(history, current, metrics, settings)
