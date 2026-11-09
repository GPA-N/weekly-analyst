"""Detection: detectors, rank blend, alert budget and items. Old function names delegate to the classes."""

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.core.items import Item
from weekly_analyst.core.store import StateStore
from weekly_analyst.data.metrics_file import MetricsFile
from weekly_analyst.detection.base import Detector, Key
from weekly_analyst.detection.blend import (
    DETECTORS,
    RankBlend,
    default_detectors,
    load_weights,
    save_weights,
)
from weekly_analyst.detection.budget import AlertBudget
from weekly_analyst.detection.cusum import CusumDetector, cusum_score
from weekly_analyst.detection.drift import DriftDetector, drift_auc
from weekly_analyst.detection.items import item_key
from weekly_analyst.detection.validation import ValidationDetector
from weekly_analyst.detection.zscore import ZScoreDetector, zscore_scores

__all__ = [
    "DETECTORS",
    "AlertBudget",
    "CusumDetector",
    "Detector",
    "DriftDetector",
    "Item",
    "Key",
    "RankBlend",
    "ValidationDetector",
    "ZScoreDetector",
    "cusum_score",
    "default_detectors",
    "drift_auc",
    "insufficient_history",
    "item_key",
    "load_weights",
    "remaining_budget",
    "save_weights",
    "score_items",
    "select_alerts",
    "zscore_scores",
]


def score_items(
    history: pl.DataFrame,
    current: pl.DataFrame,
    metrics: MetricsFile,
    settings: Settings,
    weights: dict[str, float] | None = None,
) -> list[Item]:
    """Candidate items for the current week, ranked by the rank-averaged detector blend (spec 6.6)."""
    return RankBlend(default_detectors(), weights).score_items(history, current, metrics, settings)


def remaining_budget(store: StateStore, iso_week: str, settings: Settings) -> int:
    """Alerts a run may still send in this ISO week: K minus alerts already sent (spec 6.6)."""
    return AlertBudget(store, settings).remaining(iso_week)


def insufficient_history(n_periods: int, settings: Settings) -> bool:
    """True when there is too little history to alert (cold start, spec 6.6)."""
    return AlertBudget(None, settings).insufficient_history(n_periods)


def select_alerts(
    items: list[Item],
    remaining_budget: int,
    min_change: dict[str, float],
    already_reported: set[str],
) -> list[Item]:
    """Pick alerts from rank-ordered items (spec 6.6), keeping input order (see AlertBudget.select)."""
    return AlertBudget(None, None).select(items, remaining_budget, min_change, already_reported)
