"""Known-bad controls for the eval (spec 7.5; AC-016).

The eval must tell these cases apart before its scores count as evidence: a leaky
detector whose blend weights were fitted on holdout truth, shuffled truth, and data
with no events. Every input is synthetic. No model is loaded.
"""

import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata

from eval.scoring import recall_at_k
from weekly_analyst.core.config import Settings
from weekly_analyst.detection import DETECTORS, Item, score_items
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile
from weekly_analyst.sim.events import EventSpec
from weekly_analyst.sim.generate import generate


@dataclass(frozen=True)
class Pool:
    """Candidates from several sets, ranked together. keys are (set index, metric, period, segment)."""

    keys: list[tuple]
    raw: dict[str, list[float]]
    truth: frozenset[tuple]
    k: int


def control_metrics() -> MetricsFile:
    """Two region-level sums: revenue (the injected metric) and units (a distractor)."""

    def metric(name: str, column: str) -> MetricDef:
        return MetricDef(
            name=name,
            definition=f"Sum of {column} per region.",
            owner="eval controls",
            inclusions="all rows",
            exclusions="none",
            column=column,
            agg="sum",
            segment_columns=["region"],
            min_change=0.0,
            version=1,
        )

    return MetricsFile(
        metrics=[metric("revenue", "revenue"), metric("units", "units")],
        date_column="date",
        checksum="eval-controls",
    )


def simulate(seed: int, weeks: int, events: list[EventSpec]) -> list[pl.DataFrame]:
    """Weekly frames from the replay simulator, written to a temp folder and read back."""
    with tempfile.TemporaryDirectory() as tmp:
        sim = generate(seed, weeks, events, Path(tmp))
        return [pl.read_csv(path, try_parse_dates=True) for path in sim.files]


def no_event_set(seed: int, weeks: int) -> list[pl.DataFrame]:
    return simulate(seed, weeks, [])


def scored_items(frames: list[pl.DataFrame], metrics: MetricsFile, settings: Settings) -> list[Item]:
    """Candidates for the last week, scored against every earlier week."""
    return score_items(pl.concat(frames[:-1]), frames[-1], metrics, settings)


def _segment_value(frame: pl.DataFrame, metric: MetricDef, region: str) -> float:
    column = frame.filter(pl.col("region") == region).get_column(metric.column)
    return float(column.cast(pl.Float64).sum())


def abs_change(frames: list[pl.DataFrame], metric: MetricDef, region: str) -> float:
    """|last week's regional sum - mean of the earlier weeks| in metric units (control metrics are sums)."""
    if metric.agg != "sum":
        raise ValueError("abs_change is defined for sum metrics only")
    *history, current = frames
    past = [_segment_value(frame, metric, region) for frame in history]
    return abs(_segment_value(current, metric, region) - sum(past) / len(past))


def with_abs_change(items: list[Item], frames: list[pl.DataFrame], metrics: MetricsFile) -> list[Item]:
    by_name = {m.name: m for m in metrics.metrics}
    out = []
    for item in items:
        change = abs_change(frames, by_name[item.metric], item.segment[0][1])
        out.append(replace(item, detail={**item.detail, "abs_change": change, "material_change": False}))
    return out


def build_pool(sets: Sequence[tuple[list[Item], set[tuple]]], budget_per_set: int) -> Pool:
    """Pool the candidates of several sets. events are (metric, period, segment) per set."""
    keys: list[tuple] = []
    raw: dict[str, list[float]] = {name: [] for name in DETECTORS}
    truth: set[tuple] = set()
    for index, (items, events) in enumerate(sets):
        matched = 0
        for item in items:
            key = (index, item.metric, item.period, item.segment)
            keys.append(key)
            for name in DETECTORS:
                raw[name].append(float(item.detail[name]))
            if (item.metric, item.period, item.segment) in events:
                truth.add(key)
                matched += 1
        if matched != len(events):
            raise ValueError(f"set {index}: {len(events) - matched} event(s) have no candidate")
    return Pool(keys=keys, raw=raw, truth=frozenset(truth), k=budget_per_set * len(sets))


def pool_ranking(pool: Pool, weights: dict[str, float]) -> list[tuple]:
    """Pool keys, highest blended rank first. Rank-average blend as in score_items; ties by key."""
    total = sum(weights.values())
    blended = np.zeros(len(pool.keys))
    for name in DETECTORS:
        blended += weights[name] * rankdata(pool.raw[name]) / total
    order = sorted(range(len(pool.keys)), key=lambda i: (-blended[i], pool.keys[i]))
    return [pool.keys[i] for i in order]


def pooled_recall(pool: Pool, weights: dict[str, float]) -> float:
    return recall_at_k(list(pool.truth), pool_ranking(pool, weights), pool.k)


def weight_grid() -> list[dict[str, float]]:
    """Every 0/1 weighting of the detectors except all zero (15 blends)."""
    return [dict(zip(DETECTORS, bits)) for bits in product((0.0, 1.0), repeat=len(DETECTORS)) if any(bits)]


def leaky_candidates() -> list[dict[str, float]]:
    """The 0/1 grid, then 200 seeded Dirichlet(1) blend weight vectors (positive, sum 1)."""
    draws = np.random.default_rng(0).dirichlet(np.ones(len(DETECTORS)), size=200)
    return weight_grid() + [dict(zip(DETECTORS, row.tolist())) for row in draws]


def leaky_detector(holdout: Pool) -> dict[str, float]:
    """Leaky variant: blend weights chosen to maximise recall on holdout truth (spec 7.5).

    Searches leaky_candidates(). A holdout score computed with these weights is inflated by the fit itself.
    """
    return max(leaky_candidates(), key=lambda weights: pooled_recall(holdout, weights))


def shuffled_truth(truth: list, seed: int) -> list:
    """The same labels in a seeded random order, so no label stays paired with its candidate."""
    order = np.random.default_rng(seed).permutation(len(truth))
    return [truth[i] for i in order]


def random_ranking_recall_range(
    n_events: int, k: int, n_candidates: int, sims: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """95% range of recall@k for a random ranking of n_candidates with n_events true candidates."""
    if not 1 <= n_events <= n_candidates or not 1 <= k <= n_candidates:
        raise ValueError("need 1 <= n_events <= n_candidates and 1 <= k <= n_candidates")
    rng = np.random.default_rng(seed)
    alerts = np.argsort(rng.random((sims, n_candidates)), axis=1)[:, :k]
    truth = np.argsort(rng.random((sims, n_candidates)), axis=1)[:, :n_events]
    is_truth = np.zeros((sims, n_candidates), dtype=bool)
    np.put_along_axis(is_truth, truth, True, axis=1)
    hits = np.take_along_axis(is_truth, alerts, axis=1).sum(axis=1)
    low, high = np.percentile(hits / n_events, [2.5, 97.5])
    return float(low), float(high)
