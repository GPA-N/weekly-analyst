"""Eval of the eval: known-bad controls must be told apart (spec 7.5; AC-016; ticket 026)."""

import functools
import random
from datetime import timedelta

import polars as pl
import pytest

from eval.controls import (
    abs_change,
    build_pool,
    control_metrics,
    leaky_candidates,
    leaky_detector,
    no_event_set,
    pool_ranking,
    pooled_recall,
    random_ranking_recall_range,
    scored_items,
    shuffled_truth,
    simulate,
    weight_grid,
    with_abs_change,
)
from eval.scoring import recall_at_k
from weekly_analyst.analysis.checks import ShownNumber, check_report
from weekly_analyst.core.config import Settings
from weekly_analyst.detection import DETECTORS, select_alerts
from weekly_analyst.data.recompute import recompute
from weekly_analyst.sim.events import EventSpec
from weekly_analyst.sim.generate import DEV_SEEDS, HOLDOUT_SEEDS, REGIONS, SEGMENTS, START_MONDAY

SETTINGS = Settings()
WEEKS = SETTINGS.cold_start_periods + 1  # 8 history periods, then one scored week
SCORED_WEEK = WEEKS - 1
MAGNITUDE = 1.5  # calibrated on dev seeds 0-7 only: clean blend recall 0.5 against chance 0.375
N_SETS = 30  # sets (= events) per split, at least 30 (ticket 026 rerun ruling)
N_GATE_SEEDS = 10  # quiet-gate calibration seeds, unchanged from the 10-set design
DEV_SET_SEEDS = list(DEV_SEEDS[:N_SETS])
HOLDOUT_SET_SEEDS = list(HOLDOUT_SEEDS[:N_SETS])
GATE_SEEDS = list(DEV_SEEDS[:N_GATE_SEEDS])
QUIET_TEST_SEEDS = [DEV_SEEDS[N_GATE_SEEDS], DEV_SEEDS[N_GATE_SEEDS + 1]]  # not used for calibration
METAMORPHIC_SEED = DEV_SEEDS[12]
DRAFT_SEED = DEV_SEEDS[13]
EQUAL_WEIGHTS = dict.fromkeys(DETECTORS, 1.0)


def scored_period() -> str:
    iso = (START_MONDAY + timedelta(weeks=SCORED_WEEK)).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


@functools.lru_cache(maxsize=None)  # each set is scored once per test session
def event_set(seed: int):
    """One set: a revenue outlier in one seed-chosen segment of the scored week."""
    segment = random.Random(seed).choice(SEGMENTS)
    events = [EventSpec("outlier", SCORED_WEEK, "revenue", segment, MAGNITUDE)]
    items = scored_items(simulate(seed, WEEKS, events), control_metrics(), SETTINGS)
    return items, {("revenue", scored_period(), (("region", segment[0]),))}


def quiet_gate() -> dict[str, float]:
    """Minimum-change gate per metric: largest quiet change over the calibration seeds."""
    metrics = control_metrics()
    gate = {m.name: 0.0 for m in metrics.metrics}
    for seed in GATE_SEEDS:
        frames = no_event_set(seed, WEEKS)
        for m in metrics.metrics:
            for region in REGIONS:
                gate[m.name] = max(gate[m.name], abs_change(frames, m, region))
    return gate


def _shown(value: float, metric_name: str) -> str:
    return f"{round(value):,}" if metric_name == "units" else f"{value:,.2f}"


def _worst_relative_change(base: dict, other: dict, name: str) -> float:
    return max(abs(other[k][name] - base[k][name]) / (1.0 + abs(base[k][name])) for k in base)


@pytest.fixture(scope="module")
def dev_pool():
    return build_pool([event_set(seed) for seed in DEV_SET_SEEDS], SETTINGS.alert_budget_k)


@pytest.fixture(scope="module")
def holdout_pool():
    return build_pool([event_set(seed) for seed in HOLDOUT_SET_SEEDS], SETTINGS.alert_budget_k)


@pytest.fixture(scope="module")
def gate():
    return quiet_gate()


def test_each_split_has_at_least_30_events(dev_pool, holdout_pool):
    assert len(dev_pool.truth) >= 30 and len(holdout_pool.truth) >= 30, (
        len(dev_pool.truth),
        len(holdout_pool.truth),
    )


def test_leaky_search_is_0_1_grid_plus_200_dirichlet_draws():
    candidates = leaky_candidates()
    grid = weight_grid()
    assert candidates[: len(grid)] == grid
    draws = candidates[len(grid) :]
    assert len(draws) == 200
    for weights in draws:
        assert min(weights.values()) >= 0.0
        assert abs(sum(weights.values()) - 1.0) < 1e-9


def test_leaky_variant_shows_dev_holdout_recall_gap_at_least_0_15(dev_pool, holdout_pool):
    weights = leaky_detector(holdout_pool)  # the leak: blend weights chosen on holdout truth
    gap = pooled_recall(holdout_pool, weights) - pooled_recall(dev_pool, weights)
    assert gap >= 0.15, f"holdout minus dev recall is {gap:.3f} with leaky weights {weights}"


def test_shuffled_truth_recall_inside_random_ranking_95_range(holdout_pool):
    labels = [key in holdout_pool.truth for key in holdout_pool.keys]
    shuffled = shuffled_truth(labels, seed=0)
    shuffled_events = [key for key, label in zip(holdout_pool.keys, shuffled) if label]
    recall = recall_at_k(shuffled_events, pool_ranking(holdout_pool, EQUAL_WEIGHTS), holdout_pool.k)
    low, high = random_ranking_recall_range(
        len(holdout_pool.truth), holdout_pool.k, len(holdout_pool.keys)
    )
    assert low <= recall <= high, f"shuffled recall {recall:.3f} outside [{low:.3f}, {high:.3f}]"


@pytest.mark.parametrize("seed", QUIET_TEST_SEEDS)
def test_no_event_data_zero_alerts(seed, gate):
    metrics = control_metrics()
    frames = no_event_set(seed, WEEKS)
    items = with_abs_change(scored_items(frames, metrics, SETTINGS), frames, metrics)
    alerts = select_alerts(items, SETTINGS.alert_budget_k, gate, set())
    assert alerts == []


def test_metamorphic_suite_passes_per_detector_type():
    metrics = control_metrics()
    frames = simulate(
        METAMORPHIC_SEED,
        WEEKS,
        [EventSpec("outlier", SCORED_WEEK, "revenue", SEGMENTS[0], MAGNITUDE)],
    )
    renamed_metrics = metrics.model_copy(
        update={
            "metrics": [
                m.model_copy(update={"column": "revenue_v2"}) if m.column == "revenue" else m
                for m in metrics.metrics
            ]
        }
    )
    scaled_frames = [f.with_columns(pl.col("revenue") * 2) for f in frames]
    variants = {
        "scale x2": (scaled_frames, metrics),
        "row shuffle": (
            [f.sample(fraction=1.0, shuffle=True, seed=i) for i, f in enumerate(frames)],
            metrics,
        ),
        "column rename": ([f.rename({"revenue": "revenue_v2"}) for f in frames], renamed_metrics),
    }
    base_items = scored_items(frames, metrics, SETTINGS)
    base = {(i.metric, i.segment, i.period): i.detail for i in base_items}
    base_order = [(i.metric, i.segment, i.period) for i in base_items]

    problems = []
    for label, (variant_frames, variant_metrics) in variants.items():
        items = scored_items(variant_frames, variant_metrics, SETTINGS)
        other = {(i.metric, i.segment, i.period): i.detail for i in items}
        if set(other) != set(base):
            problems.append(f"{label}: candidate set changed")
            continue
        for name in DETECTORS:
            worst = _worst_relative_change(base, other, name)
            if worst > 1e-9:
                problems.append(f"{name} raw score moved by up to {worst:.2e} under {label}")
        if [(i.metric, i.segment, i.period) for i in items] != base_order:
            problems.append(f"blended ranking changed under {label}")

    base_abs = {
        (i.metric, i.segment, i.period): i.detail["abs_change"]
        for i in with_abs_change(base_items, frames, metrics)
    }
    scaled_abs = {
        (i.metric, i.segment, i.period): i.detail["abs_change"]
        for i in with_abs_change(
            scored_items(scaled_frames, metrics, SETTINGS), scaled_frames, metrics
        )
    }
    for key, before in base_abs.items():
        expected = 2 * before if key[0] == "revenue" else before
        if abs(scaled_abs[key] - expected) > 1e-6 * (1.0 + abs(expected)):
            problems.append(f"abs_change for {key[0]} {key[1]} not scaled by 2 under scale x2")

    assert problems == [], "; ".join(problems)


def test_seeded_bad_drafts_all_rejected(tmp_path):
    metrics = control_metrics()
    snapshot = tmp_path / "snapshot.parquet"
    pl.concat(no_event_set(DRAFT_SEED, WEEKS)).write_parquet(snapshot)
    period = scored_period()
    for seed in range(12):
        metric = metrics.metrics[seed % 2]
        segment = {"region": REGIONS[seed % len(REGIONS)]}
        value = recompute(snapshot, metric, segment, period)
        assert value is not None
        right = _shown(value, metric.name)
        wrong = _shown(value * 1.2, metric.name)

        def shown(text, name):
            return ShownNumber(text=text, metric=name, segment=segment, period=period)

        clean = check_report(f"{metric.name} reached {right}.", [shown(right, metric.name)], metrics, snapshot)
        assert clean == [], f"seed {seed}: clean draft wrongly rejected: {clean}"
        bad = {
            "wrong_number": (f"{metric.name} reached {wrong}.", [shown(wrong, metric.name)]),
            "no_metric": (f"Profit reached {right}.", [shown(right, "profit")]),
            "wording": (
                f"{metric.name} reached {right} because of the spring promotion.",
                [shown(right, metric.name)],
            ),
        }
        for expected, (text, numbers) in bad.items():
            kinds = [f.kind for f in check_report(text, numbers, metrics, snapshot)]
            assert expected in kinds, f"seed {seed}: {expected} draft not rejected, got {kinds}"
