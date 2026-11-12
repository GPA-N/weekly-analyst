"""Detection blend: drift AUC, rank-averaged blend, weight holdout guard (spec 6.6; ticket 011)."""

import random
from datetime import date, timedelta

import polars as pl
import pytest

from weekly_analyst.core.config import Settings
from weekly_analyst.detection import drift_auc, load_weights, save_weights, score_items
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile
from weekly_analyst.sim.generate import DEV_SEEDS, HOLDOUT_SEEDS, generate

MONDAY_W40 = date(2026, 9, 28)  # Monday of ISO week 2026-W40
REGIONS = ("north", "south", "east", "west")
ROWS_PER_SEGMENT = 40
DRIFT_COLUMNS = ["units", "region"]


def period_of(week: int) -> str:
    iso = (MONDAY_W40 + timedelta(weeks=week)).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def metric(name: str, column: str) -> MetricDef:
    return MetricDef(
        name=name,
        definition=f"Mean of {column} per segment.",
        owner="test",
        inclusions="all rows",
        exclusions="none",
        column=column,
        agg="mean",
        segment_columns=["region"],
        min_change=0.0,
        version=1,
    )


def metrics_file() -> MetricsFile:
    metrics = [metric("units_mean", "units"), metric("discount_mean", "discount")]
    return MetricsFile(metrics=metrics, date_column="date", checksum="test")


def sample(rng: random.Random, n: int, shift: float = 0.0) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "units": [rng.gauss(50.0 + shift, 5.0) for _ in range(n)],
            "region": [rng.choice(REGIONS) for _ in range(n)],
        }
    )


def week_frame(rng: random.Random, week: int, outlier_region: str | None = None) -> pl.DataFrame:
    dates, regions, units, discounts = [], [], [], []
    for region in REGIONS:
        for _ in range(ROWS_PER_SEGMENT):
            value = round(rng.gauss(50.0, 5.0), 2)
            dates.append(MONDAY_W40 + timedelta(weeks=week, days=rng.randrange(7)))
            regions.append(region)
            units.append(round(value * 4.0, 2) if region == outlier_region else value)
            discounts.append(round(rng.uniform(0.10, 0.20), 3))
    return pl.DataFrame(
        {"date": dates, "region": regions, "units": units, "discount": discounts},
        schema={"date": pl.Date, "region": pl.Utf8, "units": pl.Float64, "discount": pl.Float64},
    )


def simulated_set() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Ten stable weeks of history, then one week where the south units are four times normal."""
    rng = random.Random(7)
    history = pl.concat([week_frame(rng, week) for week in range(10)])
    current = week_frame(rng, 10, outlier_region="south")
    return history, current


def scores_by_key(items) -> dict[tuple, float]:
    return {(item.metric, item.segment, item.period): item.score for item in items}


def test_identical_data_has_drift_auc_below_0_6():
    baseline = sample(random.Random(1), 300)

    auc = drift_auc(baseline, baseline, DRIFT_COLUMNS)

    assert 0.4 < auc < 0.6  # near 0.5 both ways: duplicate rows must not leak labels across folds


def test_shifted_data_has_drift_auc_above_0_8():
    baseline = sample(random.Random(1), 300)
    shifted = sample(random.Random(2), 300, shift=10.0)  # two standard deviations

    assert drift_auc(baseline, shifted, DRIFT_COLUMNS) > 0.8


def test_drift_auc_scale_invariant(tmp_path):
    columns = ["revenue", "region"]
    for seed in DEV_SEEDS[:5]:
        sim = generate(seed, 3, [], tmp_path / str(seed))
        frames = [pl.read_csv(path) for path in sim.files]
        history, current = pl.concat(frames[:-1]), frames[-1]

        auc = drift_auc(history, current, columns)
        scaled_auc = drift_auc(
            history.with_columns(pl.col("revenue") * 2),
            current.with_columns(pl.col("revenue") * 2),
            columns,
        )

        assert scaled_auc == pytest.approx(auc, abs=1e-12)


def test_blend_ranks_injected_outlier_item_first():
    history, current = simulated_set()

    items = score_items(history, current, metrics_file(), Settings())

    top = items[0]
    assert (top.metric, top.segment, top.period) == (
        "units_mean",
        (("region", "south"),),
        period_of(10),
    )


def test_row_shuffle_leaves_scores_unchanged():
    history, current = simulated_set()
    base = scores_by_key(score_items(history, current, metrics_file(), Settings()))

    shuffled = scores_by_key(
        score_items(
            history.sample(fraction=1.0, shuffle=True, seed=5),
            current.sample(fraction=1.0, shuffle=True, seed=6),
            metrics_file(),
            Settings(),
        )
    )

    assert shuffled.keys() == base.keys()
    for key, value in base.items():
        assert shuffled[key] == pytest.approx(value, rel=1e-9)


def test_renaming_a_column_through_the_metrics_map_leaves_scores_unchanged():
    history, current = simulated_set()
    base = scores_by_key(score_items(history, current, metrics_file(), Settings()))

    metrics = metrics_file()
    renamed_metrics = metrics.model_copy(
        update={
            "metrics": [
                m.model_copy(update={"column": "units_v2"}) if m.column == "units" else m
                for m in metrics.metrics
            ]
        }
    )
    renamed = scores_by_key(
        score_items(
            history.rename({"units": "units_v2"}),
            current.rename({"units": "units_v2"}),
            renamed_metrics,
            Settings(),
        )
    )

    assert renamed.keys() == base.keys()
    for key, value in base.items():
        assert renamed[key] == pytest.approx(value, rel=1e-9)


def test_metric_with_schema_change_gets_no_items():
    history, current = simulated_set()
    current = current.rename({"discount": "discount_v2"})

    items = score_items(history, current, metrics_file(), Settings())

    assert items
    assert {item.metric for item in items} == {"units_mean"}


def test_saving_weights_with_a_holdout_seed_raises_and_writes_nothing(tmp_path):
    path = tmp_path / "weights.json"

    with pytest.raises(ValueError):
        save_weights(path, {"zscore": 1.0}, dev_seeds=[DEV_SEEDS[0], HOLDOUT_SEEDS[0]])

    assert not path.exists()


def test_dev_seed_weights_round_trip(tmp_path):
    path = tmp_path / "weights.json"
    weights = {"zscore": 0.5, "drift_auc": 1.0, "cusum": 1.0, "validation": 2.0}

    save_weights(path, weights, dev_seeds=DEV_SEEDS)

    assert load_weights(path) == weights
