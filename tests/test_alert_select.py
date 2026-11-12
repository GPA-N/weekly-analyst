"""Weekly alert budget, minimum-change gate, reported-item gate, cold start (spec 6.6; ticket 012)."""

from dataclasses import replace
from datetime import datetime, timedelta

import polars as pl

from weekly_analyst.core.config import Settings
from weekly_analyst.detection import (
    Item,
    insufficient_history,
    item_key,
    remaining_budget,
    score_items,
    select_alerts,
)
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile
from weekly_analyst.sim.generate import START_MONDAY, generate
from weekly_analyst.core.store import StateStore, iso_week_of

METRIC = "units_mean"
MONDAY = datetime(2026, 11, 9, 9, 0)  # Monday of ISO week 2026-W46
REPLAY_SEED = 7  # a dev seed (DEV_SEEDS is range(0, 1000)); holdout seeds are never used here
REPLAY_WEEKS = 20
REPLAY_MIN_CHANGE = 2.0  # units of the weekly region mean; a test value, not a tuned default


def make_item(
    region: str,
    abs_change: float,
    period: str = "2026-W46",
    material_change: bool = False,
    schema_change: bool = False,
) -> Item:
    return Item(
        metric=METRIC,
        segment=(("region", region),),
        period=period,
        score=1.0,
        detail={
            "abs_change": abs_change,
            "material_change": material_change,
            "schema_change": schema_change,
        },
    )


def test_daily_runs_for_7_days_never_exceed_k_per_iso_week(tmp_path):
    settings = Settings()
    store = StateStore(tmp_path / "state.sqlite")
    min_change = {METRIC: 1.0}
    for day in range(7):
        week = iso_week_of(MONDAY + timedelta(days=day))
        budget = remaining_budget(store, week, settings)
        candidates = [make_item(f"day{day}-{k}", abs_change=5.0, period=week) for k in range(3)]
        for item in select_alerts(candidates, budget, min_change, store.reported_keys()):
            store.record_alert(item_key(item), week, f"run-{day}")
        assert store.alerts_sent_in_week(week) <= settings.alert_budget_k
    assert store.alerts_sent_in_week("2026-W46") == settings.alert_budget_k


def test_remaining_budget_is_k_minus_alerts_sent_this_week_floored_at_zero(tmp_path):
    settings = Settings()
    store = StateStore(tmp_path / "state.sqlite")
    assert remaining_budget(store, "2026-W46", settings) == 3
    store.record_alert("a", "2026-W46", "run-1")
    assert remaining_budget(store, "2026-W46", settings) == 2
    store.record_alert("b", "2026-W46", "run-1")
    store.record_alert("c", "2026-W46", "run-1")
    assert remaining_budget(store, "2026-W46", settings) == 0
    store.record_alert("d", "2026-W46", "run-1")
    assert remaining_budget(store, "2026-W46", settings) == 0


def test_budget_resets_in_next_iso_week(tmp_path):
    settings = Settings()
    store = StateStore(tmp_path / "state.sqlite")
    for k in range(settings.alert_budget_k):
        store.record_alert(f"k{k}", "2026-W46", "run-a")
    assert remaining_budget(store, "2026-W46", settings) == 0
    assert remaining_budget(store, "2026-W47", settings) == settings.alert_budget_k


def test_select_alerts_keeps_rank_order_and_stops_at_budget():
    ranked = [make_item(f"r{k}", abs_change=5.0) for k in range(4)]
    assert select_alerts(ranked, 2, {METRIC: 1.0}, set()) == ranked[:2]
    assert select_alerts(ranked, 0, {METRIC: 1.0}, set()) == []
    assert select_alerts(ranked, -1, {METRIC: 1.0}, set()) == []


def test_min_change_gate_drops_small_changes():
    small = make_item("north", abs_change=0.4)
    at_gate = make_item("south", abs_change=1.0)
    large = make_item("east", abs_change=3.0)
    selected = select_alerts([small, at_gate, large], 3, {METRIC: 1.0}, set())
    assert selected == [at_gate, large]


def test_reported_item_not_repeated_unless_material():
    repeated = make_item("north", abs_change=5.0)
    material = make_item("south", abs_change=5.0, material_change=True)
    new = make_item("east", abs_change=5.0)
    reported = {item_key(repeated), item_key(material)}
    selected = select_alerts([repeated, material, new], 3, {METRIC: 1.0}, reported)
    assert selected == [material, new]


def test_cold_start_true_for_7_periods_false_for_8():
    settings = Settings()
    assert insufficient_history(7, settings) is True
    assert insufficient_history(8, settings) is False


def test_schema_change_survives_cold_start():
    settings = Settings()
    schema = make_item("north", abs_change=5.0, schema_change=True)
    ordinary = make_item("south", abs_change=5.0)
    # Cold-start rule (6.6) as the run applies it: with insufficient history only
    # schema-change items go on to select_alerts. select_alerts takes no period count.
    candidates = [schema, ordinary]
    if insufficient_history(7, settings):
        candidates = [item for item in candidates if item.detail["schema_change"]]
    assert select_alerts(candidates, 3, {METRIC: 1.0}, set()) == [schema]


def segment_week_means(df: pl.DataFrame) -> dict[tuple[str, str], float]:
    weekly = (
        df.with_columns(pl.col("date").dt.strftime("%G-W%V").alias("period"))
        .group_by(["region", "period"])
        .agg(pl.col("units").mean().alias("mean"))
    )
    return {(row["region"], row["period"]): row["mean"] for row in weekly.iter_rows(named=True)}


def with_abs_change(items: list[Item], history: pl.DataFrame, current: pl.DataFrame) -> list[Item]:
    """Test helper: detail["abs_change"] = |this week's segment mean - mean of its history weeks|."""
    past = segment_week_means(history)
    now = segment_week_means(current)
    out = []
    for item in items:
        region = item.segment[0][1]
        history_means = [value for (r, _), value in past.items() if r == region]
        base = sum(history_means) / len(history_means)
        change = abs(now[(region, item.period)] - base)
        out.append(replace(item, detail={**item.detail, "abs_change": change, "material_change": False}))
    return out


def test_quiet_replay_20_weeks_zero_alerts(tmp_path):
    settings = Settings()
    sim = generate(REPLAY_SEED, REPLAY_WEEKS, [], tmp_path / "sim")
    frames = [pl.read_csv(path, try_parse_dates=True) for path in sim.files]
    metrics = MetricsFile(
        metrics=[
            MetricDef(
                name=METRIC,
                definition="Mean units per sale row.",
                owner="test",
                inclusions="all rows",
                exclusions="none",
                column="units",
                agg="mean",
                segment_columns=["region"],
                min_change=REPLAY_MIN_CHANGE,
                version=1,
            )
        ],
        date_column="date",
        checksum="test",
    )
    min_change = {m.name: m.min_change for m in metrics.metrics}
    store = StateStore(tmp_path / "state.sqlite")
    scored = 0
    sent = 0
    for week, current in enumerate(frames):
        if insufficient_history(week, settings):
            continue
        scored += 1
        history = pl.concat(frames[:week])
        items = with_abs_change(score_items(history, current, metrics, settings), history, current)
        iso_week = iso_week_of(datetime(START_MONDAY.year, START_MONDAY.month, START_MONDAY.day) + timedelta(weeks=week))
        selected = select_alerts(items, remaining_budget(store, iso_week, settings), min_change, store.reported_keys())
        for item in selected:
            store.record_alert(item_key(item), iso_week, f"replay-{week}")
        sent += len(selected)
    assert scored == REPLAY_WEEKS - settings.cold_start_periods
    assert sent == 0
