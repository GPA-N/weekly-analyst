import dataclasses

import pytest

from weekly_analyst.core.config import Settings

SPEC_VALUES = {
    "alert_budget_k": 3,
    "min_rows_per_segment": 30,
    "cold_start_periods": 8,
    "cusum_k": 0.5,
    "cusum_h": 5.0,
    "investigate_calls": 6,
    "plan_reprompts": 2,
    "repair_rounds": 2,
    "max_bytes": 50_000_000,
    "max_cells": 5_000_000,
    "max_rows": 1_000_000,
    "duckdb_memory": "2GB",
    "duckdb_threads": 2,
    "page_cap": 8,
    "headline_max": 120,
    "action_max": 300,
    "sowhat_max": 250,
    "caveat_max": 200,
    "caveats_max": 6,
    "run_timeout_s": 1200,
    "unfinished_parquet_days": 7,
    "outbox_days": 30,
}


def test_settings_values_match_spec():
    s = Settings()
    actual = {f.name: getattr(s, f.name) for f in dataclasses.fields(s)}
    assert actual == SPEC_VALUES


def test_settings_is_frozen():
    s = Settings()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.alert_budget_k = 5
