"""Frozen run settings. Values from spec section 6.9."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    alert_budget_k: int = 3
    min_rows_per_segment: int = 30
    cold_start_periods: int = 8
    cusum_k: float = 0.5
    cusum_h: float = 5.0
    investigate_calls: int = 6
    plan_reprompts: int = 2
    repair_rounds: int = 2
    max_bytes: int = 50_000_000
    max_cells: int = 5_000_000
    max_rows: int = 1_000_000
    duckdb_memory: str = "2GB"
    duckdb_threads: int = 2
    page_cap: int = 8
    headline_max: int = 120
    action_max: int = 300
    sowhat_max: int = 250
    caveat_max: int = 200
    caveats_max: int = 6
    run_timeout_s: int = 1200
    unfinished_parquet_days: int = 7
    outbox_days: int = 30
