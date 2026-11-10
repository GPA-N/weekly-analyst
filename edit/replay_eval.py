"""Scored replay on holdout seeds (AC-001/002 recall, quiet-week false alarms). VM only, no model.

Env: N_SETS=100  QUIET_SEEDS=10  OUT=/content/replay
Gate (minimum change per metric) is calibrated on dev seeds 0-9 only; every scored set uses holdout seeds.
"""
import json
import os
import random
import sys
from datetime import timedelta
from pathlib import Path

for _m in [m for m in sys.modules if m.split(".")[0] in ("weekly_analyst", "eval")]:
    del sys.modules[_m]
sys.path.insert(0, "/content/pkg/src")
sys.path.insert(0, "/content/pkg")
from eval.controls import abs_change, control_metrics, no_event_set, scored_items, simulate, with_abs_change  # noqa: E402
from eval.scoring import clopper_pearson  # noqa: E402
from weekly_analyst.core.config import Settings  # noqa: E402
from weekly_analyst.core.seeds import HOLDOUT_SEEDS  # noqa: E402
from weekly_analyst.detection import select_alerts  # noqa: E402
from weekly_analyst.sim.events import EventSpec  # noqa: E402
from weekly_analyst.sim.generate import DEV_SEEDS, REGIONS, SEGMENTS, START_MONDAY  # noqa: E402

S = Settings()
WEEKS = S.cold_start_periods + 1
SCORED = WEEKS - 1
MAGNITUDE = 1.5
n_sets = int(os.environ.get("N_SETS", "100"))
quiet_seeds = int(os.environ.get("QUIET_SEEDS", "10"))
out = Path(os.environ.get("OUT", "/content/replay"))
out.mkdir(parents=True, exist_ok=True)
metrics = control_metrics()

gate = {m.name: 0.0 for m in metrics.metrics}
for seed in list(DEV_SEEDS[:10]):
    frames = no_event_set(seed, WEEKS)
    for m in metrics.metrics:
        for region in REGIONS:
            gate[m.name] = max(gate[m.name], abs_change(frames, m, region))
print("gate", gate, flush=True)

iso = (START_MONDAY + timedelta(weeks=SCORED)).isocalendar()
period = f"{iso.year}-W{iso.week:02d}"
hits = 0
rows = []
for seed in list(HOLDOUT_SEEDS[:n_sets]):
    seg = random.Random(seed).choice(SEGMENTS)
    frames = simulate(seed, WEEKS, [EventSpec("outlier", SCORED, "revenue", seg, MAGNITUDE)])
    items = with_abs_change(scored_items(frames, metrics, S), frames, metrics)
    alerts = select_alerts(items, S.alert_budget_k, gate, set())
    hit = any(a.metric == "revenue" and a.period == period and a.segment == (("region", seg[0]),) for a in alerts)
    hits += hit
    rows.append({"seed": seed, "alerts": len(alerts), "hit": bool(hit)})
lo, hi = clopper_pearson(hits, n_sets)
recall = {"n": n_sets, "hits": hits, "recall": hits / n_sets, "ci95": [lo, hi], "ac002_pass": n_sets >= 30 and lo >= 0.80}
print("recall", recall, flush=True)

# quiet replay: no events, 20 scored weeks per seed (weeks 8..27), alerts on any week are false alarms
quiet_weeks = 0
false_alarm_weeks = 0
total_alerts = 0
for seed in list(HOLDOUT_SEEDS[-quiet_seeds:]):
    frames = no_event_set(seed, S.cold_start_periods + 20)
    for week in range(S.cold_start_periods, len(frames)):
        sub = frames[: week + 1]
        items = with_abs_change(scored_items(sub, metrics, S), sub, metrics)
        alerts = select_alerts(items, S.alert_budget_k, gate, set())
        quiet_weeks += 1
        false_alarm_weeks += bool(alerts)
        total_alerts += len(alerts)
qlo, qhi = clopper_pearson(false_alarm_weeks, quiet_weeks)
quiet = {"quiet_weeks": quiet_weeks, "false_alarm_weeks": false_alarm_weeks, "alerts": total_alerts,
         "false_alarm_rate": false_alarm_weeks / quiet_weeks, "ci95": [qlo, qhi]}
print("quiet", quiet, flush=True)
(out / "replay.json").write_text(json.dumps({"gate": gate, "magnitude": MAGNITUDE, "recall": recall, "quiet": quiet, "rows": rows}, indent=1))
