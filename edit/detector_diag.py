"""Detector diagnosis on DEV seeds only (never holdout): where does recall go? CPU, no model."""
import json, random, sys
from datetime import timedelta
sys.path.insert(0, "/content/pkg/src"); sys.path.insert(0, "/content/pkg")
import numpy as np
from eval.controls import control_metrics, simulate, scored_items, with_abs_change, abs_change, no_event_set
from weekly_analyst.core.config import Settings
from weekly_analyst.core.seeds import HOLDOUT_SEEDS
from weekly_analyst.data.metrics_file import MetricDef, MetricsFile
from weekly_analyst.detection import DETECTORS, select_alerts
from weekly_analyst.sim.events import EventSpec
from weekly_analyst.sim.generate import DEV_SEEDS, REGIONS, SEGMENTS, START_MONDAY
S = Settings(); WEEKS = S.cold_start_periods + 1; SC = WEEKS - 1
assert not set(HOLDOUT_SEEDS) & set(DEV_SEEDS)
iso = (START_MONDAY + timedelta(weeks=SC)).isocalendar(); PERIOD = f"{iso.year}-W{iso.week:02d}"
N = 100
seeds = list(DEV_SEEDS[20:20 + N])
def fine_metrics():
    def md(name, col):
        return MetricDef(name=name, definition=f"Sum of {col}.", owner="diag", inclusions="all", exclusions="none", column=col, agg="sum",
                         segment_columns=["region", "product", "channel"], min_change=0.0, version=1)
    return MetricsFile(metrics=[md("revenue", "revenue"), md("units", "units")], date_column="date", checksum="diag")
coarse, fine = control_metrics(), fine_metrics()
def rank_of(items, metric, seg):
    for i, it in enumerate(items):
        if it.metric == metric and it.period == PERIOD and it.segment == seg: return i
    return None
res = {"coarse": [], "fine": []}
for mag in (1.5, 2.0, 3.0):
    for name, mets in (("coarse", coarse), ("fine", fine)):
        ranks = {d: [] for d in list(DETECTORS) + ["blend"]}
        for seed in seeds:
            seg = random.Random(seed).choice(SEGMENTS)
            frames = simulate(seed, WEEKS, [EventSpec("outlier", SC, "revenue", seg, mag)])
            items = scored_items(frames, mets, S)
            truth = (("region", seg[0]),) if name == "coarse" else (("region", seg[0]), ("product", seg[1]), ("channel", seg[2]))
            ranks["blend"].append(rank_of(items, "revenue", truth))
            for d in DETECTORS:
                order = sorted(items, key=lambda it: -float(it.detail[d]))
                ranks[d].append(rank_of(order, "revenue", truth))
        row = {"magnitude": mag, "metrics": name, "n": len(seeds), "candidates": len(items)}
        for d, rs in ranks.items():
            row[d] = {"recall@3": float(np.mean([r is not None and r < 3 for r in rs])), "median_rank": float(np.median([r if r is not None else 999 for r in rs]))}
        res[name].append(row); print(json.dumps(row), flush=True)
# quiet false alarm structure: how large are quiet region changes relative to the 1.5x event?
chg = []
for seed in seeds[:40]:
    fr = no_event_set(seed, WEEKS)
    for m in coarse.metrics:
        for r in REGIONS: chg.append((m.name, abs_change(fr, m, r)))
rev = [c for n, c in chg if n == "revenue"]
res["quiet_revenue_region_abs_change"] = {"p50": float(np.percentile(rev, 50)), "p95": float(np.percentile(rev, 95)), "max": float(max(rev))}
print(json.dumps(res["quiet_revenue_region_abs_change"]), flush=True)
json.dump(res, open("/content/diag.json", "w"), indent=1)
