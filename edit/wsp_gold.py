import json

def gold_plan(prompt):
    rows = _data(prompt)
    steps = [{"item_key": r["item_key"], "tool": "named_query",
              "args": {"name": "by_segment", "params": {"metric": r["metric"], "segment_col": r["segment_columns"][0], "period": r["period"]}}}
             for r in rows]
    return json.dumps({"steps": steps}, indent=2)

def gold_draft(prompt):
    ev = _data(prompt)
    items = ev["items"]
    first = items[0]
    seg = " ".join(first["segment"].values())
    findings = [{"item_key": i["item_key"], "so_what": f"{i['metric'].capitalize()} moved {i['direction']} in " + " ".join(i["segment"].values()),
                 "metric_refs": [i["metric"]]} for i in items]
    return json.dumps({"headline": f"{first['metric'].capitalize()} moved {first['direction']} led by {seg}",
                       "recommended_action": "Review the affected segments with the owning team",
                       "findings": findings, "caveats": []}, indent=2)

CAL = build_tasks(range(0, 20)); DEV = build_tasks(range(20, 40)); HOLD = build_tasks(range(100, 140))
print(len(CAL), len(DEV), len(HOLD), flush=True)
for t in CAL + DEV + HOLD:
    t["gold_plan"] = gold_plan(t["plan_prompt"]); t["gold_draft"] = gold_draft(t["draft_prompt"])
assert all(passed(score_plan(t["gold_plan"], t["plan_prompt"])) and passed(score_draft(t["gold_draft"], t["draft_prompt"])) for t in CAL + DEV + HOLD), "gold fails own scorer"
print("gold ok", flush=True)
