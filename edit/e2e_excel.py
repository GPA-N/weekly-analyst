"""Real pipeline on a real Excel-derived CSV (Tableau Superstore sample, converted by tmp/xl_convert.py to /content/xl/data/sales.csv). VM only.

Runs the production runner with sim/metrics_sales.yaml unchanged on the CSV cut at several end weeks (the runner reports the latest week, so each cutoff is
a different "this week"). Env: GGUF=/path/model.gguf  OUT=/content/e2e_excel  RAW=1 (the pipeline normally constrains the plan and draft calls with a JSON schema; RAW=1 sends those calls unconstrained, to test the model alone; the free-SQL call keeps its grammar)
CUTS=2017-03-31,... (default below). Every model request and reply goes to OUT/calls.jsonl; every run's outcome to OUT/summary.json.
"""
import csv
import json
import os
import sys
import time
import uuid
from pathlib import Path

for _m in [m for m in sys.modules if m.split(".")[0] == "weekly_analyst"]:
    del sys.modules[_m]
sys.path.insert(0, "/content/pkg/src")
from weekly_analyst import cli  # noqa: E402
from weekly_analyst.analysis.model import LlamaCppClient  # noqa: E402
from weekly_analyst.core.store import StateStore  # noqa: E402
from weekly_analyst.demo import METRICS as _METRICS  # noqa: E402
from weekly_analyst.runner import RunConfig, run  # noqa: E402

METRICS = Path(os.environ.get("METRICS") or _METRICS)
gguf = Path(os.environ["GGUF"])
out = Path(os.environ.get("OUT", "/content/e2e_excel"))
out.mkdir(parents=True, exist_ok=True)
calls_path = out / "calls.jsonl"
calls_path.write_text("", encoding="utf-8")
raw = os.environ.get("RAW") == "1"
cuts = (os.environ.get("CUTS") or "2016-06-30,2016-12-31,2017-03-31,2017-06-30,2017-09-30,2017-11-30,2017-12-31").split(",")


class Recording:
    def __init__(self, inner):
        self.inner = inner

    def complete(self, prompt, json_schema, *, temperature, seed, grammar=None):
        t = time.perf_counter()
        if raw and grammar is None:  # schema-constrained call sent without the constraint
            reply = self.inner._llm.create_chat_completion(messages=[{"role": "user", "content": prompt}], temperature=temperature, seed=seed)["choices"][0]["message"]["content"]
        else:
            reply = self.inner.complete(prompt, json_schema, temperature=temperature, seed=seed, grammar=grammar)
        with open(calls_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"secs": round(time.perf_counter() - t, 1), "constrained": not (raw and grammar is None),
                                 "prompt_tail": prompt[-500:], "reply": reply}) + "\n")
        return reply


client = Recording(LlamaCppClient(gguf, threads=2, n_ctx=int(os.environ.get("N_CTX", "8192")), n_gpu_layers=-1))
src = Path(os.environ.get("SRC", "/content/xl/data/sales.csv"))
rows = list(csv.reader(open(src, encoding="utf-8")))
header, body = rows[0], rows[1:]
summary = []
for cut in cuts:
    folder = out / f"cut-{cut}-{uuid.uuid4().hex[:6]}"
    (folder / "data").mkdir(parents=True)
    (folder / "work").mkdir()
    kept = [r for r in body if r[0] <= cut]
    with open(folder / "data" / "sales.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(kept)
    cfg = RunConfig(metrics_path=METRICS, work_dir=folder / "work", out_dir=folder / "out", delivery_mode="approval", recipients=["ops@example.com"], paper="A4")
    t = time.perf_counter()
    try:
        o = run(folder / "data" / "sales.csv", cfg, client, StateStore(folder / "work" / cli.STATE_DB))
        row = {"cut": cut, "rows": len(kept), "kind": o.kind, "reasons": list(o.reasons), "report": str(o.report_path) if o.report_path else None}
    except Exception as e:  # keep going: a halted or failed run is itself a finding
        row = {"cut": cut, "rows": len(kept), "kind": "error", "reasons": [f"{type(e).__name__}: {str(e)[:300]}"], "report": None}
    row["seconds"] = round(time.perf_counter() - t, 1)
    print(json.dumps(row), flush=True)
    summary.append(row)
    (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
print("DONE-EXCEL", flush=True)
