"""Task eval for weight-space edits: real plan and draft prompts from the real runner, scored with the project's validators.

Exec this file on the VM; it defines build_tasks, run_eval and scoring helpers. Needs globals `m` (model) and `tok` for run_eval.
"""
import json
import re
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path

for _k in [k for k in sys.modules if k.split(".")[0] == "weekly_analyst"]:
    del sys.modules[_k]
sys.path.insert(0, "/content/pkg/src")
from weekly_analyst.analysis import draft as draft_mod  # noqa: E402
from weekly_analyst.analysis import plan as plan_mod  # noqa: E402
from weekly_analyst.core.config import Settings  # noqa: E402
from weekly_analyst.core.text import numerals  # noqa: E402
from weekly_analyst.demo import run_demo  # noqa: E402
from weekly_analyst.sim import demo as sim_demo  # noqa: E402

SETTINGS = Settings()


class Halt(RuntimeError):
    pass


class OracleClient:
    """Answers the plan prompt with a valid plan, keeps the draft prompt, then halts the run."""

    def __init__(self):
        self.plan_prompt = None
        self.draft_prompt = None

    def complete(self, prompt, json_schema, *, temperature, seed, grammar=None):
        data = json.loads(prompt.split("<data>\n", 1)[1].rsplit("\n</data>", 1)[0])
        if prompt.startswith(plan_mod.INSTRUCTION[:40]):
            self.plan_prompt = prompt
            steps = [{"item_key": r["item_key"], "tool": "named_query",
                      "args": {"name": "by_segment", "params": {"metric": r["metric"], "segment_col": r["segment_columns"][0], "period": r["period"]}}}
                     for r in data]
            return json.dumps({"steps": steps})
        self.draft_prompt = prompt
        raise Halt("captured")


def build_tasks(seeds, scenarios=("spike", "drop")):
    tasks = []
    base = Path(tempfile.mkdtemp(prefix="wsp-"))
    for seed in seeds:
        for scen in scenarios:
            client = OracleClient()
            start = sim_demo.START + timedelta(weeks=seed % 17)
            folder = base / f"{scen}-{seed}"
            (folder / "data").mkdir(parents=True)
            (folder / "work").mkdir()
            from weekly_analyst import cli
            from weekly_analyst.core.store import StateStore
            from weekly_analyst.demo import METRICS
            from weekly_analyst.runner import RunConfig, run
            csv = sim_demo.write_demo_csv(folder / "data" / "sales.csv", scen, start=start, seed=seed)
            cfg = RunConfig(metrics_path=METRICS, work_dir=folder / "work", out_dir=folder / "out",
                            delivery_mode="approval", recipients=["ops@example.com"], paper="A4")
            try:
                run(csv, cfg, client, StateStore(folder / "work" / cli.STATE_DB))
            except Halt:
                pass
            except Exception:
                pass
            if client.plan_prompt and client.draft_prompt:
                tasks.append({"seed": seed, "scenario": scen, "plan_prompt": client.plan_prompt, "draft_prompt": client.draft_prompt})
    return tasks


def _data(prompt):
    return json.loads(prompt.split("<data>\n", 1)[1].rsplit("\n</data>", 1)[0])


def _extract(text):
    text = text.replace("<|im_end|>", "")
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if i >= 0 and j > i else text


def score_plan(raw, prompt):
    rows = _data(prompt)
    out = {"json": False, "schema": False, "params": False, "covers": False}
    try:
        plan = plan_mod.Plan.model_validate_json(_extract(raw))
    except Exception:
        try:
            json.loads(_extract(raw)); out["json"] = True
        except Exception:
            pass
        return out
    out["json"] = out["schema"] = True
    metrics = {r["metric"] for r in rows}
    cols = {c for r in rows for c in r["segment_columns"]}
    periods = {r["period"] for r in rows}
    ok = True
    for s in plan.steps:
        p = s.args.get("params", s.args) if s.tool == "named_query" else s.args
        if s.tool in ("named_query", "attribute_change") and isinstance(p, dict):
            if "metric" in p and p["metric"] not in metrics: ok = False
            if "segment_col" in p and p["segment_col"] not in cols: ok = False
            if "period" in p and p["period"] not in periods: ok = False
    out["params"] = ok
    out["covers"] = {r["item_key"] for r in rows} <= {s.item_key for s in plan.steps}
    return out


def score_draft(raw, prompt):
    ev = _data(prompt)
    out = {"json": False, "schema": False, "length": False, "no_numerals": False, "keys": False}
    try:
        d = draft_mod.DraftFields.model_validate_json(_extract(raw))
    except Exception:
        try:
            json.loads(_extract(raw)); out["json"] = True
        except Exception:
            pass
        return out
    out["json"] = out["schema"] = True
    out["length"] = not draft_mod._length_problems(d, SETTINGS)
    text = " ".join([d.headline, d.recommended_action, *[f.so_what for f in d.findings], *d.caveats])
    out["no_numerals"] = not numerals(text)
    want = {i["item_key"] for i in ev["items"]}
    out["keys"] = {f.item_key for f in d.findings} == want
    return out


def passed(sc):
    return all(sc.values())


def generate(prompts, max_new=700, batch=8, schema=None):
    import torch
    outs = []
    tok.padding_side = "left"
    fn = None
    if schema is not None:
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in chunk]
        enc = tok(texts, return_tensors="pt", padding=True).to("cuda")
        kw = {}
        if schema is not None:
            kw["prefix_allowed_tokens_fn"] = build_transformers_prefix_allowed_tokens_fn(tok, JsonSchemaParser(schema))
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, **kw)
        outs += [tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True) for x in g]
    return outs


def run_eval(tasks, label, max_new=700, batch=8, constrained=True):
    t = time.time()
    ps_schema = plan_mod.Plan.model_json_schema() if constrained else None
    ds_schema = draft_mod.DraftFields.model_json_schema() if constrained else None
    plan_out = generate([x["plan_prompt"] for x in tasks], max_new, batch, ps_schema)
    draft_out = generate([x["draft_prompt"] for x in tasks], max_new, batch, ds_schema)
    ps = [score_plan(o, x["plan_prompt"]) for o, x in zip(plan_out, tasks)]
    ds = [score_draft(o, x["draft_prompt"]) for o, x in zip(draft_out, tasks)]
    n = len(tasks)
    res = {"label": label, "constrained": constrained, "n": n, "plan_pass": sum(map(passed, ps)), "draft_pass": sum(map(passed, ds)),
           "plan_parts": {k: sum(s[k] for s in ps) for k in ps[0]}, "draft_parts": {k: sum(s[k] for s in ds) for k in ds[0]},
           "secs": round(time.time() - t, 1)}
    return res, {"plan": plan_out, "draft": draft_out, "plan_scores": ps, "draft_scores": ds}
