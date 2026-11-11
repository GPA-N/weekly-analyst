"""Diagnosis stage: why do drafts score 0/80 without a grammar, and what does WeightWatcher say about the layers.

Exec'd by wsp_main.py with STAGE=diag (globals: m, tok, CAL, DEV, HOLD, run_eval, score_draft, draft_mod, push).
"""
import json
import time

t0 = time.time()
out = {}

# 1. raw base outputs on 6 dev tasks, with the exact validation error
raws = generate([t["draft_prompt"] for t in DEV[:6]], max_new=700, batch=6)
out["raw_base_drafts"] = []
for t, r in zip(DEV[:6], raws):
    try:
        draft_mod.DraftFields.model_validate_json(_extract(r))
        err = None
    except Exception as e:  # noqa: BLE001
        err = str(e)[:500]
    out["raw_base_drafts"].append({"raw": r[:900], "error": err, "scores": score_draft(r, t["draft_prompt"])})
    print("RAW>", r[:300].replace("\n", " ").encode("ascii", "ignore").decode(), "| ERR>", (err or "ok")[:200].replace("\n", " ").encode("ascii", "ignore").decode(), flush=True)

# 2. ceiling of the prompt alone: same prompt plus the schema in text, no grammar, no edit
schema_txt = json.dumps(draft_mod.DraftFields.model_json_schema())
def with_schema(p):
    return p + "\nThe JSON object must match this schema exactly: " + schema_txt
tasks_s = [dict(t, draft_prompt=with_schema(t["draft_prompt"])) for t in DEV]
ds_out = generate([t["draft_prompt"] for t in tasks_s], max_new=700, batch=8)
ds = [score_draft(o, t["draft_prompt"]) for o, t in zip(ds_out, DEV)]
out["schema_in_prompt_dev"] = {"pass": sum(map(passed, ds)), "n": len(ds), "parts": {k: sum(s[k] for s in ds) for k in ds[0]}}
out["schema_in_prompt_samples"] = [o[:600] for o in ds_out[:3]]
print("SCHEMA-IN-PROMPT", out["schema_in_prompt_dev"], flush=True)

# 3. WeightWatcher on the language model
try:
    import weightwatcher as ww
    w = ww.WeightWatcher(model=m.model.language_model)
    df = w.analyze(min_evals=10, plot=False)
    df.to_csv("/content/out/ww_details.csv", index=False)
    cols = [c for c in ("layer_id", "name", "alpha", "alpha_weighted", "spectral_norm", "stable_rank", "num_spikes", "D") if c in df.columns]
    out["ww_head"] = df[cols].head(80).to_dict("records")
    print("WW rows", len(df), "alpha median", float(df["alpha"].median()), flush=True)
except Exception as e:  # noqa: BLE001
    out["ww_error"] = str(e)[:400]
    print("WW ERROR", str(e)[:300].encode("ascii", "ignore").decode(), flush=True)

out["secs"] = round(time.time() - t0)
json.dump(out, open("/content/out/diag.json", "w"), indent=1)
push("diag.json")
try:
    push("ww_details.csv")
except Exception:  # noqa: BLE001
    pass
