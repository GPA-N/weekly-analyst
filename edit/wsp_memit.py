"""MEMIT-style draft-format sweep. Exec'd by wsp_main.py with STAGE=memit. Env CFG: {"sets": [...], "lam": [...], "scale": [...]}.
Directions come from CAL (seeds 0-19), configs are scored on DEV (seeds 20-39). HOLD and the SCORE seeds are never touched here."""
import json
import os
import time

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ["CFG"])
BUDGET_S = 42 * 60
t0 = time.time()
restore()
union = sorted({l for s in CFG["sets"] for l in LAYERSETS[s]})
coll = collect(CAL, union)
print("collected", round(time.time() - t0), "s", flush=True)

rows = []
base = dev_drafts("dev-base")
rows.append(base)
print({k: base[k] for k in ("label", "pass", "parts")}, flush=True)
base["fit_err"] = {"cal": struct_error_rate(CAL), "dev": struct_error_rate(DEV)}
print("teacher-forced structural error, base", base["fit_err"], flush=True)
stop = False
for name in CFG["sets"]:
    for lam in CFG["lam"]:
        for scale in CFG["scale"]:
            if time.time() - t0 > BUDGET_S:
                stop = True
                break
            restore()
            dws = build_edit(coll, LAYERSETS[name], lam, scale)
            rel = apply_dw(dws)
            r = dev_drafts(f"memit-{name}-lam{lam}-s{scale}")
            r.update(rel_norm=rel, elapsed=round(time.time() - t0), fit_err={"cal": struct_error_rate(CAL), "dev": struct_error_rate(DEV)})
            rows.append(r)
            print({x: r[x] for x in ("label", "pass", "rel_norm", "fit_err")}, r["parts"], flush=True)
            json.dump(rows, open("/content/out/sweep.json", "w"), indent=1)
            push("sweep.json")
            if r["parts"]["json"] < r["n"] * 0.5:
                break  # broken at this scale, skip larger ones
        if stop:
            break
    if stop:
        break
restore()
print("best", max(rows, key=lambda r: r["pass"])["label"], flush=True)
