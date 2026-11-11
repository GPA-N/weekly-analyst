"""Second, fresh-seed validation of the MEMIT draft edit, plus a rank sweep on DEV. Exec'd by wsp_main.py with STAGE=mfinal2.
Env CFG:
  {"set": "ww_hi", "lam": 3, "scale": 2, "rank": 8, "seeds": [200, 240], "modes": [...], "ib": true}   fresh-seed run
  {"set": "ww_hi", "lam": 3, "scale": 2, "devrank": [1, 2, 4, 8, 16]}                                   rank sweep on DEV only
Modes: base | lowrank_plan (rank-r MEMIT + earlier plan edit) | full_plan (full MEMIT + plan edit) | random_matched
(random shift targets, each layer rescaled to the Frobenius norm of the real edit). Seeds 200-239 were never used before.
"""
import json
import os

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ["CFG"])
layers = LAYERSETS[CFG["set"]]
restore()
coll = collect(CAL, layers)
dws_full = build_edit(coll, layers, CFG["lam"], CFG["scale"])


def lowrank(dws, r):
    out = {}
    for l, dW in dws.items():
        U, S, Vh = torch.linalg.svd(dW, full_matrices=False)
        out[l] = (U[:, :r] * S[:r]) @ Vh[:r]
    return out


def random_matched():
    g = torch.Generator(device="cuda").manual_seed(0)
    rc = {}
    for l in layers:
        K, R, cov, cnt = coll[l]
        rnd = torch.randn(R.shape, generator=g, device=R.device)
        rc[l] = (K, rnd / rnd.norm(dim=-1, keepdim=True) * R.norm(dim=-1, keepdim=True), cov, cnt)
    dws = build_edit(rc, layers, CFG["lam"], CFG["scale"])
    return {l: d * (dws_full[l].norm() / d.norm()) for l, d in dws.items()}


out = {"cfg": CFG}
if "devrank" in CFG:
    rows = []
    for r in CFG["devrank"]:
        restore()
        rel = apply_dw(lowrank(dws_full, r))
        res, raw = run_eval(DEV, f"dev-rank{r}", constrained=False)
        res["rel_norm"] = rel
        res["values_changed"] = r * (1024 + 3584) * len(layers)
        rows.append(res)
        print(res, flush=True)
        out["rows"] = rows
        json.dump(out, open("/content/out/mfinal2.json", "w"), indent=1)
        push("mfinal2.json")
else:
    pull("dirs.pt")
    blob = torch.load("/content/out/dirs.pt")
    tasks = build_tasks(range(*CFG["seeds"]))
    print("fresh tasks", len(tasks), flush=True)
    out["results"] = {}
    for mode in CFG["modes"]:
        restore()
        if mode == "lowrank_plan":
            apply_dw(lowrank(dws_full, CFG["rank"]))
            apply_edit(blob["d"], blob["abar"], [15, 19], 8, 6.0)
        elif mode == "full_plan":
            apply_dw(dws_full)
            apply_edit(blob["d"], blob["abar"], [15, 19], 8, 6.0)
        elif mode == "random_matched":
            apply_dw(random_matched())
        elif mode != "base":
            raise ValueError(mode)
        res, raw = run_eval(tasks, f"fresh-{mode}", constrained=False)
        res["fail_samples"] = [o[:300] for o, s in zip(raw["draft"], raw["draft_scores"]) if not passed(s)][:3]
        out["results"][mode] = res
        print({k: res[k] for k in ("label", "plan_pass", "draft_pass", "n")}, res["draft_parts"], flush=True)
        if mode == "lowrank_plan" and CFG.get("ib"):
            out["ib_lowrank_plan"], _ = run_ib(build_ib_tasks(), "ib-lowrank-plan")
            print(out["ib_lowrank_plan"], flush=True)
        json.dump(out, open("/content/out/mfinal2.json", "w"), indent=1)
        push("mfinal2.json")
restore()
