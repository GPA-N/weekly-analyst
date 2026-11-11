"""Holdout run of the MEMIT draft edit. Exec'd by wsp_main.py with STAGE=mfinal. Run ONCE per mode, after the config was chosen on DEV.
Env CFG: {"set": "ww_hi", "lam": 0.3, "scale": 2, "mode": "edit" | "random" | "combo" | "lowrank", "rank": 8}
  edit     the MEMIT edit alone (plans and drafts on HOLD, then the InsightBench narrative check)
  random   same edit but each target shift replaced by a random direction of the same length (control)
  combo    the MEMIT edit plus the earlier plan edit (layers 15 and 19, 8 columns, alpha 6)
  lowrank  the MEMIT edit truncated to `rank` singular directions per layer
"""
import json
import os

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ["CFG"])
MODE = CFG["mode"]
layers = LAYERSETS[CFG["set"]]
restore()
coll = collect(CAL, layers)
if MODE == "random":
    g = torch.Generator(device="cuda").manual_seed(0)
    for l in layers:
        K, R, cov, cnt = coll[l]
        rnd = torch.randn(R.shape, generator=g, device=R.device)
        coll[l] = (K, rnd / rnd.norm(dim=-1, keepdim=True) * R.norm(dim=-1, keepdim=True), cov, cnt)
dws = build_edit(coll, layers, CFG["lam"], CFG["scale"])
out = {"cfg": CFG}
spec = {}
for l, dW in dws.items():
    sv = torch.linalg.svdvals(dW)
    e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
    spec[l] = {str(r): round(float(e[r - 1]), 4) for r in (1, 2, 4, 8, 16, 32, 64)}
out["energy_by_rank"] = spec
if MODE == "lowrank":
    for l in list(dws):
        U, S, Vh = torch.linalg.svd(dws[l], full_matrices=False)
        r = CFG["rank"]
        dws[l] = (U[:, :r] * S[:r]) @ Vh[:r]
if MODE == "edit":
    torch.save({l: d.half().cpu() for l, d in dws.items()}, "/content/out/dw.pt")
    push("dw.pt")
out["rel_norm"] = apply_dw(dws)
if MODE == "combo":
    pull("dirs.pt")
    blob = torch.load("/content/out/dirs.pt")
    out["plan_edit_changed"] = apply_edit(blob["d"], blob["abar"], [15, 19], 8, 6.0)
res, raw = run_eval(HOLD, f"hold-{MODE}", constrained=False)
out["result"] = res
out["fail_samples"] = [o[:300] for o, s in zip(raw["draft"], raw["draft_scores"]) if not passed(s)][:3]
print(res, flush=True)
if MODE == "edit":
    IB = build_ib_tasks()
    out["ib_edited"], _ = run_ib(IB, "ib-memit")
    print(out["ib_edited"], flush=True)
restore()
json.dump(out, open("/content/out/mfinal.json", "w"), indent=1)
push("mfinal.json")
