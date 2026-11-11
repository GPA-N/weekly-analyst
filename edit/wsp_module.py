"""Weight-space programming with an attached module: the base weights are never written. Exec'd by wsp_main.py with STAGE=module.

A low-rank module (A [r,3584], B [1024,r]) is attached to mlp.down_proj of the chosen layers through a forward hook, the same shape as a LoRA,
but its values are not trained. They are edited in closed form (the MEMIT-style least-squares solve in wsp_memit_core.py: at the structural
positions where the model ranks a gold token wrong, fit the residual shift) and then cut to rank r by SVD, so B = U sqrt(S), A = sqrt(S) V^T.
Detach the hooks and the model is exactly the base model.
Fit data: CAL gold plans and drafts under the plain wording, the repair line and TUNE_WORDINGS (REWORD stays held out). Select the rank on DEV with
dev_eval(). Score once on SCORE with score_once. Env CFG: {"set": "ww_hi", "lam": 3, "scale": 2, "ranks": [4, 8, 16], "gammas": [1.0], "score": true}. With "score": false the stage
only sweeps and saves; wsp_modscore.py then scores the one config picked on DEV, so SCORE is never used to choose between configs.
lam 3, scale 2 and the layer set ww_hi were chosen on DEV earlier (colab/wsp_memit.py); they are carried over, not re-proven here.
"""
import json
import os

import torch

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ.get("CFG", "{}"))
SET, LAM, SCALE, RANKS = CFG.get("set", "ww_hi"), CFG.get("lam", 3), CFG.get("scale", 2), CFG.get("ranks", [4, 8, 16])
GAMMAS = CFG.get("gammas", [1.0])  # inference-time strength of the module, 1.0 = as fitted, 0 = base (LARA, 2607.28669)
layers = LAYERSETS[SET]
restore()

coll = collect_items([{"prompt": p, "comp": c} for p, c in tune_examples(CAL)], layers)
dws = build_edit(coll, layers, LAM, SCALE)  # {layer: dW [1024, 3584]}, fp32
svd = {l: torch.linalg.svd(dws[l], full_matrices=False) for l in layers}
print("fitted", {l: round(dws[l].norm().item(), 3) for l in layers}, flush=True)


def module(r):
    mods = {}
    for l in layers:
        U, S, Vh = svd[l]
        root = S[:r].sqrt()
        mods[l] = (Vh[:r] * root[:, None], U[:, :r] * root[None, :])  # A [r, 3584], B [1024, r]
    return mods


def attach(mods, gamma=1.0):
    def mk(l):
        A, B = mods[l]
        return lambda mod, inp, out: out + (gamma * (inp[0].float() @ A.T) @ B.T).to(out.dtype)
    return [LM.layers[l].mlp.down_proj.register_forward_hook(mk(l)) for l in mods]


def detach(handles):
    for h in handles:
        h.remove()


rows = []
for r, g in [(r, g) for r in RANKS for g in GAMMAS]:
    mods = module(r)
    kept = sum(float(svd[l][1][:r].pow(2).sum() / svd[l][1].pow(2).sum()) for l in layers) / len(layers)
    hs = attach(mods, g)
    row = {"rank": r, "gamma": g, "energy_kept": round(kept, 4), "cal_struct_err": struct_error_rate(CAL), "dev_struct_err": struct_error_rate(DEV), "dev": dev_eval()}
    detach(hs)
    rows.append(row)
    print(row, flush=True)
    json.dump({"cfg": CFG, "rows": rows}, open("/content/out/module_sweep.json", "w"), indent=1)
    push("module_sweep.json")

best = max(rows, key=lambda x: (x["dev"]["total"], -x["rank"], -abs(x["gamma"] - 1)))
mods = module(best["rank"])
torch.save({"gamma": best["gamma"], "mods": {l: (A.cpu(), B.cpu()) for l, (A, B) in mods.items()}}, "/content/out/module_ab.pt")
push("module_ab.pt")
final = {"cfg": CFG, "rows": rows, "best_rank": best["rank"], "best_gamma": best["gamma"]}
if CFG.get("score", True):
    try:
        final["score_base"] = score_once("base")
    except RuntimeError as e:
        final["score_base"] = str(e)
    hs = attach(mods, best["gamma"])
    final["score_module"] = score_once(f"{os.environ.get('TAG', 'run')}-module")
    detach(hs)
json.dump(final, open("/content/out/module_final.json", "w"), indent=1, default=str)
push("module_final.json")
print("DONE-MODULE", {"best_rank": best["rank"], "best_gamma": best["gamma"]}, flush=True)
