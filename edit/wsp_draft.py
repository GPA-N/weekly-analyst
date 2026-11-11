"""Draft-format edit sweep: bake the draft schema into the weights, so the draft prompt without the schema yields valid DraftFields.

Exec'd by wsp_main.py with STAGE=draft. Globals available: m, tok, LM, NL, CAL, DEV, HOLD, generate, score_draft, passed, trace,
directions, apply_edit, restore, _orig, draft_mod, push. Env CFG (json) selects the arm:
  {"method": "add_good" | "add_ctx" | "flip", "sets": [names], "k": [ints], "alphas": [floats]}
Selection happens on DEV (seeds 20-39); directions are fitted on CAL (seeds 0-19). Nothing here touches HOLD.
"""
import json
import os
import time

import torch

CFG = json.loads(os.environ["CFG"])
METHOD = CFG["method"]
LAYERSETS = {
    "l15_19": [15, 19],
    "mid6": [10, 12, 14, 16, 18, 20],
    "ww_hi": [16, 17, 18, 20, 21, 22],  # WeightWatcher alpha above 9 on down_proj
    "ww_lo": [9, 10, 11, 12, 13],  # WeightWatcher alpha 4.0 to 4.7 (best trained)
}
BUDGET_S = 40 * 60
t0 = time.time()
schema_txt = json.dumps(draft_mod.DraftFields.model_json_schema())


def with_schema(p):
    return p + "\nThe JSON object must match this schema exactly: " + schema_txt


def dev_drafts(label):
    out = generate([t["draft_prompt"] for t in DEV], max_new=700, batch=8)
    sc = [score_draft(o, t["draft_prompt"]) for o, t in zip(out, DEV)]
    fails = [o for o, s in zip(out, sc) if not passed(s)]
    return {"label": label, "pass": sum(map(passed, sc)), "n": len(sc), "parts": {k: sum(s[k] for s in sc) for k in sc[0]},
            "fail_sample": (fails[0][:300] if fails else "")}


# 1. calibration: failing base drafts on CAL, gold drafts, and traces with and without the schema in context
restore()
bad = generate([t["draft_prompt"] for t in CAL], max_new=700, batch=8)
print("cal base draft pass", sum(passed(score_draft(o, t["draft_prompt"])) for o, t in zip(bad, CAL)), "of", len(CAL), flush=True)
hg, ag, hb, hc = [], [], [], []
for t, b in zip(CAL, bad):
    h, a = trace(t["draft_prompt"], t["gold_draft"])
    hg.append(h); ag.append(a)
    hb.append(trace(t["draft_prompt"], b)[0])
    if METHOD == "add_ctx":
        hc.append(trace(with_schema(t["draft_prompt"]), t["gold_draft"])[0])
HG, AG, HB = torch.stack(hg), torch.stack(ag), torch.stack(hb)
S = HG.norm(dim=-1).mean(0)  # mean residual norm per layer
ABAR = AG.mean(0)
if METHOD == "add_good":
    V = (HG - HB).mean(0)
elif METHOD == "add_ctx":
    V = (torch.stack(hc) - HG).mean(0)
if METHOD in ("add_good", "add_ctx"):
    U = V / V.norm(dim=-1, keepdim=True)
else:
    pairs = [(t["draft_prompt"], t["gold_draft"], b) for t, b in zip(CAL, bad)]
    U, ABAR_F = directions(pairs)
torch.save({"U": U.cpu(), "ABAR": ABAR.cpu(), "S": S.cpu(), "ABAR_F": (ABAR_F.cpu() if METHOD == "flip" else None), "cfg": CFG}, "/content/out/dirs.pt")
push("dirs.pt")


@torch.no_grad()
def apply_add(layers, k, alpha):
    """Adds alpha * (mean residual norm) * u along the direction whenever the chosen MLP columns fire like they do on gold drafts."""
    changed = 0
    for l in layers:
        W = LM.layers[l].mlp.down_proj.weight
        if l not in _orig:
            _orig[l] = W.detach().clone()
        a = ABAR[l].to(W.device).float()
        top = torch.topk(a.abs(), k).indices
        delta = alpha * S[l].item() * torch.outer(U[l].to(W.device).float(), a[top]) / (a[top] ** 2).sum()
        W.data[:, top] = (W[:, top].float() + delta).to(W.dtype)
        changed += int(top.numel()) * W.shape[0]
    return changed


rows = []
restore()
base = dev_drafts("dev-base")
rows.append(base)
print({k: base[k] for k in ("label", "pass", "parts")}, flush=True)
done = False
for name in CFG["sets"]:
    for k in CFG["k"]:
        for alpha in CFG["alphas"]:
            if time.time() - t0 > BUDGET_S:
                done = True
                break
            restore()
            changed = apply_add(LAYERSETS[name], k, alpha) if METHOD != "flip" else apply_edit(U, ABAR_F, LAYERSETS[name], k, alpha)
            r = dev_drafts(f"{METHOD}-{name}-k{k}-a{alpha}")
            r.update(changed=changed, elapsed=round(time.time() - t0))
            rows.append(r)
            print({x: r[x] for x in ("label", "pass", "changed")}, r["parts"], flush=True)
            json.dump(rows, open("/content/out/sweep.json", "w"), indent=1)
            push("sweep.json")
            if r["parts"]["json"] < r["n"] * 0.5:
                break  # model broken at this strength, skip larger alphas
        if done:
            break
    if done:
        break
restore()
print("best", max(rows, key=lambda r: (r["pass"], -r.get("changed", 0)))["label"], flush=True)
