"""Gradient-free sparse edit of down_proj with an explicit "leave the rest alone" constraint, on top of the merged lora3 adapter.
Exec'd by wsp_main.py with STAGE=lp, env SRC_TAG=lora3. No optimizer, no loss, no training loop.

Method (nathanrs/weight-space-lp, after Takehi et al. "Sparks of Weight-Space Programming"): the edit to a layer's down_proj is rank-1, dW = u v^T,
with u a fixed residual direction and v a sparse per-neuron vector found by a linear program:
    min |v|_1 + C * sum(slack)   s.t.   v.k_t + slack_t >= c_t   at the positions R where the model writes a wrong structural token
                                        |v.k_t| <= eps          at positions P where it is already right (plans, drafts, other wordings)
k_t is the MLP input at position t (the input of down_proj), u is the top direction of the residual shifts that would fix the wrong positions
(the same gradient-direction shifts as wsp_memit_core.collect_items), c_t = (shift_t . u) * scale / (number of layers).
R comes from CAL drafts under SYN2_FIT wordings, which lora3 was not trained on and still gets wrong. P comes from correct positions of CAL plans and drafts.
Candidate neurons: the 512 with the largest mean |k| on R. Solver: scipy HiGHS. Sweep (layer set, scale, eps) on DEV with dev_eval(); the held-out
n1 and n2 wordings are in dev_eval, never in R or P. Never scores on SCORE. From nathanrs/weight-space-lp scripts/e2_sparsity_sweep.py: C=10, support = top-k by |v|*std(h) of the L1 solve (k in 8, 32), re-solve on the support with L1 cost 1e-3.
Mine, unproven: 512 pre-filtered candidates (scipy on CPU; the repo solves all columns on GPU), eps as a fraction of mean c, c_t = per-position wanted shift along u.
Env CFG: {"sets": ["l15_19", "ww_hi"], "scales": [2, 4], "eps": [0.1], "ks": [8, 32], "n_tasks": 20, "n_words": 16, "budget_min": 50}
"""
import json
import os
import random
import time

import numpy as np
import torch
from scipy.optimize import linprog

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ.get("CFG", "{}"))
SETS, SCALES, EPS, KS = CFG.get("sets", ["l15_19", "ww_hi"]), CFG.get("scales", [2, 4]), CFG.get("eps", [0.1]), CFG.get("ks", [8, 32])
NT, NW, CAND, C_SLACK = CFG.get("n_tasks", 20), CFG.get("n_words", 16), 512, 10.0
t0 = time.time()
random.seed(0)

pull("lora_ckpt.pt")  # SRC_TAG = the lora3 run
best_ck = torch.load("/content/out/lora_ckpt.pt")["best"]
restore()
apply_dw({l: best_ck["snap"]["B"][l].cuda() @ best_ck["snap"]["A"][l].cuda() for l in best_ck["snap"]["A"]})  # lora3 merged = the base of this stage
LAYERS = sorted({l for s in SETS for l in LAYERSETS[s]})
SNAP = {l: LM.layers[l].mlp.down_proj.weight.detach().clone() for l in LAYERS}  # lora3 weights, to undo each trial edit exactly


@torch.no_grad()
def collect_correct(items, layers, per_item, seed=0):
    """MLP inputs (down_proj inputs) at structural positions the model already gets right: the 'leave alone' set P."""
    rnd = random.Random(seed)
    keys = {l: [] for l in layers}
    for t in items:
        ids, p = _ids(t["prompt"], t["comp"])
        text = t["comp"].replace("<|im_end|>", "") + "<|im_end|>"
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        sm = _struct_chars(text)
        smask = torch.tensor([any(sm[c] for c in range(a, b)) if b > a else True for a, b in enc["offset_mapping"]], device="cuda")
        n = len(smask)
        acts = {}

        def mk(l):
            def hook(mod, inp):
                acts[l] = inp[0][0].detach().float()[p - 1:p - 1 + n]
            return hook

        hooks = [LM.layers[l].mlp.down_proj.register_forward_pre_hook(mk(l)) for l in layers]
        out = m(input_ids=ids)
        for h in hooks:
            h.remove()
        ok = ((out.logits[0, p - 1:p - 1 + n].argmax(-1) == ids[0, p:p + n]) & smask).nonzero()[:, 0].tolist()
        pick = rnd.sample(ok, min(per_item, len(ok)))
        for l in layers:
            keys[l].append(acts[l][pick])
    return {l: torch.cat(keys[l]) for l in layers}


tasks = CAL[:NT]
r_items = [{"prompt": reworded(t["draft_prompt"], w), "comp": t["gold_draft"]} for t in tasks for w in SYN2_FIT[:NW]]
p_items = [{"prompt": t["plan_prompt"], "comp": t["gold_plan"]} for t in tasks]
p_items += [{"prompt": reworded(t["draft_prompt"], w), "comp": t["gold_draft"]} for t in tasks for w in random.sample([x for x in FIT_WORDINGS if x not in SYN2_FIT], 4)]
coll = collect_items(r_items, LAYERS)  # wrong positions: K [n,3584], R [n,1024]
KP = collect_correct(p_items, LAYERS, per_item=20)
print("R positions", {l: coll[l][0].shape[0] for l in LAYERS}, "P positions", {l: KP[l].shape[0] for l in LAYERS}, flush=True)


def solve_layer(l, scale, eps_frac, n_layers):
    K, R = coll[l][0], coll[l][1]
    if K.shape[0] == 0:
        return None
    sel = torch.randperm(K.shape[0], generator=torch.Generator().manual_seed(l))[:600].to(K.device)  # shortcut: LP size, 600 wrong and 800 right positions
    K, R = K[sel], R[sel]
    kp_all = KP[l][torch.randperm(KP[l].shape[0], generator=torch.Generator().manual_seed(l))[:800].to(K.device)]
    u = torch.linalg.svd(R, full_matrices=False)[2][0]  # top direction of the wanted residual shifts [1024]
    if (R @ u).mean() < 0:
        u = -u
    c = ((R @ u).clamp(min=0) * scale / n_layers).cpu().numpy()
    cand = K.abs().mean(0).topk(CAND).indices
    kr, kp = K[:, cand].cpu().numpy(), kp_all[:, cand].cpu().numpy()
    nr, M = kr.shape[0], len(cand)
    eps = eps_frac * float(c.mean())
    # x = [v+ (M), v- (M), slack (nr)]
    A = np.vstack([np.hstack([-kr, kr, -np.eye(nr)]), np.hstack([kp, -kp, np.zeros((kp.shape[0], nr))]), np.hstack([-kp, kp, np.zeros((kp.shape[0], nr))])])
    b = np.concatenate([-c, np.full(2 * kp.shape[0], eps)])
    obj = np.concatenate([np.ones(2 * M), np.full(nr, C_SLACK)])
    res = linprog(obj, A_ub=A, b_ub=b, bounds=(0, None), method="highs", options={"time_limit": 180})
    if res.x is None:
        return None
    v1 = res.x[:M] - res.x[M:2 * M]
    std = np.vstack([kr, kp]).std(0)
    out = {}
    for k in KS:  # top-k by |v|*std(h), then re-solve on that support with a tiny L1 cost (weight-space-lp e2_sparsity_sweep.py)
        sup = np.sort(np.argsort(-np.abs(v1) * std)[:k])
        A2 = np.vstack([np.hstack([-kr[:, sup], kr[:, sup], -np.eye(nr)]), np.hstack([kp[:, sup], -kp[:, sup], np.zeros((kp.shape[0], nr))]),
                        np.hstack([-kp[:, sup], kp[:, sup], np.zeros((kp.shape[0], nr))])])
        r2 = linprog(np.concatenate([np.full(2 * k, 1e-3), np.full(nr, C_SLACK)]), A_ub=A2, b_ub=b, bounds=(0, None), method="highs", options={"time_limit": 180})
        if r2.x is None:
            continue
        v = np.zeros(3584, dtype=np.float32)
        v[cand.cpu().numpy()[sup]] = (r2.x[:k] - r2.x[k:2 * k]).astype(np.float32)
        out[k] = (torch.outer(u, torch.from_numpy(v).cuda()), {"nnz": int((np.abs(v) > 1e-6).sum()), "slack": round(float(r2.x[2 * k:].sum()), 3), "status": r2.status})
    return out or None


def undo():
    with torch.no_grad():
        for l in LAYERS:
            LM.layers[l].mlp.down_proj.weight.data.copy_(SNAP[l])


rows = [{"label": "lora3 only", "dev": dev_eval()}]
print(rows[0], flush=True)
best = {"total": rows[0]["dev"]["total"], "rel": 0.0, "dws": None, "label": "lora3 only"}
for s in SETS:
    for scale in SCALES:
        for e in EPS:
            sols = {l: solve_layer(l, scale, e, len(LAYERSETS[s])) for l in LAYERSETS[s]}
            for k in KS:
                label = f"{s} scale={scale} eps={e} k={k}"
                dws = {l: o[k][0] for l, o in sols.items() if o and k in o}
                info = {l: o[k][1] for l, o in sols.items() if o and k in o}
                if not dws:
                    rows.append({"label": label, "skipped": "no wrong positions or no LP solution"})
                    continue
                rel = apply_dw(dws)
                row = {"label": label, "rel_change": rel, "lp": info, "dev": dev_eval()}
                undo()
                rows.append(row)
                print(row, flush=True)
                if (row["dev"]["total"], -rel) > (best["total"], -best["rel"]):
                    best = {"total": row["dev"]["total"], "rel": rel, "dws": {l: d.half().cpu() for l, d in dws.items()}, "label": label}
                json.dump({"cfg": CFG, "rows": rows, "best": best["label"]}, open("/content/out/lp_sweep.json", "w"), indent=1, default=str)
                push("lp_sweep.json")
                if best["dws"] is not None:
                    torch.save(best["dws"], "/content/out/lp_best_dw.pt")
                    push("lp_best_dw.pt")
        if time.time() - t0 > CFG.get("budget_min", 50) * 60:
            break

# Norm-matched random control (concept.country, "Gradient-free weight editing reduces overthinking"): same sparse v and same size per layer, but u replaced by a
# random unit vector orthogonal to the fitted one. If the fitted edit beats this on dev, the direction did the work, not the size of the perturbation.
if best["dws"] is not None:
    g = torch.Generator(device="cuda").manual_seed(0)
    ctrl = {}
    for l, d in best["dws"].items():
        U, S, Vh = torch.linalg.svd(d.float().cuda(), full_matrices=False)
        r = torch.randn(U.shape[0], device="cuda", generator=g)
        r = r - (r @ U[:, 0]) * U[:, 0]
        ctrl[l] = torch.outer(r / r.norm(), S[0] * Vh[0])
    rel = apply_dw(ctrl)
    crow = {"label": "control of " + best["label"], "rel_change": rel, "dev": dev_eval()}
    undo()
    rows.append(crow)
    print(crow, flush=True)
    json.dump({"cfg": CFG, "rows": rows, "best": best["label"]}, open("/content/out/lp_sweep.json", "w"), indent=1, default=str)
    push("lp_sweep.json")
print("DONE-LP", {"best": best["label"], "best_dev_total": best["total"], "secs": round(time.time() - t0)}, flush=True)
