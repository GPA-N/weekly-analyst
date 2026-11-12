"""Sparse rewiring on top of the lora3 adapter (maxsloef/owl-rewired-neurons, train/train.py, 'reread' arm). Exec'd by wsp_main.py with STAGE=rewire, env SRC_TAG=lora3.

The lora3 adapter is merged into the weights and frozen. Per layer we add a rank-1 edit to what the MLP neurons READ: gate and up get z_i * b_i * (x . a)
with a shared read vector a, a per-neuron coefficient b (zero start, so the model starts exactly as lora3) and a hard-concrete on/off gate z_i with an L0
penalty that switches neurons off. down_proj (what a neuron writes) is untouched. Defaults taken from train.py: lr 1e-3, mask lr 0.1, L0 weight 1e-4,
AdamW (0.9, 0.99) no weight decay, 20 warmup steps then cosine, grad clip 1.0, 300 steps of 8 examples, hard-concrete BETA 2/3 GAMMA -0.1 ZETA 1.1,
mask start 1.0. NOT defaults (mine, unproven): penalty off for the first 50 steps then ramped over 100 (train.py's dense-warmup default is unknown to me),
batch of 8 built by gradient accumulation of single examples, selection on DEV by dev_eval(), evaluated every 50 steps.
Data: tune_examples(CAL) plus CAL drafts under SYN2_FIT (wordings lora3 has not fitted; its loss on the others is 0, so they would give no gradient).
Never scores on SCORE; dev only. Env CFG: {"steps": 300, "l0": 1e-4}.
"""
import json
import math
import os
import random
import time

import torch

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ.get("CFG", "{}"))
STEPS, L0, ACC, LR, MLR = CFG.get("steps", 300), CFG.get("l0", 1e-4), 8, 1e-3, 0.1
WARM, DENSE, RAMP, BUDGET_S = 20, 50, 100, 50 * 60
BETA, GAMMA, ZETA = 2 / 3, -0.1, 1.1
t0 = time.time()
random.seed(0)
torch.manual_seed(0)

pull("lora_ckpt.pt")  # SRC_TAG = the lora3 run
best_ck = torch.load("/content/out/lora_ckpt.pt")["best"]
restore()
apply_dw({l: best_ck["snap"]["B"][l].cuda() @ best_ck["snap"]["A"][l].cuda() for l in best_ck["snap"]["A"]})  # lora3 merged, now the frozen base
for p_ in m.parameters():
    p_.requires_grad_(False)

D, N = LM.layers[0].mlp.gate_proj.weight.shape[1], LM.layers[0].mlp.gate_proj.weight.shape[0]
P = {l: {"a_g": torch.nn.Parameter(torch.randn(D, device="cuda") / D ** 0.5), "a_u": torch.nn.Parameter(torch.randn(D, device="cuda") / D ** 0.5),
         "b_g": torch.nn.Parameter(torch.zeros(N, device="cuda")), "b_u": torch.nn.Parameter(torch.zeros(N, device="cuda")),
         "la": torch.nn.Parameter(torch.full((N,), 1.0, device="cuda"))} for l in range(NL)}
Z = {}
TRAIN = [True]


def gate(la):
    if TRAIN[0]:
        u = torch.rand_like(la).clamp(1e-6, 1 - 1e-6)
        s = torch.sigmoid((u.log() - (1 - u).log() + la) / BETA)
    else:
        s = torch.sigmoid(la)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0, 1)


def expected_on(la):
    return torch.sigmoid(la - BETA * math.log(-GAMMA / ZETA)).sum()


def active():
    return sum(int((gate(P[l]["la"]) > 0).sum()) for l in P)


def pre(l):
    def hook(mod, inp):
        Z[l] = gate(P[l]["la"])
    return hook


def edit(l, a, b):
    def hook(mod, inp, out):
        return out + ((inp[0].float() @ P[l][a])[..., None] * (Z[l] * P[l][b])).to(out.dtype)
    return hook


handles = []
for l in range(NL):
    mlp = LM.layers[l].mlp
    handles += [mlp.register_forward_pre_hook(pre(l)), mlp.gate_proj.register_forward_hook(edit(l, "a_g", "b_g")),
                mlp.up_proj.register_forward_hook(edit(l, "a_u", "b_u"))]


def loss_of(prompt, comp):
    text = comp.replace("<|im_end|>", "") + "<|im_end|>"
    ids, p = _ids(prompt, text)
    lg = m(input_ids=ids).logits[0, p - 1:ids.shape[1] - 1].float()
    return torch.nn.functional.cross_entropy(lg, ids[0, p:])


@torch.no_grad()
def mean_loss(exs):
    TRAIN[0] = False
    out = round(sum(loss_of(p, c).item() for p, c in exs) / len(exs), 4)
    TRAIN[0] = True
    return out


def evaluate(step):
    TRAIN[0] = False
    row = {"step": step, "active": active(), "dev": dev_eval()}
    TRAIN[0] = True
    row["syn2_dev_loss"] = mean_loss(syn2_eval)
    return row


syn2_ex = [(reworded(t["draft_prompt"], w), t["gold_draft"]) for t in CAL for w in SYN2_FIT]
syn2_eval = [(reworded(t["draft_prompt"], w), t["gold_draft"]) for t in DEV[:5] for w in NOCUE_DEV]
train = tune_examples(CAL)  # FIT_WORDINGS already holds SYN2_FIT
dense = [p for l in P for k, p in P[l].items() if k != "la"]
masks = [P[l]["la"] for l in P]
opt = torch.optim.AdamW([{"params": dense, "lr": LR}, {"params": masks, "lr": MLR}], betas=(0.9, 0.99), weight_decay=0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / WARM) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / STEPS))))
scaler = torch.amp.GradScaler("cuda")
m.eval()

hist = [evaluate(0)]
best = {"step": 0, "total": hist[0]["dev"]["total"], "loss": hist[0]["syn2_dev_loss"], "state": None}
print(hist[0], flush=True)
random.shuffle(train)
cursor = 0
for step in range(1, STEPS + 1):
    if time.time() - t0 > BUDGET_S:
        print("budget reached before step", step, flush=True)
        break
    l0_w = L0 * (0.0 if step <= DENSE else min(1.0, (step - DENSE) / RAMP))
    for _ in range(ACC):
        p, c = train[cursor % len(train)]
        cursor += 1
        with torch.autocast("cuda", dtype=torch.float16):
            ce = loss_of(p, c)
        pen = l0_w * 2 * sum(expected_on(P[l]["la"]) for l in P)
        scaler.scale((ce + pen) / ACC).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(dense, 1.0)  # masks are not clipped, as in train.py
    scaler.step(opt)
    scaler.update()
    opt.zero_grad(set_to_none=True)
    sched.step()
    if step % 50 == 0 or step == STEPS:
        row = evaluate(step)
        hist.append(row)
        if (row["dev"]["total"], -row["syn2_dev_loss"]) > (best["total"], -best["loss"]):
            best = {"step": step, "total": row["dev"]["total"], "loss": row["syn2_dev_loss"], "state": {l: {k: v.detach().cpu().clone() for k, v in P[l].items()} for l in P}}
        print(row, "best step", best["step"], flush=True)
        json.dump({"cfg": CFG, "hist": hist, "best_step": best["step"]}, open("/content/out/rewire_hist.json", "w"), indent=1)
        push("rewire_hist.json")
        if best["state"] is not None:
            torch.save(best["state"], "/content/out/rewire_best.pt")
            push("rewire_best.pt")
for h in handles:
    h.remove()
restore()
print("DONE-REWIRE", {"best_step": best["step"], "best_dev_total": best["total"]}, flush=True)
