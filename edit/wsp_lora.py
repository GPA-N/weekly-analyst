"""Small LoRA on the down projections of the language model, tuned on CAL/DEV only, scored once on SCORE. Exec'd by wsp_main.py with STAGE=lora.

Defaults (peft): r=8, alpha=8, dropout 0. Non-defaults, none yet proven: lr 2e-4 (the LoRA/QLoRA papers' convention; peft has no lr), batch 1 with
4-step accumulation, targets limited to mlp.down_proj of every language-model layer so the result merges into a dense delta that the existing
GGUF patch route (weight_patch.add_delta) can apply.
Data: CAL gold plans (plain prompt) and CAL gold drafts under the plain wording, the production repair line and TUNE_WORDINGS. REWORD is never trained on.
Select: epoch with the most DEV passes (plan plain + drafts under plain and tune wordings), ties by lower DEV loss. Score: score_once on SCORE seeds.
Logged per epoch: loss on CAL and DEV (a big CAL drop with a flat DEV loss is the overfit signal). Env CFG: {"epochs": 6, "r": 8, "lr": 2e-4}.
Checkpoints (adapter, optimizer, epoch, history) go to the runs dataset every epoch and are pulled on start.
"""
import json
import os
import random
import time

import torch

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ.get("CFG", "{}"))
EPOCHS, R, LR, ACC = CFG.get("epochs", 6), CFG.get("r", 8), CFG.get("lr", 2e-4), 4
ALPHA = CFG.get("alpha", R)  # peft default: alpha equals r when unset in its examples; scale 1
SCALE = ALPHA / R
BUDGET_S = 50 * 60
t0 = time.time()
random.seed(0)
torch.manual_seed(0)
restore()
for p_ in m.parameters():
    p_.requires_grad_(False)

A = {l: torch.nn.Parameter(torch.empty(R, 3584, device="cuda", dtype=torch.float32)) for l in range(NL)}
B = {l: torch.nn.Parameter(torch.zeros(1024, R, device="cuda", dtype=torch.float32)) for l in range(NL)}
for l in range(NL):
    torch.nn.init.kaiming_uniform_(A[l], a=5 ** 0.5)  # peft's LoRA A init


def _hook(l):
    def hook(mod, inp, out):
        return out + (((inp[0].float() @ A[l].T) @ B[l].T) * SCALE).to(out.dtype)
    return hook


handles = [LM.layers[l].mlp.down_proj.register_forward_hook(_hook(l)) for l in range(NL)]
params = [*A.values(), *B.values()]
opt = torch.optim.AdamW(params, lr=LR)
scaler = torch.amp.GradScaler("cuda")


def loss_of(prompt, comp):
    text = comp.replace("<|im_end|>", "") + "<|im_end|>"
    ids, p = _ids(prompt, text)
    lg = m(input_ids=ids).logits[0, p - 1:ids.shape[1] - 1].float()
    return torch.nn.functional.cross_entropy(lg, ids[0, p:])


@torch.no_grad()
def mean_loss(exs):
    return round(sum(loss_of(p, c).item() for p, c in exs) / len(exs), 4)


def snapshot():
    return {"A": {l: A[l].detach().cpu().clone() for l in A}, "B": {l: B[l].detach().cpu().clone() for l in B}}


def load_snapshot(s):
    for l in A:
        A[l].data.copy_(s["A"][l])
        B[l].data.copy_(s["B"][l])


train, cal_eval = tune_examples(CAL), [(t["plan_prompt"], t["gold_plan"]) for t in CAL[:10]] + [(t["draft_prompt"], t["gold_draft"]) for t in CAL[:10]]
dev_eval_ex = [(t["plan_prompt"], t["gold_plan"]) for t in DEV[:10]] + [(t["draft_prompt"], t["gold_draft"]) for t in DEV[:10]]
hist, best, start = [], None, 0
try:
    pull("lora_ckpt.pt")
    ck = torch.load("/content/out/lora_ckpt.pt")
    load_snapshot(ck["cur"])
    opt.load_state_dict(ck["opt"])
    hist, best, start = ck["hist"], ck["best"], ck["epoch"]
    print("resumed after epoch", start, flush=True)
except Exception as e:
    print("no checkpoint, starting at epoch 0", type(e).__name__, flush=True)

m.eval()  # no dropout in this LoRA; the base model stays in eval mode while A and B train
if start == 0:
    hist.append({"epoch": 0, "cal_loss": mean_loss(cal_eval), "dev_loss": mean_loss(dev_eval_ex), "dev": dev_eval()})
    best = {"epoch": 0, "snap": snapshot(), "dev_total": hist[0]["dev"]["total"], "dev_loss": hist[0]["dev_loss"]}
    print(hist[-1], flush=True)

for ep in range(start + 1, EPOCHS + 1):
    if time.time() - t0 > BUDGET_S:
        print("budget reached, stopping before epoch", ep, flush=True)
        break
    random.shuffle(train)
    run = 0.0
    for i, (p, c) in enumerate(train):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = loss_of(p, c) / ACC
        scaler.scale(loss).backward()
        run += loss.item() * ACC
        if (i + 1) % ACC == 0 or i == len(train) - 1:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
    row = {"epoch": ep, "train_loss": round(run / len(train), 4), "cal_loss": mean_loss(cal_eval), "dev_loss": mean_loss(dev_eval_ex), "dev": dev_eval()}
    hist.append(row)
    if (row["dev"]["total"], -row["dev_loss"]) > (best["dev_total"], -best["dev_loss"]):
        best = {"epoch": ep, "snap": snapshot(), "dev_total": row["dev"]["total"], "dev_loss": row["dev_loss"]}
    print(row, "best epoch", best["epoch"], flush=True)
    torch.save({"cur": snapshot(), "opt": opt.state_dict(), "hist": hist, "best": best, "epoch": ep}, "/content/out/lora_ckpt.pt")
    push("lora_ckpt.pt")
    json.dump({"cfg": CFG, "hist": hist, "best_epoch": best["epoch"]}, open("/content/out/lora_hist.json", "w"), indent=1)
    push("lora_hist.json")

for h in handles:
    h.remove()
load_snapshot(best["snap"])
dws = {l: (SCALE * (B[l].detach() @ A[l].detach())).half() for l in range(NL)}
torch.save({l: d.cpu() for l, d in dws.items()}, "/content/out/lora_dw.pt")
push("lora_dw.pt")
rel = apply_dw({l: d.float() for l, d in dws.items()})  # merged dense delta, the form that ships
print("merged relative weight change", rel, "best epoch", best["epoch"], flush=True)

final = {"cfg": CFG, "best_epoch": best["epoch"], "rel_norm": rel, "hist": hist}
try:
    final["score_base"] = None
    restore()
    final["score_base"] = score_once("base")
except RuntimeError as e:
    final["score_base"] = str(e)
apply_dw({l: d.float() for l, d in dws.items()})
final["score_lora"] = score_once(f"{os.environ.get('TAG', 'run')}-lora")
json.dump(final, open("/content/out/lora_final.json", "w"), indent=1, default=str)
push("lora_final.json")
restore()
print("DONE-LORA", {k: final[k] for k in ("best_epoch", "rel_norm")}, flush=True)
