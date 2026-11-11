"""Score the best checkpoint saved so far by wsp_lora.py, once, on the SCORE split. Exec'd by wsp_main.py with STAGE=lorascore, env SRC_TAG = tag of the LoRA run.
Merges the best adapter into a dense delta (same form that ships), scores base and the merged model with score_once, writes lorascore_final.json."""
import json
import os

import torch

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ.get("CFG", "{}"))
R = CFG.get("r", 8)
SCALE = CFG.get("alpha", R) / R
pull("lora_ckpt.pt")
ck = torch.load("/content/out/lora_ckpt.pt")
best, hist = ck["best"], ck["hist"]
dws = {l: SCALE * (best["snap"]["B"][l].cuda() @ best["snap"]["A"][l].cuda()) for l in best["snap"]["A"]}
print("scoring best epoch", best["epoch"], "of", ck["epoch"], "trained", flush=True)
restore()
final = {"best_epoch": best["epoch"], "epochs_trained": ck["epoch"], "hist": hist}
ONLY = CFG.get("only")  # "base" or "lora" to run one half; both by default
if ONLY in (None, "base"):
    try:
        final["score_base"] = score_once("base")
    except RuntimeError as e:
        final["score_base"] = str(e)
if ONLY in (None, "lora"):
    final["rel_norm"] = apply_dw(dws)
    final["score_lora"] = score_once(f"{os.environ.get('SRC_TAG', 'run')}-e{best['epoch']}-lora")
name = f"lorascore_{ONLY or 'both'}.json"
json.dump(final, open("/content/out/" + name, "w"), indent=1, default=str)
push(name)
restore()
print("DONE-LORASCORE", ONLY, flush=True)
