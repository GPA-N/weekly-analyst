"""Print what the lora2 best adapter writes under the failing dev wording NOCUE_DEV[1] (and the passing NOCUE_DEV[0]). Exec'd by wsp_main.py with STAGE=n1diag, env SRC_TAG=lora2."""
import json
import os

import torch

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

pull("lora_ckpt.pt")
best = torch.load("/content/out/lora_ckpt.pt")["best"]
restore()
apply_dw({l: best["snap"]["B"][l].cuda() @ best["snap"]["A"][l].cuda() for l in best["snap"]["A"]})
out = {}
for i in (1, 0):
    ps = [reworded(t["draft_prompt"], NOCUE_DEV[i]) for t in DEV[:4]]
    gens = generate(ps, 700, 4)
    out[f"n{i}"] = [{"text": g[:600], "score": score_draft(g, p)} for g, p in zip(gens, ps)]
    for r in out[f"n{i}"]:
        print(f"--- n{i}", r["score"], flush=True)
        print(r["text"][:600].encode("ascii", "ignore").decode(), flush=True)
json.dump(out, open("/content/out/n1diag.json", "w"), indent=1, default=str)
push("n1diag.json")
restore()
print("DONE-N1DIAG", flush=True)
