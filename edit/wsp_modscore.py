"""Score one saved module (module_ab.pt of run SRC_TAG) once on the SCORE split. Exec'd by wsp_main.py with STAGE=modscore."""
import json
import os

import torch

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

pull("module_ab.pt")
blob = torch.load("/content/out/module_ab.pt")
mods = {l: (A.cuda(), B.cuda()) for l, (A, B) in blob["mods"].items()}
restore()
hs = [LM.layers[l].mlp.down_proj.register_forward_hook(
    (lambda A, B: lambda mod, inp, out: out + (blob["gamma"] * (inp[0].float() @ A.T) @ B.T).to(out.dtype))(*mods[l])) for l in mods]
res = score_once(f"{os.environ.get('SRC_TAG', 'run')}-module")
for h in hs:
    h.remove()
json.dump({"gamma": blob["gamma"], "score": res["variants"], "plan_pass": res["plan_pass"], "draft_pass": res["draft_pass"], "n": res["n"]},
          open("/content/out/modscore.json", "w"), indent=1)
push("modscore.json")
print("DONE-MODSCORE", flush=True)
