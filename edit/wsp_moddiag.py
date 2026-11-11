"""Show what the saved module (SRC_TAG/module_ab.pt) writes on DEV drafts under DEV_WORDINGS, with the scorer's parts. Exec'd with STAGE=moddiag."""
import json
import os

import torch

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

pull("module_ab.pt")
blob = torch.load("/content/out/module_ab.pt")
restore()
hs = [LM.layers[l].mlp.down_proj.register_forward_hook(
    (lambda A, B: lambda mod, inp, out: out + (blob["gamma"] * (inp[0].float() @ A.T.cuda()) @ B.T.cuda()).to(out.dtype))(A, B)) for l, (A, B) in blob["mods"].items()]
out = {}
for name, w in (("plain", None), ("d0", DEV_WORDINGS[0]), ("d1", DEV_WORDINGS[1])):
    ps = [t["draft_prompt"] if w is None else reworded(t["draft_prompt"], w) for t in DEV[:4]]
    raws = generate(ps, 700, 4)
    out[name] = [{"parts": score_draft(o, p), "head": o[:350], "tail": o[-120:]} for o, p in zip(raws, ps)]
    print(name, [r["parts"] for r in out[name]], flush=True)
    print("SAMPLE", name, repr(out[name][0]["head"]), "...", repr(out[name][0]["tail"]), flush=True)
json.dump(out, open("/content/out/moddiag.json", "w"), indent=1, default=str)
push("moddiag.json")
print("DONE-MODDIAG", flush=True)
