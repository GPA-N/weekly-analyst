"""Out-of-distribution check of the draft edits. Exec'd by wsp_main.py with STAGE=ood. Drafts only, no grammar, greedy.

Fresh seeds 200-239 (never used to fit or select anything for these variants), the draft prompt changed in ways the edit never saw:
  plain      the unchanged prompt (in-distribution reference)
  reword     the instruction paragraph rewritten (same meaning, no schema key names spelled out)
  repair     the production re-request line appended to the prompt
  one_item   evidence cut to a single item
  renamed    metrics, columns and segment values replaced by different words (item keys change with them)
Models: base (control), gold (the gold-draft MEMIT edit from the f1 run) and onpolicy (round R of the on-policy run).
Env CFG: {"set": "ww_hi", "onpolicy_tag": "o1", "round": 3, "seeds": [200, 240], "n": 40}
"""
import json
import os
import re

from huggingface_hub import hf_hub_download

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ["CFG"])
layers = LAYERSETS[CFG["set"]]
tasks = build_tasks(range(*CFG["seeds"]))[: CFG["n"]]
sets = {k: [] for k in ("plain", *HELD_WORDINGS, "repair", "one_item", "renamed")}
for t in tasks:
    for k, p in variants(t["draft_prompt"]).items():
        sets[k].append(p)
assert all(REWORD in p for p in sets["reword"]), "reword did not apply"


def load_dws(repo_path):
    blob = torch.load(hf_hub_download(RUNS, repo_path, repo_type="dataset", token=os.environ["HF_TOKEN"]))
    return {int(l): t.float().cuda() for l, t in blob.items()}


models = {}
if CFG.get("base", True):
    models["base"] = None
if CFG.get("gold", True):
    models["gold"] = load_dws("f1/dw.pt")
if CFG.get("onpolicy_tag"):
    models["onpolicy"] = load_dws(f"{CFG['onpolicy_tag']}/dw_round{CFG['round']}.pt")
out = {"cfg": CFG, "n": len(tasks), "results": {}}
for name, dws in models.items():
    restore()
    if dws is not None:
        apply_dw(dws)
    out["results"][name] = {}
    for k, prompts in sets.items():
        raws = generate(prompts, max_new=700, batch=8)
        sc = [score_draft(o, p) for o, p in zip(raws, prompts)]
        fails = [o[:200] for o, s in zip(raws, sc) if not passed(s)][:2]
        out["results"][name][k] = {"pass": sum(map(passed, sc)), "n": len(sc), "parts": {x: sum(s[x] for s in sc) for x in sc[0]}, "fail_samples": fails}
        print(name, k, out["results"][name][k]["pass"], "/", len(sc), flush=True)
        json.dump(out, open("/content/out/ood.json", "w"), indent=1)
        push("ood.json")
restore()
print("DONE-OOD", flush=True)
