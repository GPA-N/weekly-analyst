"""On-policy rounds of the MEMIT draft edit (no gold drafts). Exec'd by wsp_main.py with STAGE=onpolicy.

Teacher = the frozen base model with the draft schema written into the prompt (it passes 40/40 drafts on dev). Student = the model being
edited, prompt without the schema. Each round: (1) the current student samples its own draft for every calibration prompt, (2) the frozen
base model with the schema reads the student's own text and gives its top-1 token at every position, (3) at structural JSON positions where
the student's top-1 differs, the same closed-form MEMIT update as before is fitted (keys from the student, target from the teacher) and
added to the running edit. Env CFG: {"set": "ww_hi", "lam": 3, "scale": 2, "rounds": 3, "seeds": [0, 20]}.
Only CAL seeds (0-19) feed the edit; DEV (20-39) scores each round; nothing else is touched.
"""
import json
import os
import time

exec(compile(open("/content/pkg/colab/wsp_memit_core.py").read(), "wsp_memit_core.py", "exec"), globals())

CFG = json.loads(os.environ["CFG"])
layers = LAYERSETS[CFG["set"]]
t0 = time.time()
schema_txt = json.dumps(draft_mod.DraftFields.model_json_schema())


def with_schema(p):
    return p + "\nThe JSON object must match this schema exactly: " + schema_txt


@torch.no_grad()
def teacher_targets(prompt, comp):
    ids, p = _ids(with_schema(prompt), comp)
    n = len(tok(comp.replace("<|im_end|>", "") + "<|im_end|>", add_special_tokens=False).input_ids)
    out = m(input_ids=ids, logits_to_keep=n + 1)
    return out.logits[0, :-1][:n].argmax(-1).cpu()


cum = {l: torch.zeros(1024, 3584, device="cuda") for l in layers}
rounds = []
for r in range(CFG["rounds"]):
    restore()
    apply_dw(cum)
    texts = generate([t["draft_prompt"] for t in CAL], max_new=700, batch=8)
    sc = [score_draft(o, t["draft_prompt"]) for o, t in zip(texts, CAL)]
    n_pass = sum(map(passed, sc))
    restore()  # teacher: base weights plus the schema in the prompt
    tops = [teacher_targets(t["draft_prompt"], o) for t, o in zip(CAL, texts)]
    restore()
    apply_dw(cum)  # back to the student
    items = [{"prompt": t["draft_prompt"], "comp": o, "target": tg} for t, o, tg in zip(CAL, texts, tops)]
    coll = collect_items(items, layers)
    new = build_edit(coll, layers, CFG["lam"], CFG["scale"])
    for l in layers:
        cum[l] = cum[l] + new[l]
    restore()
    rel = apply_dw(cum)
    dev = dev_drafts(f"onpolicy-round{r + 1}")
    dev.update(student_cal_pass_before=n_pass, rel_norm=rel, elapsed=round(time.time() - t0))
    rounds.append(dev)
    print({k: dev[k] for k in ("label", "pass", "student_cal_pass_before", "rel_norm")}, dev["parts"], flush=True)
    torch.save({l: d.half().cpu() for l, d in cum.items()}, f"/content/out/dw_round{r + 1}.pt")
    push(f"dw_round{r + 1}.pt")
    json.dump({"cfg": CFG, "rounds": rounds}, open("/content/out/onpolicy.json", "w"), indent=1)
    push("onpolicy.json")
restore()
print("DONE-ONPOLICY", flush=True)
