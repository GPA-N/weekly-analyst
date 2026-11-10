import json, itertools, time
t0 = time.time()
# 1. failing completions of the unedited model on the calibration tasks
res_c, raw_c = run_eval(CAL, "cal-base", constrained=False)
print(res_c, flush=True)
pairs = []
for i, t in enumerate(CAL):
    if not passed(raw_c["plan_scores"][i]):
        pairs.append((t["plan_prompt"], t["gold_plan"], raw_c["plan"][i]))
    if not passed(raw_c["draft_scores"][i]):
        pairs.append((t["draft_prompt"], t["gold_draft"], raw_c["draft"][i]))
print("pairs", len(pairs), flush=True)
d_dir, abar = directions(pairs[:60])
torch.save({'d': d_dir.cpu(), 'abar': abar.cpu()}, '/content/out/dirs.pt'); push('dirs.pt')
d_rand = random_directions(0)
# 2. sweep on the selection split (DEV), directions fitted on CAL only
LAYERSETS = {"l15_19": [15, 19], "l12_23": list(range(12, 24)), "all": list(range(24))}
rows = []
def trial(name, d, layers, k, alpha=1.0):
    restore()
    changed = apply_edit(d, abar, layers, k, alpha)
    r, _ = run_eval(DEV, name, constrained=False)
    r["changed"] = changed; r["elapsed"] = round(time.time() - t0)
    rows.append(r); print({k2: r[k2] for k2 in ("label", "plan_pass", "draft_pass", "changed")}, flush=True)
    json.dump(rows, open("/content/out/sweep.json", "w")); push("sweep.json")
restore()
base, _ = run_eval(DEV, "dev-base", constrained=False); rows.append(base); print(base, flush=True)
for (ln, layers), k in itertools.product(LAYERSETS.items(), (8, 64, 512)):
    trial(f"{ln}-k{k}", d_dir, layers, k)
