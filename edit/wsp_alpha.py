import json, time
blob = torch.load("/content/out/dirs.pt"); d_dir, abar = blob["d"], blob["abar"]
rows = []
CONFIGS = [([15, 19], 64, 3.0), ([15, 19], 64, 6.0), ([15, 19], 8, 6.0), (list(range(12, 24)), 64, 3.0), (list(range(24)), 64, 2.0)]
restore(); base, _ = run_eval(DEV, "dev-base", constrained=False); rows.append(base); print(base, flush=True)
for layers, k, alpha in CONFIGS:
    restore(); ch = apply_edit(d_dir, abar, layers, k, alpha)
    r, _ = run_eval(DEV, f"L{layers[0]}-{layers[-1]}n{len(layers)}-k{k}-a{alpha}", constrained=False); r["changed"] = ch; rows.append(r)
    print({x: r[x] for x in ("label", "plan_pass", "draft_pass", "changed")}, flush=True)
    json.dump(rows, open("/content/out/alpha.json", "w")); push("alpha.json")
