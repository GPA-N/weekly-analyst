import json, os
cfg = json.loads(os.environ["CONFIG"])  # {"layers": [...], "k": int, "alpha": float}
blob = torch.load("/content/out/dirs.pt")
d_dir, abar = blob["d"], blob["abar"]
out = {}
restore(); out["base"], _ = run_eval(HOLD, "hold-base", constrained=False); print(out["base"], flush=True)
changed = apply_edit(d_dir, abar, cfg["layers"], cfg["k"], cfg.get("alpha", 1.0)); out["edit_changed"] = changed
out["edited"], raw_e = run_eval(HOLD, "hold-edited", constrained=False); print(out["edited"], flush=True)
restore(); apply_edit(random_directions(0), abar, cfg["layers"], cfg["k"], cfg.get("alpha", 1.0))
out["random"], _ = run_eval(HOLD, "hold-random", constrained=False); print(out["random"], flush=True)
IB = build_ib_tasks()
restore(); out["ib_base"], _ = run_ib(IB, "ib-base"); print(out["ib_base"], flush=True)
apply_edit(d_dir, abar, cfg["layers"], cfg["k"], cfg.get("alpha", 1.0)); out["ib_edited"], _ = run_ib(IB, "ib-edited"); print(out["ib_edited"], flush=True)
restore()
out["cfg"] = cfg
json.dump(out, open("/content/out/final.json", "w")); push("final.json")
