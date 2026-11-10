import json, os, subprocess, sys
import numpy as np
cfg = json.loads(os.environ["CONFIG"])
blob = torch.load("/content/out/dirs.pt"); d_dir, abar = blob["d"], blob["abar"]
restore()
changed = apply_edit(d_dir, abar, cfg["layers"], cfg["k"], cfg.get("alpha", 1.0))
# sparse patch: edited columns only
patch = {}
for l in cfg["layers"]:
    W = LM.layers[l].mlp.down_proj.weight.detach()
    diff = (W.float() - _orig[l].float()).abs().sum(0) > 0
    idx = diff.nonzero().flatten().cpu().numpy()
    patch[f"L{l}_cols"] = idx; patch[f"L{l}_vals"] = W[:, idx].float().cpu().numpy()
np.savez("/content/out/patch.npz", **patch)
json.dump({"cfg": cfg, "changed_values": changed}, open("/content/out/edit.json", "w"))
m.save_pretrained("/content/edited", safe_serialization=True); tok.save_pretrained("/content/edited")
REPO = "ewin-reg/MiniCPM-V-4.7-1B-weekly-analyst-edit"
_api.create_repo(REPO, repo_type="model", private=True, exist_ok=True)
_api.upload_folder(folder_path="/content/edited", repo_id=REPO, repo_type="model", commit_message="edited weights (safetensors, f16)")
for f in ("patch.npz", "edit.json"):
    _api.upload_file(path_or_fileobj=f"/content/out/{f}", path_in_repo=f, repo_id=REPO, repo_type="model")
print("safetensors uploaded", flush=True)
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print("$", c[:70], "rc", r.returncode, (r.stderr or r.stdout)[-300:].replace("\n", " "), flush=True); return r.returncode
sh("git clone --depth 1 https://github.com/ggml-org/llama.cpp /content/llama.cpp")
sh(f"{sys.executable} -m pip install -q gguf sentencepiece protobuf")
rc = sh(f"{sys.executable} /content/llama.cpp/convert_hf_to_gguf.py /content/edited --outfile /content/out/edited-f16.gguf --outtype f16")
if rc == 0:
    _api.upload_file(path_or_fileobj="/content/out/edited-f16.gguf", path_in_repo="MiniCPM-V-4_7-1B-edited-F16.gguf", repo_id=REPO, repo_type="model")
    print("gguf uploaded", flush=True)
