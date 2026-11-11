"""Build an f16 GGUF with both weight edits (the 16,384-value plan edit and the dense draft edit) and check it in llama.cpp.

Standalone script, run detached on a Colab GPU VM: python /content/pkg/colab/gguf_v2.py   (env HF_TOKEN)
1. official f16 GGUF + sparse plan patch (patch.npz of the model repo) + dense draft delta (dw.pt of run f1, layers 16 17 18 20 21 22)
2. tensor checks against the official file
3. llama-cpp-python 0.3.36 (CUDA wheel if it loads, else CPU) on fresh seeds 200-211, draft prompts, no grammar, greedy; plus the
   unedited official GGUF on 8 of the same tasks as a control
4. upload to the model repo only if the edited file passes at least 80% of the drafts
Nothing is quantized: the output is f16.
"""
import json
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print("$", c[:100], "rc", r.returncode, (r.stderr or r.stdout)[-200:].replace("\n", " ").encode("ascii", "ignore").decode(), flush=True)
    return r.returncode


t0 = time.time()
sh(f"{sys.executable} -m pip install -q gguf huggingface_hub")
sh(f"{sys.executable} -m pip install -q -r /content/pkg/requirements.lock")
cuda = sh(f"{sys.executable} -m pip install -q llama-cpp-python==0.3.36 --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124") == 0
import torch  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, "/content/pkg/src")
from weekly_analyst.analysis.weight_patch import add_delta, apply_patch  # noqa: E402

TOK = os.environ["HF_TOKEN"]
MODEL_REPO = "ewin-reg/MiniCPM-V-4.7-1B-weekly-analyst-edit"
OUT_NAME = "MiniCPM-V-4_7-1B-weekly-analyst-v2-F16.gguf"
off = hf_hub_download("openbmb/MiniCPM-V-4.7-1B-gguf", "MiniCPM-V-4_7-1B-F16.gguf")
plan_patch = hf_hub_download(MODEL_REPO, "patch.npz", token=TOK)
dw_path = hf_hub_download("ewin-reg/weekly-analyst-wsp-runs", "f1/dw.pt", repo_type="dataset", token=TOK)
out = f"/content/{OUT_NAME}"
plan_layers = apply_patch(off, plan_patch, out)
dw = {int(l): t.float().numpy() for l, t in torch.load(dw_path).items()}
add_delta(out, dw)
print("patched plan layers", plan_layers, "dense layers", sorted(dw), round(time.time() - t0), "s", flush=True)

# tensor checks
a, o = GGUFReader(out), GGUFReader(off)
ta, to = {t.name: t for t in a.tensors}, {t.name: t for t in o.tensors}
edited = {f"blk.{l}.ffn_down.weight" for l in set(plan_layers) | set(dw)}
chk = {"tensor_count_equal": len(ta) == len(to), "untouched_equal": all(np.array_equal(ta[n].data, to[n].data) for n in ta if n not in edited)}
for l, d in dw.items():
    n = f"blk.{l}.ffn_down.weight"
    want = (to[n].data.astype(np.float32) + d).astype(np.float16)
    chk[n] = {"max_abs_diff_vs_official_plus_delta": float(np.abs(ta[n].data.astype(np.float32) - want.astype(np.float32)).max()),
              "changed_vs_official": int((ta[n].data != to[n].data).sum())}
print(json.dumps(chk), flush=True)

# llama.cpp check
import llama_cpp  # noqa: E402

gpu = bool(llama_cpp.llama_supports_gpu_offload())
print("llama_cpp", llama_cpp.__version__, "gpu offload", gpu, "cuda wheel installed", cuda, flush=True)
exec(compile(open("/content/pkg/colab/wsp_lib.py").read(), "wsp_lib.py", "exec"), globals())
tasks = build_tasks(range(200, 212))[:24 if gpu else 6]
print("tasks", len(tasks), flush=True)


def run(path, label, n):
    llm = llama_cpp.Llama(model_path=path, n_ctx=8192, n_threads=2, n_gpu_layers=-1 if gpu else 0, verbose=False)
    sc, fails = [], []
    for t in tasks[:n]:
        r = llm.create_chat_completion(messages=[{"role": "user", "content": t["draft_prompt"]}], temperature=0.0, max_tokens=700)
        txt = r["choices"][0]["message"]["content"]
        s = score_draft(txt, t["draft_prompt"])
        sc.append(passed(s))
        if not passed(s):
            fails.append(txt[:200])
    del llm
    res = {"label": label, "pass": sum(sc), "n": len(sc), "fail_samples": fails[:2]}
    print(res, flush=True)
    return res


results = {"checks": chk, "gpu": gpu, "llama_cpp": llama_cpp.__version__}
results["edited"] = run(out, "edited-gguf", len(tasks))
results["official"] = run(off, "official-gguf", min(8, len(tasks)))
os.makedirs("/content/out", exist_ok=True)
json.dump(results, open("/content/out/gguf_v2.json", "w"), indent=1)
api = HfApi(token=TOK)
api.create_repo("ewin-reg/weekly-analyst-wsp-runs", repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj="/content/out/gguf_v2.json", path_in_repo="gguf_v2/gguf_v2.json", repo_id="ewin-reg/weekly-analyst-wsp-runs", repo_type="dataset")
if results["edited"]["pass"] >= 0.8 * results["edited"]["n"] and results["official"]["pass"] == 0:
    api.upload_file(path_or_fileobj=out, path_in_repo=OUT_NAME, repo_id=MODEL_REPO, repo_type="model",
                    commit_message="f16 GGUF with the plan edit and the dense draft edit (checked in llama-cpp-python 0.3.36)")
    api.upload_file(path_or_fileobj=dw_path, path_in_repo="draft_edit_dw_fp16.pt", repo_id=MODEL_REPO, repo_type="model",
                    commit_message="dense draft-edit delta (6 layers, fp16)")
    print("UPLOADED", OUT_NAME, flush=True)
else:
    print("NOT UPLOADED: the edited file did not pass the gate", flush=True)
print("DONE-GGUF", round(time.time() - t0), "s", flush=True)
