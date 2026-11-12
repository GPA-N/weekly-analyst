"""Round 2: build an f16 GGUF with the lora3 delta (best epoch of run lora3, dense, all 24 down_proj layers) on the official f16 GGUF and check it in llama.cpp.
Differs from gguf_v3.py: nothing is uploaded to the public model repo (the file and delta go to the private runs dataset under gguf_v4/), the check uses unspent seeds 500-511 plus a held-out wording, and DONE-GGUF4 is printed.

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
import ctypes, glob, sysconfig  # noqa: E402
sh(f"{sys.executable} -m pip install -q nvidia-cuda-runtime-cu12 nvidia-cublas-cu12")  # the CUDA wheel needs libcudart.so.12, which this VM lacks
for _pat in ("nvidia/cuda_runtime/lib/libcudart.so.12*", "nvidia/cublas/lib/libcublasLt.so.12*", "nvidia/cublas/lib/libcublas.so.12*"):
    for _p in sorted(glob.glob(os.path.join(sysconfig.get_paths()["purelib"], _pat))):
        ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        break
import torch  # noqa: E402
from gguf import GGUFReader  # noqa: E402
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

sys.path.insert(0, "/content/pkg/src")
from weekly_analyst.analysis.weight_patch import add_delta  # noqa: E402

TOK = os.environ["HF_TOKEN"]
MODEL_REPO = "ewin-reg/MiniCPM-V-4.7-1B-weekly-analyst-edit"
OUT_NAME = "MiniCPM-V-4_7-1B-weekly-analyst-lora3-F16.gguf"
off = hf_hub_download("openbmb/MiniCPM-V-4.7-1B-gguf", "MiniCPM-V-4_7-1B-F16.gguf")
ck = torch.load(hf_hub_download("ewin-reg/weekly-analyst-wsp-runs", "lora3/lora_ckpt.pt", repo_type="dataset", token=TOK), map_location="cpu")
snap = ck["best"]["snap"]
print("lora3 best epoch", ck["best"]["epoch"], flush=True)
dw_path = "/content/lora3_dw_fp16.pt"
torch.save({l: (snap["B"][l] @ snap["A"][l]).half() for l in snap["A"]}, dw_path)  # alpha = r = 8, scale 1
out = f"/content/{OUT_NAME}"
shutil.copyfile(off, out)
plan_layers = []
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
tasks = build_tasks(range(500, 512))[:24 if gpu else 6]
print("tasks", len(tasks), flush=True)


def run(path, label, n):
    llm = llama_cpp.Llama(model_path=path, n_ctx=8192, n_threads=2, n_gpu_layers=-1 if gpu else 0, verbose=False)
    sc, fails, plan_ok, rb = [], [], [], []
    for t in tasks[:n]:
        r = llm.create_chat_completion(messages=[{"role": "user", "content": t["draft_prompt"]}], temperature=0.0, max_tokens=700)
        txt = r["choices"][0]["message"]["content"]
        s = score_draft(txt, t["draft_prompt"])
        rp = llm.create_chat_completion(messages=[{"role": "user", "content": t["plan_prompt"]}], temperature=0.0, max_tokens=700)
        plan_ok.append(passed(score_plan(rp["choices"][0]["message"]["content"], t["plan_prompt"])))
        sc.append(passed(s))
        if label.startswith('edited') and len(rb) < 8:
            pb = variants(t['draft_prompt'])['reword_b']
            rb.append(passed(score_draft(llm.create_chat_completion(messages=[{'role': 'user', 'content': pb}], temperature=0.0, max_tokens=700)['choices'][0]['message']['content'], pb)))
        if not passed(s):
            fails.append(txt[:200])
    del llm
    res = {"label": label, "pass": sum(sc), "plan_pass": sum(plan_ok), "n": len(sc), "reword_b_pass": sum(rb), "reword_b_n": len(rb), "fail_samples": fails[:2]}
    print(res, flush=True)
    return res


results = {"checks": chk, "gpu": gpu, "llama_cpp": llama_cpp.__version__}
results["edited"] = run(out, "edited-gguf", len(tasks))
results["official"] = run(off, "official-gguf", min(8, len(tasks)))
os.makedirs("/content/out", exist_ok=True)
json.dump(results, open("/content/out/gguf_v4.json", "w"), indent=1)
api = HfApi(token=TOK)
api.create_repo("ewin-reg/weekly-analyst-wsp-runs", repo_type="dataset", private=True, exist_ok=True)
api.upload_file(path_or_fileobj="/content/out/gguf_v4.json", path_in_repo="gguf_v4/gguf_v4.json", repo_id="ewin-reg/weekly-analyst-wsp-runs", repo_type="dataset")
api.upload_file(path_or_fileobj=dw_path, path_in_repo="gguf_v4/lora3_dw_fp16.pt", repo_id="ewin-reg/weekly-analyst-wsp-runs", repo_type="dataset")
print("DONE-GGUF4", round(time.time() - t0), "s", flush=True)  # the GGUF stays on this VM (/content); upload to the model repo only when the owner says so
