import subprocess, sys, os, json, time
os.makedirs("/content/out", exist_ok=True)
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True); print("$", c[:80], "rc", r.returncode, flush=True)
sh(f"{sys.executable} -m pip install -q -U transformers accelerate weightwatcher lm-format-enforcer huggingface_hub rouge_score")
sh(f"{sys.executable} -m pip install -q -r /content/pkg/requirements.lock")
import torch, transformers
import transformers.tokenization_utils as tu
tu.PreTrainedTokenizerBase = transformers.PreTrainedTokenizerBase
from transformers import AutoModelForImageTextToText, AutoTokenizer
from huggingface_hub import HfApi
_api = HfApi(token=os.environ["HF_TOKEN"])
RUNS = "ewin-reg/weekly-analyst-wsp-runs"
_api.create_repo(RUNS, repo_type="dataset", private=True, exist_ok=True)
def push(name):
    _api.upload_file(path_or_fileobj=f"/content/out/{name}", path_in_repo=f"{os.environ.get('TAG','run')}/{name}", repo_id=RUNS, repo_type="dataset")
def pull(name):
    from huggingface_hub import hf_hub_download
    import shutil
    shutil.copy(hf_hub_download(RUNS, f"{os.environ.get('SRC_TAG', os.environ.get('TAG','run'))}/{name}", repo_type="dataset", token=os.environ["HF_TOKEN"]), f"/content/out/{name}")
tok = AutoTokenizer.from_pretrained("openbmb/MiniCPM-V-4.7-1B")
m = AutoModelForImageTextToText.from_pretrained("openbmb/MiniCPM-V-4.7-1B", dtype=torch.float16, device_map="cuda")
m.eval(); print("loaded", flush=True)
for f in ("wsp_lib.py", "wsp_gold.py", "wsp_edit.py", "wsp_ib.py"):
    exec(compile(open("/content/pkg/colab/" + f).read(), f, "exec"), globals())
if os.environ.get("STAGE") in ("final", "alpha", "export"):
    pull("dirs.pt")
exec(compile(open("/content/pkg/colab/wsp_" + os.environ.get("STAGE", "sweep") + ".py").read(), "stage", "exec"), globals())
print("DONE", flush=True)
