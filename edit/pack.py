"""Pack src/ and requirements.lock into a self-extracting script for `colab exec -f` (writes colab/_put_pkg.py)."""
import base64
import io
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    skip = lambda info: None if "__pycache__" in info.name or info.name.endswith(".pyc") else info  # noqa: E731
    tar.add(ROOT / "src", arcname="pkg/src", filter=skip)
    tar.add(ROOT / "eval", arcname="pkg/eval", filter=skip)
    tar.add(ROOT / "space" / "app.py", arcname="pkg/app.py")
    for f in ("wsp_lib.py", "wsp_gold.py", "wsp_edit.py", "wsp_sweep.py", "wsp_final.py", "wsp_main.py", "detector_diag.py", "wsp_alpha.py", "wsp_ib.py", "wsp_export.py", "e2e_demo.py", "wsp_diag.py", "wsp_draft.py", "wsp_memit.py", "wsp_memit_core.py", "wsp_mfinal.py", "wsp_mfinal2.py", "wsp_onpolicy.py", "wsp_ood.py", "gguf_v2.py", "gguf_v3.py", "wsp_lora.py", "wsp_module.py", "wsp_lorascore.py", "wsp_modscore.py", "wsp_moddiag.py", "wsp_n1diag.py", "wsp_rewire.py", "wsp_lp.py", "gguf_v4.py", "e2e_excel.py"):
        tar.add(ROOT / "colab" / f, arcname=f"pkg/colab/{f}")
    tar.add(ROOT / "requirements.lock", arcname="pkg/requirements.lock")
payload = base64.b64encode(buf.getvalue()).decode()
script = (
    "import base64, io, tarfile, shutil, warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "shutil.rmtree('/content/pkg/src', ignore_errors=True)\n"
    f"tarfile.open(fileobj=io.BytesIO(base64.b64decode('{payload}')), mode='r:gz').extractall('/content')\n"
    "print('pkg ok')\n"
)
(ROOT / "colab" / "_put_pkg.py").write_text(script, encoding="utf-8")
print(len(payload) // 1024, "KB")
