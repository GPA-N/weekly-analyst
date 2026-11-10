"""Weekly Analyst: hosted demo of an autonomous on-device data analyst (MiniCPM-V 4.7, 1B, f16; CPU, or CUDA when the llama.cpp build supports it).

Synthetic data only. There is no upload path. The agent reads a weekly sales CSV, stays silent when nothing
changed, and otherwise writes a stakeholder PDF whose every number is recomputed by code. The model only writes
the words. Reports go nowhere except the local sent folder of this Space.
"""

import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402
import pypdfium2 as pdfium  # noqa: E402

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE / "src"))

from weekly_analyst.analysis.model import LlamaCppClient  # noqa: E402
from weekly_analyst.demo import DemoRun, approve_demo, run_demo  # noqa: E402

MODEL_REPO = "openbmb/MiniCPM-V-4.7-1B-gguf"
MODEL_FILE = "MiniCPM-V-4_7-1B-F16.gguf"  # full precision on purpose: quantization degrades quality
N_CTX = 4096
BASE = Path(tempfile.mkdtemp(prefix="weekly-analyst-"))
EXAMPLES = HERE / "examples"

_client = None
_busy = threading.Lock()  # llama.cpp is not thread-safe: one run at a time
_runs: dict[str, DemoRun] = {}

SCENARIO_LABELS = {
    "Quiet week: nothing changed, the agent stays silent": "quiet",
    "Sales drop: units and revenue fall in the newest week": "drop",
    "Sales spike: units and revenue jump in the newest week": "spike",
}


try:  # ZeroGPU Spaces attach a GPU only inside @spaces.GPU calls; anywhere else this is a no-op
    import spaces

    gpu = spaces.GPU(duration=300)
except ImportError:

    def gpu(fn):
        return fn


def _model_path() -> str:
    """The official f16 GGUF, with the weight edit applied unless WEEKLY_ANALYST_EDIT=0 (the patch is tiny, the model is not)."""
    local = HERE / MODEL_FILE
    if local.exists():
        base = str(local)
    else:
        from huggingface_hub import hf_hub_download

        base = hf_hub_download(MODEL_REPO, MODEL_FILE)
    if os.environ.get("WEEKLY_ANALYST_EDIT", "1") == "0":
        return base
    out = BASE / "MiniCPM-V-4_7-1B-edited-F16.gguf"
    if not out.exists():
        from weekly_analyst.analysis.weight_patch import apply_patch

        apply_patch(Path(base), HERE / "assets" / "edit_patch.npz", out)
    return str(out)


def _gpu_layers() -> int:
    try:
        from llama_cpp import llama_supports_gpu_offload

        return -1 if llama_supports_gpu_offload() else 0
    except ImportError:
        return 0


def get_client() -> LlamaCppClient:
    """CPU: load once and keep. CUDA: load per run, because a ZeroGPU context ends with the call."""
    global _client
    layers = _gpu_layers()
    if layers:
        return LlamaCppClient(Path(_model_path()), threads=os.cpu_count() or 2, n_ctx=N_CTX, n_gpu_layers=layers)
    if _client is None:
        _client = LlamaCppClient(Path(_model_path()), threads=os.cpu_count() or 2, n_ctx=N_CTX)
    return _client


@gpu
def run_on_device(scenario: str, base: Path) -> DemoRun:
    return run_demo(scenario, get_client(), base)


def page_images(pdf_path: Path, scale: float = 1.2) -> list[str]:
    out_dir = pdf_path.parent / f"{pdf_path.stem}-pages"
    out_dir.mkdir(exist_ok=True)
    document = pdfium.PdfDocument(str(pdf_path))
    paths = []
    for index in range(len(document)):
        target = out_dir / f"page-{index + 1}.png"
        document[index].render(scale=scale).to_pil().save(target)
        paths.append(str(target))
    document.close()
    return paths


def run_scenario(label: str):
    """Generator: status text, page images, PDF file, run key, approve button visibility."""
    scenario = SCENARIO_LABELS[label]
    if not _busy.acquire(blocking=False):
        yield "Another run is in progress on this Space. Try again in a few minutes.", [], None, "", gr.update(visible=False)
        return
    try:
        result: dict = {}

        def work() -> None:
            try:
                result["demo"] = run_on_device(scenario, BASE)
            except Exception as err:  # noqa: BLE001 - shown to the visitor, never silent
                result["error"] = f"{type(err).__name__}: {err}"

        thread = threading.Thread(target=work, daemon=True)
        started = time.perf_counter()
        thread.start()
        while thread.is_alive():
            elapsed = int(time.perf_counter() - started)
            yield (
                f"Running the nine states on CPU ({elapsed} s). The first run also downloads the 2 GB model. "
                "A report takes about 3 to 4 minutes.",
                [],
                None,
                "",
                gr.update(visible=False),
            )
            time.sleep(3)
        if "error" in result:
            yield f"The run failed: {result['error']}", [], None, "", gr.update(visible=False)
            return
        demo: DemoRun = result["demo"]
        key = uuid.uuid4().hex
        _runs[key] = demo
        if demo.kind == "silent":
            yield (
                f"**Silent.** Nothing material changed, so the agent made no model call and sent nothing "
                f"({demo.seconds} s).",
                [],
                None,
                key,
                gr.update(visible=False),
            )
        elif demo.kind == "brief" and demo.draft:
            yield (
                f"**Draft report ready** in {demo.seconds} s, waiting for approval. Every number in it was recomputed "
                "by separate code and the PDF passed the safety scan. Press Approve to stamp and send it.",
                page_images(demo.draft),
                str(demo.draft),
                key,
                gr.update(visible=True),
            )
        else:
            reasons = "; ".join(demo.reasons) or "no reason recorded"
            yield (
                f"**Halted, nothing sent.** {reasons}. The agent refuses to send a report it cannot verify.",
                [],
                None,
                key,
                gr.update(visible=False),
            )
    finally:
        _busy.release()


def approve(key: str):
    demo = _runs.get(key)
    if demo is None:
        return "There is no run to approve. Generate a report first.", [], None, gr.update(visible=False)
    ok, message, pdf = approve_demo(demo)
    if ok and pdf:
        return (
            f"**Approved.** {message}. The stamp area now names the approver and time, and the stamped PDF was "
            "re-checked before sending.",
            page_images(pdf),
            str(pdf),
            gr.update(visible=False),
        )
    return f"**Approval refused.** {message}", [], None, gr.update(visible=False)


def example_files(prefix: str) -> tuple[list[str], str | None]:
    pages = sorted(str(p) for p in EXAMPLES.glob(f"{prefix}-p*.png"))
    pdf = EXAMPLES / f"{prefix}.pdf"
    return pages, (str(pdf) if pdf.exists() else None)


ABOUT = """
### What this is
An agent that attaches to a CSV that is replaced every week. When new data lands it decides, with no prompt,
whether anything material changed. If not, it stays silent. If so, it writes a stakeholder PDF: executive summary,
findings with charts, risks and data quality, next checks, and appendices for definitions, data tables and method.

### How it stays honest
- **The model writes words only.** Code computes every number, chart and table. A separate implementation
  recomputes each number from the data and the PDF is rejected if any figure does not match.
- **Data cannot instruct the agent.** Segment labels are sanitized, the model's data tool is a locked-down
  read-only query engine, and 62 injection attempts replayed against a model that obeys them leaked nothing.
- **A person approves.** Sending is code, outside the model. Approval re-renders the stamped PDF and re-checks it.
- **Read-only, local, offline.** The source file is never modified, and nothing leaves the machine.

### The model
Official MiniCPM-V 4.7 1B in full precision (f16 GGUF, no quantization), running on CPU through llama.cpp.
Measured on a 2-core CPU: 4.8 tokens per second, 5.2 GB peak memory, plan JSON valid in 10 of 10 trials
(50 of 50 on GPU). A fine-tuned variant was tested and not adopted: 48 of 50 valid plans, 4 to 9 times longer replies.

### Honest limits
- Synthetic sales data only, so reliability figures are from synthetic replays.
- For free SQL the 1B model got 21 of 30 questions right under a grammar, short of the 24 of 30 bar, so named,
  parameterized queries are the main path and free SQL is a last resort.
- Chart reading by the vision encoder is not claimed: the projector would not load in the runtime used.
- CPU only here; Apple Silicon speed is untested.
"""

with gr.Blocks(title="Weekly Analyst") as demo_app:
    gr.Markdown(
        "# Weekly Analyst\nAn autonomous data analyst for a weekly CSV, running on a 1B model, on CPU. "
        "Synthetic data only; nothing is uploaded."
    )
    with gr.Tab("Try it"):
        scenario = gr.Radio(list(SCENARIO_LABELS), value=list(SCENARIO_LABELS)[1], label="This week's file")
        run_button = gr.Button("Run the agent", variant="primary")
        status = gr.Markdown()
        gallery = gr.Gallery(label="Report pages", columns=3, height=520, object_fit="contain")
        pdf_file = gr.File(label="PDF")
        key_state = gr.State("")
        approve_button = gr.Button("Approve and send", visible=False)
        run_button.click(
            run_scenario,
            inputs=scenario,
            outputs=[status, gallery, pdf_file, key_state, approve_button],
            concurrency_limit=1,
        )
        approve_button.click(approve, inputs=key_state, outputs=[status, gallery, pdf_file, approve_button])
    with gr.Tab("Example reports (already run)"):
        gr.Markdown(
            "These two reports were produced by this same code with the real model, on a 2-core CPU, then "
            "approved with the real approve command."
        )
        for prefix, title in (("drop", "Sales drop"), ("spike", "Sales spike")):
            pages, pdf = example_files(prefix)
            gr.Markdown(f"#### {title}")
            gr.Gallery(value=pages, label=title, columns=3, height=420, object_fit="contain")
            if pdf:
                gr.File(value=pdf, label=f"{title} PDF")
    with gr.Tab("How it works and how reliable it is"):
        gr.Markdown(ABOUT)

demo_app.queue()

if __name__ == "__main__":
    demo_app.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")), theme=gr.themes.Soft())
