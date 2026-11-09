
# Weekly Analyst

- Live demo (Hugging Face Space): https://huggingface.co/spaces/ewin-reg/weekly-analyst
- Pages site with real example reports: https://gpa-n.github.io/weekly-analyst/

An autonomous data analyst for a CSV that is replaced every week, built for the MiniCPM Spark Program (On-Device Agent Challenge).
It runs MiniCPM-V 4.7 (1B, full precision f16, no quantization) on CPU through llama.cpp.

The Pages site shows real reports produced by this code and the real model. This repository also holds everything needed to run it:

- `src/weekly_analyst/`: the agent (extract, validate, detect, plan, investigate, draft, check, outbox, remember), the PDF renderer, the checks and the approve command.
- `app.py` and `requirements.txt`: a Gradio app that runs the real agent on CPU. Run it locally with `pip install -r requirements.txt` and `python app.py`, or duplicate this Space, set `sdk: gradio` and choose CPU hardware (free hardware needs a PRO plan).
- `examples/`: the report PDFs and page images shown above.
- `results/`: the measurements behind the numbers on the page, including what failed.
- `docs/`: the design document and the architecture notes.

Synthetic data only. Nothing is uploaded and the source file is never modified. A person approves a report before it is sent.

## Run it

Click **Code > Codespaces > Create codespace** on this repository. The dev container installs the requirements and starts the Gradio app on port 7860; open the forwarded port. The first run downloads the full precision (f16) MiniCPM-V 4.7 1B GGUF from Hugging Face, about 2 GB, and runs it on CPU.

Locally: `pip install -r requirements.txt && python app.py`.

## Measured limits

Scored replay on 100 holdout sets (one revenue outlier of x1.5 in one region, alert budget 3 per week, gate calibrated on dev seeds only):

- Recall of the injected event: 21 of 100 (0.21, 95% Clopper-Pearson interval 0.13 to 0.30). The bar (lower bound at least 0.80) is not met.
- Quiet weeks: 28 of 200 raised at least one alert (0.14, interval 0.10 to 0.20).

Raw numbers: `results/replay-holdout-results.json`. Not yet done: injection corpus against the real model, a second person scoring brief quality, the reliability report and the demo video.

The agent uses CUDA when the llama.cpp build supports it (`requirements.cuda.txt`); the default is CPU.
