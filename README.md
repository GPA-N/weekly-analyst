# Weekly Analyst

**Drop a CSV in a folder. Do nothing else. If something material changed, a stakeholder-ready PDF appears. If not, it stays quiet.**

Weekly Analyst is an autonomous data analyst that lives next to a file you replace every week. It runs fully on your machine, on CPU, with MiniCPM-V 4.7 (1B, full precision f16, no quantization). Built for the MiniCPM Spark Program, On-Device Agent Challenge.

- Live demo: https://huggingface.co/spaces/ewin-reg/weekly-analyst (a spike or drop scenario takes about 5 minutes on the Space's CPU; a quiet week takes seconds)
- Example reports: https://gpa-n.github.io/weekly-analyst/
- Edited model: https://huggingface.co/ewin-reg/MiniCPM-V-4.7-1B-weekly-analyst-edit

| ![page 1](examples/drop-p1.png) | ![page 2](examples/drop-p2.png) | ![page 3](examples/drop-p3.png) |
|---|---|---|

## The pitch in three sentences

Most "AI analyst" demos let a language model write the numbers, and a fluent report with one wrong figure is worse than no report. Here the model never touches a number: code detects the change, runs the queries, draws the charts and fills every table, and an independent recompute step checks each figure before a person approves the send. The small model does the part it is good at, deciding which analysis to run and writing the words around the evidence.

## What happens each week

1. **Detect.** Code scores every metric and segment for the newest week (z-score, drift, CUSUM, validation checks, blended by rank) and keeps at most 3 alerts per ISO week.
2. **Stay silent if nothing material changed.** On a quiet week the agent makes no model call and sends nothing.
3. **Plan.** MiniCPM picks a tool and its parameters for each finding (named, parameterised queries; no SQL in the plan).
4. **Investigate.** Code runs the queries on a Parquet snapshot in a locked-down DuckDB.
5. **Draft.** MiniCPM writes headline, recommended action and one "so what" per finding, in words only. A digit in model text is rejected.
6. **Check.** A separate recompute verifies every number in the PDF, scans the PDF for links, scripts and embedded files, and re-checks the stamped file before sending.
7. **Approve.** A person approves (needs a terminal or a token file). Sending is code-only.

## Why MiniCPM, and what it actually does here

MiniCPM-V 4.7 (1B, f16) runs locally through llama.cpp. It understands the request, selects tools and parameters for each finding, writes the narrative, and its output is checked by code. That is a real job in local execution, on a model small enough for CPU, with no data leaving the machine. We use the text path of the model; the vision projector did not load in llama-cpp-python, so we make no vision claims.

## Measured, not promised

| What | Result |
|---|---|
| Real model, quiet week | silent in about 1 s |
| Real model, spike or drop week | PDF draft in about 195 s on a 2-vCPU Colab, about 5 min on the Space; approve re-renders, re-checks and writes it to the sent folder |
| Peak memory | 5.2 GB |
| Test suite (full project) | 654 passed |
| Weight edit, held-out plans (80 tasks, no grammar) | 64/80 to 80/80; a random edit of the same size stayed at 64/80 |
| Change detector, holdout recall at the quiet gate | 0.21 (95% interval 0.13 to 0.30), not good enough yet, see below |

## A small weight edit, found without training

We edited 16,384 values (about 0.001% of the model) in two MLP layers. The direction came from the agent's own good and failing plan outputs, the columns were ranked by attribution, and the settings were chosen on separate dev seeds. On 80 held-out tasks, plans went from 64/80 to 80/80 with no grammar, and a random direction of the same size did nothing. The Space applies the edit as a 66 KB patch (`assets/edit_patch.npz`) to the official f16 GGUF at start; set `WEEKLY_ANALYST_EDIT=0` for the unedited model. **Second edit, for drafts (measured 2026-11-11, not shipped).** Drafts scored 0/80 without a grammar because the draft prompt never shows the schema, so the base model invents keys. A closed-form MEMIT-style update of `down_proj` in six layers, fitted at the JSON positions where the model's top token is wrong, took drafts from 0/80 to 80/80 on 80 held-out tasks (seeds 100-139), and on fresh seeds 200-239 an edit plus the plan edit scored 80/80 plans and 75/80 drafts (base: 62/80 and 0/80; a random edit of the same size: 62/80 and 0/80). It changes about 2.6% of those matrices' norm, so it is much larger than the first edit (a rank-8 version is about 221,000 values). Adding the schema to the prompt text also fixes the no-grammar case for the unedited model on dev (40/40), so this edit is research, not a necessity. On other prompts it held for new data, one item and renamed metrics (40/40 each) and for the production re-request line (32/40), but scored **0/40 when the instruction text was reworded**, so it overfits the wording; a refit on the model's own samples also failed. It is research, not a general fix. Details, failed attempts and caveats: `results/wsp-draft-edit.md`.

Method and limits are in the [model card](https://huggingface.co/ewin-reg/MiniCPM-V-4.7-1B-weekly-analyst-edit); scripts are in `edit/`.

## Problems we hit, and what fixed them

| Problem | What we did |
|---|---|
| Fluent reports can carry wrong numbers | Code supplies every number. A separate recompute checks each one, a digit in model text is rejected, the PDF is scanned and the stamped file is re-checked before send. |
| Untrusted CSV labels can carry prompt injection | Segment labels become letters-only hash codes before the model or the PDF sees them. The plan never contains SQL, sending is code-only, and approval needs a terminal or a token file. |
| First real-model runs halted: unknown query name, wrong `segment_col`, digits in drafts | The plan prompt lists the query catalogue with a worked example, a parameter check re-prompts with the exact problem, and draft evidence is words only (direction, largest segments, no numbers). |
| Free-form SQL from a 1B model is risky | Named, parameterised queries are the main path. Free SQL is the last resort under a GBNF grammar, in DuckDB locked down in the right order (autoload off before lock). |
| Quantization hurt quality | Full precision f16 everywhere, no quantized model. |
| Our own finetune vs the official model | Compared on the same tasks; the official model won on length and out-of-domain scores, so we did not adopt the finetune. |
| Report layout bugs: footer overlapped the page number, raw item keys in findings | Shortened the footer hash to 16 characters and rendered readable segment names. |
| The vision projector would not load in llama-cpp-python | Dropped the vision claim; the agent uses the text path. |
| Colab VMs died during long runs and took results with them (twice) | Long jobs run detached and push every result to a private Hugging Face dataset; a lost VM costs one trial. |
| A plain weight edit did nothing (alpha 1); strong or wide edits broke the model (0/40 plans) | Searched strength and width on dev seeds; the working edit is narrow (8 columns, 2 layers, alpha 6), then validated once on held-out tasks against a random-direction control. |
| The converted edited GGUF would not load in llama-cpp-python 0.3.36 | Ship the edit as a patch on the official GGUF and verify it matches the converted file on the edited tensors. |
| The Space repo has a 1 GB cap and the model is 2 GB | The app downloads the official GGUF at start and patches a copy. |
| ZeroGPU Space: the agent run was killed with "GPU task aborted" and the host reported 192 CPUs, which thrashed llama.cpp (545 s) | The agent runs on CPU outside the GPU time cap with threads capped at 8; one tiny `@spaces.GPU` function stays so the Space starts. |
| Gradio Spaces could not be created through the API on a free account (402) | The Space was created in the web UI and updated through the API. |
| Drafts scored 0/80 without a grammar and the first edit did not help | Found by printing raw outputs: the draft prompt never shows the schema, so the model invents keys. Direction add/flip edits stayed at 0/40; a MEMIT-style closed-form edit fitted at the wrong JSON positions reached 80/80 held-out drafts (fresh seeds: 75/80 with the plan edit). |
| Detector recall was 0.21 on holdout | Diagnosed on dev seeds: ranking alone reaches 0.61 recall at the top 3 for a 1.5x outlier (0.65 for z-score alone), so the strict minimum-change gate costs most of it. Not fixed yet. |

## Honest limits

- Detection: recall 0.21 on the holdout replay at the quiet gate (the bar was 0.80), and 14% of quiet weeks raised an alert. We know why, we have not fixed it.
- The weight-edit gains are on synthetic tasks, without a JSON grammar, from one greedy run each. In production the agent uses a grammar and retries, so the gain there is probably smaller. The shipped 16,384-value edit does not make drafts valid without a grammar. A second, larger edit does (below), but it is only measured on the transformers model and is not yet applied to the GGUF the Space runs.
- Not done yet: injection corpus against the real model, a second person scoring brief quality, a reliability report, a demo video, CUDA on the Space (the code supports it; it did not run on the GPU there).
- Synthetic data only. Nothing is uploaded, and the source file is never modified.

## Run it

`pip install -r requirements.txt && python app.py`, or open the Space. With CUDA: `requirements.cuda.txt`. The first run downloads the 2 GB f16 GGUF from `openbmb/MiniCPM-V-4.7-1B-gguf`.

## Layout

- `src/weekly_analyst/`: the agent (extract, validate, detect, plan, investigate, draft, check, outbox, remember), renderer, checks, approve command, `analysis/weight_patch.py`.
- `app.py`: the Gradio demo. `assets/`: the weight patch. `examples/`: sample reports.
- `edit/`: weight-edit and evaluation scripts for Colab. `results/`: raw measurements, including the ones that went badly.
- `docs/`: design document and architecture notes.
