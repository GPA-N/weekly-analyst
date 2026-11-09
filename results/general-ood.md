# 049 out-of-distribution check, finetune vs official (safetensors, transformers, T4)

Status: DONE. Run by the router on a Colab T4 (session stopped). Records: spike/results/general-*.jsonl. Numbers: docs/spike-general-results.json. Script: spike/spike_general.py.

Setup: the non-GGUF weights, openbmb/MiniCPM-V-4.7-1B and ewin-reg/MiniCPM-V-4.7-1B-Agentic-FwS-FlashREINFORCE, transformers 5.22.1, float32 (no quantization, no half-precision rounding), greedy decoding, max 1024 new tokens, 100 GSM8K and 100 ARC-Easy questions picked with a fixed seed, same prompts. Thinking mode is the chat template's enable_thinking flag. These tasks are unlike the finetune's training data (text tool calling from Nemotron-SFT-Agentic-v4).

| Config | GSM8K (of 100) | ARC-Easy (of 100) | Mean new tokens (GSM8K / ARC) |
| --- | --- | --- | --- |
| official, instruct | 82 (0.73 to 0.89) | 88 (0.80 to 0.94) | 298 / 3 |
| official, thinking | 80 (0.71 to 0.87) | 64 (0.54 to 0.73) | 523 / 232 |
| finetune, instruct | 76 (0.66 to 0.84) | 84 (0.75 to 0.91) | 257 / 3 |
| finetune, thinking | 73 (0.63 to 0.81) | 71 (0.61 to 0.80) | 294 / 233 |

Reading: the finetune is 4 to 6 points below the official model on tasks unlike its training data (82 vs 76 and 88 vs 84 in instruct mode), but every gap is inside the 95% intervals, so this does not show a loss of general ability, and does not rule out a small one either. In thinking mode the official model is also much worse on ARC (64), so thinking mode is not the better mode for either model on these tasks. The finetune is not clearly overfit by this test; it is simply not better at anything we need.

Decision: the product uses the official MiniCPM-V-4.7-1B in instruct mode. Together with tickets 045 (plan validity 50/50 vs 48/50) and 046 (SQL), the finetune is not adopted.

Caveats: 100 questions per task; one seed; greedy decoding; the model card's in-distribution numbers (exact 0.414 on 1000 val rows, base 0.047) were not reproduced here. The earlier GGUF f16 tests (045, 046) used llama.cpp; this one used the safetensors weights, so the three results are not from a single runtime.
