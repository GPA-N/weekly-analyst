# 045 finetune compare (f16, GPU accuracy trials)

Status: PARTIAL. Plan validity measured on Colab T4 at full precision (f16 GGUF, no quantization). CPU bars and the query-correctness arm were not run for the finetune (the adoption rule already failed on plan validity).

## Pre-registered adoption rule (written in the worker brief before any result was seen)

Adopt the finetune only if plan_valid_rate(B) >= plan_valid_rate(A) AND query correct_rate(B) exceeds A by at least 0.10 AND peak memory is under 12GB. Otherwise keep the official model.

## Results (same 50 trials, same prompt and schema, library-default sampling, fixed seeds, llama.cpp b12223, T4 offload)

| Arm | File | plan_valid | Typical completion tokens |
| --- | --- | --- | --- |
| A official | openbmb MiniCPM-V-4_7-1B-F16.gguf | 50 of 50 | 36 to 58 |
| B finetune | ewin-reg ...FlashREINFORCE-f16.gguf | 48 of 50 | 176 to 512 (one reply hit the 512-token cap) |

Verdict by the rule: B (48 of 50) is below A (50 of 50) on plan validity, so the finetune is NOT adopted. The official f16 model stays.

## Caveats

- GPU decoding can differ slightly from CPU decoding. Time and memory bars still need a CPU-only f16 run of the official model (not done yet).
- 50 trials: the difference of 2 trials is not statistically firm, but the rule was fixed in advance and it is not met.
- The finetune writes 4 to 9 times as many tokens per plan, which also costs CPU time.
- The raw per-trial records were on the Colab VM only; the download step failed (path rewriting by Git Bash) and the VM was stopped before the records were saved, so only the counts above exist. Re-running takes about 5 minutes on a T4.
- The first spike's Q4_K_M numbers are superseded.
