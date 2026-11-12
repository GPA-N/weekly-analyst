# Tune/score split, LoRA and closed-form edits (2026-11-11), round 2 added 2026-11-12

All numbers: synthetic tasks, no JSON grammar, greedy decoding, f16, MiniCPM-V 4.7 1B on a Colab T4.

## Protocol

- Fit data: seeds 0 to 19. Selection: seeds 20 to 39 and two dev wordings that no edit was fitted on. Scoring: seeds 300 to 339, scored once per candidate; a ledger file in a private dataset refuses a second run of the same candidate tag.
- Wordings: fitted wordings (8 hand-written plus composed paraphrases, later also the original reworded instruction), 2 dev wordings, and 3 held-out wordings used only for scoring (`reword`, `reword_b` without field names, `reword_c` with them).
- Read before the work (excerpts, not full text unless stated): 2410.07819, 2506.19540, 2606.28374, 2602.04998, 2512.15634, 2604.13627, 2605.22579, 2611.09382, 2607.28669, the full "Sparks of Weight-Space Programming" post, the weight-space-lp post and repo, and the WeightWatcher documentation.

## Scored once on seeds 300 to 339

| | Plans /80 | Drafts, plain /80 | Renamed /40 | Repair line /40 | One item /40 | Reworded, no field names /40 |
|---|---|---|---|---|---|---|
| Base | 65 | 0 | 0 | 0 | 0 | 0 |
| LoRA, rank 8, every `down_proj`, lr 2e-4, epoch 2 | 80 | 80 | 40 | 40 | 40 | 0 |

Training loss fell from 0.30 to 0.0001 with equal loss on fit and dev rows, and the merged change is 1.2% of the weight norm. The LoRA is hand-written (a hook on `down_proj`); lr 2e-4 and accumulation 4 are conventions, not tuned here. One run, no variance estimate.

## Closed-form (no gradient) edits, dev only, `ww_hi` layers 16 17 18 20 21 22

| scale | relative change | plans /40 (base 36) | plain drafts /40 | dev wording, no field names /20 | text loss (base 2.993) |
|---|---|---|---|---|---|
| 2 | 0.116 | 35 | 40 | 0 | n/a |
| 3 | 0.174 | 27 | 40 | 5 | 2.996 |
| 4, lam 3 | 0.232 | 12 | 40 | 20 | 3.005 |
| 4, lam 1 | 0.276 | 11 | 40 | 20 | 3.020 |

Not scored on seeds 300 to 339 because scale 4 breaks the plan step.

## Cause of the failing wording

The evidence items carry a field `direction`. When the instruction names no JSON fields, the model copies that key instead of writing `so_what` and `metric_refs`. After the scale-2 edit, only that token (`so` then `direction`) is still wrong under that wording.

## Caveats

- The layer set `ww_hi` came from one WeightWatcher run with `min_evals=10` and alpha above 9, which its documentation calls spurious (above 8). The choice is not validated.
- Mid-layer sets (12 14 16 18 20 22 and eight cached layers) returned NaN deltas at scale 2; not resolved.
- Deployed: the LoRA GGUF (`MiniCPM-V-4_7-1B-weekly-analyst-lora-F16.gguf`, official f16 GGUF plus the merged delta; all other tensors identical) passed a llama.cpp check on a T4 (24/24 plans, 24/24 drafts on seeds 200 to 211, which are tuning data; official GGUF 4/8 and 0/8) and a live Space run (spike draft in 297 s, approve in 17 s). The closed-form edits are not deployed.


## Round 2 (2026-11-12): fixing the wording failure with data, then a real-data test

The first LoRA scored 0/40 on an instruction that names no JSON field. Round 2 kept the recipe (rank 8, alpha 8, every `down_proj`, lr 2e-4, batch 1 with 4-step accumulation) and changed only the data, continuing from the previous adapter each time.

### Protocol

- Fit seeds 0 to 19; dev seeds 20 to 39 with 2 old dev wordings and 3 new dev wordings that name no field (`n0`, `n1`, `n2`; never fitted); fresh scoring seeds 400 to 439 (seeds 300 to 339 were spent in round 1); the ledger refuses a second run of a tag (`lora3-e4-lora`, `base-s400`).
- Training wordings: 53 distinct wordings by the end of `lora3` (the first LoRA's wordings, 12 composed wordings that name no JSON field, 8 synonym wordings for the action field). 16 more synonym wordings for every field were written afterwards and were not used for `lora3`.
- `reword` was added to the training wordings by decision, so on fresh seeds it only tests new seeds. `reword_b` (no field names) and `reword_c` were never trained on.

### Dev results (20 tasks per wording, plans and plain drafts 40)

| Run | Plans | Plain drafts | `d0` | `d1` | `n0` | `n1` | `n2` | Total |
|---|---|---|---|---|---|---|---|---|
| Base | 36/40 | 0/40 | 0 | 0 | 0 | 0 | 0 | 36 |
| `lora2`, epoch 1 (12 no-field wordings added) | 40 | 40 | 20 | 20 | 20 | **0** | 20 | 160 |
| `lora3`, epoch 3 (8 synonym wordings added, resumed from `lora2`) | 40 | 40 | 20 | 20 | 20 | 14 | 20 | 174 |
| `lora3`, epoch 4 | 40 | 40 | 20 | 20 | 20 | **20** | 20 | 180 |

Reading raw outputs showed the `n1` failure was not the `direction` key: the wording said "a recommended step" and the model wrote `"recommended_step"` instead of `"recommended_action"`; `so_what` and `metric_refs` were correct. The fix was wording variety for the action field, with the exact phrases "recommended step" and "what to do next" kept out of training. Dev is optimistic: the dev wordings were written in the same session as the training wordings.

### Scored once on seeds 400 to 439 (no grammar, greedy)

| | Plans /80 | Drafts /80 | `reword` /40 | `reword_b` /40 | `reword_c` /40 | Renamed /40 | Repair /40 | One item /40 |
|---|---|---|---|---|---|---|---|---|
| Base (`base-s400`) | 65 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `lora3`, epoch 4 (`lora3-e4-lora`) | 80 | 80 | 40 | 40 | 40 | 40 | 40 | 40 |

Wilson 95% lower bounds: about 0.954 for 80/80, 0.912 for 40/40. The merged change is 1.46% of the weight norm.

GGUF check (llama-cpp-python 0.3.36, T4, unspent seeds 500 to 511, greedy, no grammar): `lora3` file plans 24/24, drafts 24/24, 8/8 under `reword_b`; official file drafts 0/8. All other tensors match the official file exactly and the 24 `down_proj` tensors equal official plus delta.

### First real-data test and a bug it found

Data: the UCI Online Retail sheet, converted to our weekly CSV (country group as region, unit-price tercile as product, registered or guest customer as channel, cancellation invoices as returns; discount fixed at 0). 539,392 rows, 53 weeks, 21 segments, median 61 rows per segment per week. Run on a T4 with the `lora3` GGUF, model output unconstrained, three cutoff weeks.

- Before the fix: valid plan and draft each time, but the pipeline halted on all three weeks with `wrong_number ... not confirmed by recompute`. Cause: the attribution rows showed each segment's contribution to the total change, while the independent number check recomputes the segment's own change and returns nothing for a segment with fewer than 30 rows. Mean-type metrics (return rate, discount) differ between contribution and own change; small segments produced `value None`.
- Fix: `attribution_rows` shows each segment's own change and leaves out segments the check cannot recompute. Two unit tests added; the changed test files (report, checks, runner, demo run, approve, architecture, draft) passed.
- After the fix: a full brief and PDF for 3 of 3 weeks.
- An earlier run on the Tableau Superstore sample (2,000 orders, median 8 rows a week) stayed silent for all 7 cutoffs, correctly, because its segments are far under the size minimum.

### Built or read, not applied

An LP-style gradient-free edit (`weight-space-lp` recipe) and an L0-gated sparse rewiring stage were written (`wsp_lp.py`, `wsp_rewire.py`) but not run, because `lora3` already passed. Papers read in full for this round: 2405.09673 (LoRA), 2604.13627 (learning rates), 2606.01967 (training prompts), 2609.36659 (OPSFT), 2405.14734 (SimPO), plus the concept.country edit paper, the owl-rewired-neurons code and the weight-space-lp code. None changed the plan.

### Limits

Synthetic tasks from one generator; no general-capability regression check for the LoRA; one real dataset, one run per week, not a quality score; the change detector's recall (0.21) is unchanged.
