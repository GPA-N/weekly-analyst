# Tune/score split, LoRA and closed-form edits (2026-11-11)

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
- Nothing here is deployed to the Space or the GGUF.
