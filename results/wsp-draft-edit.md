# Draft-format weight edit: what was tried, what worked, what is still open (2026-11-11)

## The problem

On the held-out synthetic tasks the model scored 0/80 on drafts when no grammar was used. The first weight edit fixed the plans (64/80 to 80/80) but left drafts at 0/80.
A diagnostic run found the cause: the draft prompt never contains the JSON schema, only the grammar does. Without a grammar the base model invents keys (`finding` instead of `findings`, an `items` list, a `cold_start` key).
With the schema written into the prompt text, the unedited model passes 40 of 40 dev drafts. So the failing target is "write the exact schema without being shown it", which is a format default to be written into the weights.

## What did not work

Adding or flipping a residual direction through a few MLP columns (the method of the first edit) stayed at 0/40 dev drafts in every configuration that still produced JSON. Layers 15 and 19, six mid layers, the six layers WeightWatcher flags as under-trained (alpha above 9), and the five best-trained ones all failed this way; stronger edits broke the model (JSON validity fell to 0 to 13 of 40). Files: `wsp-draft-sweep-d1..d4.json`.
This matches IWIBench (arXiv 2611.08647), where no editor model could set a structured-output default by weight editing.

## What worked: a closed-form MEMIT-style edit of down_proj

Scripts: `edit/wsp_memit_core.py`, `wsp_memit.py`, `wsp_mfinal.py`, `wsp_mfinal2.py`.
1. Take the gold draft (built by code) for each calibration prompt (seeds 0 to 19). Feed it in with the correct prefix. At every structural position (a key name, bracket, quote or comma, not a free-text word) where the model's top-1 token is not the gold token, record the MLP input there (the key) and the residual shift that would raise the gold token by 3 logits (the value, one gradient-direction estimate; no weight is trained).
2. For each chosen layer solve the least-squares update dW = R^T K (K^T K + lam * n * C + eps I)^-1, where C is the mean covariance of all MLP inputs on the calibration prompts. The ridge term keeps other positions close to unchanged. The target shift is split evenly across the layers.
3. Layers 16, 17, 18, 20, 21, 22 (WeightWatcher alpha above 9 on down_proj), lam 3, scale 2, chosen on dev seeds 20 to 39 by the rule "best dev score, then smallest edit". The weight change is 2.6% of the norm of those matrices. Dev drafts went from 0/40 to 40/40 at scale 2 on the three leading layer sets (13 to 19 of 40 at scale 1; at scale 4 the same sets still scored 39 to 40 of 40, with a weight change twice as large).
WeightWatcher result: the least-trained layers by alpha gave the smallest edit with the best score; the best-trained layers (alpha 4.0 to 4.7) did worst (at most 33/40). The difference among the three leading layer sets was small (40/40 each), so this is a weak signal, not a proof.

## Held-out results (no grammar, greedy, 80 tasks each)

Seeds 100 to 139 (the same set the first edit was scored on; variants were compared here, so these numbers are optimistic):

| Model | Plans | Drafts |
|---|---|---|
| Base | 64/80 | 0/80 |
| MEMIT edit alone | 66/80 | 80/80 |
| MEMIT edit cut to rank 8 per layer | 66/80 | 80/80 |
| MEMIT edit plus the earlier plan edit | 80/80 | 75/80 |
| Random shift targets (not size matched, 9 times larger change) | 54/80 | 0/80 |

Seeds 200 to 239 (fresh, nothing was chosen on them):

| Model | Plans | Drafts |
|---|---|---|
| Base | 62/80 | 0/80 |
| MEMIT edit plus the earlier plan edit | 80/80 | 75/80 (93.8%) |
| Rank-8 MEMIT edit plus the plan edit | 80/80 | 73/80 (91.3%) |
| Random shift targets rescaled to the same norm, no plan edit | 62/80 | 0/80 |

InsightBench narrative check (100 cases, ROUGE-L, weak): base 0.1305, MEMIT edit 0.1316, rank-8 plus plan edit 0.1331. Read it only as "no sign of damage".
Failing drafts that were inspected show one token-level slip, for example the key written as `"findings[":` (one string for the key and the bracket). The cause is a guess: the edit was fitted along the gold prefix only, so a path the model takes on its own is not covered.

## Limits that must be stated with these numbers

- Synthetic tasks from one generator, one prompt template, one scorer that checks format (schema, length, no digits, item keys), not writing quality.
- The edit is dense inside six layers (a full 1024 by 3584 update each). The rank-8 form is about 221,000 values, not the 16,384 values of the first edit.
- These edits are measured in the transformers (safetensors) model. They are NOT yet applied to the GGUF in the Space, and the live agent still runs the old 16,384-value patch.
- Production decodes under a grammar, where drafts already pass, so the real-world gain from this edit is small; it matters for runs without a grammar.

## Rank sweep (dev seeds 20-39, 40 tasks, no grammar)

Truncating the MEMIT update to rank r per layer (six layers): rank 1 scored 0/40 drafts, rank 2 0/40, rank 4 5/40, rank 8 reached 73 to 80 of 80 on the held-out seeds. So rank 8 (about 221,000 values) is the smallest rank tested that works; rank 16 was not run (the session was lost). Plans stayed at the base level (36/40 on dev) in these truncated runs without the plan edit. Files: `wsp-g4-mfinal2.json` (rank 1) and `wsp-g5-mfinal2.json` (ranks 2 and 4).

## On-policy rounds: a negative result

Job o1 (`edit/wsp_onpolicy.py`): the model samples its own draft, the frozen base model with the schema in the prompt gives its top-1 token at every position of that text, and the MEMIT update is fitted where the two disagree on structural tokens, added to a running edit.
Round 1: dev drafts 0/40, only 20/40 outputs were valid JSON (relative weight change 7.4%, three times the gold-fitted edit). Round 2: 0/40 and 28/40 valid JSON (9.5%). The edit damaged the output format without producing the schema. So this on-policy variant, as implemented, did not work and is not a fix for the 5 to 7 failing drafts. Possible reasons, not tested: the student's own text on the unedited model is a long way from the schema, so the teacher's top-1 at each position is a noisy target; one gradient step per position is a weak estimate when many positions are corrected at once. Files: `wsp-o1-onpolicy.json`.

## Open at the time of writing

- Out-of-distribution prompts (reworded instruction, the production re-request line, one evidence item, renamed metrics and segments): job h1 for base and the gold-fitted edit; the on-policy edit is checked after o1 finishes.
- Applying the edit to the GGUF and re-running the live Space check.
