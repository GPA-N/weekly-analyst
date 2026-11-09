# 046 SQL writing at f16 (official vs finetune), 30 questions

Status: DONE (run by the router on a Colab T4, session stopped). Records: spike/results/sql-f16-official.jsonl and sql-f16-finetune.jsonl. Numbers: docs/spike-sql-f16-results.json. Script: spike/spike_sql.py, grammar: spike/grammar/select_sales.gbnf.

Correction: the first SQL spike (ticket 030, Q4_K_M) used temperature 0 (greedy); the temperature 0.7 setting belonged to the model spike (028). The only things that changed here are f16 instead of Q4_K_M and the arms below.

## Rules (set before any f16 number was seen)

- An arm is adopted over the baseline (arm "default": server-default sampling, one attempt) only if it gets at least 3 more of the 30 questions correct.
- Bar for the shipped SQL path: at least 24 of 30 correct (0.80) after all adopted techniques; otherwise the fallback is: the agent leans on the fixed tools (attribute_change, wilson, concentration) plus a small library of named parameterised queries that the model fills, and free SQL is only a last resort.
- The finetune is adopted only if plan_valid_rate >= official AND query correct exceeds official by at least 3 of 30 AND memory fits (ticket 045).

## Results (correct / wrong / invalid of 30, Clopper-Pearson 95% interval on correct)

| Arm | Official f16 | Finetune f16 |
| --- | --- | --- |
| default sampling, 1 attempt | 18 / 4 / 8 (0.41 to 0.77) | 20 / 7 / 3 (0.47 to 0.83) |
| greedy, 1 attempt | 16 / 3 / 11 (0.34 to 0.72) | 18 / 6 / 6 (0.41 to 0.77) |
| greedy + GBNF grammar | 21 / 4 / 5 (0.51 to 0.85) | 0 / 0 / 30 (artifact, see below) |
| greedy + up to 2 repair rounds | 17 / 10 / 3 (0.37 to 0.75), 1.6 attempts | 19 / 9 / 2 (0.44 to 0.80), 1.3 attempts |

Mean completion tokens per query: official about 21, finetune about 94 to 136.

## Verdicts

- Official: grammar gets +3 over default (21 vs 18), which meets the adoption threshold, so constrained decoding is adopted for the model's SQL. Greedy (-2) and repair (-1) are not adopted: repair turns invalid queries into valid but wrong ones (wrong 4 to 10). Best official arm: 21 of 30 = 0.70, below the 24 of 30 bar. The fallback applies: fixed tools plus named parameterised queries; free SQL is a last resort.
- Finetune: best arm 20 of 30 (default), which is +2 over the official default and +/-0 vs official's best, so it does not exceed the official model by 3, and its plan validity (45: 48 of 50 vs 50 of 50) is lower. Not adopted.
- The finetune grammar result (0 of 30) is an artifact, not a measure of SQL skill: the grammar forces the reply to start with SELECT, but the finetune writes a wrapper first (it averages 4 to 5 times more tokens), so message content came back empty. Not tested fairly; a grammar that allows its own format would be needed.

## What failed

Official, grammar arm, still failing: E05, E07, E08, E17 (the same four failed with the default arm: product units, region revenue for online channel, returned-order counts, return share), plus N02, N03, N04, N07, N09 (top-3, HAVING, ratio, STDDEV, CASE share). The repeated easy failures (E05, E07, E08, E17) suggest schema ambiguity in the prompt (the returned boolean, qty versus units). A prompt change would have to be measured on the same 30 questions and adopted only if it wins.

## Caveats

- 30 questions on one small synthetic table: every interval is wide (about plus or minus 0.17). A 3-question difference is at the edge of what this can show.
- GPU f16 decoding; CPU decoding can differ slightly.
- The 24 of 30 bar is not met by any arm, so this is a design input, not a pass.
