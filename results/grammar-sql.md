# 048 grammar-constrained SQL: result

Status: DONE (finished and verified by the router; the haiku worker was stopped after it had written all four steps and before it wrote this file).

Verification: `python -m pytest tests/test_sqlgrammar.py tests/test_plan.py tests/test_query.py` gave 40 passed. Full suite `python -m pytest -q`: 619 passed in 558.29 s, zero failures (605 before plus 14 new), including tests/test_injection.py.

What exists: analysis/sqlgrammar.py (build_select_grammar, build_sql_prompt), ModelClient.complete(..., grammar=None) with a lazy LlamaGrammar in LlamaCppClient and recording in ScriptedClient, plan.py (local_query args are question and required_columns, SQL in a plan is rejected, named_query is preferred in the plan prompt), systems/tools.py (_free_sql: one grammar-constrained call, no repair, LocalQuery lockdown decides).

Not yet verified: the grammar has been tested with llama-server on Colab (ticket 046, 21 of 30 correct) in its spike form, but not through llama-cpp-python's LlamaGrammar with the generated text. That check runs on Colab with the real model before the scored evals.
