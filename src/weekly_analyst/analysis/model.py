"""Model client interface: real llama.cpp client and a scripted stand-in for tests."""

from pathlib import Path
from typing import Protocol


class ModelClient(Protocol):
    def complete(
        self, prompt: str, json_schema: dict, *, temperature: float, seed: int, grammar: str | None = None
    ) -> str: ...


class LlamaCppClient:
    def __init__(self, gguf_path: Path, threads: int, n_ctx: int) -> None:
        from llama_cpp import Llama  # lazy: only real runs need llama-cpp-python

        self._llm = Llama(
            model_path=str(gguf_path), n_threads=threads, n_ctx=n_ctx, verbose=False
        )

    def complete(
        self, prompt: str, json_schema: dict, *, temperature: float, seed: int, grammar: str | None = None
    ) -> str:
        if grammar is None:
            constraint = {"response_format": {"type": "json_object", "schema": json_schema}}
        else:
            # A GBNF grammar replaces the JSON response format: both at once leave no valid output.
            from llama_cpp import LlamaGrammar  # lazy: only the free-SQL call needs it

            constraint = {"grammar": LlamaGrammar.from_string(grammar)}
        out = self._llm.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            seed=seed,
            **constraint,
        )
        return out["choices"][0]["message"]["content"]


class ScriptedClient:
    """Returns the scripted responses in order. Asking for more than scripted raises IndexError.

    grammars records the grammar of every call, None for calls without one.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls = 0
        self.grammars: list[str | None] = []

    def complete(
        self, prompt: str, json_schema: dict, *, temperature: float, seed: int, grammar: str | None = None
    ) -> str:
        self.calls += 1
        self.grammars.append(grammar)
        return self._responses[self.calls - 1]
