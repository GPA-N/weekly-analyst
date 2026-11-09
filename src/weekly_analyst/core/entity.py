"""Entity: one run. It is an id and nothing else; the run's data lives in its components (architecture.md)."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Entity:
    run_id: str
