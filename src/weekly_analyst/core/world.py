"""World: the component store for all runs (architecture.md).

A run's components save to ``runs_dir/<run_id>/progress.json`` through an atomic write (temp file, fsync,
replace), the same write the runner used before this module existed.
"""

import json
import os
from pathlib import Path
from typing import Any, TypeVar

from weekly_analyst.core.components import COMPONENTS
from weekly_analyst.core.entity import Entity

PROGRESS_FILE = "progress.json"
T = TypeVar("T")


class World:
    def __init__(self, runs_dir: Path) -> None:
        self.runs_dir = Path(runs_dir)
        self._parts: dict[Entity, dict[type, Any]] = {}

    def run_dir(self, entity: Entity) -> Path:
        return self.runs_dir / entity.run_id

    def add(self, entity: Entity, component: Any) -> None:
        self._parts.setdefault(entity, {})[type(component)] = component

    def get(self, entity: Entity, kind: type[T]) -> T:
        return self._parts[entity][kind]

    def has(self, entity: Entity, kind: type) -> bool:
        return kind in self._parts.get(entity, {})

    def save(self, entity: Entity) -> None:
        run_dir = self.run_dir(entity)
        components = self._parts.get(entity, {})
        payload = {
            "run_id": entity.run_id,
            "components": {kind.__name__: part.to_json() for kind, part in components.items()},
        }
        tmp = run_dir / f"{PROGRESS_FILE}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True, default=str))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, run_dir / PROGRESS_FILE)

    def load(self, run_id: str) -> Entity:
        raw = (self.runs_dir / run_id / PROGRESS_FILE).read_text(encoding="utf-8")
        entity = Entity(run_id)
        parts = json.loads(raw)["components"]
        self._parts[entity] = {
            COMPONENTS[name]: COMPONENTS[name].from_json(body) for name, body in parts.items()
        }
        return entity
