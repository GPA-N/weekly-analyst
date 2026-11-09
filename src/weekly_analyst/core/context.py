"""RunContext: the non-component services a system needs, passed in and never global (architecture.md).

core imports nothing from the other packages, so the metrics file and the model client are typed Any here.
Their real types are MetricsFile (data) and ModelClient (analysis); the systems use them by those names.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from weekly_analyst.core.config import Settings
from weekly_analyst.core.store import StateStore
from weekly_analyst.core.system import Halt


@dataclass
class RunContext:
    settings: Settings
    metrics: Any  # MetricsFile
    client: Any  # ModelClient
    store: StateStore
    out_dir: Path
    paper: str
    recipients: list[str]
    started: float
    clock: Callable[[], float] = time.monotonic
    cache: dict[str, Any] = field(default_factory=dict)  # per-attempt memo: snapshot frame, query engine

    def check_time(self) -> None:
        """Halt the run once its wall time reaches run_timeout_s (spec 6.2)."""
        if self.clock() - self.started >= self.settings.run_timeout_s:
            raise Halt(f"timeout: run stopped at {self.settings.run_timeout_s} s")
