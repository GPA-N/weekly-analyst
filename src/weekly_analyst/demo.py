"""Hosted-demo driver: one scenario through the real runner, then the real approve command.

Used by the Hugging Face Space and by the end-to-end checks on Colab. Synthetic data only, nothing is uploaded,
and the only place a report goes is the local sent folder. Not part of the watch/run product path.
"""

import contextlib
import io
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from weekly_analyst import cli
from weekly_analyst.analysis.model import ModelClient
from weekly_analyst.core.store import StateStore
from weekly_analyst.runner import RunConfig, run
from weekly_analyst.sim.demo import write_demo_csv

METRICS = Path(__file__).parent / "sim" / "metrics_sales.yaml"


@dataclass
class DemoRun:
    scenario: str
    folder: Path
    run_id: str
    kind: str  # silent, brief or halted
    reasons: list[str]
    draft: Path | None
    seconds: float
    log: list[str] = field(default_factory=list)

    @property
    def csv_folder(self) -> Path:
        return self.folder / "data"

    @property
    def work_dir(self) -> Path:
        return self.folder / "work"

    @property
    def out_dir(self) -> Path:
        return self.folder / "out"


def run_demo(scenario: str, client: ModelClient, base_dir: Path) -> DemoRun:
    folder = Path(base_dir) / f"{scenario}-{uuid.uuid4().hex[:8]}"
    (folder / "data").mkdir(parents=True)
    (folder / "work").mkdir()
    csv = write_demo_csv(folder / "data" / "sales.csv", scenario)
    cfg = RunConfig(
        metrics_path=METRICS,
        work_dir=folder / "work",
        out_dir=folder / "out",
        delivery_mode="approval",
        recipients=["ops@example.com"],
        paper="A4",
    )
    store = StateStore(folder / "work" / cli.STATE_DB)
    started = time.perf_counter()
    outcome = run(csv, cfg, client, store)
    seconds = round(time.perf_counter() - started, 1)
    return DemoRun(scenario, folder, outcome.run_id, outcome.kind, list(outcome.reasons), outcome.report_path, seconds)


def approve_demo(demo: DemoRun) -> tuple[bool, str, Path | None]:
    """Run the real approve command as the local user (a token file stands in for the terminal). Returns ok, message, PDF."""
    if demo.kind != "brief":
        return False, "there is no brief waiting for approval", None
    (demo.work_dir / cli.TOKEN_FILE).write_text("local demo user\n", encoding="utf-8")
    argv = [
        "approve",
        demo.run_id,
        "--folder",
        str(demo.csv_folder),
        "--metrics",
        str(METRICS),
        "--work-dir",
        str(demo.work_dir),
        "--out-dir",
        str(demo.out_dir),
    ]
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code = cli.main(argv)
    sent = sorted((demo.out_dir / "sent").glob("*.pdf")) if (demo.out_dir / "sent").exists() else []
    if code == 0 and sent:
        return True, "approved, re-checked and written to the sent folder", sent[-1]
    return False, err.getvalue().strip() or f"approve failed with code {code}", None
