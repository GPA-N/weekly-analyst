"""Outbox, approval and send gates (spec 6.7, AC-009).

Sending is code only, outside model tools. approve() runs the gates on the stamped report and returns
its path; send() runs the delivery gates and writes to the local sent folder. The caller sends after approve.
"""

import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Literal, Protocol

import pypdf

from weekly_analyst.core.config import Settings
from weekly_analyst.core.pdfscan import scan_pdf
from weekly_analyst.core.store import StateStore

SENT = "sent"
_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# A CSV-shaped source row: an ISO date as the first field, then a delimiter.
_RAW_ROW = re.compile(r"^\s*\d{4}-\d{2}-\d{2}[,;\t]")


class DeliveryRefused(RuntimeError):
    """A gate refused the brief. Nothing was sent."""


@dataclass(frozen=True, kw_only=True)
class DeliveryConfig:
    mode: Literal["approval", "auto"] = "approval"
    recipients: set[str]
    max_bytes: int


class Outbox:
    def __init__(self, dir: Path, store: StateStore, settings: Settings) -> None:
        self._dir = Path(dir)
        self._store = store
        self._settings = settings
        self._dir.mkdir(parents=True, exist_ok=True)

    def put(
        self, run_id: str, pdf_path: Path, snapshot_hash: str, metrics_checksum: str, report: dict | None = None
    ) -> None:
        """report is the stored report bundle approve re-renders and re-checks from; the run folder is deleted at finish."""
        _check_run_id(run_id)
        draft = self._dir / f"{run_id}.pdf"
        shutil.copyfile(pdf_path, draft)
        self._store.save_state(
            run_id,
            "outbox",
            {
                "draft": str(draft),
                "snapshot_hash": snapshot_hash,
                "metrics_checksum": metrics_checksum,
                "report": report,
            },
        )

    def approve(
        self,
        run_id: str,
        approver: str,
        latest_snapshot_hash: str,
        render_stamped: Callable[[Path, str], Path],
        recheck: Callable[[Path], list],
    ) -> Path:
        _check_run_id(run_id)
        run = self._store.load_run(run_id)
        if run is None or run["state"] != "outbox":
            raise DeliveryRefused(f"no pending brief for run {run_id}")
        payload = run["payload"]
        if payload["snapshot_hash"] != latest_snapshot_hash:
            raise DeliveryRefused("brief is for a superseded snapshot; nothing approved")
        stamped = render_stamped(Path(payload["draft"]), approver)
        problems = [*scan_pdf(stamped), *recheck(stamped)]
        if problems:
            raise DeliveryRefused(f"stamped report failed the final checks: {problems}")
        self._store.save_state(run_id, "approved", {**payload, "approver": approver, "stamped": str(stamped)})
        return stamped

    def send(self, path: Path, recipient: str, config: DeliveryConfig) -> None:
        if recipient not in config.recipients:
            raise DeliveryRefused("recipient is not on the allowlist")
        if Path(path).stat().st_size > config.max_bytes:
            raise DeliveryRefused("report is over the size cap")
        if scan_pdf(path):
            raise DeliveryRefused("report failed the static scan")
        if _has_raw_rows(path):
            raise DeliveryRefused("body contains raw data rows")
        self._transmit(path, recipient)

    def purge_expired(self, now: datetime) -> int:
        cutoff = (now - timedelta(days=self._settings.outbox_days)).timestamp()
        removed = 0
        for folder in (self._dir, self._dir / SENT):
            if not folder.is_dir():
                continue
            for item in folder.iterdir():
                if item.is_file() and item.stat().st_mtime < cutoff:
                    item.unlink()
                    removed += 1
        return removed

    def _transmit(self, path: Path, recipient: str) -> None:
        """Local delivery: the PDF is written to the sent folder."""
        sent = self._dir / SENT
        sent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, sent / Path(path).name)


class Sender(Protocol):
    """Delivers one approved PDF to a recipient. An implementation refuses what the gates refuse."""

    def send(self, path: Path, recipient: str, config: DeliveryConfig) -> None: ...


class FileSender:
    """Default Sender: the gated local send, which writes the PDF to the sent folder."""

    def __init__(self, outbox: Outbox) -> None:
        self._outbox = outbox

    def send(self, path: Path, recipient: str, config: DeliveryConfig) -> None:
        self._outbox.send(path, recipient, config)


def _check_run_id(run_id: str) -> None:
    if not _RUN_ID.match(run_id):
        raise ValueError(f"run id {run_id!r} is not a plain identifier")


def _has_raw_rows(path: Path) -> bool:
    reader = pypdf.PdfReader(str(path))
    for page in reader.pages:
        for line in (page.extract_text() or "").splitlines():
            if _RAW_ROW.match(line):
                return True
    return False
