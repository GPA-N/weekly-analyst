"""Command line: watch, run, approve (spec 6.5, 6.7; ticket 023).

approve is the only path that can send. It needs an interactive terminal or a local token file that the
local user creates by hand. No model tool reaches it (plan.STATE_ALLOWLIST has no approve entry).
Run as: python -m weekly_analyst.cli <command> ...
"""

import argparse
import getpass
import logging
import os
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

from weekly_analyst.analysis.checks import check_report
from weekly_analyst.analysis.model import LlamaCppClient
from weekly_analyst.core.config import Settings
from weekly_analyst.core.scheduler import file_sha256
from weekly_analyst.core.store import StateStore
from weekly_analyst.core.text import numerals
from weekly_analyst.data.extract import ExtractError, extract
from weekly_analyst.data.metrics_file import load_metrics_file
from weekly_analyst.delivery import watch
from weekly_analyst.delivery.outbox import DeliveryConfig, DeliveryRefused, Outbox
from weekly_analyst.report import render_pdf
from weekly_analyst.report.layout import stamp_label
from weekly_analyst.report.model import ReportModel, ReportTooLong, ShownNumber
from weekly_analyst.runner import RunConfig, RunLocked, run
from weekly_analyst.systems.check import pdf_text

TOKEN_FILE = "approve.token"
STATE_DB = "state.db"
LOCAL_SENT = "local-sent-folder"  # the local sent folder is the only transport; approve sends to it through the gates
POLL_S = 30.0  # watch polls the folder this often (seconds)
N_CTX = 2048  # llama.cpp context length; a library-style default, not measured (spike ticket 028 decides)


class ApproveRefused(RuntimeError):
    """approve ran without an interactive terminal or a local token file."""


def approver_for(token_path: Path, *, interactive: bool) -> str:
    token = Path(token_path)
    local_token = token.is_file() and not token.is_symlink() and token.stat().st_size > 0
    if not (interactive or local_token):
        raise ApproveRefused("approve needs an interactive terminal or a local token file; nothing approved")
    return getpass.getuser()


def _is_interactive() -> bool:
    # isatty() is True for the Windows NUL device; a real console also answers the terminal-size query.
    try:
        os.get_terminal_size(sys.stdin.fileno())
    except (OSError, ValueError, AttributeError):
        return False
    return True


class _Approval:
    """What approve re-renders and re-checks: the stored report bundle, the metrics file and the latest CSV."""

    def __init__(self, report: dict, approver: str, approved_at: str, metrics, csv: Path, work: Path) -> None:
        self._report = report
        self._approver = approver
        self._approved_at = approved_at
        self._metrics = metrics
        self._csv = csv
        self._work = work

    def render_stamped(self, draft: Path, approver: str) -> Path:
        """Re-render the stored ReportModel as approved: the same model, so the same text and numbers as the draft."""
        out = self._work / Path(draft).name
        try:
            render_pdf(
                ReportModel.from_dict(self._report["model"]),
                out,
                status="approved",
                approver=approver,
                paper=self._report["paper"],
                approved_at=self._approved_at,
            )
        except (ReportTooLong, ValueError) as err:
            raise DeliveryRefused(f"stamp: {err}") from None
        return out

    def recheck(self, path: Path) -> list[str]:
        """Check the stamped text against the stored numbers, on a snapshot rebuilt from the same CSV."""
        snapshot_dir = self._work / "snapshot"
        snapshot_dir.mkdir()
        try:
            snapshot = extract(self._csv, self._metrics, snapshot_dir, Settings())
        except ExtractError as err:
            return [f"snapshot: {err}"]
        if snapshot.snapshot_hash != self._report["model"]["snapshot_hash"]:
            return ["snapshot: the source changed during approve"]
        shown = [ShownNumber.from_dict(d) for d in self._report["shown_numbers"]]
        # The approved stamp has numerals of its own (the time), so they are exempt on top of the draft's list.
        stamp = stamp_label("approved", self._approver, self._approved_at)
        exempt = [*self._report["structural_numbers"], *numerals(stamp)]
        failures = check_report(
            pdf_text(path),
            shown,
            self._metrics,
            snapshot.parquet_path,
            exempt=exempt,
            model_text=self._report["model_text"],
        )
        return [f"{failure.kind}: {failure.detail}" for failure in failures]


def _stored_payload(store: StateStore, run_id: str) -> dict:
    row = store.load_run(run_id)
    if row is None or row["state"] != "outbox" or not (row["payload"].get("report") or {}).get("model"):
        raise DeliveryRefused(f"no pending brief with a stored report for run {run_id}; nothing approved")
    return row["payload"]


def _latest_csv(folder: Path) -> Path:
    files = watch.scan(folder)
    if not files:
        raise DeliveryRefused("no accepted CSV in the watched folder; nothing approved")
    return max(files, key=lambda path: files[path][0])


def _model(args) -> LlamaCppClient:
    return LlamaCppClient(args.model, threads=os.cpu_count() or 1, n_ctx=N_CTX)


def _run_config(args) -> RunConfig:
    return RunConfig(
        metrics_path=args.metrics,
        work_dir=args.work_dir,
        out_dir=args.out_dir,
        delivery_mode="approval",
        recipients=list(args.recipient),
        paper=args.paper,
        settings=Settings(),
    )


def _print_outcome(outcome) -> None:
    print(f"{outcome.kind} {outcome.run_id}")
    if outcome.report_path is not None:
        print(f"report {outcome.report_path}")
    for reason in outcome.reasons:
        print(f"reason {reason}")


def _run(args) -> int:
    store = StateStore(args.work_dir / STATE_DB)
    outcome = run(args.csv, _run_config(args), _model(args), store)
    _print_outcome(outcome)
    return 1 if outcome.kind == "halted" else 0


def _watch(args) -> int:
    folder = args.folder.resolve()
    cfg = _run_config(args)
    store = StateStore(args.work_dir / STATE_DB)
    client = _model(args)
    seen = watch.scan(folder)  # files already in the folder at start are baselined, not run
    while True:
        time.sleep(POLL_S)
        current = watch.scan(folder)
        for path, signature in current.items():
            if seen.get(path) == signature:
                continue
            try:
                outcome = run(path, cfg, client, store)
            except RunLocked:
                continue  # not marked seen, so the next poll retries it
            seen[path] = signature
            _print_outcome(outcome)


def _approve(args) -> int:
    approver = approver_for(args.work_dir / TOKEN_FILE, interactive=_is_interactive())
    store = StateStore(args.work_dir / STATE_DB)
    payload = _stored_payload(store, args.run_id)
    metrics = load_metrics_file(args.metrics)
    if metrics.checksum != payload["metrics_checksum"]:
        raise DeliveryRefused("metrics file changed since the brief was built; nothing approved")
    csv = _latest_csv(args.folder)
    approved_at = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    outbox = Outbox(args.out_dir, store, Settings())
    with tempfile.TemporaryDirectory() as tmp:
        approval = _Approval(payload["report"], approver, approved_at, metrics, csv, Path(tmp))
        stamped = outbox.approve(args.run_id, approver, file_sha256(csv), approval.render_stamped, approval.recheck)
        outbox.send(stamped, LOCAL_SENT, DeliveryConfig(recipients={LOCAL_SENT}, max_bytes=Settings().max_bytes))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="weekly-analyst")
    sub = parser.add_subparsers(dest="command", required=True)

    def run_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--metrics", type=Path, required=True, help="metrics file (YAML or JSON)")
        p.add_argument("--model", type=Path, required=True, help="GGUF model path")
        p.add_argument("--work-dir", type=Path, default=Path("work"))
        p.add_argument("--out-dir", type=Path, default=Path("out"))
        p.add_argument("--recipient", action="append", default=[], help="allowlisted recipient (repeatable)")
        p.add_argument("--paper", choices=["A4", "Letter"], default="A4")

    p_watch = sub.add_parser("watch", help="run each new or replaced CSV in a folder")
    p_watch.add_argument("folder", type=Path)
    run_options(p_watch)
    p_watch.set_defaults(handler=_watch)

    p_run = sub.add_parser("run", help="run one CSV snapshot")
    p_run.add_argument("csv", type=Path)
    run_options(p_run)
    p_run.set_defaults(handler=_run)

    p_approve = sub.add_parser("approve", help="approve a pending brief (local user only)")
    p_approve.add_argument("run_id")
    p_approve.add_argument("--folder", type=Path, required=True, help="watched folder holding the latest CSV")
    p_approve.add_argument("--metrics", type=Path, required=True, help="metrics file the brief was built with")
    p_approve.add_argument("--work-dir", type=Path, default=Path("work"))
    p_approve.add_argument("--out-dir", type=Path, default=Path("out"))
    p_approve.set_defaults(handler=_approve)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (ApproveRefused, DeliveryRefused, RunLocked) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
