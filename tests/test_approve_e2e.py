"""Approve end to end: the stored report model, the stamped re-render, the re-check and the sent copy (spec 6.7, AC-009; ticket 044)."""

import os
import re
import time
from types import SimpleNamespace

import pikepdf
import pypdf
import pytest
from test_runner import METRICS, DataClient, make_cfg, pdf_text, write_csv

from weekly_analyst import cli
from weekly_analyst.core.pdfscan import scan_pdf
from weekly_analyst.core.store import StateStore
from weekly_analyst.delivery.outbox import Outbox
from weekly_analyst.report.model import ReportModel
from weekly_analyst.runner import run

STAMP = re.compile(r"Approved by ops-lead \d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC")


def _squash(text: str) -> str:
    return " ".join(text.split())


def _body(text: str) -> str:
    """Page text without the stamp: the draft and the approved copy must agree on everything else."""
    return _squash(STAMP.sub("", text).replace("Draft, pending approval", ""))


@pytest.fixture
def briefed(tmp_path, monkeypatch):
    """A finished run that left a brief in the outbox, its CSV in the watched folder, and a token file."""
    cfg = make_cfg(tmp_path)
    cfg.work_dir.mkdir(parents=True)
    folder = tmp_path / "incoming"
    folder.mkdir()
    csv = write_csv(folder / "sales.csv", bump=1.5)
    store = StateStore(cfg.work_dir / "state.db")
    outcome = run(csv, cfg, DataClient(), store)
    assert outcome.kind == "brief", outcome.reasons
    (cfg.work_dir / cli.TOKEN_FILE).write_text("local\n", encoding="utf-8")
    monkeypatch.setattr(cli.getpass, "getuser", lambda: "ops-lead")
    return SimpleNamespace(cfg=cfg, folder=folder, run_id=outcome.run_id)


def approve_argv(b) -> list[str]:
    return [
        "approve", b.run_id,
        "--folder", str(b.folder),
        "--metrics", str(METRICS),
        "--work-dir", str(b.cfg.work_dir),
        "--out-dir", str(b.cfg.out_dir),
    ]


def test_token_approve_writes_stamped_copy_to_sent_folder(briefed):
    assert cli.main(approve_argv(briefed)) == 0
    sent = briefed.cfg.out_dir / "sent" / f"{briefed.run_id}.pdf"
    text = _squash(pdf_text(sent))
    assert "Draft, pending approval" not in text
    assert STAMP.search(text)  # approver name and time
    assert scan_pdf(sent) == []


def test_approved_copy_has_the_draft_body_text_and_pages(briefed):
    assert cli.main(approve_argv(briefed)) == 0
    draft = briefed.cfg.out_dir / f"{briefed.run_id}.pdf"
    sent = briefed.cfg.out_dir / "sent" / f"{briefed.run_id}.pdf"
    assert _body(pdf_text(sent)) == _body(pdf_text(draft))
    assert len(pypdf.PdfReader(str(sent)).pages) == len(pypdf.PdfReader(str(draft)).pages)


def test_approve_refused_when_snapshot_superseded(briefed, capsys):
    newer = write_csv(briefed.folder / "sales-next.csv", bump=2.0)
    later = time.time() + 60
    os.utime(newer, (later, later))  # the newest file in the watched folder is the latest snapshot
    assert cli.main(approve_argv(briefed)) == 1
    assert "superseded" in capsys.readouterr().err
    assert not (briefed.cfg.out_dir / "sent").exists()


def test_approve_without_token_or_tty_sends_nothing(briefed, monkeypatch, capsys):
    (briefed.cfg.work_dir / cli.TOKEN_FILE).unlink()
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)
    assert cli.main(approve_argv(briefed)) == 1
    assert "approve needs" in capsys.readouterr().err
    assert not (briefed.cfg.out_dir / "sent").exists()


def test_stamped_copy_with_planted_uri_is_not_sent(briefed, monkeypatch, capsys):
    real_render = cli.render_pdf

    def render_with_uri(model, out, **kwargs):
        result = real_render(model, out, **kwargs)
        with pikepdf.open(out, allow_overwriting_input=True) as pdf:
            link = pikepdf.Dictionary(
                Type=pikepdf.Name.Annot,
                Subtype=pikepdf.Name.Link,
                Rect=[0, 0, 40, 10],
                A=pikepdf.Dictionary(S=pikepdf.Name.URI, URI=pikepdf.String("https://canary.example/x")),
            )
            pdf.pages[0].obj.Annots = pdf.make_indirect(pikepdf.Array([link]))
            pdf.save(out)
        return result

    monkeypatch.setattr(cli, "render_pdf", render_with_uri)
    assert cli.main(approve_argv(briefed)) == 1
    assert "final checks" in capsys.readouterr().err
    assert not (briefed.cfg.out_dir / "sent").exists()


def test_recheck_blocks_stamped_copy_whose_number_differs_from_draft(briefed, monkeypatch, capsys):
    real_render = cli.render_pdf

    def render_changed(model, out, **kwargs):
        model.kpis[0].value.text = "999,999.00"  # a number no stored shown number confirms
        return real_render(model, out, **kwargs)

    monkeypatch.setattr(cli, "render_pdf", render_changed)
    assert cli.main(approve_argv(briefed)) == 1
    assert "final checks" in capsys.readouterr().err
    assert not (briefed.cfg.out_dir / "sent").exists()


def test_approve_twice_sends_once(briefed, monkeypatch):
    sends: list[str] = []
    real_transmit = Outbox._transmit

    def counting_transmit(self, path, recipient):
        sends.append(recipient)
        real_transmit(self, path, recipient)

    monkeypatch.setattr(Outbox, "_transmit", counting_transmit)
    assert cli.main(approve_argv(briefed)) == 0
    assert cli.main(approve_argv(briefed)) == 1
    assert sends == [cli.LOCAL_SENT]


def test_outbox_row_keeps_report_model_after_run_finishes(tmp_path):
    cfg = make_cfg(tmp_path)
    store = StateStore(tmp_path / "state.db")
    outcome = run(write_csv(tmp_path / "sales.csv", bump=1.5), cfg, DataClient(), store)
    assert outcome.kind == "brief", outcome.reasons
    assert not (cfg.work_dir / "runs" / outcome.run_id).exists()  # the progress file is deleted at finish

    report = store.load_run(outcome.run_id)["payload"]["report"]  # the outbox row is not
    assert ReportModel.from_dict(report["model"]).run_id == outcome.run_id
    assert report["shown_numbers"] and report["structural_numbers"]
    assert report["model_text"].startswith("Weekly sales moved")
    assert report["paper"] == "A4"
