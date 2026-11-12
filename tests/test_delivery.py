"""Outbox, approval and send gates (spec 6.7, AC-009). Tests replace the local transport with a fake sender."""

import os
from datetime import datetime, timedelta
from pathlib import Path

import pikepdf
import pytest
from reportlab.pdfgen import canvas

from weekly_analyst.core.config import Settings
from weekly_analyst.delivery.outbox import DeliveryConfig, DeliveryRefused, Outbox
from weekly_analyst.core.store import StateStore

RECIPIENT = "ops@example.com"
CFG = DeliveryConfig(recipients={RECIPIENT}, max_bytes=10_000_000)


def make_pdf(path: Path, lines: list[str]) -> Path:
    c = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()
    return path


def make_js_pdf(path: Path) -> Path:
    pdf = pikepdf.new()
    pdf.Root.OpenAction = pdf.make_indirect(
        pikepdf.Dictionary(S=pikepdf.Name.JavaScript, JS=pikepdf.String("app.alert(1)"))
    )
    pdf.save(path)
    return path


def ok_stamp(draft_path: Path, approver: str) -> Path:
    return make_pdf(draft_path.with_name("stamped.pdf"), ["Weekly summary", f"Approved by {approver}"])


def no_findings(path: Path) -> list:
    return []


@pytest.fixture
def sent(monkeypatch):
    calls: list[tuple[Path, str]] = []
    monkeypatch.setattr(Outbox, "_transmit", lambda self, path, recipient: calls.append((path, recipient)))
    return calls


@pytest.fixture
def outbox(tmp_path):
    return Outbox(tmp_path / "outbox", StateStore(tmp_path / "state.db"), Settings())


@pytest.fixture
def draft(tmp_path):
    return make_pdf(tmp_path / "draft.pdf", ["Weekly summary", "Revenue held steady"])


def test_nothing_sent_before_approve_in_default_mode(outbox, draft, sent):
    assert DeliveryConfig(recipients={RECIPIENT}, max_bytes=10).mode == "approval"
    outbox.put("run-1", draft, "hash-1", "sum-1")
    assert sent == []


def test_approve_refused_when_snapshot_superseded(outbox, draft, sent):
    outbox.put("run-1", draft, "hash-1", "sum-1")
    stamp_calls: list[Path] = []

    def stamp(path: Path, approver: str) -> Path:
        stamp_calls.append(path)
        return ok_stamp(path, approver)

    with pytest.raises(DeliveryRefused, match="superseded"):
        outbox.approve("run-1", "ops-lead", "hash-2", stamp, no_findings)
    assert stamp_calls == []
    assert sent == []


def test_approve_returns_stamped_path_when_gates_pass(outbox, draft, sent):
    outbox.put("run-1", draft, "hash-1", "sum-1")
    stamped = outbox.approve("run-1", "ops-lead", "hash-1", ok_stamp, no_findings)
    assert stamped.exists()
    assert sent == []  # approve runs the gates and returns the stamped file; the caller sends it
    outbox.send(stamped, RECIPIENT, CFG)
    assert sent == [(stamped, RECIPIENT)]


def test_recipient_outside_allowlist_refused(outbox, draft, sent):
    with pytest.raises(DeliveryRefused, match="allowlist"):
        outbox.send(draft, "eve@example.net", CFG)
    assert sent == []


def test_body_with_raw_rows_refused(outbox, tmp_path, sent):
    rows = make_pdf(tmp_path / "rows.pdf", ["2026-11-01,EU,Widget,12.50"])
    with pytest.raises(DeliveryRefused, match="raw"):
        outbox.send(rows, RECIPIENT, CFG)
    assert sent == []


def test_oversize_refused(outbox, draft, sent):
    tiny = DeliveryConfig(recipients={RECIPIENT}, max_bytes=10)
    with pytest.raises(DeliveryRefused, match="size"):
        outbox.send(draft, RECIPIENT, tiny)
    assert sent == []


def test_stamped_pdf_rescanned_before_send(outbox, draft, sent):
    outbox.put("run-1", draft, "hash-1", "sum-1")
    with pytest.raises(DeliveryRefused, match="final checks"):
        outbox.approve(
            "run-1",
            "ops-lead",
            "hash-1",
            lambda path, approver: make_js_pdf(path.with_name("stamped.pdf")),
            no_findings,
        )
    assert sent == []


def test_stamped_pdf_rechecked_before_send(outbox, draft, sent):
    outbox.put("run-1", draft, "hash-1", "sum-1")
    with pytest.raises(DeliveryRefused, match="final checks"):
        outbox.approve("run-1", "ops-lead", "hash-1", ok_stamp, lambda path: ["wording: because of"])
    assert sent == []


def test_send_refuses_hazardous_file(outbox, tmp_path, sent):
    hazard = make_js_pdf(tmp_path / "hazard.pdf")
    with pytest.raises(DeliveryRefused, match="scan"):
        outbox.send(hazard, RECIPIENT, CFG)
    assert sent == []


def test_auto_mode_still_applies_gates(outbox, draft, tmp_path, sent):
    auto = DeliveryConfig(mode="auto", recipients={RECIPIENT}, max_bytes=10_000_000)
    rows = make_pdf(tmp_path / "rows.pdf", ["2026-11-01,EU,Widget,12.50"])
    with pytest.raises(DeliveryRefused):
        outbox.send(rows, RECIPIENT, auto)
    with pytest.raises(DeliveryRefused):
        outbox.send(draft, "eve@example.net", auto)
    assert sent == []
    outbox.send(draft, RECIPIENT, auto)
    assert sent == [(draft, RECIPIENT)]


def test_purge_after_30_days(tmp_path, outbox, draft):
    now = datetime(2026, 11, 9, 12, 0)
    old = (now - timedelta(days=31)).timestamp()
    outbox.put("old-run", draft, "hash-1", "sum-1")
    outbox.put("new-run", draft, "hash-1", "sum-1")
    outbox.send(draft, RECIPIENT, CFG)  # real local transport writes a sent copy
    os.utime(tmp_path / "outbox" / "old-run.pdf", (old, old))
    os.utime(tmp_path / "outbox" / "sent" / "draft.pdf", (old, old))
    assert outbox.purge_expired(now) == 2
    assert (tmp_path / "outbox" / "new-run.pdf").exists()


def test_run_id_with_path_parts_refused(outbox, draft):
    with pytest.raises(ValueError):
        outbox.put("../escape", draft, "hash-1", "sum-1")
