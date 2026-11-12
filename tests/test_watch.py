"""Folder watcher containment and the approve gate (spec 6.5, 6.7; ticket 023)."""

import os

import pytest

from weekly_analyst import cli
from weekly_analyst.analysis import plan
from weekly_analyst.delivery import watch


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "incoming"
    root.mkdir()
    return root


@pytest.fixture
def csv(folder):
    path = folder / "sales.csv"
    path.write_text("date,units\n2026-11-02,5\n", encoding="utf-8")
    return path


@pytest.fixture
def outside(tmp_path):
    path = tmp_path / "elsewhere.csv"
    path.write_text("date,units\n2026-11-02,9\n", encoding="utf-8")
    return path


def test_file_inside_folder_is_accepted(csv, folder):
    assert watch.accept(csv, folder) is True


def test_path_outside_folder_is_rejected(outside, folder):
    assert watch.accept(outside, folder) is False


def test_dotdot_is_rejected_even_when_it_resolves_inside(csv, folder):
    (folder / "sub").mkdir()
    assert watch.accept(folder / "sub" / ".." / csv.name, folder) is False


def test_symlink_escape_is_rejected(folder, outside):
    link = folder / "link.csv"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks cannot be created on this machine ({exc}); escape check not exercised")
    assert watch.accept(link, folder) is False


def test_scan_lists_only_accepted_csv_files(folder, csv, outside):
    (folder / "notes.txt").write_text("not data", encoding="utf-8")
    try:
        os.symlink(outside, folder / "escape.csv")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks cannot be created on this machine ({exc}); escape check not exercised")
    assert set(watch.scan(folder)) == {csv}


def test_approver_needs_interactive_terminal_or_token_file(tmp_path):
    token = tmp_path / "approve.token"
    with pytest.raises(cli.ApproveRefused):
        cli.approver_for(token, interactive=False)
    token.write_text("local\n", encoding="utf-8")
    assert cli.approver_for(token, interactive=False)
    assert cli.approver_for(token, interactive=True)


def test_nul_stdin_is_not_an_interactive_terminal(monkeypatch):
    # Windows reports the NUL device as a TTY to isatty(); redirected stdin must not count as interactive.
    with open(os.devnull, encoding="utf-8") as null:
        monkeypatch.setattr("sys.stdin", null)
        assert cli._is_interactive() is False


def test_approve_without_tty_or_token_refuses_and_sends_nothing(tmp_path, folder, csv, monkeypatch, capsys):
    from weekly_analyst.delivery.outbox import Outbox

    sent: list = []
    monkeypatch.setattr(Outbox, "_transmit", lambda self, path, recipient: sent.append(path))
    monkeypatch.setattr(cli, "_is_interactive", lambda: False)
    rc = cli.main(
        [
            "approve",
            "run-20261109T120000-abcd1234",
            "--folder",
            str(folder),
            "--metrics",
            str(tmp_path / "metrics.yaml"),
            "--work-dir",
            str(tmp_path / "work"),
            "--out-dir",
            str(tmp_path / "out"),
        ]
    )
    assert rc != 0
    assert sent == []
    assert "approve needs" in capsys.readouterr().err


def test_approve_is_not_reachable_from_any_model_tool():
    tools = set().union(*plan.STATE_ALLOWLIST.values())
    assert "approve" not in tools
    assert "send" not in tools
