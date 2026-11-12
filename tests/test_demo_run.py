"""Hosted-demo driver end to end with a stand-in model (the real model is exercised on Colab)."""

import pypdf

from test_runner import DataClient
from weekly_analyst.analysis.model import ScriptedClient
from weekly_analyst.demo import approve_demo, run_demo


def _text(path):
    return "\n".join(page.extract_text() or "" for page in pypdf.PdfReader(str(path)).pages)


def test_quiet_scenario_is_silent_with_no_model_call(tmp_path):
    demo = run_demo("quiet", ScriptedClient([]), tmp_path)  # any model call raises IndexError
    assert demo.kind == "silent"
    ok, message, pdf = approve_demo(demo)
    assert not ok and pdf is None and "no brief" in message


def test_spike_scenario_makes_a_brief_and_approve_stamps_it(tmp_path):
    demo = run_demo("spike", DataClient(), tmp_path)
    assert demo.kind == "brief", demo.reasons
    assert demo.draft is not None and "Draft, pending approval" in _text(demo.draft)
    ok, message, pdf = approve_demo(demo)
    assert ok, message
    stamped = _text(pdf)
    assert "Draft, pending approval" not in stamped
    assert "approved" in stamped.lower()


def test_approve_twice_does_not_send_twice(tmp_path):
    demo = run_demo("drop", DataClient(), tmp_path)
    assert demo.kind == "brief", demo.reasons
    assert approve_demo(demo)[0]
    ok, _, _ = approve_demo(demo)
    assert not ok
    assert len(list((demo.out_dir / "sent").glob("*.pdf"))) == 1
