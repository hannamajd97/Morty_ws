"""GUI recording logic without a display (QT_QPA_PLATFORM=offscreen): the
run acknowledgment and the post-processing verdict label (2026-09-30)."""
import json
import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PyQt5.QtWidgets")


class FakeBridge:
    def __init__(self, status):
        self.status = status
        self.cmds = []

    def vla_status(self):
        return dict(self.status)

    def vla_cmd(self, text):
        self.cmds.append(text)

    def vla_node_up(self):
        return True


@pytest.fixture(scope="module")
def app():
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_vla_ack_needs_this_run_id(app):
    from morty_gui.vla_link import VlaLink
    b = FakeBridge({"stale": [], "recording": True, "run_id": "someone-else"})
    link = VlaLink(b)
    assert link.begin("task", "/tmp/x") == ""
    assert b.cmds[0].startswith("run ") and b.cmds[-1].startswith("trial ")
    assert link.ack() == ("waiting", "")                        # a cached status of another run is no ack
    b.status = {"recording": True, "run_id": link.run_id}
    assert link.ack() == ("recording", "")
    b.status = {"recording": False, "run_id": "", "last_run": {"run_id": link.run_id, "state": "refused",
                                                                "reason": "not started: stale wrist"}}
    assert link.ack() == ("refused", "not started: stale wrist")


def test_postprocess_verdict_is_shown_and_never_assumed(app, tmp_path):
    from PyQt5.QtTest import QTest
    from morty_gui.mmhri_window import MMHRIWindow
    w = MMHRIWindow(bridge=None)
    assert hasattr(w, "assist_btn") and hasattr(w, "framing_btn") and not w.assist_btn.isEnabled()
    ok_dir, bad_dir = tmp_path / "trial_ok", tmp_path / "trial_bad"
    ok_dir.mkdir(); bad_dir.mkdir()
    w._watch_postprocess(str(ok_dir))
    assert "verifying" in w.pp_label.text()
    (ok_dir / "postprocess_status.json").write_text(json.dumps(
        {"ok": True, "vla": {"ok": True, "ready": {"episode": "complete"}}, "trial_times": {"ok": True}}))
    QTest.qWait(2600)
    assert w.pp_label.text().startswith("✔") and "complete" in w.pp_label.text()
    w._watch_postprocess(str(bad_dir))
    (bad_dir / "postprocess_status.json").write_text(json.dumps(
        {"ok": False, "vla": {"ok": False, "errors": ["bag: bag_0.db3: missing"]}, "trial_times": {"ok": True}}))
    QTest.qWait(2600)
    assert w.pp_label.text().startswith("⚠") and "NOT verified" in w.pp_label.text() and "missing" in w.pp_label.text()
