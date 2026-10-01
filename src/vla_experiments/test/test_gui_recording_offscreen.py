"""GUI recording logic without a display (QT_QPA_PLATFORM=offscreen): the
run acknowledgment and the post-processing verdict label (2026-09-30)."""
import csv
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
    assert hasattr(w, "handover_btn") and hasattr(w, "framing_btn") and not w.handover_btn.isEnabled()
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


def test_trial_window_layout_stop_then_save(app):
    """2026-10-01 (Majd): Stop and Save are separate buttons, in his order;
    assist / rearranged / camera blocked / abort are gone; the notes field
    takes several lines; the progress table shows only what is done."""
    from morty_gui.mmhri_window import MMHRIWindow
    w = MMHRIWindow(bridge=None)
    def row_of(btn):
        lay = btn.parentWidget().layout()
        for i in range(lay.count()):
            sub = lay.itemAt(i).layout()
            if sub is not None and any(sub.itemAt(j).widget() is btn for j in range(sub.count())):
                return [sub.itemAt(j).widget().text() for j in range(sub.count()) if sub.itemAt(j).widget()]
        return []
    assert row_of(w.start_btn) == ["⏺ Start trial", "⏹ Stop", "💾 Save", "✖ discard", "📊 Send last trial", "🗑 delete"]
    assert row_of(w.fix_btn) == ["✎ fix trial", "📊 Excel export", "🤝 hand-over", "🎯 framing check…"]
    assert not hasattr(w, "assist_btn") and not w.save_btn.isEnabled()
    assert isinstance(w.notes, QtWidgets.QPlainTextEdit) and w.notes.height() > 40
    w._refresh_progress()
    txt = w.progress_lbl.text()
    assert "/5" not in txt and "/105" not in txt and "total done:" in txt
    # a stopped, unsaved trial blocks the next Start
    w._stopped, w._saved, w._cancelled = True, False, False
    w._on_start()
    assert "not saved yet" in w.status.text()


def test_save_writes_the_form_as_it_is_after_stop(app, tmp_path, monkeypatch):
    """💾 Save after ⏹ Stop: trial.json and the CSV rows from the form at
    Save time (multi-line notes kept), and "Full Task Success?" given after
    Stop becomes the VLA episode's outcome once its manifest is complete."""
    import morty_gui.mmhri_window as mw
    from PyQt5.QtTest import QTest
    monkeypatch.setattr(mw, "MMHRI_DIR", str(tmp_path / "mmhri"))
    (tmp_path / "mmhri").mkdir()
    trial = tmp_path / "trial_999_LA_S3"
    trial.mkdir()
    (trial / "meta.json").write_text(json.dumps({"outcome": "unknown", "success": None}))
    (trial / "recording_manifest.json").write_text(json.dumps({"state": "complete", "outcome": "unknown"}))
    w = mw.MMHRIWindow(bridge=None)
    monkeypatch.setattr(w, "_start_postprocess", lambda root: None)
    w._bag = type("B", (), {"last_dir": str(trial / "bag")})()
    w._vla = type("V", (), {"last_episode": trial.name})()
    w._trial_dir = w._trial_root = str(trial)
    w._rec_header = {"tid": "999", "layout": "Layout A", "scenario": mw.SCENARIOS[2], "level": "L1"}
    w._stopped, w._saved, w._cancelled, w._stage_times = True, False, False, {"total_s": 12.0}
    w.chips["full"].setCurrentText("yes")
    w.notes.setPlainText("first line\nsecond line")
    w._on_save()
    tj = json.loads((trial / "trial.json").read_text())
    assert tj["Full Task Success?"] == "yes" and tj["Failure Category / Notes"] == "first line\nsecond line"
    rows = list(csv.DictReader(open(tmp_path / "mmhri" / "mmhri_experiments.csv")))
    assert rows[-1]["Trial ID"] == "999" and rows[-1]["Failure Category / Notes"] == "first line\nsecond line"
    assert "saved:" in w.status.text() and w.start_btn.isEnabled() and not w.save_btn.isEnabled()
    for _ in range(40):
        QTest.qWait(100)
        if json.loads((trial / "meta.json").read_text())["outcome"] == "success":
            break
    assert json.loads((trial / "meta.json").read_text())["outcome"] == "success"
    assert json.loads((trial / "recording_manifest.json").read_text())["outcome"] == "success"
