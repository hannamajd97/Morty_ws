"""Recording I/O without a robot: stop verdicts, the image writer, bag
verification and the export rules (2026-09-30). Needs the ROS Python
environment sourced (episode_recorder imports rclpy / cv2)."""
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from vla_experiments.recording_contract import parse_stop_arg


# ------------------------------------------------------------- stop verdicts ----
@pytest.mark.parametrize("arg,outcome", [("ok", "success"), ("success", "success"), ("fail", "failure"),
                                         ("unknown", "unknown"), ("interrupted", "interrupted"),
                                         ("", "unknown"), ("sucess", "unknown"), ("  ", "unknown")])
def test_stop_argument_never_defaults_to_success(arg, outcome):
    got, note, _known = parse_stop_arg(arg)
    assert got == outcome
    if outcome == "unknown" and arg.strip() not in ("unknown",):
        assert "unknown" in note


def test_misspelled_stop_is_reported_as_unrecognised():
    assert parse_stop_arg("sucess")[2] is False


# -------------------------------------------------------------- image writer ----
class _Log:
    def __init__(self):
        self.lines = []

    def warn(self, m):
        self.lines.append(m)


def _writer():
    from vla_experiments.episode_recorder import _ImageWriter
    return _ImageWriter(_Log())


def test_drain_waits_for_the_write_in_progress(tmp_path, monkeypatch):
    """The queue is empty while the worker still writes the last image:
    drain must wait for THAT write (it used to return at the empty queue)."""
    import vla_experiments.episode_recorder as er
    started, release = threading.Event(), threading.Event()

    def slow_imwrite(path, img, params):
        started.set()
        release.wait(5.0)
        Path(path).write_bytes(b"x")
        return True

    monkeypatch.setattr(er.cv2, "imwrite", slow_imwrite)
    w = _writer()
    w.put(tmp_path / "a.jpg", np.zeros((4, 4, 3), np.uint8), 90)
    assert started.wait(2.0)
    assert w._q.empty()                      # the item is taken, the write is not done
    assert w.drain(timeout_s=0.3) is False   # outstanding write -> not drained
    release.set()
    assert w.drain(timeout_s=3.0) is True
    assert (tmp_path / "a.jpg").exists()


def test_imwrite_false_and_exceptions_are_failures(tmp_path, monkeypatch):
    import vla_experiments.episode_recorder as er
    calls = iter([False, RuntimeError("disk full")])

    def bad_imwrite(path, img, params):
        r = next(calls)
        if isinstance(r, Exception):
            raise r
        return r

    monkeypatch.setattr(er.cv2, "imwrite", bad_imwrite)
    w = _writer()
    w.put(tmp_path / "a.jpg", np.zeros((4, 4, 3), np.uint8), 90)
    w.put(tmp_path / "b.jpg", np.zeros((4, 4, 3), np.uint8), 90)
    assert w.drain(3.0)
    assert w.failed == 2
    assert any("returned False" in p for p in w.failed_paths)
    assert any("disk full" in p for p in w.failed_paths)


def test_jpeg_bytes_are_written_as_is(tmp_path):
    from vla_experiments.episode_recorder import _Jpeg
    w = _writer()
    w.put(tmp_path / "a.jpg", _Jpeg(b"\xff\xd8abc"), 90)
    assert w.drain(3.0) and (tmp_path / "a.jpg").read_bytes() == b"\xff\xd8abc" and w.failed == 0


# ------------------------------------------------------------ bag checking ----
def _make_bag(d: Path, n_js=5, n_meta=None, with_db=True):
    d.mkdir(parents=True)
    if with_db:
        con = sqlite3.connect(d / "bag_0.db3")
        con.execute("create table topics(id integer primary key, name text, type text, serialization_format text,"
                    " offered_qos_profiles text)")
        con.execute("create table messages(id integer primary key, topic_id integer, timestamp integer, data blob)")
        con.execute("insert into topics values (1, '/joint_states', 'sensor_msgs/msg/JointState', 'cdr', '')")
        for i in range(n_js):
            con.execute("insert into messages(topic_id, timestamp, data) values (1, ?, x'00')", (int(1e18) + i,))
        con.commit()
        con.close()
    n_meta = n_js if n_meta is None else n_meta
    (d / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  version: 5\n  storage_identifier: sqlite3\n"
        "  relative_file_paths:\n    - bag_0.db3\n"
        f"  message_count: {n_meta}\n  topics_with_message_count:\n"
        f"    - topic_metadata:\n        name: /joint_states\n        type: sensor_msgs/msg/JointState\n"
        f"      message_count: {n_meta}\n")


def test_bag_ok(tmp_path):
    from vla_experiments.bag_check import verify_bag
    _make_bag(tmp_path / "bag")
    r = verify_bag(tmp_path / "bag", required_topics=["/joint_states"])
    assert r["ok"], r["issues"]


def test_bag_metadata_without_database(tmp_path):
    from vla_experiments.bag_check import verify_bag
    _make_bag(tmp_path / "bag", with_db=False)
    r = verify_bag(tmp_path / "bag")
    assert not r["ok"] and "missing" in r["issues"][0]


def test_bag_lfs_pointer_and_corrupt_database(tmp_path):
    from vla_experiments.bag_check import verify_bag
    _make_bag(tmp_path / "a/bag", with_db=False)
    (tmp_path / "a/bag/bag_0.db3").write_bytes(b"version https://git-lfs.github.com/spec/v1\noid sha256:1\n")
    assert "LFS pointer" in verify_bag(tmp_path / "a/bag")["issues"][0]
    _make_bag(tmp_path / "b/bag", with_db=False)
    (tmp_path / "b/bag/bag_0.db3").write_bytes(b"SQLite format 3\x00" + b"\x00" * 200)
    assert not verify_bag(tmp_path / "b/bag")["ok"]


def test_bag_count_mismatch_and_missing_topic_and_dirty_exit(tmp_path):
    from vla_experiments.bag_check import verify_bag
    _make_bag(tmp_path / "bag", n_js=5, n_meta=9)
    r = verify_bag(tmp_path / "bag", required_topics=["/joint_states", "/tf"])
    assert any("metadata says 9" in i for i in r["issues"])
    assert any("/tf has no messages" in i for i in r["issues"])
    (tmp_path / "bag_status.json").write_text(json.dumps({"state": "exited", "exit_code": -9, "stop_stage": 3}))
    assert any("did not exit cleanly" in i for i in verify_bag(tmp_path / "bag")["issues"])


def test_bag_startup_failure_leaves_no_metadata(tmp_path):
    from vla_experiments.bag_check import verify_bag
    (tmp_path / "bag").mkdir()
    r = verify_bag(tmp_path / "bag")
    assert not r["ok"] and "no metadata.yaml" in r["issues"][0]


# ------------------------------------------------------------------- export ----
def _episode(tmp_path, segs, idx, outcome="success", quality=None, label_source="command_lifecycle"):
    from vla_experiments.recording_contract import RAW_FORMAT
    n = len(segs)
    d = tmp_path / f"ep_{outcome}_{label_source}"
    d.mkdir()
    t = np.arange(n) / 10.0
    q = np.cumsum(np.full((n, 6), 0.01), axis=0)
    steps = {"t": t, "stamp_ros": 1000 + t, "joint_pos": q, "joint_vel": np.zeros((n, 6)),
             "joint_cmd": np.full((n, 6), np.nan), "tcp_pos": q[:, :3].copy(), "tcp_quat": np.tile([0, 0, 0, 1.0], (n, 1)),
             "gripper": np.ones(n), "gripper_cmd": np.ones(n), "segment": np.asarray(segs, float),
             "instruction_idx": np.asarray(idx, float)}
    np.savez(d / "steps.npz", **steps)
    meta = {"format": RAW_FORMAT, "task": "a", "fps": 10, "outcome": outcome, "label_source": label_source,
            "instructions": [{"idx": 0, "instruction": "a"}, {"idx": 1, "instruction": "b"}],
            "quality": quality or {"integrity": {"ok": True, "issues": []}, "alignment": {"ok": True, "issues": []},
                                   "semantic": {"review_required": label_source != "command_lifecycle", "flags": []},
                                   "issues": []}}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def test_segments_keep_instructions_and_actions_stop_at_gaps(tmp_path):
    from vla_experiments.raw_dataset import Episode, derive_actions, training_segments
    from vla_experiments.recording_contract import SEGMENT_HOLD as H, SEGMENT_IDLE as I, SEGMENT_TASK as T
    segs = [I, T, T, H, T, T, I, T, T, T]
    idx = [-1, 0, 0, 0, 0, 0, -1, 1, 1, 1]
    ep = Episode(_episode(tmp_path, segs, idx))
    s = dict(training_segments(ep))
    assert list(s["a"]) == [1, 2, 4, 5] and list(s["b"]) == [7, 8, 9]
    a = derive_actions(ep, s["a"])["joint_delta"]
    assert np.allclose(a[1, :6], 0.0)          # step 2 is before the excluded hold: no motion to step 4
    assert np.allclose(a[0, :6], 0.01)         # step 1 -> 2, inside the run
    assert np.allclose(a[-1, :6], 0.0)         # run end
    with_hold = dict(training_segments(ep, include=("hold",)))
    assert list(with_hold["a"]) == [1, 2, 3, 4, 5] and list(with_hold["b"]) == [7, 8, 9]
    with_idle = dict(training_segments(ep, include=("idle",)))
    assert list(with_idle["a"]) == [0, 1, 2, 4, 5] and list(with_idle["b"]) == [6, 7, 8, 9]
    with pytest.raises(ValueError):
        training_segments(ep, include=("everything",))


def test_select_episodes_rules_are_separate(tmp_path):
    from vla_experiments.raw_dataset import select_episodes
    from vla_experiments.recording_contract import SEGMENT_TASK as T
    ok = _episode(tmp_path, [T] * 4, [0] * 4)
    fail = _episode(tmp_path, [T] * 4, [0] * 4, outcome="failure")
    legacy = _episode(tmp_path, [T] * 4, [0] * 4, outcome="success", label_source="legacy_intent_arrival")
    names = lambda r: sorted(p.name for p in r)  # noqa: E731
    assert names(select_episodes(tmp_path)) == [ok.name]
    assert names(select_episodes(tmp_path, outcomes=("success", "failure"))) == sorted([ok.name, fail.name])
    assert names(select_episodes(tmp_path, allow_review=True)) == sorted([ok.name, legacy.name])


def test_builder_options_and_legacy_all():
    import importlib.util
    p = Path(__file__).resolve().parents[1] / "rlds_builder" / "morty_ur3e" / "morty_ur3e_dataset_builder.py"
    src = p.read_text()
    ns = {}
    # export_options is plain Python; run just that function without tensorflow
    start = src.index("def export_options")
    end = src.index("\nclass ", start)
    exec("import os\n" + src[start:end], ns)
    opt = ns["export_options"]({})
    assert opt["outcomes"] == ("success",) and not opt["allow_quality"] and not opt["allow_review"] and opt["include"] == ()
    legacy = ns["export_options"]({"MORTY_VLA_ALL": "1"})
    assert set(legacy["outcomes"]) == {"success", "failure", "unknown", "interrupted"}
    assert legacy["allow_quality"] and legacy["allow_review"] and legacy["include"] == ()   # segmentation kept
    mixed = ns["export_options"]({"MORTY_VLA_OUTCOMES": "success,unknown", "MORTY_VLA_INCLUDE": "hold"})
    assert mixed["outcomes"] == ("success", "unknown") and mixed["include"] == ("hold",)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
