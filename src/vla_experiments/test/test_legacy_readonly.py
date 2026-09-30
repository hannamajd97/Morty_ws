"""The named historical trials, READ-ONLY (skipped where absent): the legacy
reader still works, the known defects are found, and validation / repair
never modify an original (hashes before and after)."""
import hashlib
import json
from pathlib import Path

import pytest

DAY = Path.home() / "morty_ws" / "experiments" / "experiments_29_09_2026"


def _trial(prefix):
    hits = sorted(DAY.glob(prefix + "_*")) if DAY.exists() else []
    if not hits:
        pytest.skip(f"{prefix} not on this machine")
    return hits[0]


def _hashes(d: Path):
    out = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and "images" not in p.parts:
            out[str(p.relative_to(d))] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_legacy_reader_still_reads_old_episodes():
    from vla_experiments.raw_dataset import Episode, derive_actions, training_segments
    ep = Episode(_trial("trial_114"))
    assert ep.label_source == "legacy_intent_arrival" and ep.review_required
    segs = training_segments(ep)
    assert segs and derive_actions(ep, segs[0][1])["eef_delta"].shape[1] == 7


def test_validate_finds_114_121_126_and_changes_nothing(tmp_path):
    from vla_experiments.validate import validate_trial, write_repair
    trials = [_trial(p) for p in ("trial_114", "trial_121", "trial_126")]
    before = {t: _hashes(t) for t in trials}
    reports = {t.name[:9]: validate_trial(t) for t in trials}
    f114 = " ".join(reports["trial_114"]["labels"]["findings"])
    assert "8 ms apart" in f114 and "0 step(s)" in f114
    assert any("spans a grasp motion start" in f for f in reports["trial_121"]["labels"]["findings"])
    assert any("then=release_home" in f for f in reports["trial_126"]["labels"]["findings"])
    rec114 = [i["instruction"] for i in reports["trial_114"]["labels"]["reconstruction"]["instructions"]]
    assert rec114[:2] == ["look to the right", "go back home"]
    rec126 = [i["instruction"] for i in reports["trial_126"]["labels"]["reconstruction"]["instructions"]]
    assert "open the gripper and go back home" in rec126
    for t in trials:
        write_repair(t, tmp_path / "repaired", *reports[t.name[:9]]["_repair"])
    for t in trials:
        assert _hashes(t) == before[t], f"{t.name} was modified"
    m = json.loads((tmp_path / "repaired" / trials[0].name / "meta.json").read_text())
    assert m["label_source"] == "reconstructed_from_router_log" and m["quality"]["semantic"]["review_required"]
    assert m["relabelled_from"]["path"] == str(trials[0])


def test_relabel_is_dry_run_by_default_and_refuses_in_place(tmp_path, capsys):
    from vla_experiments.relabel import main
    t = _trial("trial_126")
    before = _hashes(t)
    assert main([str(t)]) == 0
    assert "dry run" in capsys.readouterr().out
    assert main(["--apply", str(t)]) == 2
    assert main(["--out", str(tmp_path / "out"), str(t)]) == 0
    assert _hashes(t) == before
    assert (tmp_path / "out" / t.name / "meta.json").exists()


def test_bags_of_dialogue_only_trials_verify_on_this_machine():
    from vla_experiments.bag_check import verify_bag
    for p in ("trial_116", "trial_117", "trial_124"):
        r = verify_bag(_trial(p) / "bag")
        assert r["ok"], (p, r["issues"])
