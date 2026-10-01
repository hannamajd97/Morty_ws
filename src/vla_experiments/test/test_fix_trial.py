"""✎ fix trial (morty_gui.fix_trial) on COPIES of a saved trial, the CSVs and
the workbook (skipped where absent): every place the form lives is
corrected, the original data never touched."""
import csv
import json
import shutil
from pathlib import Path

import pytest

EXP = Path.home() / "morty_ws" / "experiments"
SRC = EXP / "experiments_30_09_2026" / "trial_143_LA_S3"


@pytest.fixture
def copies(tmp_path, monkeypatch):
    if not (SRC / "trial.json").exists() or not (EXP / "MMHRI_experiments.xlsx").exists():
        pytest.skip("trial 143 / the workbook not on this machine")
    F = pytest.importorskip("morty_gui.fix_trial")
    d = tmp_path / "experiments_30_09_2026" / SRC.name
    d.mkdir(parents=True)
    for p in SRC.iterdir():
        if p.is_file() and p.stat().st_size < 50_000_000:
            shutil.copy2(p, d / p.name)
    (tmp_path / "mmhri").mkdir()
    for p in (EXP / "mmhri").glob("*.csv"):
        shutil.copy2(p, tmp_path / "mmhri" / p.name)
    shutil.copy2(EXP / "MMHRI_experiments.xlsx", tmp_path / "wb.xlsx")
    monkeypatch.setattr(F, "MMHRI_DIR", str(tmp_path / "mmhri"))
    monkeypatch.setattr(F, "BACKUPS", tmp_path / "backups")
    monkeypatch.setattr(F, "EXPERIMENTS", tmp_path)
    return F, d, tmp_path


def test_fix_corrects_every_place_and_renames(copies):
    F, d, tmp = copies
    before = (SRC / "trial.json").read_bytes()
    lines = F.fix_trial(d, {"Scenario": "S4", "Full Task Success?": "no", "Failure Category / Notes": "t"},
                        xlsx=str(tmp / "wb.xlsx"))
    nd = d.with_name("trial_143_LA_S4")
    assert nd.exists() and not d.exists(), lines
    tj = json.loads((nd / "trial.json").read_text())
    assert tj["Scenario"].startswith("S4:") and tj["Full Task Success?"] == "no" and tj["Corrections"]
    m = json.loads((nd / "meta.json").read_text())
    assert m["outcome"] == "failure" and m["success"] is False and nd.name in m["episode_dir"]
    rows = [r for r in csv.DictReader(open(tmp / "mmhri" / "mmhri_meta.csv")) if r["Trial ID"] == "143"]
    assert rows[-1]["Scenario"].startswith("S4:")
    import openpyxl
    ws = openpyxl.load_workbook(tmp / "wb.xlsx")["Experiments"]
    h = {c.value: i + 1 for i, c in enumerate(ws[1])}
    r = next(r for r in range(2, ws.max_row + 1) if str(ws.cell(r, h["Trial ID"]).value) == "143")
    assert ws.cell(r, h["Scenario ID"]).value == "S4" and ws.cell(r, h["VLA outcome"]).value == "failure"
    assert ws.cell(r, h["Folder"]).value == nd.name
    assert (SRC / "trial.json").read_bytes() == before            # the real trial untouched


def test_no_change_and_dry_run_write_nothing(copies):
    F, d, tmp = copies
    form = F.read_form(d)
    assert F.fix_trial(d, form, xlsx=str(tmp / "wb.xlsx"))[0].endswith("no change")
    out = F.fix_trial(d, {"Scene Layout": "Layout C"}, dry_run=True, xlsx=str(tmp / "wb.xlsx"))
    assert out[-1].startswith("dry run") and d.exists() and not (tmp / "backups").exists()
    with pytest.raises(ValueError):
        F.fix_trial(d, {"Scenario": "S9"}, xlsx=str(tmp / "wb.xlsx"))
