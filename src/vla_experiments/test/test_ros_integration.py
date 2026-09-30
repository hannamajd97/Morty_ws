"""The ROS integration checks (real router + recorder against fakes) on an
isolated domain — run with MORTY_ROS_INTEGRATION=1 (they take ~2 min and
need the workspace sourced); skipped otherwise."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
pytestmark = pytest.mark.skipif(os.environ.get("MORTY_ROS_INTEGRATION") != "1",
                                reason="set MORTY_ROS_INTEGRATION=1 (isolated ROS domain 77)")


def _run(script, *args):
    env = dict(os.environ, ROS_DOMAIN_ID=os.environ.get("MORTY_TEST_DOMAIN", "77"))
    r = subprocess.run([sys.executable, str(HERE / script), *args], env=env, capture_output=True, text=True,
                       timeout=600)
    print(r.stdout[-4000:])
    assert r.returncode == 0 and "ALL PASSED" in r.stdout, r.stdout[-2000:] + r.stderr[-2000:]


def test_router_pick_lifecycle():
    _run("ros_integration_router_pick.py")


def test_recording_lifecycle_end_to_end(tmp_path):
    _run("ros_integration_lifecycle.py", str(tmp_path / "work"))
