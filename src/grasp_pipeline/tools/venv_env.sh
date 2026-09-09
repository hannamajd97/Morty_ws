#!/usr/bin/env bash
# Source this AFTER activating your venv.
#
# Purpose: keep ROS2 Humble Python packages + this workspace packages visible,
# while avoiding accidentally importing system `/usr/local` NumPy 2.x (which
# breaks ROS Humble cv_bridge built against NumPy 1.x).
#
# Usage:
#   cd ~/morty_ws
#   source .venv_grasp/bin/activate
#   source src/grasp_pipeline/venv_env.sh

# NOTE: Do not use `set -u` here. ROS2 setup scripts reference unset variables
# and `set -u` would make sourcing / chaining fragile.
set -e
set -o pipefail

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "venv_env.sh: activate a venv first (VIRTUAL_ENV is empty)" >&2
  return 1
fi

PYVER="$("$VIRTUAL_ENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
WS_DIR="${WS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"

export PYTHONNOUSERSITE=1
unset PYTHONHOME || true

# Force a clean python path so we don't accidentally inherit `/usr/local/...` NumPy 2.x
PYTHONPATH=""

PYTHONPATH="${VIRTUAL_ENV}/lib/python${PYVER}/site-packages:${PYTHONPATH}"
PYTHONPATH="${WS_DIR}/install/lib/python${PYVER}/site-packages:${PYTHONPATH}"
PYTHONPATH="${WS_DIR}/install/local/lib/python${PYVER}/dist-packages:${PYTHONPATH}"
PYTHONPATH="/opt/ros/humble/lib/python${PYVER}/site-packages:${PYTHONPATH}"
PYTHONPATH="/opt/ros/humble/local/lib/python${PYVER}/dist-packages:${PYTHONPATH}"

export PYTHONPATH

export COLCON_PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
export AMENT_PYTHON_EXECUTABLE="${VIRTUAL_ENV}/bin/python"
