#!/usr/bin/env bash
# start_graspgen.sh — one-terminal launcher for the GraspGen service + MeshCat.
#
# Starts meshcat-server, waits until it's listening, then launches the GraspGen
# ZMQ service with --viz so clouds + grasps render in the browser. Ctrl-C cleanly
# stops BOTH.
#
# Usage:
#   ./start_graspgen.sh                 # robotiq_2f_140, with MeshCat viz
#   ./start_graspgen.sh --no-viz        # service only, no MeshCat
#   ./start_graspgen.sh --gripper franka_panda
#   ./start_graspgen.sh --bind tcp://0.0.0.0:5555
#   ./start_graspgen.sh --help

set -o pipefail

# ── Defaults ─────────────────────────────────────────────────────────────
# Resolve the workspace from this script's own location (<ws>/scripts/).
# (Edited 2026-08-06 — this copy lives in ~/morty_ws.)
WS="${WS:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
GG_DIR="$WS/GraspGen"
MODELS_DIR="$WS/models/graspgen"
GRIPPER="robotiq_2f_140"
BIND="tcp://127.0.0.1:5555"
THRESHOLD="0.4"
USE_VIZ=1
MESHCAT_BIN="$HOME/.local/bin/meshcat-server"
MESHCAT_WEB="http://127.0.0.1:7000/static/"
MESHCAT_ZMQ_PORT=6000

# ── Parse args ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-viz)   USE_VIZ=0; shift ;;
    --gripper)  GRIPPER="$2"; shift 2 ;;
    --bind)     BIND="$2"; shift 2 ;;
    --threshold) THRESHOLD="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "[start_graspgen] unknown option: $1 (try --help)"; exit 1 ;;
  esac
done

CONFIG="$MODELS_DIR/graspgen_${GRIPPER}.yml"

# ── Sanity checks ────────────────────────────────────────────────────────
if [[ ! -d "$GG_DIR" ]]; then
  echo "[start_graspgen] ERROR: GraspGen not found at $GG_DIR" >&2; exit 1
fi
if [[ ! -e "$CONFIG" ]]; then
  echo "[start_graspgen] ERROR: gripper config not found: $CONFIG" >&2
  echo "  Available:" >&2
  ls "$MODELS_DIR"/*.yml 2>/dev/null | sed 's/^/    /' >&2
  exit 1
fi

MESHCAT_PID=""
SERVICE_PID=""

cleanup() {
  echo ""
  echo "[start_graspgen] shutting down…"
  [[ -n "$SERVICE_PID" ]] && kill "$SERVICE_PID" 2>/dev/null
  [[ -n "$MESHCAT_PID" ]] && kill "$MESHCAT_PID" 2>/dev/null
  sleep 1
  [[ -n "$SERVICE_PID" ]] && kill -9 "$SERVICE_PID" 2>/dev/null
  [[ -n "$MESHCAT_PID" ]] && kill -9 "$MESHCAT_PID" 2>/dev/null
  echo "[start_graspgen] done."
  exit 0
}
trap cleanup INT TERM

# ── Start MeshCat (unless --no-viz) ──────────────────────────────────────
VIZ_FLAG=()
if [[ "$USE_VIZ" -eq 1 ]]; then
  if [[ ! -x "$MESHCAT_BIN" ]]; then
    # fall back to module form if the binary isn't where we expect
    MESHCAT_BIN="python3 -m meshcat.servers.zmqserver"
  fi
  echo "[start_graspgen] starting meshcat-server…"
  $MESHCAT_BIN > /tmp/meshcat_server.log 2>&1 &
  MESHCAT_PID=$!

  # Wait until the meshcat ZMQ port (6000) is listening
  echo "[start_graspgen] waiting for meshcat on port ${MESHCAT_ZMQ_PORT}…"
  for i in $(seq 1 30); do
    if ss -ltn 2>/dev/null | grep -q ":${MESHCAT_ZMQ_PORT}\b"; then
      break
    fi
    if ! kill -0 "$MESHCAT_PID" 2>/dev/null; then
      echo "[start_graspgen] ERROR: meshcat-server died. Log:" >&2
      tail -10 /tmp/meshcat_server.log >&2
      exit 1
    fi
    sleep 0.5
  done
  echo "[start_graspgen] meshcat ready."
  echo "[start_graspgen] >>> open the viewer in your browser: ${MESHCAT_WEB}"
  VIZ_FLAG=(--viz)
fi

# ── Start the GraspGen service ───────────────────────────────────────────
echo "[start_graspgen] launching GraspGen service:"
echo "                 gripper=${GRIPPER}  bind=${BIND}  thr=${THRESHOLD}  viz=${USE_VIZ}"
echo "[start_graspgen] press Ctrl-C to stop everything."
echo ""

cd "$GG_DIR"
python3 scripts/graspgen_service.py \
    --bind "$BIND" \
    --config "$CONFIG" \
    --threshold "$THRESHOLD" \
    "${VIZ_FLAG[@]}" &
SERVICE_PID=$!

wait "$SERVICE_PID"
cleanup
