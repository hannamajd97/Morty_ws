#!/usr/bin/env bash
set -euo pipefail

# Resolve the workspace from this script's own location (<ws>/tools/) rather than
# a hardcoded name, so experiments run against the workspace they live in.
# (Edited 2026-08-06 — this copy lives in ~/morty_ws.)
WS_DIR="${WS_DIR:-$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)}"
LOG_DIR="${GRASP_LOG_DIR:-$WS_DIR/src/experiment_logs}"
OLD_YOLO_MODEL="$WS_DIR/src/grasp_pipeline/new_y26s_seg.pt"
OLD_YOLO_CLASSES="kitchen_towel, retail_box, vitamin_tube"
NEW9_YOLO_MODEL="$WS_DIR/yolo26s-seg_24_04.pt"
NEW9_YOLO_CLASSES="tape_dispenser, cd_case, toy_faucet, kitchen_towel, nutmeg_jar, retail_box, small_bottle, wooden_maze_puzzle, vitamin_c"

usage() {
  cat <<'EOF'
Usage:
  run_grasp_experiment.sh [--tag NAME] [--model-preset old|new9] [extra ros2 launch args...]

Examples:
  run_grasp_experiment.sh --tag vitamin_tube_run14
  run_grasp_experiment.sh --tag vitamin_tube_old_run1 --model-preset old
  run_grasp_experiment.sh --tag retail_box_run2 registration_motion_timeout_s:=150

This wrapper:
  - sources the workspace
  - launches the grasp pipeline with the common experiment defaults
  - saves the full raw log
  - saves a compact summary log with the key experiment lines
EOF
}

tag="run"
model_preset="new9"
extra_args=()

while (($# > 0)); do
  case "$1" in
    --tag)
      if (($# < 2)); then
        echo "--tag requires a value" >&2
        exit 2
      fi
      tag="$2"
      shift 2
      ;;
    --model-preset)
      if (($# < 2)); then
        echo "--model-preset requires old or new9" >&2
        exit 2
      fi
      model_preset="$2"
      shift 2
      ;;
    --new9)
      model_preset="new9"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      extra_args+=("$1")
      shift
      ;;
  esac
done

case "$model_preset" in
  old)
    ACTIVE_YOLO_MODEL="$OLD_YOLO_MODEL"
    ACTIVE_YOLO_CLASSES="$OLD_YOLO_CLASSES"
    ;;
  new9)
    ACTIVE_YOLO_MODEL="$NEW9_YOLO_MODEL"
    ACTIVE_YOLO_CLASSES="$NEW9_YOLO_CLASSES"
    ;;
  *)
    echo "unknown --model-preset '$model_preset' (expected old or new9)" >&2
    exit 2
    ;;
esac

mkdir -p "$LOG_DIR"
timestamp="$(date +%Y%m%d_%H%M%S)"
safe_tag="$(printf '%s' "$tag" | tr ' /:' '___')"
raw_log="$LOG_DIR/${timestamp}_${safe_tag}.log"
summary_log="$LOG_DIR/${timestamp}_${safe_tag}_summary.log"
meta_log="$LOG_DIR/${timestamp}_${safe_tag}_meta.txt"
structured_json="$LOG_DIR/${timestamp}_${safe_tag}_structured.json"
run_csv="$LOG_DIR/${timestamp}_${safe_tag}_run.csv"
views_csv="$LOG_DIR/${timestamp}_${safe_tag}_views.csv"

write_summary() {
  if [[ ! -f "$raw_log" ]]; then
    return 0
  fi

  grep -E \
    '\[continuous\] direction score details:|\[continuous\] one-direction score selected|\[continuous\] accepted frame|\[continuous\] accepted final frame|\[continuous\] sweep leg|\[continuous\] stopping sweep|rejected for border clipping|\[grasp-utility\] view_[0-9_]+: voxel_gain|\[grasp-utility\] view_[0-9_]+: U=|\[grasp-utility\] view_[0-9_]+: delta_U=|\[grasp-utility\] final_cloud: U=' \
    "$raw_log" > "$summary_log" || true

  {
    echo "timestamp=$timestamp"
    echo "tag=$tag"
    echo "raw_log=$raw_log"
    echo "summary_log=$summary_log"
    echo "workspace=$WS_DIR"
    echo "structured_json=$structured_json"
    echo "run_csv=$run_csv"
    echo "views_csv=$views_csv"
    echo "model_preset=$model_preset"
    echo "yolo_model=$ACTIVE_YOLO_MODEL"
    echo "yolo_classes=$ACTIVE_YOLO_CLASSES"
    echo "launch_defaults=yolo_world_model:=$ACTIVE_YOLO_MODEL sam_model_path:=$ACTIVE_YOLO_MODEL classes=$ACTIVE_YOLO_CLASSES registration_dynamic_nbv_enabled:=true registration_continuous_observation_enabled:=true registration_tsdf_enabled:=true registration_motion_timeout_s:=120"
    if ((${#extra_args[@]} > 0)); then
      printf 'extra_args='
      printf '%s ' "${extra_args[@]}"
      printf '\n'
    else
      echo "extra_args="
    fi
  } > "$meta_log"

  python3 "$WS_DIR/src/tools/parse_grasp_experiment.py" \
    --raw-log "$raw_log" \
    --json-out "$structured_json" \
    --run-csv-out "$run_csv" \
    --views-csv-out "$views_csv" >/dev/null 2>&1 || true
}

trap write_summary EXIT

set +u
source "$WS_DIR/install/setup.bash"
set -u

echo "raw_log=$raw_log"
echo "summary_log=$summary_log"
echo "meta_log=$meta_log"
echo "structured_json=$structured_json"
echo "run_csv=$run_csv"
echo "views_csv=$views_csv"
echo "model_preset=$model_preset"
echo "yolo_model=$ACTIVE_YOLO_MODEL"
echo "yolo_classes=$ACTIVE_YOLO_CLASSES"

set +e
ros2 launch grasp_pipeline run_grasp_pipeline.launch.py \
  yolo_world_model:="$ACTIVE_YOLO_MODEL" \
  sam_model_path:="$ACTIVE_YOLO_MODEL" \
  classes:="$ACTIVE_YOLO_CLASSES" \
  registration_dynamic_nbv_enabled:=true \
  registration_continuous_observation_enabled:=true \
  registration_tsdf_enabled:=true \
  registration_motion_timeout_s:=120 \
  "${extra_args[@]}" 2>&1 | tee "$raw_log"
launch_rc=${PIPESTATUS[0]}
set -e

exit "$launch_rc"
