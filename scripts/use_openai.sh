#!/usr/bin/env bash
# Source this file to point Morty at OpenAI directly. Requires OPENAI_API_KEY_DIRECT
# in ~/morty_ws/.morty_env (a real sk-... key from platform.openai.com).
#   source ~/morty_ws/use_openai.sh
#   ros2 launch communication_llm voice_robot.launch.py

# Resolve the workspace from this script's own location (…/<ws>/scripts/), so it
# works whatever the folder is called (normally ~/morty_ws).
WS_DIR="${WS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$WS_DIR/.morty_env"

if [[ -z "${OPENAI_API_KEY_DIRECT:-}" ]]; then
    echo "[morty] ERROR: OPENAI_API_KEY_DIRECT is not set in $WS_DIR/.morty_env"
    echo "[morty] Add a line like:  export OPENAI_API_KEY_DIRECT=\"sk-...\""
    return 1 2>/dev/null || exit 1
fi

export OPENAI_API_KEY="$OPENAI_API_KEY_DIRECT"
unset OPENAI_BASE_URL
# MORTY_MODEL_OVERRIDE (set by morty_gui) wins over .morty_env (2026-08-11).
export MORTY_LLM_MODEL="${MORTY_MODEL_OVERRIDE:-${OPENAI_MODEL:-gpt-5.4-mini}}"
unset MORTY_REASONING_EFFORT   # not needed for native OpenAI models

echo "[morty] provider = OpenAI direct"
echo "[morty]   model  = $MORTY_LLM_MODEL"
