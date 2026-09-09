#!/usr/bin/env bash
# Source this file to point Morty at AWS Bedrock.
# Automatically picks the right backend based on the model ID:
#   anthropic.* / meta.* / mistral.* / ...  -> native Bedrock Converse (via boto3)
#   openai.* (e.g. openai.gpt-oss-20b-1:0)  -> Bedrock's OpenAI-compatible /openai/v1 endpoint
# Switch models by editing BEDROCK_MODEL in ~/new_ur_gripper_ws/.morty_env.

# Resolve the workspace from this script's own location (…/<ws>/scripts/), so it
# works whatever the folder is called. On the Alienware the name
# ~/new_ur_gripper_ws was already taken by a pre-existing workspace that must not
# be touched, so this copy lives in ~/morty_ws. (Edited 2026-08-06.)
WS_DIR="${WS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "$WS_DIR/.morty_env"

# MORTY_MODEL_OVERRIDE (set by morty_gui) wins over .morty_env (2026-08-11).
export MORTY_LLM_MODEL="${MORTY_MODEL_OVERRIDE:-${BEDROCK_MODEL:-anthropic.claude-haiku-4-5-20251001-v1:0}}"

# Detect backend based on model prefix.
case "$MORTY_LLM_MODEL" in
    openai.*|gpt-*)
        # OpenAI-compatible Bedrock endpoint. Needs OPENAI_API_KEY (use the
        # Bedrock bearer token) and OPENAI_BASE_URL pointing at /openai/v1.
        export OPENAI_API_KEY="$AWS_BEARER_TOKEN_BEDROCK"
        export OPENAI_BASE_URL="https://bedrock-runtime.${AWS_REGION}.amazonaws.com/openai/v1"
        BACKEND="OpenAI-compat (Bedrock /openai/v1)"
        ;;
    *)
        # Native Bedrock Converse via boto3. Reads AWS_BEARER_TOKEN_BEDROCK +
        # AWS_REGION directly from env. Unset OPENAI_BASE_URL so the OpenAI
        # SDK fallback path doesn't kick in if a future call uses an OpenAI model.
        unset OPENAI_BASE_URL
        unset OPENAI_API_KEY
        BACKEND="Bedrock Converse (boto3)"
        ;;
esac

# low | medium | high - only meaningful for gpt-oss-* models
export MORTY_REASONING_EFFORT="${MORTY_REASONING_EFFORT:-low}"

echo "[morty] provider = AWS Bedrock"
echo "[morty]   backend = $BACKEND"
echo "[morty]   region  = $AWS_REGION"
echo "[morty]   model   = $MORTY_LLM_MODEL"
case "$MORTY_LLM_MODEL" in
    openai.*|gpt-*) echo "[morty]   reasoning_effort = $MORTY_REASONING_EFFORT" ;;
esac
