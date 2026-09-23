"""
llm_service.py
==============

LLM backend abstraction for ``chat_node``. Hides which provider is in use
(OpenAI direct, AWS Bedrock, or OpenAI-compatible Bedrock) behind a small,
uniform API: take a user utterance + recent dialog, return a *validated*
structured intent.

Provides:

* ``LLMService`` — wraps the OpenAI / Bedrock client and exposes a single
  ``chat()`` entry that respects the bounded-intent contract.
* ``extract_intent()`` / ``intent_to_json()`` — parse the LLM's structured
  output (a tool call shaped like ``INTENT_TOOL``) into a typed
  :class:`Intent` (via Pydantic) and serialize it back to JSON for the
  ``/robot/intent`` ROS topic.
* ``INTENT_TOOL`` — the OpenAI-style "tool" / function schema the LLM is
  instructed to call. Adding a new robot intent means extending this schema
  (so the LLM knows about it) AND the ``intent_router_node`` (so the robot
  reacts to it).

The provider is chosen at runtime by env vars:

* ``MORTY_LLM_MODEL``       — model id (``gpt-5.4-mini``, ``claude-haiku-4-5``,
                              ``openai.gpt-oss-20b-1:0``, ...).
* ``OPENAI_API_KEY``        — OpenAI key.
* ``OPENAI_BASE_URL``       — set when using Bedrock's OpenAI-compatible
                              endpoint (the model id starts with ``openai.``
                              or ``gpt-``).
* ``AWS_BEARER_TOKEN_BEDROCK`` + ``AWS_REGION`` — native Bedrock Converse.

The ``<reasoning>...</reasoning>`` stripping below handles a quirk of the
open-weight ``gpt-oss-*`` models on Bedrock: they emit their chain-of-thought
inline; without stripping, the TTS would read the entire reasoning trace.
"""
from __future__ import annotations

import json
import os
import sys
import re
from typing import Any

from openai import OpenAI
from pydantic import BaseModel


# OpenAI's open-weight gpt-oss-* models on Bedrock emit their chain-of-thought
# inside <reasoning>...</reasoning> at the start of the response. Strip it
# before exposing the text to the rest of the package, otherwise the TTS reads
# the entire reasoning trace aloud.
_REASONING_RE = re.compile(r"<reasoning\b[^>]*>.*?</reasoning>", re.DOTALL | re.IGNORECASE)


def _strip_reasoning(text: str | None) -> str:
    if not text:
        return ""
    return _REASONING_RE.sub("", text).strip()


class ObjectAttributes(BaseModel):
    colour: str | None = None
    shape: str | None = None
    # Spatial qualifier the user added to disambiguate (e.g. "leftmost",
    # "rightmost", "middle"). intent_router uses this to pick a specific
    # detection when several share the same label.
    position: str | None = None


class IntentExtraction(BaseModel):
    action_intent: str | None = None
    object_intent: str | None = None
    object_attributes: ObjectAttributes | None = None


INTENT_TOOL = {
    "type": "function",
    "name": "intent_extractor",
    "description": (
        "Extract the robot action intent in a single call. Returns the "
        "structured fields directly so the router can act with no second "
        "LLM round-trip."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action_intent": {
                "type": "string",
                "enum": [
                    "pick",
                    "place",
                    "pick_and_place",
                    "confirm_drop",
                    "release",
                    "open_gripper",
                    "close_gripper",
                    "stop",
                    "go_home",
                    "go_to_ready",
                    "go_to_start",
                    "sleep",
                    "look_left",
                    "look_right",
                    "look_up",
                    "drop_right",
                    "drop_left",
                    "enable_multiview",
                    "disable_multiview",
                ],
                "description": (
                    "Single action keyword. There is NO generic 'drop' or "
                    "'go to drop' — drops always go to a SIDE: use "
                    "'drop_right' or 'drop_left'. If the user says just "
                    "'drop' / 'drop position' / 'put it down' without a "
                    "side, ASK them which side ('left or right?') instead "
                    "of guessing. Use 'sleep' for stow/park. Use "
                    "'pick_and_place' as a COMPOUND action when the user "
                    "says something like 'pick the X and put it on the "
                    "Y'; the router runs pick→move→release as one unit. "
                    "Set object_attributes.destination to 'drop_right' / "
                    "'drop_left' / 'home' (NO drop_center). "
                    "Use 'enable_multiview' when the user asks for "
                    "multi-view, multi view, multiple views, registration, "
                    "fused view, scan around, look from multiple angles, "
                    "Mehrfachansicht. Use 'disable_multiview' for the "
                    "opposite: single view, just one view, one shot, "
                    "Einzelansicht, no scanning. Default behaviour is "
                    "single-view; the user has to ask to switch.\n"
                    "Use 'release' ONLY when the pick is paused at the "
                    "DROP pose waiting for permission to open the "
                    "gripper (Morty just said 'Should I open the "
                    "gripper?'). User phrases: 'release', 'open it', "
                    "'give it to me', 'drop it', 'let go', 'yes'. "
                    "Use 'confirm_drop' ONLY when an interactive pick is "
                    "paused waiting for a destination — i.e. AFTER Morty "
                    "has just said 'where should I drop it?'. Set "
                    "object_attributes.destination to 'drop_right' / "
                    "'drop_left' / 'home' based on what the user said. "
                    "If the user says 'drop on the right and then go "
                    "home', emit ONE confirm_drop with destination = "
                    "'drop_right' (the executor sequence already returns "
                    "to home after dropping). Do NOT emit confirm_drop "
                    "for a 'go to the drop right position' command when "
                    "no pick is in progress — that's a plain named-pose "
                    "move, use 'drop_right' instead."
                ),
            },
            "object_intent": {
                "type": "string",
                "description": (
                    "Target object's name for pick/place (e.g. 'vitamin "
                    "tube'). Empty string for non-pick actions."
                ),
            },
            "colour": {
                "type": "string",
                "description": (
                    "Optional colour qualifier ('blue', 'red', ...). Also "
                    "used to disambiguate between several visible objects "
                    "of the same type: 'pick the RED tube' -> colour='red'."
                ),
            },
            "shape": {
                "type": "string",
                "description": "Optional shape qualifier.",
            },
            "ordinal": {
                "type": "integer",
                "description": (
                    "1-based counting qualifier when the user counts "
                    "objects: 'the second vitamin tube' -> 2, 'the third "
                    "one from the right' -> 3 (with ordinal_from='right'). "
                    "Omit or 0 when the user did not count."
                ),
            },
            "ordinal_from": {
                "type": "string",
                "enum": ["", "left", "right"],
                "description": (
                    "Counting direction for 'ordinal'. 'the second from "
                    "the RIGHT' -> 'right'. Default 'left' when the user "
                    "counts without naming a side."
                ),
            },
            "position": {
                "type": "string",
                "enum": [
                    "",
                    "leftmost",
                    "rightmost",
                    "left",
                    "right",
                    "middle",
                    "biggest",
                    "smallest",
                ],
                "description": (
                    "Spatial / size qualifier when several matching objects "
                    "are visible, OR which side for 'place' actions."
                ),
            },
            "speed": {
                "type": "string",
                "enum": ["", "very_slow", "slow", "normal", "fast", "very_fast"],
                "description": (
                    "Optional speed qualifier for arm motion. Five tiers, "
                    "mapped roughly as percentages of full speed:\n"
                    "  'very slow' / 'very carefully' / 'extra slow' -> "
                    "'very_slow' (~5%)\n"
                    "  'slow' / 'slowly' / 'carefully' -> 'slow' (~18%, "
                    "DEFAULT when no qualifier given)\n"
                    "  'normal' / 'normal speed' / 'regular' / 'medium' -> "
                    "'normal' (~40%)\n"
                    "  'fast' / 'quickly' / 'quick' -> 'fast' (~60%)\n"
                    "  'very fast' / 'very quickly' / 'as fast as possible' "
                    "/ 'maximum speed' -> 'very_fast' (~80%)\n"
                    "Leave empty if no speed qualifier; the router applies "
                    "'slow' by default."
                ),
            },
            "destination": {
                "type": "string",
                "enum": [
                    "",
                    "drop_right",
                    "drop_left",
                    "home",
                ],
                "description": (
                    "Only used with action_intent='pick_and_place'. Where to "
                    "place the picked object. Map: 'on the right' / 'right "
                    "side' / 'right drop' -> 'drop_right'. 'on the left' / "
                    "'left side' / 'left drop' -> 'drop_left'. 'back home' "
                    "/ 'on the home spot' -> 'home'. If the user did NOT "
                    "specify a side, ask 'left or right?' instead of "
                    "guessing — do NOT use a 'centre' destination."
                ),
            },
        },
        "required": ["action_intent"],
    },
}


DEFAULT_MODEL = os.environ.get("MORTY_LLM_MODEL", "gpt-4o-mini")

# Backend selection is driven by the model string + env vars:
#   - Model starts with "anthropic." (or any non-openai Bedrock provider
#     prefix below): use the native Bedrock Converse API via boto3.
#   - OPENAI_BASE_URL set + OpenAI-style model: use Chat Completions
#     (Bedrock's /openai/v1 route, LiteLLM, Ollama, vLLM, ...).
#   - Otherwise: use the OpenAI Responses API.
BEDROCK_PROVIDER_PREFIXES = (
    "anthropic.", "meta.", "mistral.", "cohere.", "ai21.", "amazon.",
    # Cross-region inference profile prefixes used by Bedrock
    "us.anthropic.", "eu.anthropic.", "apac.anthropic.",
)


def _use_bedrock_converse(model: str) -> bool:
    return any(model.startswith(p) for p in BEDROCK_PROVIDER_PREFIXES)


USE_CHAT_COMPLETIONS = bool(os.environ.get("OPENAI_BASE_URL"))

# gpt-oss-* models (the OpenAI open-weight family on Bedrock) emit a
# <reasoning>...</reasoning> chain-of-thought before answering, which adds
# 5-15s of latency on every turn. Setting reasoning_effort="low" tells the
# model to skip the heavy thinking and respond directly. Configurable via
# MORTY_REASONING_EFFORT (default low for fast responses).
REASONING_EFFORT = os.environ.get("MORTY_REASONING_EFFORT", "low")


def _extra_body_for_model(model: str) -> dict[str, Any]:
    """Per-model overrides passed via extra_body to the OpenAI SDK."""
    if model.startswith("openai.gpt-oss") or model.startswith("gpt-oss"):
        return {"reasoning_effort": REASONING_EFFORT}
    return {}


# --------------------------------------------------------------------------
# Chat-completions <-> Responses-API adapter objects.
# These let the rest of chat_node.py keep using `response.output`,
# `response.output_text`, `item.type == "function_call"`, `.call_id`, etc.
# while the underlying call is a chat-completions POST.
# --------------------------------------------------------------------------

class _AdaptedFunctionCall:
    """Mimics a Responses-API function_call item."""
    def __init__(self, name: str, arguments: str, call_id: str) -> None:
        self.type = "function_call"
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _AdaptedTextItem:
    def __init__(self, text: str) -> None:
        self.type = "message"
        self.role = "assistant"
        self.content = text


class _AdaptedResponse:
    """Wraps a chat-completion choice in the Responses-API shape used by chat_node."""
    def __init__(self, raw: Any) -> None:
        self._raw = raw
        msg = raw.choices[0].message
        items: list[Any] = []
        text = _strip_reasoning(msg.content)
        if text:
            items.append(_AdaptedTextItem(text))
        for tc in (getattr(msg, "tool_calls", None) or []):
            items.append(_AdaptedFunctionCall(
                name=tc.function.name,
                arguments=tc.function.arguments,
                call_id=tc.id,
            ))
        self.output = items
        self.output_text = text


class _AdaptedParsedResponse:
    def __init__(self, parsed: Any) -> None:
        self.output_parsed = parsed


def _responses_tool_to_chat_tool(tool: dict[str, Any]) -> dict[str, Any]:
    """Convert a Responses-API tool def to the chat.completions tools= shape."""
    if tool.get("type") == "function" and "function" not in tool:
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters", {}),
            },
        }
    return tool


def _messages_for_chat_completions(messages: list[Any]) -> list[dict[str, Any]]:
    """Flatten the Responses-API message list into chat.completions messages."""
    out: list[dict[str, Any]] = []
    for m in messages:
        # _AdaptedTextItem assistant turn
        if hasattr(m, "type") and m.type == "message":
            out.append({"role": "assistant", "content": m.content})
            continue
        # _AdaptedFunctionCall (tool call from the model)
        if hasattr(m, "type") and m.type == "function_call":
            out.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": m.call_id,
                    "type": "function",
                    "function": {"name": m.name, "arguments": m.arguments},
                }],
            })
            continue
        # Dicts
        if isinstance(m, dict):
            if m.get("type") == "function_call_output":
                out.append({
                    "role": "tool",
                    "tool_call_id": m["call_id"],
                    "content": m["output"],
                })
                continue
            role = m.get("role", "user")
            # The Responses API has a "developer" role; chat-completions uses "system".
            if role == "developer":
                role = "system"
            content = m.get("content", "")
            out.append({"role": role, "content": content})
    return out


# --------------------------------------------------------------------------
# Bedrock Converse adapter (boto3-based, for Claude / Llama / Mistral / etc.)
# Translates messages + tool spec into the Bedrock Converse format and
# wraps the response back into the same _AdaptedResponse shape used above.
# --------------------------------------------------------------------------

def _messages_for_converse(messages: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (messages, system) where messages is the Converse-format list and
    system holds any system/developer turns. Converse requires user/assistant
    turns to alternate strictly, so consecutive turns of the same role are
    merged into one message with multiple content blocks.
    """
    system: list[dict[str, Any]] = []
    msgs: list[dict[str, Any]] = []

    def _append(role: str, block: dict[str, Any]) -> None:
        # Merge into the previous message if same role (Converse rule).
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"].append(block)
        else:
            msgs.append({"role": role, "content": [block]})

    for m in messages:
        # Adapted text item from a previous chat() call
        if hasattr(m, "type") and m.type == "message":
            _append("assistant", {"text": m.content})
            continue
        # Adapted function call (model invoked a tool)
        if hasattr(m, "type") and m.type == "function_call":
            args = m.arguments
            try:
                args = json.loads(args) if isinstance(args, str) else args
            except Exception:
                args = {"_raw": str(args)}
            _append("assistant", {
                "toolUse": {
                    "toolUseId": m.call_id,
                    "name": m.name,
                    "input": args,
                },
            })
            continue
        if isinstance(m, dict):
            if m.get("type") == "function_call_output":
                _append("user", {
                    "toolResult": {
                        "toolUseId": m["call_id"],
                        "content": [{"text": str(m["output"])}],
                    },
                })
                continue
            role = m.get("role", "user")
            content = m.get("content", "")
            if role in ("system", "developer"):
                system.append({"text": content})
                continue
            _append("assistant" if role == "assistant" else "user", {"text": content})
    return msgs, system


def _tools_for_converse(tools: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not tools:
        return None
    out_tools: list[dict[str, Any]] = []
    for tool in tools:
        # Accept both Responses-API shape (top-level name/parameters) and
        # OpenAI chat-completions shape ({"function": {...}}).
        if "function" in tool:
            fn = tool["function"]
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
        else:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
        out_tools.append({
            "toolSpec": {
                "name": name,
                "description": desc,
                "inputSchema": {"json": params},
            },
        })
    return {"tools": out_tools}


class _BedrockConverseResponse:
    """Wraps a Bedrock Converse response in the Responses-API shape used by chat_node."""
    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = raw
        items: list[Any] = []
        text_parts: list[str] = []
        output_msg = raw.get("output", {}).get("message", {})
        for block in output_msg.get("content", []):
            if "text" in block:
                text = block["text"] or ""
                cleaned = _strip_reasoning(text)
                if cleaned:
                    text_parts.append(cleaned)
            elif "toolUse" in block:
                tu = block["toolUse"]
                args = tu.get("input", {}) or {}
                items.append(_AdaptedFunctionCall(
                    name=tu.get("name", ""),
                    arguments=json.dumps(args),
                    call_id=tu.get("toolUseId", ""),
                ))
        full_text = " ".join(text_parts).strip()
        if full_text:
            items.insert(0, _AdaptedTextItem(full_text))
        self.output = items
        self.output_text = full_text


# Cached Bedrock client — created lazily on first use so importing this
# module never requires boto3 unless a Bedrock model is selected.
_BEDROCK_CLIENT = None


def _bedrock_runtime_client():
    global _BEDROCK_CLIENT
    if _BEDROCK_CLIENT is None:
        import boto3  # noqa: WPS433
        # Region: AWS_REGION env var takes precedence; boto3's defaults otherwise.
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        kwargs = {}
        if region:
            kwargs["region_name"] = region
        # Auth: Bedrock API key path. AWS_BEARER_TOKEN_BEDROCK is a recent
        # auth method; if set we wire it as a fake static credential pair
        # (the bearer token goes in the secret slot). Otherwise boto3 uses
        # the standard credential chain (AWS_ACCESS_KEY_ID / IAM role / ...).
        bearer = os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if bearer and not os.environ.get("AWS_ACCESS_KEY_ID"):
            kwargs["aws_access_key_id"] = "bedrock-api-key"
            kwargs["aws_secret_access_key"] = bearer
        _BEDROCK_CLIENT = boto3.client("bedrock-runtime", **kwargs)
    return _BEDROCK_CLIENT


_CACHE_POINTS_OK = True   # cleared when the model/region rejects cache points


def _bedrock_converse_stream(client, kwargs: dict[str, Any], on_tool_call, on_text=None) -> dict[str, Any]:
    """converse_stream, rebuilt into the non-streaming response shape.

    Why streaming (2026-09-18, Majd: "make the model reply faster"): a
    tool-use block (the intent) is complete long before the model has
    finished its prose; ``on_tool_call`` fires the moment the block closes,
    so the pick is dispatched while the rest of the reply still streams."""
    resp = client.converse_stream(**kwargs)
    blocks: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    usage: dict[str, Any] = {}

    def _block(idx: int, kind: str) -> dict[str, Any]:
        if idx not in blocks:
            blocks[idx] = {"text": ""} if kind == "text" else {"toolUse": {"toolUseId": "", "name": "", "input_str": ""}}
            order.append(idx)
        return blocks[idx]

    for ev in resp["stream"]:
        if "contentBlockStart" in ev:
            cs = ev["contentBlockStart"]
            start = cs.get("start", {}) or {}
            if "toolUse" in start:
                b = _block(cs["contentBlockIndex"], "toolUse")
                b["toolUse"]["toolUseId"] = start["toolUse"].get("toolUseId", "")
                b["toolUse"]["name"] = start["toolUse"].get("name", "")
            else:
                _block(cs["contentBlockIndex"], "text")
        elif "contentBlockDelta" in ev:
            d = ev["contentBlockDelta"]
            delta = d.get("delta", {}) or {}
            if "text" in delta:
                b = _block(d["contentBlockIndex"], "text")
                b["text"] = b.get("text", "") + (delta["text"] or "")
                if on_text is not None and delta["text"]:
                    try:
                        on_text(delta["text"])     # chat_node speaks finished sentences early
                    except Exception:
                        pass
            elif "toolUse" in delta:
                b = _block(d["contentBlockIndex"], "toolUse")
                b["toolUse"]["input_str"] += delta["toolUse"].get("input", "") or ""
        elif "contentBlockStop" in ev:
            b = blocks.get(ev["contentBlockStop"]["contentBlockIndex"])
            if b and "toolUse" in b and "input" not in b["toolUse"]:
                raw_in = b["toolUse"].pop("input_str", "")
                try:
                    b["toolUse"]["input"] = json.loads(raw_in) if raw_in else {}
                except Exception:
                    b["toolUse"]["input"] = {"_raw": raw_in}
                if on_tool_call is not None:
                    try:
                        on_tool_call(_AdaptedFunctionCall(
                            name=b["toolUse"]["name"],
                            arguments=json.dumps(b["toolUse"]["input"]),
                            call_id=b["toolUse"]["toolUseId"],
                        ))
                    except Exception:
                        pass
        elif "metadata" in ev:
            usage = ev["metadata"].get("usage", {}) or {}
    content = []
    for idx in order:
        b = blocks[idx]
        if "toolUse" in b and "input" not in b["toolUse"]:
            raw_in = b["toolUse"].pop("input_str", "")
            try:
                b["toolUse"]["input"] = json.loads(raw_in) if raw_in else {}
            except Exception:
                b["toolUse"]["input"] = {"_raw": raw_in}
        content.append(b)
    return {"output": {"message": {"role": "assistant", "content": content}}, "usage": usage}


def _bedrock_converse_call(
    model: str,
    messages: list[Any],
    tools: list[dict[str, Any]],
    timeout: float,
    on_tool_call=None,
    on_text=None,
) -> _BedrockConverseResponse:
    """Bedrock Converse with prompt caching and streaming (2026-09-18).

    Caching: the 12 k-character system prompt and the tool list are the
    same on every call; a cache point after each lets Bedrock skip
    re-reading them (measured elsewhere as 0.3-0.5 s per call). If the
    model or region rejects cache points, the call is retried without
    them and caching stays off for the process."""
    global _CACHE_POINTS_OK
    msgs, system = _messages_for_converse(messages)

    def _build(cache: bool) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "modelId": model,
            "messages": msgs,
            "inferenceConfig": {"maxTokens": 1024},
        }
        sys_blocks = list(system)
        if sys_blocks and cache:
            sys_blocks.append({"cachePoint": {"type": "default"}})
        if sys_blocks:
            kwargs["system"] = sys_blocks
        tool_config = _tools_for_converse(tools)
        if tool_config is not None:
            if cache:
                tool_config = {"tools": list(tool_config["tools"]) + [{"cachePoint": {"type": "default"}}]}
            kwargs["toolConfig"] = tool_config
        return kwargs

    client = _bedrock_runtime_client()
    last_exc: Exception | None = None
    for attempt in (0, 1):
        cache = _CACHE_POINTS_OK and attempt == 0
        try:
            raw = _bedrock_converse_stream(client, _build(cache), on_tool_call, on_text)
            out = _BedrockConverseResponse(raw)
            u = raw.get("usage", {}) or {}
            out.cache_read_tokens = int(u.get("cacheReadInputTokens", 0) or 0)
            out.cache_write_tokens = int(u.get("cacheWriteInputTokens", 0) or 0)
            return out
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            text = str(exc)
            if cache and ("cachePoint" in text or "cache" in text.lower() or "ValidationException" in text):
                _CACHE_POINTS_OK = False
                continue
            raise
    raise last_exc  # type: ignore[misc]


def _fallback_model_for(model: str) -> str:
    """The route to try when ``model`` is unavailable: the env override, else
    the same model id under the global inference profile (eu./us./apac. →
    global.), else nothing."""
    env = os.environ.get("MORTY_LLM_FALLBACK_MODEL", "").strip()
    if env and env != model:
        return env
    for pfx in ("eu.", "us.", "apac."):
        if model.startswith(pfx):
            return "global." + model[len(pfx):]
    return ""


def _is_capacity_error(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}"
    return any(k in text for k in ("ServiceUnavailable", "Throttling", "ModelNotReady",
                                    "InternalServer", "unable to process", "TooManyRequests"))


class LLMService:
    def __init__(
        self,
        api_key: str | None = None,
        tools: list[dict[str, Any]] | None = None,
        messages: list[Any] | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # OPENAI_API_KEY is only required for the OpenAI-SDK backends.
        # Bedrock Converse uses AWS_BEARER_TOKEN_BEDROCK or the standard
        # AWS credential chain instead.
        if not self.api_key:
            if not (os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
                    or os.environ.get("AWS_ACCESS_KEY_ID")):
                raise RuntimeError(
                    "No credentials found: set OPENAI_API_KEY for OpenAI-SDK "
                    "models, or AWS_BEARER_TOKEN_BEDROCK / AWS_ACCESS_KEY_ID "
                    "for Bedrock Converse models."
                )
            # Placeholder so the OpenAI client constructor doesn't error if a
            # subsequent call falls back to the OpenAI path.
            self.api_key = self.api_key or "bedrock"
        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            self.client = OpenAI(api_key=self.api_key, base_url=base_url)
        else:
            self.client = OpenAI(api_key=self.api_key)
        self.tools = tools or []
        self.messages = messages if messages is not None else []

    def chat(self, model: str = DEFAULT_MODEL, timeout: float = 20.0, on_tool_call=None, on_text=None) -> Any:
        # 20s ceiling — without this the openai client's default is 600s, so
        # a single slow turn would hang the whole rclpy executor thread for
        # 10 minutes and the user sees an unresponsive Morty.
        # on_tool_call: called with each completed tool call WHILE the reply
        # streams (Bedrock path only) — the intent is dispatched early.
        if _use_bedrock_converse(model):
            try:
                return _bedrock_converse_call(model, self.messages, self.tools, timeout, on_tool_call, on_text)
            except Exception as exc:
                # 2026-09-23: Bedrock's eu profile answered ServiceUnavailable
                # twice in one afternoon ("Sorry, I couldn't reach the language
                # model"). The same model exists under the global routing
                # profile; one retry there costs ~1 s and the user hears an
                # answer instead of an apology. MORTY_LLM_FALLBACK_MODEL
                # overrides the choice.
                fb = _fallback_model_for(model)
                if fb and _is_capacity_error(exc):
                    print(f"[llm] {model} unavailable ({type(exc).__name__}); retrying once on {fb}",
                          file=sys.stderr, flush=True)
                    return _bedrock_converse_call(fb, self.messages, self.tools, timeout, on_tool_call, on_text)
                raise
        extra_body = _extra_body_for_model(model)
        if USE_CHAT_COMPLETIONS:
            raw = self.client.with_options(timeout=timeout).chat.completions.create(
                model=model,
                tools=[_responses_tool_to_chat_tool(t) for t in self.tools] or None,
                messages=_messages_for_chat_completions(self.messages),
                extra_body=extra_body or None,
            )
            return _AdaptedResponse(raw)
        return self.client.with_options(timeout=timeout).responses.create(
            model=model,
            tools=self.tools,
            input=self.messages,
        )

    def single_json_response(
        self,
        model: str = DEFAULT_MODEL,
        text_format: type[BaseModel] | None = None,
    ) -> Any:
        if _use_bedrock_converse(model):
            # Bedrock doesn't have a "respect pydantic schema" mode the way
            # OpenAI does, so we ask the model in the system prompt to return
            # JSON matching the schema and then parse it ourselves.
            schema_hint = ""
            if text_format is not None:
                schema_hint = (
                    "\n\nReturn ONLY a valid JSON object matching this schema "
                    "(no markdown fences, no commentary):\n"
                    + json.dumps(text_format.model_json_schema())
                )
            # Append the schema instruction as a transient system message.
            augmented = list(self.messages) + [{
                "role": "system",
                "content": schema_hint.strip(),
            }] if schema_hint else self.messages
            resp = _bedrock_converse_call(model, augmented, [], timeout)
            text = resp.output_text or "{}"
            try:
                data = json.loads(text)
            except Exception:
                # Strip ```json fences if the model added them anyway.
                cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
                data = json.loads(cleaned)
            parsed = text_format(**data) if text_format is not None else data
            return _AdaptedParsedResponse(parsed)
        extra_body = _extra_body_for_model(model)
        if USE_CHAT_COMPLETIONS:
            raw = self.client.beta.chat.completions.parse(
                model=model,
                messages=_messages_for_chat_completions(self.messages),
                response_format=text_format,
                extra_body=extra_body or None,
            )
            return _AdaptedParsedResponse(raw.choices[0].message.parsed)
        return self.client.responses.parse(
            model=model,
            input=self.messages,
            text_format=text_format,
        )

    def add_developer_message(self, content: str) -> None:
        self.messages.append({"role": "developer", "content": content})

    def add_user_message(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})

    def add_assistant_response(self, response: Any) -> None:
        self.messages.extend(response.output)

    def add_function_call_output(self, item: Any, output: str) -> None:
        self.messages.append(
            {
                "type": "function_call_output",
                "call_id": item.call_id,
                "output": output,
            }
        )


def extract_intent(intent_summary: str, model: str = DEFAULT_MODEL) -> IntentExtraction:
    json_agent = LLMService()
    json_agent.add_developer_message(
        "You extract a robot task intent from a one-line summary. "
        "Return only the structured fields requested by the schema; do NOT "
        "leave action_intent null when the summary clearly specifies an action.\n"
        "\n"
        "Supported action_intent values (use exactly one):\n"
        "  - 'pick' (object manipulation; pick up, grab, take)\n"
        "  - 'place' (place an object somewhere)\n"
        "  - 'open_gripper' (open, release, let go)\n"
        "  - 'close_gripper' (close, grip, clamp)\n"
        "  - 'stop' (cancel, abort)\n"
        "  - 'go_home' (go home, return home)\n"
        "  - 'go_to_ready' / 'go_to_start' (ready position, start position)\n"
        "  - 'sleep' (sleep, stow, park, sleep position)\n"
        "  - 'look_left' / 'look_right' / 'look_up'\n"
        "  - 'go_to_drop' (go to drop, drop position; NOT go_to_ready!)\n"
        "  - 'drop_right' (drop right, drop right side, right drop position)\n"
        "  - 'drop_left' (drop left, drop left side, left drop position)\n"
        "\n"
        "CRITICAL DISAMBIGUATION:\n"
        "  - 'go to drop position' / 'drop position' -> action_intent='go_to_drop'\n"
        "    NOT 'go_to_ready' or 'go_to_start'.\n"
        "  - 'drop right [position/side]' / 'right drop position' -> 'drop_right'\n"
        "    NOT 'place'.\n"
        "  - 'drop left [position/side]' / 'left drop position' -> 'drop_left'\n"
        "    NOT 'go_to_start' or 'place'.\n"
        "  - The word 'drop' in any direction context = a drop_* intent,\n"
        "    not place. 'place' is only when an object is being placed.\n"
        "\n"
        "For pick/place: fill object_intent with the target's name and use\n"
        "object_attributes for colour, shape, and position.\n"
        "object_attributes.position values: 'leftmost', 'rightmost', 'left',\n"
        "'right', 'middle', 'biggest', 'smallest'.\n"
        "For all other actions: object_intent = null and object_attributes = null."
    )
    json_agent.add_user_message(intent_summary)
    response = json_agent.single_json_response(
        model=model,
        text_format=IntentExtraction,
    )
    return response.output_parsed


def intent_to_json(intent: IntentExtraction) -> str:
    return json.dumps(intent.model_dump(), separators=(",", ":"))
