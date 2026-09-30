"""
Minimal two-agent dialogue API.

Personas: data/agent_personas.json — LLM wording: backend/prompts.py
"""

from __future__ import annotations

import contextvars
import json
import math
import os
import random
import re
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import httpx
from openai import AuthenticationError, OpenAI
from pydantic import BaseModel, Field

from backend import exp_control, prompts, tts

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MAX_TURNS = int(exp_control.conversation_max_turns)

# Edit this string to change which model writes initial_view + personal_story via /generate-personas-from-topic.
PERSONA_AUTHORING_MODEL = "gpt-4o"

# Splits each gpt-4o reply into pause-separated line's for display (transcript context still uses raw text).
# UTTERANCE_SEGMENT_MODEL is used only when the LLM splitter below is re-enabled.
UTTERANCE_SEGMENT_MODEL = "gpt-4o-mini"
BACKCHANNEL_INSERT_MODEL = "gpt-4o"
DISFLUENCY_INSERT_MODEL = "gpt-4o"
EXPRESSION_TAG_MODEL = "gpt-4o"
# input_backchannels = 0 # Manually control the number of backchannels inserted
DISFLUENCY_RATE_PER_WORD = float(os.getenv("DISFLUENCY_RATE_PER_WORD", "0.09"))
DISFLUENCY_TYPE_WEIGHTS: dict[str, float] = {
    "filled_pause": 0.31,
    "discourse_marker": 0.24,
    "prolongation": 0.28,
    "self_repair": 0,
    "repetition": 0.16,
}
VALID_DISFLUENCY_TYPES = frozenset(DISFLUENCY_TYPE_WEIGHTS)

class _RequestApiKeysMiddleware:
    """Bind X-OpenAI-Key / X-ElevenLabs-Key headers to the current request."""

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {
            k.decode("latin-1").lower(): v.decode("latin-1").strip()
            for k, v in scope.get("headers") or []
        }
        openai_token = _request_openai_key.set(headers.get("x-openai-key") or None)
        eleven_token = _request_elevenlabs_key.set(headers.get("x-elevenlabs-key") or None)
        try:
            await self.app(scope, receive, send)
        finally:
            _request_openai_key.reset(openai_token)
            _request_elevenlabs_key.reset(eleven_token)


app = FastAPI(title="two-agent-chat")
app.add_middleware(_RequestApiKeysMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_json(rel: Path) -> Any:
    with open(ROOT / rel, encoding="utf-8") as f:
        return json.load(f)


def _personas_document() -> dict[str, Any]:
    raw = _load_json(Path("data") / "agent_personas.json")
    agents = raw.get("agents")
    if not isinstance(agents, list) or len(agents) < 2:
        raise RuntimeError("agent_personas.json must contain two agents.")
    return raw


_sessions: dict[str, dict[str, Any]] = {}
# Sessions hold the visitor's ElevenLabs key in memory, so drop them once they go stale.
SESSION_TTL_S = 60 * 60


def _purge_stale_sessions() -> None:
    now = time.time()
    for stale_id, stale in list(_sessions.items()):
        if now - float(stale.get("created_at", now)) <= SESSION_TTL_S:
            continue
        pipeline = stale.get("pipeline")
        if pipeline is not None:
            pipeline.cancel()
        _sessions.pop(stale_id, None)
_cached_doc: dict[str, Any] | None = None

PERSONAS_PATH = ROOT / "data" / "agent_personas.json"


def personas_document() -> dict[str, Any]:
    global _cached_doc
    if _cached_doc is None:
        _cached_doc = _personas_document()
    return _cached_doc


def _invalidate_personas_cache() -> None:
    global _cached_doc
    _cached_doc = None


def _write_personas_document(doc: dict[str, Any]) -> None:
    PERSONAS_PATH.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _invalidate_personas_cache()


def _merge_agent_identity(prev: dict[str, Any], *, initial_view: str, personal_story: str) -> dict[str, Any]:
    """Preserve name, optional verbal_style_id, participation_score; drop legacy keys like voice."""
    out: dict[str, Any] = {
        "name": prev["name"],
        "initial_view": initial_view.strip(),
        "personal_story": personal_story.strip(),
        "participation_score": prev.get("participation_score", 0.5),
    }
    if "verbal_style_id" in prev:
        out["verbal_style_id"] = prev["verbal_style_id"]
    if "gender" in prev:
        out["gender"] = prev["gender"]
    if "elevenlabs_voice_id" in prev:
        out["elevenlabs_voice_id"] = prev["elevenlabs_voice_id"]
    if "elevenlabs_speaking_speed" in prev:
        out["elevenlabs_speaking_speed"] = prev["elevenlabs_speaking_speed"]
    return out


def _elapsed_ms(t0: float) -> int:
    return round((time.perf_counter() - t0) * 1000)


def _openai_json_object(
    *,
    client: OpenAI,
    model: str,
    system: str,
    user: str,
) -> dict[str, Any]:
    rsp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = (rsp.choices[0].message.content or "").strip()
    if not raw:
        raise HTTPException(status_code=502, detail="empty JSON from model")
    try:
        out: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=502,
            detail=f"invalid JSON from model: {e}",
        ) from e
    if not isinstance(out, dict):
        raise HTTPException(status_code=502, detail="model JSON was not an object")
    return out


TURN_REQUEST_TIMEOUT_S = 15.0
PERSONA_REQUEST_TIMEOUT_S = 20.0


MISSING_OPENAI_KEY = "OpenAI API key is missing. Enter it on the API keys page."
MISSING_ELEVENLABS_KEY = "ElevenLabs API key is missing. Enter it on the API keys page."

# Visitors bring their own keys (request headers). Server .env keys are only used as a
# fallback when ALLOW_SERVER_KEYS is on, so a public deployment never spends the owner's quota.
ALLOW_SERVER_KEYS = os.getenv("ALLOW_SERVER_KEYS", "").strip().lower() in {"1", "true", "yes"}
_request_openai_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_openai_key", default=None
)
_request_elevenlabs_key: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_elevenlabs_key", default=None
)


def _openai_key() -> str | None:
    key = _request_openai_key.get()
    if key:
        return key
    return os.getenv("OPENAI_API_KEY") if ALLOW_SERVER_KEYS else None


def _elevenlabs_key() -> str | None:
    key = _request_elevenlabs_key.get()
    if key:
        return key
    return os.getenv("ELEVENLABS_API_KEY") if ALLOW_SERVER_KEYS else None


def _client(*, timeout: float = 60.0, max_retries: int = 1) -> OpenAI | None:
    key = _openai_key()
    if not key:
        return None
    # Default client timeout is 10 minutes, so a stalled call looks like a frozen button.
    # Turns normally return in ~2s; a stalled connection should be retried quickly.
    return OpenAI(api_key=key, timeout=timeout, max_retries=max_retries)


# Break after . , ? ! ... (when followed by whitespace) or on " - ".
_SEGMENT_BREAK_RE = re.compile(r"(?<=[.,?!])\s+|(?<=\.\.\.)\s*|(?:\s+-\s+)")


def _split_utterance_segments_deterministic(utterance: str) -> list[str]:
    """Fast rule-based split at punctuation / dash pauses. Keeps trailing . , ? ! on each segment."""
    inner = utterance.strip()
    if not inner:
        return []
    parts = _SEGMENT_BREAK_RE.split(inner)
    segments = [p.strip() for p in parts if p.strip()]
    return segments if segments else [inner]


# def _split_utterance_segments_llm(
#     client: OpenAI, speaker_name: str, utterance: str
# ) -> list[str]:
#     """Return segment texts (no speaker prefix). Falls back to one segment."""
#     inner = utterance.strip()
#     if not inner:
#         return []
#     try:
#         data = _openai_json_object(
#             client=client,
#             model=UTTERANCE_SEGMENT_MODEL,
#             system=prompts.UTTERANCE_SEGMENT_SYSTEM,
#             user=prompts.utterance_segment_user_prompt(speaker_name, utterance),
#         )
#     except Exception:
#         return [inner]
#     segs = data.get("segments")
#     if not isinstance(segs, list) or not segs:
#         return [inner]
#     parts = [str(s).strip() for s in segs if str(s).strip()]
#     return parts if parts else [inner]


def _split_utterance_segments(client: OpenAI, speaker_name: str, utterance: str) -> list[str]:
    """Return segment texts (no speaker prefix). Uses deterministic split by default."""
    _ = client, speaker_name  # kept for LLM swap-in
    return _split_utterance_segments_deterministic(utterance)
    # return _split_utterance_segments_llm(client, speaker_name, utterance)


def _format_tts_unit_lines_with_backchannels(
    speaker_name: str,
    listener_name: str,
    tts_units: list[str],
    backchannels: list[dict[str, Any]],
) -> str:
    bc_by_unit = {
        int(bc.get("tts_unit_index", bc.get("segment_index", -1))): bc
        for bc in backchannels
    }
    lines: list[str] = []
    for i, unit in enumerate(tts_units):
        line = f"{speaker_name}: {unit}"
        bc = bc_by_unit.get(i)
        if bc:
            line += f" (<Backchannel> {listener_name}: {bc['text']})"
        lines.append(line)
    return "\n".join(lines)


def _format_segment_lines_with_backchannels(
    speaker_name: str,
    listener_name: str,
    segments: list[str],
    backchannels: list[dict[str, Any]],
) -> str:
    bc_by_idx = {int(bc["segment_index"]): bc for bc in backchannels}
    lines: list[str] = []
    for i, seg in enumerate(segments):
        line = f"{speaker_name}: {seg}"
        bc = bc_by_idx.get(i)
        if bc:
            line += f" (<Backchannel> {listener_name}: {bc['text']})"
        lines.append(line)
    return "\n".join(lines)


def _choose_backchannels(
    client: OpenAI,
    *,
    speaker_name: str,
    listener_name: str,
    tts_units: list[str],
    unit_micro_indices: list[list[int]],
) -> tuple[list[dict[str, Any]], int]:
    """Backchannels keyed to TTS units; segment_index = last micro line for transcript display."""
    if not tts_units:
        return [], 0

    max_backchannels = random.randint(0, len(tts_units) // 2)

    # max_backchannels = input_backchannels
    if max_backchannels == 0 or not exp_control.backchannel_speech:
        return [], 0

    try:
        data = _openai_json_object(
            client=client,
            model=BACKCHANNEL_INSERT_MODEL,
            system=prompts.backchannel_insert_system_prompt(
                speaker_name=speaker_name,
                listener_name=listener_name,
                max_backchannels=max_backchannels,
            ),
            user=prompts.backchannel_insert_user_prompt(
                speaker_name=speaker_name,
                listener_name=listener_name,
                segments=tts_units,
                max_backchannels=max_backchannels,
            ),
        )
    except Exception:
        return [], max_backchannels

    raw = data.get("backchannels")
    if not isinstance(raw, list) or not raw:
        return [], max_backchannels

    chosen: list[dict[str, Any]] = []
    used: set[int] = set()
    for item in raw:
        if len(chosen) >= max_backchannels:
            break
        if not isinstance(item, dict):
            continue
        idx = item.get("unit_index", item.get("segment_index"))
        bc_text = str(item.get("text", "")).strip()
        if bc_text == "":
            continue
        try:
            ui = int(idx)
        except (TypeError, ValueError):
            continue
        if ui < 0 or ui >= len(tts_units) or ui in used:
            continue
        used.add(ui)
        micro_idxs = unit_micro_indices[ui]
        display_seg = micro_idxs[-1] if micro_idxs else ui
        chosen.append(
            {
                "tts_unit_index": ui,
                "segment_index": display_seg,
                "text": bc_text,
                "listener": listener_name,
            }
        )
    return chosen, max_backchannels


def _word_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _poisson_sample(lam: float) -> int:
    if lam <= 0:
        return 0
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while p > limit:
        k += 1
        p *= random.random()
    return k - 1


def _pick_disfluency_type() -> str:
    items = [(k, w) for k, w in DISFLUENCY_TYPE_WEIGHTS.items() if w > 0]
    keys, vals = zip(*items)
    return random.choices(keys, weights=vals, k=1)[0]


def _sample_disfluency_count(segments: list[str]) -> int:
    total_words = sum(_word_count(s) for s in segments)
    if total_words <= 0:
        return 0
    lam = int(total_words * DISFLUENCY_RATE_PER_WORD)
    # k = _poisson_sample(lam)
    # cap = max(1, total_words // 3)
    return lam


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text)))


def _duplicate_clause_in_text(modified: str, original: str) -> bool:
    """True when modified repeats a substantial chunk of the original clause."""
    o = _normalize_turn_plain(original)
    m = _normalize_turn_plain(modified)
    if len(o) < 12:
        return False
    for length in range(min(len(o), 72), 11, -1):
        phrase = o[:length]
        if m.count(phrase) >= 2:
            return True
    return False


def _choose_disfluencies(
    client: OpenAI,
    *,
    speaker_name: str,
    segments: list[str],
    disfluency_enabled: bool | None = None,
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Returns (disfluencies, segments_for_tts, disfluency_count)."""

    # None keeps the global exp_control switch. Comparison sessions pass True/False explicitly.
    enabled = exp_control.disfluency_speech if disfluency_enabled is None else disfluency_enabled
    # If exp control does not allow disfluency speech, don't insert disfluencies
    if not enabled or not segments:
        return [], list(segments), 0

    disfluency_count = _sample_disfluency_count(segments)
    if disfluency_count == 0:
        return [], list(segments), 0

    requested_types = [_pick_disfluency_type() for _ in range(disfluency_count)]

    try:
        data = _openai_json_object(
            client=client,
            model=DISFLUENCY_INSERT_MODEL,
            system=prompts.disfluency_insert_system_prompt(
                speaker_name=speaker_name,
                max_disfluencies=disfluency_count,
            ),
            user=prompts.disfluency_insert_user_prompt(
                speaker_name=speaker_name,
                segments=segments,
                max_disfluencies=disfluency_count,
                requested_types=requested_types,
            ),
        )
    except Exception:
        return [], list(segments), disfluency_count

    raw_tts = data.get("segments_for_tts")
    if (
        isinstance(raw_tts, list)
        and len(raw_tts) == len(segments)
        and all(str(s).strip() for s in raw_tts)
    ):
        segments_for_tts: list[str] = []
        for _clean, modified in zip(segments, raw_tts):
            mod = str(modified).strip()
            # Guard disabled — testing prompt-only rule against duplication.
            # if _duplicate_clause_in_text(mod, clean):
            #     segments_for_tts.append(str(clean).strip())
            # else:
            segments_for_tts.append(mod)
    else:
        segments_for_tts = list(segments)

    raw = data.get("disfluencies")
    if not isinstance(raw, list) or not raw:
        return [], segments_for_tts, disfluency_count

    chosen: list[dict[str, Any]] = []
    for i, item in enumerate(raw):
        if len(chosen) >= disfluency_count:
            break
        if not isinstance(item, dict):
            continue
        try:
            idx_i = int(item.get("segment_index"))
        except (TypeError, ValueError):
            continue
        if idx_i < 0 or idx_i >= len(segments):
            continue
        dtype = str(item.get("type", requested_types[i] if i < len(requested_types) else "")).strip()
        if dtype not in VALID_DISFLUENCY_TYPES:
            dtype = requested_types[len(chosen)] if len(chosen) < len(requested_types) else "filled_pause"
        insert = str(item.get("insert", item.get("text", ""))).strip()
        if not insert:
            continue
        spoken = segments_for_tts[idx_i]
        if insert not in spoken:
            pos = spoken.lower().find(insert.lower())
            if pos < 0:
                continue
            insert = spoken[pos : pos + len(insert)]
        chosen.append(
            {
                "segment_index": idx_i,
                "type": dtype,
                "insert": insert,
            }
        )

    return chosen, segments_for_tts, len(chosen)


def _normalize_turn_plain(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _derive_expressions_from_tagged_segments(
    segments_expressive: list[str],
) -> list[dict[str, Any]]:
    """UI metadata: tags found in each display segment (derived locally, not from LLM)."""
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(segments_expressive):
        tags = re.findall(r"\[([^\]]+)\]", str(seg))
        if not tags:
            continue
        out.append({"segment_index": i, "tags": tags, "note": ""})
    return out


def _expressive_units_to_micro_segments(
    units_expressive: list[str],
    unit_micro_indices: list[list[int]],
    segments_for_tts: list[str],
) -> list[str]:
    """Split tagged TTS units back to per-line micro segments for transcript display."""
    micro = list(segments_for_tts)
    for ui, tagged in enumerate(units_expressive):
        if ui >= len(unit_micro_indices):
            break
        micro_idxs = unit_micro_indices[ui]
        if not micro_idxs:
            continue
        micro_parts = [segments_for_tts[i] for i in micro_idxs]
        ranges = tts.char_ranges_for_micro_parts(micro_parts)
        split = tts.split_tagged_turn_into_segments(tagged, ranges)
        if len(split) != len(micro_idxs):
            continue
        for mi, seg_exp in zip(micro_idxs, split):
            micro[mi] = seg_exp
    return micro


def _apply_expression_tags(
    client: OpenAI,
    *,
    speaker_name: str,
    listener_name: str,
    tts_units: list[str],
    unit_micro_indices: list[list[int]],
    segments_for_tts: list[str],
    backchannels: list[dict[str, Any]],
    discussion_topic: str = "",
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Tag TTS units for Eleven v3; split back to micro segments for display."""
    if not tts_units:
        return [], list(segments_for_tts), [dict(b) for b in backchannels], []

    if not exp_control.audio_tag_speech:
        return list(tts_units), list(segments_for_tts), [dict(b) for b in backchannels], []

    fallback_units = list(tts_units)

    try:
        data = _openai_json_object(
            client=client,
            model=EXPRESSION_TAG_MODEL,
            system=prompts.expression_tag_system_prompt(speaker_name=speaker_name),
            user=prompts.expression_tag_user_prompt(
                speaker_name=speaker_name,
                listener_name=listener_name,
                tts_units=tts_units,
                backchannels=backchannels,
                discussion_topic=discussion_topic,
            ),
        )
        print(f"[EXPRESSION TAGS] LLM returned keys: {list(data.keys())}")
    except Exception as exc:
        print(f"[EXPRESSION TAGS] EXCEPTION during LLM call: {exc}")
        return fallback_units, list(segments_for_tts), [dict(b) for b in backchannels], []

    raw_units = data.get("speaker_units_for_tts") or data.get("speaker_segments")
    print(f"[EXPRESSION TAGS] raw_units type={type(raw_units).__name__}, "
          f"len={len(raw_units) if isinstance(raw_units, list) else 'N/A'}, "
          f"expected len={len(tts_units)}")
    if isinstance(raw_units, list):
        for i, ru in enumerate(raw_units):
            print(f"[EXPRESSION TAGS]   unit[{i}] = {str(ru)[:120]}")

    units_expressive: list[str] = []
    if isinstance(raw_units, list) and len(raw_units) == len(tts_units):
        for i, (plain, raw) in enumerate(zip(tts_units, raw_units)):
            tagged = str(raw).strip()
            stripped = _normalize_turn_plain(tts.strip_audio_tags(tagged))
            original = _normalize_turn_plain(plain)
            if tagged and stripped == original:
                units_expressive.append(tagged)
                if tagged != plain:
                    print(f"[EXPRESSION TAGS]   unit[{i}] ACCEPTED (has tags)")
            else:
                units_expressive.append(plain)
                print(f"[EXPRESSION TAGS]   unit[{i}] REJECTED: "
                      f"stripped={stripped!r} != original={original!r}")
    else:
        print(f"[EXPRESSION TAGS] LENGTH MISMATCH or bad type — using fallback")
        units_expressive = fallback_units

    segments_expressive = _expressive_units_to_micro_segments(
        units_expressive, unit_micro_indices, segments_for_tts
    )

    bc_tts: dict[int, str] = {}
    bc_ordered = sorted(
        backchannels, key=lambda b: int(b.get("tts_unit_index", b.get("segment_index", 0)))
    )
    raw_bc = data.get("backchannel_clips")
    if isinstance(raw_bc, list):
        for i, bc in enumerate(bc_ordered):
            if i >= len(raw_bc):
                break
            item = raw_bc[i]
            if isinstance(item, str):
                txt = item.strip()
            elif isinstance(item, dict):
                txt = str(item.get("text_for_tts", "")).strip()
            else:
                continue
            if txt:
                ui = int(bc.get("tts_unit_index", bc.get("segment_index", i)))
                bc_tts[ui] = txt

    bc_out: list[dict[str, Any]] = []
    for bc in backchannels:
        item = dict(bc)
        ui = int(item.get("tts_unit_index", item.get("segment_index", 0)))
        item["text_for_tts"] = bc_tts.get(ui) or str(item.get("text", "")).strip()
        bc_out.append(item)

    expressions = _derive_expressions_from_tagged_segments(segments_expressive)

    return units_expressive, segments_expressive, bc_out, expressions


def _tts_units_for_synthesis(display: dict[str, Any]) -> list[str]:
    expressive = display.get("units_expressive")
    if isinstance(expressive, list) and expressive:
        return [str(u) for u in expressive]
    plain = display.get("tts_units")
    if isinstance(plain, list) and plain:
        return [str(u) for u in plain]
    spoken = display.get("segments_for_tts") or display.get("segments") or []
    units, _ = tts.group_segments_for_tts_units([str(s) for s in spoken])
    return units


def _backchannels_for_tts(display: dict[str, Any]) -> list[dict[str, Any]]:
    """Backchannels with text_for_tts set when expression tagging ran."""
    out: list[dict[str, Any]] = []
    for bc in display.get("backchannels") or []:
        if not isinstance(bc, dict):
            continue
        item = dict(bc)
        tts_text = str(item.get("text_for_tts") or item.get("text") or "").strip()
        if tts_text:
            item["text_for_tts"] = tts_text
        out.append(item)
    return out


def _segment_utterance_for_display(
    client: OpenAI,
    speaker_name: str,
    listener_name: str,
    utterance: str,
    discussion_topic: str = "",
    disfluency_enabled: bool | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """Segment + backchannels + disfluencies. Returns (display dict, partial stage timings)."""
    inner = utterance.strip()
    if not inner:
        line = f"{speaker_name}: {utterance}".strip()
        return (
            {
                "segments": [utterance.strip()] if utterance.strip() else [],
                "segments_for_tts": [utterance.strip()] if utterance.strip() else [],
                "segments_expressive": [utterance.strip()] if utterance.strip() else [],
                "tts_units": [utterance.strip()] if utterance.strip() else [],
                "units_expressive": [utterance.strip()] if utterance.strip() else [],
                "unit_micro_indices": [[0]] if utterance.strip() else [],
                "backchannels": [],
                "disfluencies": [],
                "expressions": [],
                "backchannel_max": 0,
                "disfluency_count": 0,
                "listener": listener_name,
                "segmented_dialogue": line,
            },
            {
                "segment_ms": 0,
                "backchannel_ms": 0,
                "disfluency_ms": 0,
                "expression_ms": 0,
            },
        )

    t0 = time.perf_counter()
    segments = _split_utterance_segments(client, speaker_name, utterance)
    segment_ms = _elapsed_ms(t0)
    print(f"\n[OUTPUT 2 - SEGMENT SPLIT] {speaker_name} ({segment_ms}ms) — {len(segments)} segments")
    for i, s in enumerate(segments):
        print(f"  [{i}] {s}")

    t0 = time.perf_counter()
    disfluencies, segments_for_tts, disfluency_count = _choose_disfluencies(
        client,
        speaker_name=speaker_name,
        segments=segments,
        disfluency_enabled=disfluency_enabled,
    )
    disfluency_ms = _elapsed_ms(t0)
    print(f"\n[OUTPUT 3 - DISFLUENCY] {speaker_name} ({disfluency_ms}ms) — {disfluency_count} inserted")
    for i, s in enumerate(segments_for_tts):
        marker = " *" if s != segments[i] else ""
        print(f"  [{i}]{marker} {s}")

    tts_units, unit_micro_indices = tts.group_segments_for_tts_units(segments_for_tts)

    t0 = time.perf_counter()
    backchannels, backchannel_max = _choose_backchannels(
        client,
        speaker_name=speaker_name,
        listener_name=listener_name,
        tts_units=tts_units,
        unit_micro_indices=unit_micro_indices,
    )
    backchannel_ms = _elapsed_ms(t0)
    print(f"\n[OUTPUT 4 - BACKCHANNELS] {speaker_name} ({backchannel_ms}ms) — {len(backchannels)}/{backchannel_max} inserted")
    for bc in backchannels:
        print(f"  @ unit {bc.get('tts_unit_index', '?')}: {bc.get('listener')} \"{bc.get('text')}\"")

    t0 = time.perf_counter()
    units_expressive, segments_expressive, backchannels, expressions = _apply_expression_tags(
        client,
        speaker_name=speaker_name,
        listener_name=listener_name,
        tts_units=tts_units,
        unit_micro_indices=unit_micro_indices,
        segments_for_tts=segments_for_tts,
        backchannels=backchannels,
        discussion_topic=discussion_topic,
    )
    expression_ms = _elapsed_ms(t0)
    print(f"\n[OUTPUT 5 - EMOTION TAGS] {speaker_name} ({expression_ms}ms)")
    for i, (plain, expressive) in enumerate(zip(tts_units, units_expressive)):
        marker = " *" if expressive != plain else ""
        print(f"  [{i}]{marker} {expressive}")

    segmented_dialogue = _format_tts_unit_lines_with_backchannels(
        speaker_name, listener_name, tts_units, backchannels
    )
    display = {
        "segments": segments,
        "segments_for_tts": segments_for_tts,
        "segments_expressive": segments_expressive,
        "tts_units": tts_units,
        "units_expressive": units_expressive,
        "unit_micro_indices": unit_micro_indices,
        "backchannels": backchannels,
        "disfluencies": disfluencies,
        "expressions": expressions,
        "backchannel_max": backchannel_max,
        "disfluency_count": disfluency_count,
        "listener": listener_name,
        "segmented_dialogue": segmented_dialogue,
    }
    return display, {
        "segment_ms": segment_ms,
        "backchannel_ms": backchannel_ms,
        "disfluency_ms": disfluency_ms,
        "expression_ms": expression_ms,
    }


def _speakable_tts_text(text: str) -> bool:
    return bool(tts.strip_audio_tags(str(text)).strip())


def _empty_tts_line(
    *,
    kind: str,
    segment_index: int,
    speaker: str,
    text: str,
    error: str | None = None,
) -> dict[str, Any]:
    line: dict[str, Any] = {
        "kind": kind,
        "segment_index": segment_index,
        "speaker": speaker,
        "text": text,
        "audio_base64": "",
        "duration_s": 0.0,
        "api_ms": 0,
        "alignment": None,
    }
    if error:
        line["error"] = error
    return line


def _tts_line_timing(line: dict[str, Any]) -> dict[str, Any]:
    kind = line["kind"]
    idx = int(line["segment_index"])
    if kind == "backchannel":
        label = f"bc @ unit {idx}"
    else:
        label = f"unit {idx}"
    return {
        "kind": kind,
        "segment_index": idx,
        "label": label,
        "api_ms": line.get("api_ms", 0),
        "duration_s": line.get("duration_s", 0),
    }


def _synthesize_with_retry(
    *,
    text: str,
    voice_id: str,
    speed: float,
    api_key: str,
    retries: int = 3,
) -> dict[str, Any]:
    last_err: str | None = None
    t0 = time.perf_counter()
    for attempt in range(max(1, retries)):
        try:
            result = tts.synthesize_with_timestamps(
                text=text,
                voice_id=voice_id,
                speed=speed,
                api_key=api_key,
            )
            if result.get("audio_base64"):
                return result
            last_err = "empty audio response"
            print(f"[TTS RETRY] attempt={attempt + 1}/{retries} empty audio — text={text[:80]!r}")
        except Exception as exc:
            last_err = str(exc)
            print(f"[TTS RETRY] attempt={attempt + 1}/{retries} exception={last_err!r} — text={text[:80]!r}")
        if attempt + 1 < retries:
            time.sleep(0.5 * (attempt + 1))
    total_ms = _elapsed_ms(t0)
    print(f"[TTS FAIL] all {retries} attempts failed — text={text!r} voice={voice_id} "
          f"elapsed={total_ms}ms error={last_err}")
    return {
        "audio_base64": "",
        "duration_s": 0.0,
        "api_ms": total_ms,
        "alignment": None,
        "error": last_err or "synthesis failed",
    }


def _synthesize_turn_audio(
    *,
    speaker: dict[str, Any],
    listener: dict[str, Any],
    tts_units: list[str],
    backchannels: list[dict[str, Any]],
    on_line_ready: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
    """One clip per TTS unit + separate backchannel clips (parallel)."""
    if not api_key:
        return [], [], None

    speaker_vid = tts.voice_id_for_agent(speaker)
    listener_vid = tts.voice_id_for_agent(listener)
    speaker_speed = tts.speaking_speed_for_agent(speaker)
    listener_speed = tts.speaking_speed_for_agent(listener)
    if not speaker_vid or not listener_vid:
        return [], [], None

    if not tts_units:
        return [], [], 0

    bc_by_unit: dict[int, dict[str, Any]] = {}
    for bc in backchannels:
        ui = int(bc.get("tts_unit_index", bc.get("segment_index", 0)))
        bc_by_unit[ui] = bc

    results: dict[tuple[str, int], dict[str, Any]] = {}

    def _store_line(line: dict[str, Any]) -> None:
        kind = str(line["kind"])
        idx = int(line["segment_index"])
        results[(kind, idx)] = line
        if on_line_ready is not None:
            on_line_ready(line, _tts_line_timing(line))

    for ui, unit in enumerate(tts_units):
        text = str(unit)
        if not _speakable_tts_text(text):
            _store_line(
                _empty_tts_line(
                    kind="segment",
                    segment_index=ui,
                    speaker=str(speaker["name"]),
                    text=text,
                    error="no speakable text after stripping audio tags",
                )
            )

    # Build jobs: segment units in ascending order first, then backchannels in ascending order.
    # Processed sequentially so unit 0 is always synthesized before unit 1, and the current
    # turn fully completes before the next turn starts (tts_pool max_workers=1).
    jobs: list[tuple[str, int, str, str, str, float]] = []
    for ui, unit in enumerate(tts_units):
        text = str(unit)
        if (("segment", ui) in results) or not _speakable_tts_text(text):
            continue
        jobs.append(("segment", ui, text, str(speaker["name"]), speaker_vid, speaker_speed))
    for ui in sorted(bc_by_unit.keys()):
        bc = bc_by_unit[ui]
        text = str(bc.get("text_for_tts") or bc["text"])
        if not _speakable_tts_text(text):
            _store_line(
                _empty_tts_line(
                    kind="backchannel",
                    segment_index=ui,
                    speaker=str(bc.get("listener", listener["name"])),
                    text=text,
                    error="no speakable text after stripping audio tags",
                )
            )
            continue
        jobs.append(
            (
                "backchannel",
                ui,
                text,
                str(bc.get("listener", listener["name"])),
                listener_vid,
                listener_speed,
            )
        )

    # Submit jobs in priority order (unit 0 first, ascending) into a pool of 4.
    # ThreadPoolExecutor's internal queue is FIFO, so earlier-submitted jobs are
    # always picked up first when a worker slot frees up.
    # tts_pool (max_workers=1) ensures turn N fully completes before turn N+1 starts.
    _MAX_CONCURRENT_TTS = 4
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=_MAX_CONCURRENT_TTS) as pool:
        future_map = {}
        for job in jobs:
            kind, idx, text, name, voice_id, speed = job
            fut = pool.submit(
                _synthesize_with_retry, text=text, voice_id=voice_id, speed=speed, api_key=api_key
            )
            future_map[fut] = (kind, idx, name, text)
        for fut in as_completed(future_map):
            kind, idx, name, text = future_map[fut]
            synth = fut.result()
            line = {
                **synth,
                "kind": kind,
                "segment_index": idx,
                "speaker": name,
                "text": text,
            }
            _store_line(line)

    tts_total_ms = _elapsed_ms(t0)
    audio_lines: list[dict[str, Any]] = []
    tts_line_timings: list[dict[str, Any]] = []

    for ui in range(len(tts_units)):
        seg_key = ("segment", ui)
        if seg_key in results:
            r = results[seg_key]
        else:
            r = _empty_tts_line(
                kind="segment",
                segment_index=ui,
                speaker=str(speaker["name"]),
                text=str(tts_units[ui]),
                error="missing synthesis result",
            )
            results[seg_key] = r
        line = {
            "kind": "segment",
            "segment_index": ui,
            "speaker": r["speaker"],
            "text": r["text"],
            "audio_base64": r.get("audio_base64") or "",
            "duration_s": r.get("duration_s", 0),
            "api_ms": r.get("api_ms", 0),
            "alignment": r.get("alignment"),
        }
        if r.get("error"):
            line["error"] = r["error"]
        audio_lines.append(line)
        tts_line_timings.append(_tts_line_timing(line))

        if ui in bc_by_unit:
            bc_key = ("backchannel", ui)
            if bc_key in results:
                r = results[bc_key]
            else:
                bc = bc_by_unit[ui]
                r = _empty_tts_line(
                    kind="backchannel",
                    segment_index=ui,
                    speaker=str(bc.get("listener", listener["name"])),
                    text=str(bc.get("text_for_tts") or bc["text"]),
                    error="missing synthesis result",
                )
                results[bc_key] = r
            line = {
                "kind": "backchannel",
                "segment_index": ui,
                "speaker": r["speaker"],
                "text": r["text"],
                "audio_base64": r.get("audio_base64") or "",
                "duration_s": r.get("duration_s", 0),
                "api_ms": r.get("api_ms", 0),
                "alignment": r.get("alignment"),
            }
            if r.get("error"):
                line["error"] = r["error"]
            audio_lines.append(line)
            tts_line_timings.append(_tts_line_timing(line))

    return audio_lines, tts_line_timings, tts_total_ms


@dataclass
class TurnTtsState:
    turn_no: int
    unit_count: int
    segments: dict[int, dict[str, Any]] = field(default_factory=dict)
    backchannels: dict[int, dict[str, Any]] = field(default_factory=dict)
    tts_line_timings: list[dict[str, Any]] = field(default_factory=list)
    complete: bool = False
    tts_total_ms: int | None = None
    error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)


def _start_turn_tts_background(
    sess: dict[str, Any],
    *,
    executor: ThreadPoolExecutor,
    turn_no: int,
    speaker: dict[str, Any],
    listener: dict[str, Any],
    tts_units: list[str],
    backchannels: list[dict[str, Any]],
) -> TurnTtsState:
    state = TurnTtsState(turn_no=turn_no, unit_count=len(tts_units))
    by_turn: dict[int, TurnTtsState] = sess.setdefault("turn_tts_by_turn", {})
    by_turn[turn_no] = state
    sess["turn_tts"] = state

    def run() -> None:
        def on_ready(line: dict[str, Any], timing: dict[str, Any]) -> None:
            ui = int(line["segment_index"])
            with state.lock:
                if line["kind"] == "segment":
                    state.segments[ui] = line
                else:
                    state.backchannels[ui] = line
                state.tts_line_timings.append(timing)

        try:
            _, _, tts_total_ms = _synthesize_turn_audio(
                speaker=speaker,
                listener=listener,
                tts_units=tts_units,
                backchannels=backchannels,
                on_line_ready=on_ready,
                api_key=sess.get("elevenlabs_api_key"),
            )
            with state.lock:
                state.complete = True
                state.tts_total_ms = tts_total_ms
        except Exception as exc:
            with state.lock:
                state.complete = True
                state.error = str(exc)

    executor.submit(run)
    return state


def _generate_turn_text(
    client: OpenAI,
    *,
    agents: list[dict[str, Any]],
    speaker_idx: int,
    discussion_topic: str,
    transcript_so_far: list[dict[str, str]],
    max_attempts: int = 3,
    include_conversation_instructions: bool | None = None,
) -> tuple[str, int]:
    speaker = agents[speaker_idx]
    partner = agents[1 - speaker_idx]
    system = prompts.agent_system_prompt(
        discussion_topic=discussion_topic,
        speaker_name=speaker["name"],
        partner_name=partner["name"],
        gender=str(speaker.get("gender", "")),
        partner_gender=str(partner.get("gender", "")),
        stance_on_topic=str(speaker.get("initial_view", "")),
        personal_story=str(speaker.get("personal_story", "")),
        voice_style=str(speaker.get("voice", "")),
        include_conversation_instructions=include_conversation_instructions,
    )
    user_content = prompts.agent_user_prompt(
        speaker_name=speaker["name"],
        transcript_so_far=transcript_so_far,
    )

    last_detail = "empty model reply"
    total_ms = 0
    for attempt in range(max(1, max_attempts)):
        t0 = time.perf_counter()
        rsp = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ],
        )
        total_ms += _elapsed_ms(t0)
        choice = rsp.choices[0]
        lt = (choice.message.content or "").strip()
        if lt:
            pref = speaker["name"] + ":"
            if lt.lower().startswith(pref.lower()):
                lt = lt[len(pref) :].lstrip()
            text = lt[:2000]
            print(f"\n{'='*60}")
            print(f"[OUTPUT 1 - RESPONSE] {speaker['name']} ({total_ms}ms)")
            print(f"{'='*60}")
            print(text)
            return text, total_ms
        finish = getattr(choice, "finish_reason", None)
        last_detail = f"empty model reply (finish_reason={finish})"
        if attempt + 1 < max_attempts:
            time.sleep(0.4 * (attempt + 1))

    raise HTTPException(status_code=502, detail=last_detail)


def _finalize_display_timings(
    display: dict[str, Any],
    *,
    text_generation_ms: int,
    stage_partial: dict[str, int],
    t_all_start: float,
    tts_enabled: bool,
    audio_lines: list[dict[str, Any]],
    tts_line_timings: list[dict[str, Any]],
    tts_total_ms: int | None,
    tts_wall_ms: int,
    prefetched: bool = False,
    wait_ms: int = 0,
    tts_streaming: bool = False,
) -> dict[str, Any]:
    timings: dict[str, Any] = {
        "text_generation_ms": text_generation_ms,
        "segment_ms": stage_partial["segment_ms"],
        "backchannel_ms": stage_partial["backchannel_ms"],
        "disfluency_ms": stage_partial.get("disfluency_ms", 0),
        "expression_ms": stage_partial.get("expression_ms", 0),
        "tts_requested": tts_enabled,
        "tts_streaming": tts_streaming,
        "tts_synthesized": (not tts_streaming) and tts_total_ms is not None,
        "tts_total_ms": tts_total_ms if tts_total_ms is not None else 0,
        "tts_wall_ms": tts_wall_ms,
        "tts_lines": tts_line_timings,
        "total_ms": _elapsed_ms(t_all_start),
        "prefetched": prefetched,
        "wait_ms": wait_ms,
    }
    display = dict(display)
    display["timings"] = timings
    display["audio_lines"] = audio_lines
    display["tts_streaming"] = tts_streaming
    return display


def _turn_api_payload(
    turn_no: int,
    speaker: dict[str, Any],
    partner: dict[str, Any],
    text: str,
    display: dict[str, Any],
) -> dict[str, Any]:
    return {
        "turn": turn_no,
        "speaker": speaker["name"],
        "listener": partner["name"],
        "text": text,
        "segments": display["segments"],
        "segments_for_tts": display.get("segments_for_tts", display["segments"]),
        "segments_expressive": display.get("segments_expressive", display.get("segments_for_tts", display["segments"])),
        "tts_units": display.get("tts_units", []),
        "units_expressive": display.get("units_expressive", []),
        "unit_micro_indices": display.get("unit_micro_indices", []),
        "backchannels": display["backchannels"],
        "disfluencies": display.get("disfluencies", []),
        "expressions": display.get("expressions", []),
        "backchannel_max": display["backchannel_max"],
        "disfluency_count": display.get("disfluency_count", 0),
        "segmented_dialogue": display["segmented_dialogue"],
        "timings": display.get("timings"),
        "audio_lines": display.get("audio_lines", []),
        "tts_streaming": bool(display.get("tts_streaming", False)),
    }


class SessionPipeline:
    """
    TTS-on prefetch pipeline:
    - Each turn: text → post-process → background TTS → commit to transcript.
    - After finalize for turn N, immediately start building turn N+1 (like turn 1 → 2).
    - Delivery via /next_turn only hands off prefetch; it does not start generation.
    """

    def __init__(self, sess: dict[str, Any]):
        self.sess = sess
        self.max_turns = int(sess.get("max_turns") or MAX_TURNS)
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.consumed = threading.Event()
        self.consumed.set()
        self.prefetch: dict[str, Any] | None = None
        self.prefetch_turn_no: int | None = None
        self.error: str | None = None
        self.cancelled = False
        self.tts_pool = ThreadPoolExecutor(max_workers=1)
        self._build_executor = ThreadPoolExecutor(max_workers=1)
        self._worker: threading.Thread | None = None
        client = _client(timeout=TURN_REQUEST_TIMEOUT_S, max_retries=2)
        if client is None:
            raise HTTPException(status_code=401, detail=MISSING_OPENAI_KEY)
        self._client = client

    def cancel(self) -> None:
        self.cancelled = True
        self.consumed.set()
        self.ready.set()
        self.tts_pool.shutdown(wait=False)
        self._build_executor.shutdown(wait=False)

    def _commit_turn_to_transcript(self, turn_payload: dict[str, Any]) -> None:
        self.sess["transcript"].append(
            {"speaker": turn_payload["speaker"], "text": turn_payload["text"]}
        )

    def _build_turn_committed(self, turn_no: int) -> dict[str, Any]:
        """Full pipeline for one turn; commit to transcript when finalize returns."""
        turn_payload = self._build_turn(turn_no=turn_no, prefetched=True)
        self._commit_turn_to_transcript(turn_payload)
        return turn_payload

    def _postprocess_turn_text(
        self, *, speaker_idx: int, text: str
    ) -> tuple[dict[str, Any], dict[str, int]]:
        speaker = self.sess["agents"][speaker_idx]
        partner = self.sess["agents"][1 - speaker_idx]
        return _segment_utterance_for_display(
            self._client, speaker["name"], partner["name"], text,
            discussion_topic=str(self.sess.get("discussion_topic", "")),
            disfluency_enabled=self.sess.get("disfluency_enabled"),
        )

    def _build_turn(
        self,
        *,
        turn_no: int,
        prefetched: bool,
        wait_ms: int = 0,
    ) -> dict[str, Any]:
        """Generate one turn from the committed transcript only."""
        speaker_idx = (turn_no - 1) % 2
        text, text_ms = _generate_turn_text(
            self._client,
            agents=self.sess["agents"],
            speaker_idx=speaker_idx,
            discussion_topic=str(self.sess["discussion_topic"]),
            transcript_so_far=list(self.sess["transcript"]),
            include_conversation_instructions=self.sess.get("conversation_instructions"),
        )
        display, stage_partial = self._postprocess_turn_text(
            speaker_idx=speaker_idx, text=text
        )
        return self._finalize_turn_with_tts(
            turn_no=turn_no,
            text=text,
            text_ms=text_ms,
            speaker_idx=speaker_idx,
            display=display,
            stage_partial=stage_partial,
            prefetched=prefetched,
            wait_ms=wait_ms,
        )

    def _finalize_turn_with_tts(
        self,
        *,
        turn_no: int,
        text: str,
        text_ms: int,
        speaker_idx: int,
        display: dict[str, Any],
        stage_partial: dict[str, int],
        prefetched: bool,
        wait_ms: int = 0,
    ) -> dict[str, Any]:
        agents: list[dict[str, Any]] = self.sess["agents"]
        speaker = agents[speaker_idx]
        partner = agents[1 - speaker_idx]
        t_all = time.perf_counter()

        tts_units = _tts_units_for_synthesis(display)
        _start_turn_tts_background(
            self.sess,
            executor=self.tts_pool,
            turn_no=turn_no,
            speaker=speaker,
            listener=partner,
            tts_units=tts_units,
            backchannels=_backchannels_for_tts(display),
        )

        display = _finalize_display_timings(
            display,
            text_generation_ms=text_ms,
            stage_partial=stage_partial,
            t_all_start=t_all,
            tts_enabled=True,
            audio_lines=[],
            tts_line_timings=[],
            tts_total_ms=None,
            tts_wall_ms=0,
            prefetched=prefetched,
            wait_ms=wait_ms,
            tts_streaming=True,
        )
        return _turn_api_payload(turn_no, speaker, partner, text, display)

    def build_first_turn(self, speaker_idx: int) -> dict[str, Any]:
        """Turn 1: TEXT → post-process → background TTS → return."""
        t_all = time.perf_counter()
        speaker = self.sess["agents"][speaker_idx]
        text, text_ms = _generate_turn_text(
            self._client,
            agents=self.sess["agents"],
            speaker_idx=speaker_idx,
            discussion_topic=str(self.sess["discussion_topic"]),
            transcript_so_far=self.sess["transcript"],
            include_conversation_instructions=self.sess.get("conversation_instructions"),
        )
        display, stage_partial = self._postprocess_turn_text(
            speaker_idx=speaker_idx, text=text
        )
        self.sess["transcript"].append({"speaker": speaker["name"], "text": text})

        turn_payload = self._finalize_turn_with_tts(
            turn_no=1,
            text=text,
            text_ms=text_ms,
            speaker_idx=speaker_idx,
            display=display,
            stage_partial=stage_partial,
            prefetched=False,
        )
        if turn_payload.get("timings"):
            turn_payload["timings"]["total_ms"] = _elapsed_ms(t_all)

        if self.max_turns > 1:
            self._worker = threading.Thread(
                target=self._worker_loop,
                args=(2,),
                daemon=True,
            )
            self._worker.start()

        return turn_payload

    def _worker_loop(self, next_turn_no: int) -> None:
        """Prefetch one turn for delivery; start the next build after each finalize."""
        pending_next: Future[dict[str, Any]] | None = None
        try:
            turn_no = next_turn_no
            while turn_no <= self.max_turns and not self.cancelled:
                if pending_next is not None:
                    turn_payload = pending_next.result()
                    pending_next = None
                else:
                    turn_payload = self._build_turn_committed(turn_no)

                with self.lock:
                    self.prefetch = turn_payload
                    self.prefetch_turn_no = turn_no
                    self.error = None
                    self.consumed.clear()
                    self.ready.set()

                if turn_no < self.max_turns and not self.cancelled:
                    nxt = turn_no + 1
                    pending_next = self._build_executor.submit(
                        self._build_turn_committed, nxt
                    )

                self.consumed.wait()
                if self.cancelled:
                    break
                turn_no += 1
        except HTTPException as exc:
            with self.lock:
                self.error = str(exc.detail)
                self.ready.set()
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
                self.ready.set()

    def take_next_turn(self, expected_turn_no: int) -> dict[str, Any]:
        t_wait = time.perf_counter()
        while True:
            if self.cancelled:
                raise HTTPException(status_code=499, detail="prefetch cancelled")
            if self.ready.wait(timeout=0.25):
                break
        wait_ms = _elapsed_ms(t_wait)

        with self.lock:
            if self.error:
                raise HTTPException(status_code=502, detail=f"prefetch failed: {self.error}")
            if (
                self.prefetch is None
                or self.prefetch_turn_no != expected_turn_no
            ):
                raise HTTPException(
                    status_code=502,
                    detail=f"prefetch turn mismatch (expected {expected_turn_no})",
                )
            payload = self.prefetch
            self.prefetch = None
            self.prefetch_turn_no = None
            self.ready.clear()

        self.consumed.set()
        if payload.get("timings"):
            payload["timings"] = dict(payload["timings"])
            payload["timings"]["wait_ms"] = wait_ms
            payload["timings"]["prefetched"] = True
        return payload


def _reply_for_speaker(
    *,
    agents: list[dict[str, Any]],
    speaker_idx: int,
    discussion_topic: str,
    transcript_so_far: list[dict[str, str]],
    tts_enabled: bool = False,
    include_conversation_instructions: bool | None = None,
    disfluency_enabled: bool | None = None,
) -> tuple[str, dict[str, Any]]:
    client = _client(timeout=TURN_REQUEST_TIMEOUT_S, max_retries=2)
    if client is None:
        raise HTTPException(
            status_code=401,
            detail=MISSING_OPENAI_KEY,
        )

    speaker = agents[speaker_idx]
    partner = agents[1 - speaker_idx]

    system = prompts.agent_system_prompt(
        discussion_topic=discussion_topic,
        speaker_name=speaker["name"],
        partner_name=partner["name"],
        gender=str(speaker.get("gender", "")),
        partner_gender=str(partner.get("gender", "")),
        stance_on_topic=str(speaker.get("initial_view", "")),
        personal_story=str(speaker.get("personal_story", "")),
        voice_style=str(speaker.get("voice", "")),
        include_conversation_instructions=include_conversation_instructions,
    )
    user_content = prompts.agent_user_prompt(
        speaker_name=speaker["name"],
        transcript_so_far=transcript_so_far,
    )

    model = "gpt-4o"
    t_all = time.perf_counter()
    t0 = time.perf_counter()
    rsp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
    )
    text_generation_ms = _elapsed_ms(t0)
    lt = (rsp.choices[0].message.content or "").strip()
    if not lt:
        raise HTTPException(status_code=502, detail="empty model reply")

    pref = speaker["name"] + ":"
    if lt.lower().startswith(pref.lower()):
        lt = lt[len(pref) :].lstrip()
    text = lt[:2000]

    display, stage_partial = _segment_utterance_for_display(
        client, speaker["name"], partner["name"], text,
        discussion_topic=discussion_topic,
        disfluency_enabled=disfluency_enabled,
    )

    audio_lines: list[dict[str, Any]] = []
    tts_line_timings: list[dict[str, Any]] = []
    tts_total_ms: int | None = None
    tts_wall_ms = 0
    if tts_enabled:
        t0 = time.perf_counter()
        tts_units = _tts_units_for_synthesis(display)
        audio_lines, tts_line_timings, tts_total_ms = _synthesize_turn_audio(
            speaker=speaker,
            listener=partner,
            tts_units=tts_units,
            backchannels=_backchannels_for_tts(display),
            api_key=_elevenlabs_key(),
        )
        tts_wall_ms = _elapsed_ms(t0)

    timings: dict[str, Any] = {
        "text_generation_ms": text_generation_ms,
        "segment_ms": stage_partial["segment_ms"],
        "backchannel_ms": stage_partial["backchannel_ms"],
        "disfluency_ms": stage_partial.get("disfluency_ms", 0),
        "expression_ms": stage_partial.get("expression_ms", 0),
        "tts_requested": tts_enabled,
        "tts_synthesized": tts_total_ms is not None,
        "tts_total_ms": tts_total_ms if tts_total_ms is not None else 0,
        "tts_wall_ms": tts_wall_ms,
        "tts_lines": tts_line_timings,
        "total_ms": _elapsed_ms(t_all),
    }
    display["timings"] = timings
    display["audio_lines"] = audio_lines
    return text, display


@app.get("/turn_audio")
def get_turn_audio(session_id: str, turn: int):
    sess = _sessions.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="unknown session_id")

    state: TurnTtsState | None = sess.get("turn_tts_by_turn", {}).get(turn)
    if state is None:
        state = sess.get("turn_tts")
    if state is None or state.turn_no != turn:
        return {
            "turn": turn,
            "expected_units": 0,
            "segments": {},
            "backchannels": {},
            "tts_line_timings": [],
            "complete": True,
        }

    with state.lock:
        return {
            "turn": turn,
            "expected_units": state.unit_count,
            "segments": {str(k): v for k, v in state.segments.items()},
            "backchannels": {str(k): v for k, v in state.backchannels.items()},
            "tts_line_timings": list(state.tts_line_timings),
            "complete": state.complete,
            "tts_total_ms": state.tts_total_ms,
            "error": state.error,
        }


@app.get("/")
def serve_index():
    path = ROOT / "frontend" / "index.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="frontend/index.html missing")
    return FileResponse(path)


@app.get("/topics")
def get_topics():
    """Single-topic list sourced from prompts.DISCUSSION_TOPIC (UI convenience)."""
    t = prompts.discussion_topic_strip()
    return {"topics": [t]}


@app.get("/personas")
def get_personas_doc():
    return personas_document()


class GeneratePersonasBody(BaseModel):
    """Topic used to synthesize initial_view + personal_story for both agents via the model."""
    topic: str = Field(min_length=1)


@app.get("/key-config")
def key_config():
    return {
        "allow_server_keys": ALLOW_SERVER_KEYS,
        "server_has_openai_key": ALLOW_SERVER_KEYS and bool(os.getenv("OPENAI_API_KEY")),
        "server_has_elevenlabs_key": ALLOW_SERVER_KEYS and bool(os.getenv("ELEVENLABS_API_KEY")),
    }


def _check_openai_key(key: str | None) -> dict[str, Any]:
    if not key:
        return {"ok": False, "error": "Missing."}
    try:
        OpenAI(api_key=key, timeout=15.0, max_retries=1).models.list()
        return {"ok": True}
    except AuthenticationError:
        return {"ok": False, "error": "OpenAI rejected this key."}
    except Exception as exc:
        return {"ok": False, "error": f"Could not reach OpenAI: {exc}"}


def _check_elevenlabs_key(key: str | None) -> dict[str, Any]:
    if not key:
        return {"ok": False, "error": "Missing."}
    try:
        rsp = httpx.get(
            "https://api.elevenlabs.io/v1/models",
            headers={"xi-api-key": key},
            timeout=15.0,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Could not reach ElevenLabs: {exc}"}
    if rsp.status_code in (400, 401, 403):
        try:
            status = str((rsp.json().get("detail") or {}).get("status", ""))
        except Exception:
            status = ""
        # Restricted keys may lack the models scope but still allow text-to-speech.
        if status != "missing_permissions":
            return {"ok": False, "error": "ElevenLabs rejected this key."}
    elif rsp.status_code >= 400:
        return {"ok": False, "error": f"ElevenLabs returned HTTP {rsp.status_code}."}
    return {"ok": True}


@app.post("/validate-keys")
def validate_keys():
    openai_result = _check_openai_key(_openai_key())
    elevenlabs_result = _check_elevenlabs_key(_elevenlabs_key())
    return {
        "ok": openai_result["ok"] and elevenlabs_result["ok"],
        "openai": openai_result,
        "elevenlabs": elevenlabs_result,
    }


@app.post("/generate-personas-from-topic")
def generate_personas_from_topic(body: GeneratePersonasBody):
    """
    1) One model call → two distinct second-person initial_view strings (JSON).
    2) Two model calls → personal_story per agent from topic + that agent’s view.
    Persists result to data/agent_personas.json (names and other metadata preserved).
    If exp_control.use_fixed_personas is True, skips LLM generation and returns the current personas as-is.
    """
    if exp_control.use_fixed_personas:
        doc = personas_document()
        return {"ok": True, "path": str(PERSONAS_PATH.relative_to(ROOT)), "personas": doc, "fixed": True}

    client = _client(timeout=PERSONA_REQUEST_TIMEOUT_S, max_retries=2)
    if client is None:
        raise HTTPException(status_code=401, detail=MISSING_OPENAI_KEY)

    topic = body.topic.strip()
    model = PERSONA_AUTHORING_MODEL
    doc = personas_document()
    agents: list[dict[str, Any]] = doc["agents"]
    name_a = str(agents[0]["name"])
    name_b = str(agents[1]["name"])

    t0 = time.perf_counter()
    views_payload = _openai_json_object(
        client=client,
        model=model,
        system=prompts.PERSONA_TWO_VIEWS_SYSTEM,
        user=prompts.persona_two_views_user_prompt(topic, name_a, name_b),
    )
    print(f"[PERSONAS] views {_elapsed_ms(t0)}ms")
    view_a = str(views_payload.get("initial_view_a", "")).strip()
    view_b = str(views_payload.get("initial_view_b", "")).strip()
    if not view_a or not view_b:
        raise HTTPException(
            status_code=502,
            detail="model did not return initial_view_a / initial_view_b",
        )

    def story_for(name: str, view: str) -> dict[str, Any]:
        return _openai_json_object(
            client=client,
            model=model,
            system=prompts.PERSONA_STORY_SYSTEM,
            user=prompts.persona_story_user_prompt(topic, name, view),
        )

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(story_for, name_a, view_a)
        fut_b = pool.submit(story_for, name_b, view_b)
        story_a_payload = fut_a.result()
        story_b_payload = fut_b.result()
    print(f"[PERSONAS] stories {_elapsed_ms(t0)}ms")
    story_a = str(story_a_payload.get("personal_story", "")).strip()
    story_b = str(story_b_payload.get("personal_story", "")).strip()
    if not story_a or not story_b:
        raise HTTPException(
            status_code=502,
            detail="model did not return personal_story for one or both agents",
        )

    new_doc: dict[str, Any] = {
        "agents": [
            _merge_agent_identity(agents[0], initial_view=view_a, personal_story=story_a),
            _merge_agent_identity(agents[1], initial_view=view_b, personal_story=story_b),
        ]
    }
    _write_personas_document(new_doc)
    return {"ok": True, "path": str(PERSONAS_PATH.relative_to(ROOT)), "personas": new_doc}


class StartBody(BaseModel):
    """Optional override replaces prompts.DISCUSSION_TOPIC for that session only."""
    topic_override: str | None = Field(default=None)
    tts_enabled: bool = Field(default=False)
    max_turns: int | None = Field(default=None, ge=1, le=40)
    mode: Literal["baseline", "exp"] | None = None


def _comparison_flags(mode: str | None) -> tuple[bool | None, bool | None]:
    """Return (conversation_instructions, disfluency_enabled). None keeps global exp_control."""
    if mode == "baseline":
        return False, False
    if mode == "exp":
        return True, True
    return None, None


@app.post("/start")
def start_session(body: StartBody):
    doc = personas_document()
    agents: list[dict[str, Any]] = doc["agents"]
    _purge_stale_sessions()
    sid = uuid.uuid4().hex[:16]

    ov = (body.topic_override or "").strip()
    discussion_topic = ov or prompts.discussion_topic_strip()
    tts_enabled = body.tts_enabled
    max_turns = int(body.max_turns) if body.max_turns is not None else MAX_TURNS
    conversation_instructions, disfluency_enabled = _comparison_flags(body.mode)
    elevenlabs_api_key = _elevenlabs_key()
    if tts_enabled and not elevenlabs_api_key:
        raise HTTPException(status_code=401, detail=MISSING_ELEVENLABS_KEY)

    transcript: list[dict[str, str]] = []
    speaker_idx = len(transcript) % 2

    sess: dict[str, Any] = {
        "discussion_topic": discussion_topic,
        "agents": agents,
        "transcript": transcript,
        "tts_enabled": tts_enabled,
        "max_turns": max_turns,
        "mode": body.mode,
        "conversation_instructions": conversation_instructions,
        "disfluency_enabled": disfluency_enabled,
        # Background TTS threads run outside the request, so the key travels with the session.
        "elevenlabs_api_key": elevenlabs_api_key,
        "created_at": time.time(),
    }
    _sessions[sid] = sess
    print(
        f"[SESSION] mode={body.mode} max_turns={max_turns} "
        f"conversation_instructions={conversation_instructions} disfluency={disfluency_enabled}"
    )

    if tts_enabled:
        pipeline = SessionPipeline(sess)
        sess["pipeline"] = pipeline
        turn_payload = pipeline.build_first_turn(speaker_idx)
    else:
        text, display = _reply_for_speaker(
            agents=agents,
            speaker_idx=speaker_idx,
            discussion_topic=discussion_topic,
            transcript_so_far=transcript,
            tts_enabled=False,
            include_conversation_instructions=conversation_instructions,
            disfluency_enabled=disfluency_enabled,
        )
        first = agents[speaker_idx]
        transcript.append({"speaker": first["name"], "text": text})
        partner = agents[1 - speaker_idx]
        turn_payload = _turn_api_payload(1, first, partner, text, display)

    return {
        "session_id": sid,
        "max_turns": max_turns,
        "mode": body.mode,
        "discussion_topic": discussion_topic,
        "tts_enabled": tts_enabled,
        "debug_setting": bool(exp_control.debug_setting),
        "use_pause_and_gap": bool(exp_control.use_pause_and_gap),
        "personas": {"agents": agents},
        "turn": turn_payload,
    }


class NextBody(BaseModel):
    session_id: str = Field(min_length=8)


@app.post("/next_turn")
def next_turn(body: NextBody):
    sid = body.session_id
    sess = _sessions.get(sid)
    if not sess:
        raise HTTPException(status_code=404, detail="unknown session_id")

    transcript: list[dict[str, str]] = sess["transcript"]
    max_turns = int(sess.get("max_turns") or MAX_TURNS)
    pipeline_early: SessionPipeline | None = sess.get("pipeline")
    if len(transcript) >= max_turns:
        if pipeline_early is None:
            return {"done": True}
        with pipeline_early.lock:
            if pipeline_early.prefetch is None:
                return {"done": True}

    tts_enabled = bool(sess.get("tts_enabled", False))
    pipeline: SessionPipeline | None = sess.get("pipeline")

    if tts_enabled and pipeline is not None:
        with pipeline.lock:
            turn_no = pipeline.prefetch_turn_no or (len(transcript) + 1)
        try:
            turn_payload = pipeline.take_next_turn(turn_no)
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"prefetch failed: {exc}") from exc
    else:
        turn_no = len(transcript) + 1
        agents: list[dict[str, Any]] = sess["agents"]
        speaker_idx = len(transcript) % 2
        discussion_topic = str(sess["discussion_topic"])
        text, display = _reply_for_speaker(
            agents=agents,
            speaker_idx=speaker_idx,
            discussion_topic=discussion_topic,
            transcript_so_far=transcript,
            tts_enabled=tts_enabled,
            include_conversation_instructions=sess.get("conversation_instructions"),
            disfluency_enabled=sess.get("disfluency_enabled"),
        )
        speaker = agents[speaker_idx]
        partner = agents[1 - speaker_idx]
        transcript.append({"speaker": speaker["name"], "text": text})
        turn_payload = _turn_api_payload(turn_no, speaker, partner, text, display)

    done = len(transcript) >= max_turns
    if done and pipeline is not None:
        with pipeline.lock:
            if pipeline.prefetch is not None:
                done = False
    payload: dict[str, Any] = {"turn": turn_payload, "prefetched": tts_enabled}
    if done:
        payload["done"] = True
        pl: SessionPipeline | None = sess.get("pipeline")
        if pl is not None:
            pl.cancel()
    return payload


class CancelBody(BaseModel):
    session_id: str = Field(min_length=8)


@app.post("/cancel")
def cancel_session(body: CancelBody):
    """Stop prefetch for this session. Turns already started can still finish TTS."""
    sess = _sessions.get(body.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="unknown session_id")
    pipeline: SessionPipeline | None = sess.get("pipeline")
    if pipeline is not None:
        pipeline.cancel()
    return {"ok": True}
