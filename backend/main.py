"""
Minimal two-agent dialogue API.

Personas: data/agent_personas.json — LLM wording: backend/prompts.py
"""

from __future__ import annotations

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
from typing import Any, Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel, Field

from backend import prompts, tts

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MAX_TURNS = int(os.getenv("CONVERSATION_MAX_TURNS", "12"))

# Edit this string to change which model writes initial_view + personal_story via /generate-personas-from-topic.
PERSONA_AUTHORING_MODEL = "gpt-4o"

# Splits each gpt-4o reply into pause-separated line's for display (transcript context still uses raw text).
# UTTERANCE_SEGMENT_MODEL is used only when the LLM splitter below is re-enabled.
UTTERANCE_SEGMENT_MODEL = "gpt-4o-mini"
BACKCHANNEL_INSERT_MODEL = "gpt-4o"
DISFLUENCY_INSERT_MODEL = "gpt-4o"
EXPRESSION_TAG_MODEL = "gpt-4o-mini"
# input_backchannels = 0 # Manually control the number of backchannels inserted
DISFLUENCY_RATE_PER_WORD = float(os.getenv("DISFLUENCY_RATE_PER_WORD", "0.14"))
DISFLUENCY_TIER_COMMON = 1
DISFLUENCY_TYPE_WEIGHTS_COMMON: dict[str, float] = {
    "filled_pause": 0.45,
    "discourse_marker": 0.30,
    "elongation": 0.25
}
DISFLUENCY_TYPE_WEIGHTS_RARE: dict[str, float] = {
    "self_repair": 0.7,
    "stumble": 0.3
}
VALID_DISFLUENCY_TYPES = frozenset(
    set(DISFLUENCY_TYPE_WEIGHTS_COMMON) | set(DISFLUENCY_TYPE_WEIGHTS_RARE)
)

app = FastAPI(title="two-agent-chat")
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


def _client() -> OpenAI | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    return OpenAI(api_key=key)


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
    if max_backchannels == 0:
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
    if random.random() < DISFLUENCY_TIER_COMMON:
        weights = DISFLUENCY_TYPE_WEIGHTS_COMMON
    else:
        weights = DISFLUENCY_TYPE_WEIGHTS_RARE
    keys = list(weights.keys())
    vals = list(weights.values())
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
) -> tuple[list[dict[str, Any]], list[str], int]:
    """Returns (disfluencies, segments_for_tts, disfluency_count)."""
    if not segments:
        return [], [], 0

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
) -> tuple[list[str], list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Tag TTS units for Eleven v3; split back to micro segments for display."""
    if not tts_units:
        return [], list(segments_for_tts), [dict(b) for b in backchannels], []

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
            ),
        )
    except Exception:
        return fallback_units, list(segments_for_tts), [dict(b) for b in backchannels], []

    raw_units = data.get("speaker_units_for_tts") or data.get("speaker_segments")
    units_expressive: list[str] = []
    if isinstance(raw_units, list) and len(raw_units) == len(tts_units):
        for plain, raw in zip(tts_units, raw_units):
            tagged = str(raw).strip()
            if tagged and _normalize_turn_plain(
                tts.strip_audio_tags(tagged)
            ) == _normalize_turn_plain(plain):
                units_expressive.append(tagged)
            else:
                units_expressive.append(plain)
    else:
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

    t0 = time.perf_counter()
    disfluencies, segments_for_tts, disfluency_count = _choose_disfluencies(
        client,
        speaker_name=speaker_name,
        segments=segments,
    )
    disfluency_ms = _elapsed_ms(t0)

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

    t0 = time.perf_counter()
    units_expressive, segments_expressive, backchannels, expressions = _apply_expression_tags(
        client,
        speaker_name=speaker_name,
        listener_name=listener_name,
        tts_units=tts_units,
        unit_micro_indices=unit_micro_indices,
        segments_for_tts=segments_for_tts,
        backchannels=backchannels,
    )
    expression_ms = _elapsed_ms(t0)

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
    retries: int = 2,
) -> dict[str, Any]:
    last_err: str | None = None
    for _ in range(max(1, retries)):
        try:
            result = tts.synthesize_with_timestamps(
                text=text,
                voice_id=voice_id,
                speed=speed,
            )
            if result.get("audio_base64"):
                return result
            last_err = "empty audio response"
        except Exception as exc:
            last_err = str(exc)
    return {
        "audio_base64": "",
        "duration_s": 0.0,
        "api_ms": 0,
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None]:
    """One clip per TTS unit + separate backchannel clips (parallel)."""
    if not os.getenv("ELEVENLABS_API_KEY"):
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

    t0 = time.perf_counter()
    workers = min(6, max(1, len(jobs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                _synthesize_with_retry,
                text=text,
                voice_id=voice_id,
                speed=speed,
            ): (
                kind,
                idx,
                name,
                text,
            )
            for kind, idx, text, name, voice_id, speed in jobs
        }
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
            return lt[:2000], total_ms
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
        self.lock = threading.Lock()
        self.ready = threading.Event()
        self.consumed = threading.Event()
        self.consumed.set()
        self.prefetch: dict[str, Any] | None = None
        self.prefetch_turn_no: int | None = None
        self.error: str | None = None
        self.cancelled = False
        self.tts_pool = ThreadPoolExecutor(max_workers=2)
        self._build_executor = ThreadPoolExecutor(max_workers=1)
        self._worker: threading.Thread | None = None
        client = _client()
        if client is None:
            raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set.")
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
            self._client, speaker["name"], partner["name"], text
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

        if MAX_TURNS > 1:
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
            while turn_no <= MAX_TURNS and not self.cancelled:
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

                if turn_no < MAX_TURNS and not self.cancelled:
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
) -> tuple[str, dict[str, Any]]:
    client = _client()
    if client is None:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set.",
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
        client, speaker["name"], partner["name"], text
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


@app.post("/generate-personas-from-topic")
def generate_personas_from_topic(body: GeneratePersonasBody):
    """
    1) One model call → two distinct second-person initial_view strings (JSON).
    2) Two model calls → personal_story per agent from topic + that agent’s view.
    Persists result to data/agent_personas.json (names and other metadata preserved).
    """
    client = _client()
    if client is None:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not set.")

    topic = body.topic.strip()
    model = PERSONA_AUTHORING_MODEL
    doc = personas_document()
    agents: list[dict[str, Any]] = doc["agents"]
    name_a = str(agents[0]["name"])
    name_b = str(agents[1]["name"])

    views_payload = _openai_json_object(
        client=client,
        model=model,
        system=prompts.PERSONA_TWO_VIEWS_SYSTEM,
        user=prompts.persona_two_views_user_prompt(topic, name_a, name_b),
    )
    view_a = str(views_payload.get("initial_view_a", "")).strip()
    view_b = str(views_payload.get("initial_view_b", "")).strip()
    if not view_a or not view_b:
        raise HTTPException(
            status_code=502,
            detail="model did not return initial_view_a / initial_view_b",
        )

    story_a_payload = _openai_json_object(
        client=client,
        model=model,
        system=prompts.PERSONA_STORY_SYSTEM,
        user=prompts.persona_story_user_prompt(topic, name_a, view_a),
    )
    story_b_payload = _openai_json_object(
        client=client,
        model=model,
        system=prompts.PERSONA_STORY_SYSTEM,
        user=prompts.persona_story_user_prompt(topic, name_b, view_b),
    )
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


@app.post("/start")
def start_session(body: StartBody):
    doc = personas_document()
    agents: list[dict[str, Any]] = doc["agents"]
    sid = uuid.uuid4().hex[:16]

    ov = (body.topic_override or "").strip()
    discussion_topic = ov or prompts.discussion_topic_strip()
    tts_enabled = body.tts_enabled

    transcript: list[dict[str, str]] = []
    speaker_idx = len(transcript) % 2

    sess: dict[str, Any] = {
        "discussion_topic": discussion_topic,
        "agents": agents,
        "transcript": transcript,
        "tts_enabled": tts_enabled,
    }
    _sessions[sid] = sess

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
        )
        first = agents[speaker_idx]
        transcript.append({"speaker": first["name"], "text": text})
        partner = agents[1 - speaker_idx]
        turn_payload = _turn_api_payload(1, first, partner, text, display)

    return {
        "session_id": sid,
        "max_turns": MAX_TURNS,
        "discussion_topic": discussion_topic,
        "tts_enabled": tts_enabled,
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
    pipeline_early: SessionPipeline | None = sess.get("pipeline")
    if len(transcript) >= MAX_TURNS:
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
        agents: list[dict[str, Any]] = sess["agents"]
        speaker_idx = len(transcript) % 2
        discussion_topic = str(sess["discussion_topic"])
        text, display = _reply_for_speaker(
            agents=agents,
            speaker_idx=speaker_idx,
            discussion_topic=discussion_topic,
            transcript_so_far=transcript,
            tts_enabled=tts_enabled,
        )
        speaker = agents[speaker_idx]
        partner = agents[1 - speaker_idx]
        transcript.append({"speaker": speaker["name"], "text": text})
        turn_payload = _turn_api_payload(turn_no, speaker, partner, text, display)

    done = len(transcript) >= MAX_TURNS
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
