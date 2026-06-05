"""ElevenLabs TTS via the with-timestamps endpoint (full clip + character alignment)."""

from __future__ import annotations

import os
import re
import time
from typing import Any

ELEVENLABS_MODEL = os.getenv("ELEVENLABS_MODEL", "eleven_v3")
ELEVENLABS_OUTPUT_FORMAT = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "mp3_44100_128")
DEFAULT_ELEVENLABS_SPEAKING_SPEED = 1.2
ELEVENLABS_SPEED_MIN = 0.7
ELEVENLABS_SPEED_MAX = 1.2


def _client():
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        return None
    from elevenlabs import ElevenLabs

    return ElevenLabs(api_key=key)


def _alignment_to_dict(alignment: Any) -> dict[str, Any] | None:
    if alignment is None:
        return None
    chars = getattr(alignment, "characters", None)
    starts = getattr(alignment, "character_start_times_seconds", None)
    ends = getattr(alignment, "character_end_times_seconds", None)
    if not chars or not ends:
        return None
    return {
        "characters": list(chars),
        "character_start_times_seconds": list(starts or []),
        "character_end_times_seconds": list(ends),
    }


def _duration_from_alignment(alignment: Any) -> float:
    d = _alignment_to_dict(alignment)
    if not d:
        return 0.0
    ends = d["character_end_times_seconds"]
    if not ends:
        return 0.0
    return float(ends[-1])


def speaking_speed_for_agent(agent: dict[str, Any]) -> float:
    raw = agent.get("elevenlabs_speaking_speed", DEFAULT_ELEVENLABS_SPEAKING_SPEED)
    try:
        speed = float(raw)
    except (TypeError, ValueError):
        speed = DEFAULT_ELEVENLABS_SPEAKING_SPEED
    return max(ELEVENLABS_SPEED_MIN, min(ELEVENLABS_SPEED_MAX, speed))


def synthesize_with_timestamps(
    *,
    text: str,
    voice_id: str,
    speed: float | None = None,
) -> dict[str, Any]:
    """
    One full clip per call. Returns audio_base64, alignment, duration_s, api_ms.
    Raises on missing API key or HTTP errors.
    """
    client = _client()
    if client is None:
        raise RuntimeError("ELEVENLABS_API_KEY is not set")

    t0 = time.perf_counter()
    request_kwargs: dict[str, Any] = {
        "voice_id": voice_id,
        "text": text,
        "model_id": ELEVENLABS_MODEL,
        "output_format": ELEVENLABS_OUTPUT_FORMAT,
    }
    if speed is not None:
        request_kwargs["voice_settings"] = {"speed": speed}
    resp = client.text_to_speech.convert_with_timestamps(**request_kwargs)
    api_ms = round((time.perf_counter() - t0) * 1000)

    alignment = getattr(resp, "alignment", None) or getattr(resp, "normalized_alignment", None)
    align_dict = _alignment_to_dict(alignment)
    duration_s = round(_duration_from_alignment(alignment), 3)

    audio_b64 = (
        getattr(resp, "audio_base64", None)
        or getattr(resp, "audio_base_64", None)
        or ""
    )
    return {
        "audio_base64": audio_b64,
        "duration_s": duration_s,
        "api_ms": api_ms,
        "alignment": align_dict,
    }


def voice_id_for_agent(agent: dict[str, Any]) -> str | None:
    vid = str(agent.get("elevenlabs_voice_id", "")).strip()
    return vid or None


_SENTENCE_END_RE = re.compile(r"[.!?]\s*$")


def sew_tts_unit_parts(parts: list[str]) -> str:
    """Join comma-clause micro-segments within one TTS unit."""
    if not parts:
        return ""
    if len(parts) == 1:
        return str(parts[0]).strip()
    out = str(parts[0]).strip()
    for p in parts[1:]:
        left = out.rstrip()
        right = str(p).strip()
        if left.endswith(","):
            out = left + " " + right
        else:
            out = left + ", " + right
    return out


def char_ranges_for_micro_parts(parts: list[str]) -> list[tuple[int, int]]:
    """Char spans of each micro-segment inside a sewn TTS unit string."""
    ranges: list[tuple[int, int]] = []
    pos = 0
    for j, p in enumerate(parts):
        text = str(p).strip()
        if j > 0 and ranges:
            prev = str(parts[j - 1]).rstrip()
            pos += 1 if prev.endswith(",") else 2
        start = pos
        pos += len(text)
        ranges.append((start, pos))
    return ranges


def group_segments_for_tts_units(
    segments: list[str],
) -> tuple[list[str], list[list[int]]]:
    """
    TTS playback units: comma-clauses sewn with ', '; . ! ? keep sentence boundaries.
    Returns (unit_texts, unit_micro_indices).
    """
    units: list[str] = []
    unit_micro_indices: list[list[int]] = []
    parts: list[str] = []
    indices: list[int] = []

    for i, seg in enumerate(segments):
        text = str(seg).strip()
        if not text:
            continue
        parts.append(text)
        indices.append(i)
        if _SENTENCE_END_RE.search(text):
            units.append(sew_tts_unit_parts(parts))
            unit_micro_indices.append(indices)
            parts = []
            indices = []

    if parts:
        units.append(sew_tts_unit_parts(parts))
        unit_micro_indices.append(indices)

    return units, unit_micro_indices


def join_segments_for_turn_tts(segments: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Join speaker segments with a single space; return char ranges per segment index."""
    joined_parts: list[str] = []
    ranges: list[tuple[int, int]] = []
    pos = 0
    for i, seg in enumerate(segments):
        text = str(seg).strip()
        if i > 0 and joined_parts:
            pos += 1
        start = pos
        if text:
            joined_parts.append(text)
            pos += len(text)
        ranges.append((start, pos))
    return " ".join(joined_parts), ranges


def _plain_idx_to_segment(plain_idx: int, char_ranges: list[tuple[int, int]]) -> int:
    for i, (start, end) in enumerate(char_ranges):
        if start <= plain_idx < end:
            return i
        if plain_idx < start:
            return max(0, i - 1)
    return max(0, len(char_ranges) - 1)


def strip_audio_tags(tagged: str) -> str:
    return re.sub(r"\[[^\]]+\]", "", tagged)


def split_tagged_turn_into_segments(
    tagged_turn: str,
    char_ranges: list[tuple[int, int]],
) -> list[str]:
    """Split one tagged turn back into per-segment strings for transcript display."""
    if not char_ranges:
        text = tagged_turn.strip()
        return [text] if text else []
    buffers: list[list[str]] = [[] for _ in char_ranges]
    plain_idx = 0
    i = 0
    while i < len(tagged_turn):
        if tagged_turn[i] == "[":
            close = tagged_turn.find("]", i)
            if close < 0:
                seg_i = _plain_idx_to_segment(plain_idx, char_ranges)
                buffers[seg_i].append(tagged_turn[i:])
                break
            tag = tagged_turn[i : close + 1]
            seg_i = _plain_idx_to_segment(plain_idx, char_ranges)
            buffers[seg_i].append(tag)
            i = close + 1
            continue
        seg_i = _plain_idx_to_segment(plain_idx, char_ranges)
        buffers[seg_i].append(tagged_turn[i])
        plain_idx += 1
        i += 1
    return ["".join(buf).strip() for buf in buffers]


def segment_timings_from_alignment(
    alignment: dict[str, Any] | None,
    char_ranges: list[tuple[int, int]],
    *,
    total_duration_s: float,
) -> list[dict[str, Any]]:
    """Map each segment's char span to start/end seconds using ElevenLabs alignment."""
    if not char_ranges:
        return []

    starts = (alignment or {}).get("character_start_times_seconds") or []
    ends = (alignment or {}).get("character_end_times_seconds") or []
    n_align = min(len(starts), len(ends))

    timings: list[dict[str, Any]] = []
    prev_end = 0.0
    for i, (c_start, c_end) in enumerate(char_ranges):
        if c_start >= c_end or n_align == 0:
            timings.append(
                {
                    "segment_index": i,
                    "start_s": round(prev_end, 3),
                    "end_s": round(prev_end, 3),
                }
            )
            continue

        start_idx = min(max(c_start, 0), n_align - 1)
        end_idx = min(max(c_end - 1, 0), n_align - 1)
        seg_start = float(starts[start_idx])
        seg_end = float(ends[end_idx])
        if seg_end < seg_start:
            seg_end = seg_start
        prev_end = seg_end
        timings.append(
            {
                "segment_index": i,
                "start_s": round(seg_start, 3),
                "end_s": round(seg_end, 3),
            }
        )

    if timings and total_duration_s > 0:
        last = timings[-1]
        if last["end_s"] < total_duration_s:
            last["end_s"] = round(total_duration_s, 3)

    return timings
