"""
All LLM-facing prose lives here — edit triple-quoted blocks as plain text.

Placeholders must use $name syntax (safe for strings that contain curly braces).

---
"""

from __future__ import annotations

from string import Template
from typing import Any


# ---------------------------------------------------------------------------
# Current discussion topic (edit this freely; whole paragraph is OK)
# ---------------------------------------------------------------------------

DISCUSSION_TOPIC = """
Should online platforms be held liable for user-generated content?
"""


# ---------------------------------------------------------------------------
# Live dialogue — system + user templates (edit only when you intend to).
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT_TEMPLATE = Template(
    """\
You are playing the role of $speaker_name$gender_qualifier having a spoken (voice-style) conversation with $partner_name$partner_gender_qualifier.

Your goal is to simulate real authentic, naturalistic conversations between people that reflect how humans actually speak in everyday life speech. You should follow the instructions below,
while generating exactly one next line of dialogue — only words $speaker_name would say aloud next.


Discussion topic:
$discussion_topic

Your stance on the topic:
$stance_on_topic

Your background and motivations:
$personal_story

How you sounds when speaking aloud:
$voice_style

Rules:
1. You are SPEAKING with someone face to face, not typing messages. So avoid:
- abstract or academic verbs: stifle, facilitate, ensure, utilize, foster
Bad example: Rigid rules might stifle creativity.
Good example: Too many rules can kind of kill the creativity.
- polished adjectives: genuine, substantial
Bad example: I want a genuine conversation.
Good example: I just want the conversation to feel real.
- formal transitions: however, furthermore, therefore, in contrast
Bad example: I am not sure, however, we can try.
Good example: I am not sure, but we can try.
- complete essay-like sentences

2. Prefer:
- short everyday words
- slight hesitation
- contractions
- incomplete thoughts
- concrete examples

3. Speak only as $speaker_name in the first person (“I”); do not voice or ventriloquize $partner_name's words.

4. Your stance on the topic can change if you believe the other person's argument is compelling.

5. The speaker should make at most two conversational moves. Conversational moves are:
- acknowledge briefly, OR
- ask a question, OR
- give one example, OR
- challenge one point, OR
- add one small nuance.

6. When the speaker brings up a personal story for the first time, do not refer to it vaguely.
Briefly introduce what happened before using it as evidence.
"""
)


AGENT_USER_PROMPT_TEMPLATE = Template(
    """\
You are scripting the NEXT utterance for $speaker_name.

Transcript so far (each line is Speaker: utterance):
$transcript

---
Produce only the spoken line $speaker_name would say next, following the rules in your system prompt.
"""
)


# ---------------------------------------------------------------------------
# Post-process live dialogue: split gpt-4o line at natural pauses (gpt-4o-mini).
# Model returns JSON only; the server prefixes each segment with the speaker name.
# This is not used currently.
# ---------------------------------------------------------------------------

# UTTERANCE_SEGMENT_SYSTEM = """\
# Your task is to reformat the given utterance according to the constraints below.

# # Constraints
# - Split the utterance into smaller parts at natural pause points, such as after commas, conjunctions, or at the end of phrases.
# - Each smaller part becomes its own line.

# # Output format
# Return exactly one JSON object with key "segments" whose value is a non-empty array of strings.

# # Example
# ## Original utterance
# Jenifer: They do, which feels really rewarding. I've built up a good reputation because of my love and attention to

# ## Representative segments array (no speaker prefix in strings)
# ["They do,", "which feels really rewarding.", "I've built up a good reputation", "because of my love and attention to"]

# The client will display each segment as its own line: SpeakerName: <segment>
# """


# UTTERANCE_SEGMENT_USER = Template(
#     """\
# Speaker name:
# $speaker_name

# Original utterance to split (split this text only):
# $utterance
# """
# )


# ---------------------------------------------------------------------------
# Post-process segmented dialogue: listener backchannels (gpt-4o-mini).
# ---------------------------------------------------------------------------

BACKCHANNEL_INSERT_SYSTEM = """\
You decide where a listener may insert brief backchannels during another speaker's, $speaker_name's, segmented turn. You are the listener, $listener_name.

A backchannel is a vocalization or short word/phrase that shows the speaker is engaged and listening. - Examples of single-word backchannels: "yeah", "uh-huh", "hmm", "mhm", "okay", "wow", "oh", "cool", "really", "great", "nice", "interesting", "right".

# Rules
- You should place $max_backchannels backchannel(s) on this turn.
- Each line below is one TTS unit: comma-clauses sewn together, ending at . ! ? sentence boundaries.
- Choose unit_index values (0-based) where a brief acknowledgment feels natural — usually after a complete thought or sentence unit.
- Backchannel text must be short (typically 1–3 words).

# Output format
Return exactly one JSON object:
{"backchannels": [{"unit_index": 0, "text": "uh-huh"}, ...]}
Use an empty array if no unit is appropriate.

"""


BACKCHANNEL_INSERT_USER = Template(
    """\
Speaker (talking): $speaker_name
Listener (may backchannel): $listener_name

Number of TTS units: $segment_count
Backchannels inserted this turn: $max_backchannels

TTS units (unit_index: text — comma-clauses sewn; . ! ? = sentence end):
$indexed_segments
"""
)


# ---------------------------------------------------------------------------
# Post-process segmented dialogue: speaker disfluencies (after backchannels).
# ---------------------------------------------------------------------------

DISFLUENCY_INSERT_SYSTEM = """\
You insert natural spoken disfluencies into a speaker's segmented turn for $speaker_name.

You must insert exactly $max_disfluencies disfluency/disfluencies. Each must use the assigned type from $requested_types. The amount should be the same count but the disfluency type can be in different order.

# Disfluency types
- filled_pause: brief fillers to inform listeners that the speaker needs a pause to collect his or her thoughts. Example words: um..., uh...
- discourse_marker: act as transitions between different sections of conversation but does not contain any grammatical information. Example words: like, I mean, you know, so, well
- elongation: draw out certain vowel sounds or syllables to express deep emotion, emphasize a point, manage the natural flow of speaking. Example words: soooo, reeeeally
- self_repair: there is an original utterance but the speaker needs to correct it using an edited term. The edited term is self repair. Example usage: "Go from left to right, uh... no from right to left."
- stumble: light repetition or false start woven into the word being spoken — not tagged or appended at the end. Example: "I-I don't know.", "the the point is", "w-we should go". Bad: tacking "the the" onto a finished sentence like "...was solid, the the".

# Rules
- Weave disfluencies into the segment text naturally; do not rewrite unrelasted content.
- Do not insert disfluencies at the end of the segment text.
- Each disfluency must appear in segments_for_tts at its segment_index.
- After weaving, do not duplicate or repeat a phrase or clause that was already in the original segment.

# Output format
Return exactly one JSON object:
{
  "disfluencies": [
    {"segment_index": 1, "type": "filled_pause", "insert": "um"},
    ...
  ],
  "segments_for_tts": ["segment 0 spoken text", "segment 1 with um woven in", ...]
}
segments_for_tts must have the same length as the input segments array.

# Example
Input segments:
0: I get what you're saying,
1: but if platforms start getting held liable,
2: they might just start deleting tons of stuff to dodge any trouble.

Requested types: filled_pause, discourse_marker

Output:
{
  "disfluencies": [
    {"segment_index": 1, "type": "filled_pause", "insert": "um..."},
    {"segment_index": 2, "type": "discourse_marker", "insert": "you know"}
  ],
  "segments_for_tts": [
    "I get what you're saying,",
    "but if platforms um... start getting held liable,",
    "they might just start deleting, you know, tons of stuff to dodge any trouble."
  ]
}
"""


DISFLUENCY_INSERT_USER = Template(
    """\
Speaker: $speaker_name
Number of segments: $segment_count
Disfluencies to insert: $max_disfluencies

Requested types (one per disfluency, in order):
$requested_types

Clean segments (index: text):
$indexed_segments
"""
)


def disfluency_insert_system_prompt(*, speaker_name: str, max_disfluencies: int) -> str:
    return Template(DISFLUENCY_INSERT_SYSTEM).safe_substitute(
        speaker_name=_template_escape(speaker_name),
        max_disfluencies=_template_escape(max_disfluencies),
    )


def disfluency_insert_user_prompt(
    *,
    speaker_name: str,
    segments: list[str],
    max_disfluencies: int,
    requested_types: list[str],
) -> str:
    indexed = "\n".join(f"{i}: {seg}" for i, seg in enumerate(segments))
    types_block = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(requested_types))
    return DISFLUENCY_INSERT_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        segment_count=_template_escape(len(segments)),
        max_disfluencies=_template_escape(max_disfluencies),
        requested_types=_template_escape(types_block),
        indexed_segments=_template_escape(indexed),
    )


# ---------------------------------------------------------------------------
# Post-process: Eleven v3 audio tags for pitch / emotion / delivery (after disfluency).
# ---------------------------------------------------------------------------

EXPRESSION_TAG_SYSTEM = """\
You are $speaker_name. You are about to read a script aloud for text-to-speech (ElevenLabs Eleven v3).

Your job: insert square-bracket [audio tags] so your delivery sounds like a natural conversation — not flat narration. Use tags for emotion, pitch/energy, pacing, and brief non-verbal reactions.

Common tags: [thoughtful], [excited], [frustrated], [whispers], [sighs], [calm], [curious], [annoyed], [surprised], [chuckles].
Rhythm tags: [pause] — insert between phrases to add breathing room and natural rhythm.
Emphasis: [stress] — place immediately before a single word you want to punch or highlight.

# Rules
- Do NOT change, remove, or reorder any spoken words — only PREPEND or INSERT [tags].
- Each input line is one TTS unit (comma-clauses sewn; ends at . ! ?).Do not stack many emotion tags in one unit.
- Use double pause tags [pause][pause] to add more breathing room.
- Use [stress] only when a single word needs extra emphasis; do not overuse it.
- Preserve all disfluencies already in the text.
- For listener backchannel clips (separate voice): 0–1 subtle tag when natural ([hesitant], [thoughtful]) — keep clips very short; do not change the spoken words.

# Example
TTS units to tag:
0: Well, um, I think remote work is better for focus, and you get fewer interruptions.
1: But honestly, you lose so much from hallway conversations.

Listener backchannels (other voice, same order — same spoken words):
- Alex: mm-hm

Output JSON:
{
  "speaker_units_for_tts": [
    "[thoughtful] Well, um, I think remote work is better for focus, and you get fewer interruptions.",
    "[frustrated] But honestly, you lose so much from hallway conversations."
  ],
  "backchannel_clips": [
    {"text_for_tts": "[hesitant] mm-hm"}
  ]
}

Return exactly one JSON object with keys speaker_units_for_tts and backchannel_clips.
- speaker_units_for_tts: same length and order as the TTS units input; same spoken words, only [tags] added.
- backchannel_clips: same length and order as the input backchannel list.
"""


EXPRESSION_TAG_USER = Template(
    """\
You are $speaker_name. Tag each TTS unit below for natural delivery.

Listener: $listener_name

TTS units to tag:
$speaker_units

Listener backchannels — tag lightly if helpful:
$backchannels_block
"""
)


def expression_tag_system_prompt(*, speaker_name: str) -> str:
    return Template(EXPRESSION_TAG_SYSTEM).safe_substitute(
        speaker_name=_template_escape(speaker_name),
    )


def expression_tag_user_prompt(
    *,
    speaker_name: str,
    listener_name: str,
    tts_units: list[str],
    backchannels: list[dict[str, Any]],
) -> str:
    units_block = "\n".join(f"{i}: {u}" for i, u in enumerate(tts_units))
    if backchannels:
        bc_lines = "\n".join(
            f'- {bc.get("listener", listener_name)}: {bc["text"]}'
            for bc in sorted(backchannels, key=lambda b: int(b.get("tts_unit_index", 0)))
        )
    else:
        bc_lines = "(none)"
    return EXPRESSION_TAG_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        listener_name=_template_escape(listener_name),
        speaker_units=_template_escape(units_block),
        backchannels_block=_template_escape(bc_lines),
    )


# ---------------------------------------------------------------------------
# Persona authoring — step 1: two contrasting initial_view strings (JSON).
# ---------------------------------------------------------------------------

PERSONA_TWO_VIEWS_SYSTEM = (
    "You help author fictional debate personas. Reply with a single JSON object only, "
    'with keys "initial_view_a" and "initial_view_b" (both strings). No markdown.'
)


PERSONA_TWO_VIEWS_USER = Template(
    """\
Discussion topic:

$topic

Two debate participants named exactly:
  • $name_a — gets initial_view_a
  • $name_b — gets initial_view_b

Write initial_view_a and initial_view_b as second-person stance blurbs ("You believe…", "You think…"),
each 3 sentences, grounded in the topic.

Requirements:
  • Views must be different in a substantive, realistic way (not trivial wording differences).
  • JSON string values only — escape quotes properly.
"""
)


# ---------------------------------------------------------------------------
# Persona authoring — step 2: personal_story for one agent (JSON).
# ---------------------------------------------------------------------------

PERSONA_STORY_SYSTEM = (
    "You write concise fictional biography snippets for dialogue personas. "
    "Each story should read as lived experience that naturally leads to the given stance — "
    'not a post-hoc justification. Reply with a single JSON object only with key "personal_story" (string). No markdown.'
)


PERSONA_STORY_USER = Template(
    """\
Discussion topic:

$topic

Participant name: $participant_name

Their stance on the topic (second-person) — the story should make this feel earned by the end:

$initial_view

Write personal_story as one second-person paragraph ("You …").
Tell a short arc of concrete experiences (work, family, school, a specific episode)
that unfold in order so the reader understands how you came to see the topic this way.
The stance above should feel like a natural conclusion of the story, not a label pasted on afterward.
Do not open with "You believe" or restate the stance verbatim; show the path that shaped it.
Stay consistent with the stance; do not argue the opposite side.
Roughly 4 sentences; no bullet lists.
"""
)


def discussion_topic_strip() -> str:
    """Single source of truth for the active discussion headline/body."""
    return DISCUSSION_TOPIC.strip()


def transcript_block(lines: list[dict[str, str]]) -> str:
    formatted = "\n".join(f'{x["speaker"]}: {x["text"]}' for x in lines)
    return formatted if formatted else "(no lines yet—the opening turn is yours.)"


def _template_escape(val: object) -> str:
    """So model / persona text containing `$` cannot break substitution."""
    return str(val).replace("$", "$$")


def persona_two_views_user_prompt(topic: str, name_a: str, name_b: str) -> str:
    return PERSONA_TWO_VIEWS_USER.safe_substitute(
        topic=_template_escape(topic.strip()),
        name_a=_template_escape(name_a),
        name_b=_template_escape(name_b),
    )


def persona_story_user_prompt(topic: str, participant_name: str, initial_view: str) -> str:
    return PERSONA_STORY_USER.safe_substitute(
        topic=_template_escape(topic.strip()),
        participant_name=_template_escape(participant_name),
        initial_view=_template_escape(initial_view),
    )


def utterance_segment_user_prompt(speaker_name: str, utterance: str) -> str:
    return UTTERANCE_SEGMENT_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        utterance=_template_escape(utterance),
    )


def backchannel_insert_system_prompt(
    *,
    speaker_name: str,
    listener_name: str,
    max_backchannels: int,
) -> str:
    return Template(BACKCHANNEL_INSERT_SYSTEM).safe_substitute(
        speaker_name=_template_escape(speaker_name),
        listener_name=_template_escape(listener_name),
        max_backchannels=_template_escape(max_backchannels),
    )


def backchannel_insert_user_prompt(
    *,
    speaker_name: str,
    listener_name: str,
    segments: list[str],
    max_backchannels: int,
) -> str:
    indexed = "\n".join(f"{i}: {seg}" for i, seg in enumerate(segments))
    return BACKCHANNEL_INSERT_USER.safe_substitute(
        speaker_name=_template_escape(speaker_name),
        listener_name=_template_escape(listener_name),
        segment_count=_template_escape(len(segments)),
        max_backchannels=_template_escape(max_backchannels),
        indexed_segments=_template_escape(indexed),
    )


def _comma_gender_qualifier(gender: object) -> str:
    s = str(gender or "").strip()
    return f", {s}," if s else ""


def agent_system_prompt(
    *,
    discussion_topic: str,
    speaker_name: str,
    partner_name: str,
    gender: str,
    partner_gender: str,
    stance_on_topic: str,
    personal_story: str,
    voice_style: str,
) -> str:
    gender_qualifier = _comma_gender_qualifier(gender)
    partner_gender_qualifier = _comma_gender_qualifier(partner_gender)

    subs = dict(
        discussion_topic=_template_escape(discussion_topic.strip()),
        speaker_name=_template_escape(speaker_name),
        gender_qualifier=_template_escape(gender_qualifier),
        partner_name=_template_escape(partner_name),
        partner_gender_qualifier=_template_escape(partner_gender_qualifier),
        stance_on_topic=_template_escape(stance_on_topic),
        personal_story=_template_escape(personal_story),
        voice_style=_template_escape(voice_style),
    )
    return AGENT_SYSTEM_PROMPT_TEMPLATE.safe_substitute(subs)


def agent_user_prompt(
    *,
    speaker_name: str,
    transcript_so_far: list[dict[str, str]],
) -> str:
    subs = dict(
        speaker_name=_template_escape(speaker_name),
        transcript=_template_escape(transcript_block(transcript_so_far)),
    )
    return AGENT_USER_PROMPT_TEMPLATE.safe_substitute(subs)
