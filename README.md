# Audio-exp: Two-agent dialogue with TTS

A minimal app for two-persona debates: LLM turn generation, spoken-style post-processing (disfluencies, backchannels, ElevenLabs audio tags), and streaming TTS playback in the browser.

## Requirements

- **Python** 3.10 or newer
- **pip** (venv recommended)

### Python packages

Install from `requirements.txt`:


| Package             | Purpose                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| `openai`            | GPT for dialogue, personas, disfluencies, backchannels, expression tags |
| `elevenlabs`        | Text-to-speech (optional if TTS is off)                                 |
| `python-dotenv`     | Load `.env`                                                             |
| `fastapi`           | HTTP API                                                                |
| `uvicorn[standard]` | ASGI server                                                             |
| `pydantic`          | Request/response models                                                 |


```bash
pip install -r requirements.txt
```

## Environment variables

Copy the example file and fill in keys:

```bash
cp .env.example .env
```


| Variable                   | Required | Description                                                       |
| -------------------------- | -------- | ----------------------------------------------------------------- |
| `OPENAI_API_KEY`           | **Yes**  | All LLM steps (personas, turns, disfluencies, backchannels, tags) |
| `ELEVENLABS_API_KEY`       | For TTS  | Synthesis of speaker units and listener backchannels              |
| `CONVERSATION_MAX_TURNS`   | No       | Max turns per session (default `12`)                              |
| `DISFLUENCY_RATE_PER_WORD` | No       | Disfluency count scale (default `0.14`)                           |
| `ELEVENLABS_MODEL`         | No       | Default `eleven_v3`                                               |
| `ELEVENLABS_OUTPUT_FORMAT` | No       | Default `mp3_44100_128`                                           |


Without `ELEVENLABS_API_KEY`, leave **TTS** unchecked in the UI; text and post-processing still run.

Personas live in `data/agent_personas.json` (names, voices, `initial_view`, `personal_story`). The UI can regenerate views/stories via **Generate personas from topic** (`POST /generate-personas-from-topic`).

## Run the server

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** — the app serves `frontend/index.html` and API routes on the same origin.

### Typical UI flow

1. Set discussion topic (or use default from prompts).
2. Generate personas from the topic.
3. Enable **TTS** if ElevenLabs is configured.
4. **Start** — first turn returns; with auto-advance, later turns are fetched via `/next_turn`.
5. Export HTML/audio or a clean transcript when finished.

---

## Pipeline overview

### Persona generation

**Step 1: Generate two contrasting initial views**


|              |                                                                                                                                                         |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location** | System: `backend/prompts.py` — `PERSONA_TWO_VIEWS_SYSTEM` User: `backend/prompts.py` — `PERSONA_TWO_VIEWS_USER` (via `persona_two_views_user_prompt()`) |
| **Function** | Produce `initial_view_a` and `initial_view_b` as JSON from the discussion topic and agent names.                                                        |
| **API**      | `POST /generate-personas-from-topic` (model: `PERSONA_AUTHORING_MODEL`, default `gpt-4o`)                                                               |


**Step 2: Generate two persona stories**


|                 |                                                                                                                                             |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location**    | System: `backend/prompts.py` — `PERSONA_STORY_SYSTEM` User: `backend/prompts.py` — `PERSONA_STORY_USER` (via `persona_story_user_prompt()`) |
| **Function**    | One call per agent: a `personal_story` that explains how they came to hold their `initial_view`.                                            |
| **Persistence** | Written to `data/agent_personas.json`                                                                                                       |


---

### Response generation (per turn)

Orchestrated in `backend/main.py` — `_segment_utterance_for_display()` (post-process) and `_finalize_turn_with_tts()` (TTS).

**Step 1: Prompt for response generation**


|              |                                                                                                                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location** | System: `backend/prompts.py` — `AGENT_SYSTEM_PROMPT_TEMPLATE` (via `agent_system_prompt()`) User: `backend/prompts.py` — `AGENT_USER_PROMPT_TEMPLATE` (via `agent_user_prompt()`) |
| **Function** | Generate a full reply for the current speaker from topic, personas, and transcript.                                                                                               |
| **Code**     | `_generate_turn_text()`                                                                                                                                                           |


**Step 2: Split the response into segments**


|              |                                                                                                                                   |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/main.py` — `_split_utterance_segments()` → `_split_utterance_segments_deterministic()`                                   |
| **Function** | Rule-based split after `. , ? !` (when followed by whitespace), after `...`, or on `-`. LLM splitter exists but is commented out. |


**Step 3: Sample number of disfluencies and types**


|              |                                                                                                                                                                                    |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Count**    | `backend/main.py` — `_sample_disfluency_count()`                                                                                                                                   |
| **Function** | `disfluency_count = int(word_count × DISFLUENCY_RATE_PER_WORD)` (default rate `0.14` from env).                                                                                    |
| **Types**    | `backend/main.py` — `_choose_disfluencies()`, `_pick_disfluency_type()`                                                                                                            |
| **Function** | For each disfluency slot, pick one type: • **88%** common tier: filled pause (45%), discourse marker (30%), elongation (25%) • **12%** rare tier: self-repair (70%), stumble (30%) |


**Step 4: Insert disfluencies**


|              |                                                                                                                   |
| ------------ | ----------------------------------------------------------------------------------------------------------------- |
| **Location** | System: `backend/prompts.py` — `DISFLUENCY_INSERT_SYSTEM` User: `backend/prompts.py` — `DISFLUENCY_INSERT_USER`   |
| **Function** | LLM (`DISFLUENCY_INSERT_MODEL`, default `gpt-4o-mini`) weaves requested types into segments → `segments_for_tts`. |
| **Code**     | `_choose_disfluencies()`                                                                                          |


**Group segments into TTS units** (between steps 4 and 5)


|              |                                                                                                                             |
| ------------ | --------------------------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/tts.py` — `group_segments_for_tts_units()`                                                                         |
| **Function** | Sew comma-clauses into units; sentence boundaries at `. ! ?`. Backchannels attach to **TTS units**, not raw micro-segments. |


**Step 5: Add backchannels**


|              |                                                                                                                                                                       |
| ------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/main.py` — `_choose_backchannels()` Prompts: `backend/prompts.py` — `BACKCHANNEL_INSERT_SYSTEM`, `BACKCHANNEL_INSERT_USER`                                   |
| **Function** | Sample `max_backchannels = random.randint(0, len(tts_units) // 2)`. LLM (`BACKCHANNEL_INSERT_MODEL`, default `gpt-4o`) picks unit indices and short listener phrases. |


**Step 6: Insert audio tags (ElevenLabs v3)**


|              |                                                                                                            |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/prompts.py` — `EXPRESSION_TAG_SYSTEM`, `EXPRESSION_TAG_USER` (via `expression_tag_user_prompt()`) |
| **Function** | Add `[audio tags]` to speaker TTS units and optionally to backchannel clips (`text_for_tts`).              |
| **Code**     | `_apply_expression_tags()`                                                                                 |


**Step 7: TTS**


|              |                                                                                                                                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/main.py` — `_finalize_turn_with_tts()` → `_start_turn_tts_background()` / `_synthesize_turn_audio()` `backend/tts.py` — ElevenLabs client                                                                              |
| **Function** | One clip per speaker TTS unit (speaker voice) plus one per backchannel (listener voice), synthesized in parallel. Browser polls `GET /turn_audio` and plays units sequentially; backchannels overlap the **next** speaker unit. |


---

### Turn handling for latency reduction

When TTS is enabled, `SessionPipeline` prefetches the next turn while you listen to the current one.


| Transition      | When the next turn starts                                                                                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Turn 1 → 2**  | Right after turn 1’s step 7 is **started** (`_finalize_turn_with_tts` returns); worker thread runs `_build_turn_committed(2)`.                                                              |
| **Turn 2 → 3+** | Right after the previous turn’s step 7 is **started** and that turn is committed to `transcript`; the worker submits the next `_build_turn_committed(N+1)` **before** waiting for delivery. |


Delivery to the browser still happens on `POST /next_turn` (auto-advance calls this after each turn’s audio). That handoff does **not** start generation; it only releases the prefetched payload.

So **turn 3 text generation can start while turn 1 audio is still playing**, as soon as turn 2’s full pipeline (steps 1–7 start) has finished — same pattern as turn 2 starting while turn 1 audio prepares.

Relevant code:

- `SessionPipeline.build_first_turn()` — turn 1 + `worker.start()`
- `SessionPipeline._worker_loop()` — prefetch, `_build_executor.submit()` for next turn
- `SessionPipeline.take_next_turn()` — deliver prefetched turn

---

## Project layout

```
audio-exp/
├── backend/
│   ├── main.py      # API, pipeline, post-process, TTS orchestration
│   ├── prompts.py   # All LLM prompts
│   └── tts.py       # Segment grouping, ElevenLabs
├── frontend/
│   └── index.html   # UI, playback, export
├── data/
│   └── agent_personas.json
├── requirements.txt
├── .env.example
└── README.md
```

