# Audio-exp: Two-agent dialogue with TTS

A web app that has two LLM personas discuss a topic out loud, and plays the same conversation side by side in two conditions:

- **Off-the-shelf agent** (left): disfluency off, conversation-style instructions off, no added pauses.
- **Imperfect agent** (right): disfluency on, conversation-style instructions (`AGENT_SYSTEM_PROMPT_CONVERSATION_INSTRUCTIONS`) on, random pauses within and between turns.

Both sides share the same topic and the same generated stances and personal stories. Dialogue is written by OpenAI and spoken by ElevenLabs.

## Requirements

- **Python** 3.10 or newer
- An **OpenAI API key** and an **ElevenLabs API key** (each visitor uses their own; see [API keys](#api-keys))

| Package             | Purpose                                                   |
| ------------------- | --------------------------------------------------------- |
| `openai`            | Personas, dialogue turns, disfluency insertion            |
| `elevenlabs`        | Text-to-speech                                            |
| `fastapi`           | HTTP API                                                  |
| `uvicorn[standard]` | ASGI server                                               |
| `pydantic`          | Request/response models                                   |
| `httpx`             | ElevenLabs key check                                      |
| `python-dotenv`     | Load optional settings from `.env`                        |

## Run locally

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**. The server serves `frontend/index.html` and the API on the same origin.

## API keys

Keys are entered on the first page of the app, not configured on the server.

- The browser sends them with each request in the `X-OpenAI-Key` and `X-ElevenLabs-Key` headers.
- The server uses them only for that visitor's requests and never writes them to disk. The ElevenLabs key is held in memory with the conversation session (needed by background TTS) and dropped when the session expires after 1 hour.
- In the browser, keys live in `sessionStorage` (cleared when the tab closes), or in `localStorage` if the visitor ticks **Remember on this browser**.
- **Check keys and continue** calls `POST /validate-keys` before moving on.

### Using your own server keys (local only)

To skip typing keys on your own machine, put them in `.env` and turn on the fallback:

```bash
cp .env.example .env              # then fill in OPENAI_API_KEY and ELEVENLABS_API_KEY
ALLOW_SERVER_KEYS=1 python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The keys page then shows a **Use server keys** button. **Never set `ALLOW_SERVER_KEYS` on a public deployment**, or every visitor will spend your quota. `.env` is git-ignored; do not commit it.

## Environment variables

All optional.

| Variable                   | Default          | Description                                                   |
| -------------------------- | ---------------- | ------------------------------------------------------------- |
| `ALLOW_SERVER_KEYS`        | off              | `1` lets requests without key headers fall back to `.env` keys |
| `OPENAI_API_KEY`           | —                | Used only when `ALLOW_SERVER_KEYS=1`                          |
| `ELEVENLABS_API_KEY`       | —                | Used only when `ALLOW_SERVER_KEYS=1`                          |
| `DISFLUENCY_RATE_PER_WORD` | `0.09`           | Disfluency count scale                                        |
| `ELEVENLABS_MODEL`         | `eleven_v3`      | TTS model                                                     |
| `ELEVENLABS_OUTPUT_FORMAT` | `mp3_44100_128`  | TTS output format                                             |

Experiment switches (conversation instructions, disfluency, backchannels, audio tags, max turns, debug timings) are Python constants in `backend/exp_control.py`. The comparison view overrides conversation instructions and disfluency per side; everything else follows that file.

## UI flow

1. **API keys** — enter and check both keys.
2. **Topic** — type a discussion topic (required) and click **Generate agent profiles**.
3. **Agent profiles** — review each agent's stance and personal story, set the number of turns (default 6), then **Continue to comparison**.
4. **Side by side** — click **Start** on either column. The other column's Start is disabled until this one finishes. Each turn's text appears, its audio plays live, and the next turn appears once that audio ends. After a turn plays, a **Replay** player appears under its text; **Play all turns** replays the whole column. **Stop** ends the run; **Start** again begins from turn 1.

On the imperfect side, the sampled pause lengths are labeled on the transcript, and replay uses the same pauses heard live.

---

## Pipeline overview

### Persona generation

`POST /generate-personas-from-topic` (model `PERSONA_AUTHORING_MODEL`, default `gpt-4o`).

**Step 1: Two contrasting stances**

|              |                                                                                                                              |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/prompts.py` — `PERSONA_TWO_VIEWS_SYSTEM`, `persona_two_views_user_prompt()`                                         |
| **Function** | Produce `initial_view_a` and `initial_view_b` as JSON from the topic and agent names.                                        |

**Step 2: Two personal stories**

|                 |                                                                                                   |
| --------------- | ------------------------------------------------------------------------------------------------- |
| **Location**    | `backend/prompts.py` — `PERSONA_STORY_SYSTEM`, `persona_story_user_prompt()`                      |
| **Function**    | One call per agent (run in parallel): a `personal_story` explaining how they came to their stance. |
| **Persistence** | Written to `data/agent_personas.json`                                                             |

---

### Response generation (per turn)

Orchestrated in `backend/main.py` — `_segment_utterance_for_display()` (post-process) and `_finalize_turn_with_tts()` (TTS).

**Step 1: Generate the reply**

|              |                                                                                                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/prompts.py` — `agent_system_prompt()`, `agent_user_prompt()`                                                                                            |
| **Function** | Full reply for the current speaker from topic, personas, and transcript. `AGENT_SYSTEM_PROMPT_CONVERSATION_INSTRUCTIONS` is appended only on the imperfect side. |
| **Code**     | `_generate_turn_text()`                                                                                                                                          |

**Step 2: Split into segments**

|              |                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------- |
| **Location** | `backend/main.py` — `_split_utterance_segments()` → `_split_utterance_segments_deterministic()`   |
| **Function** | Rule-based split after `. , ? !` (when followed by whitespace), after `...`, or on `-`.           |

**Step 3: Sample disfluency count and types** (imperfect side only)

|              |                                                                                                                                  |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| **Count**    | `_sample_disfluency_count()` — `int(word_count × DISFLUENCY_RATE_PER_WORD)`                                                      |
| **Types**    | `_pick_disfluency_type()` using `DISFLUENCY_TYPE_WEIGHTS`: filled pause 0.31, prolongation 0.28, discourse marker 0.24, repetition 0.16, self-repair 0 |

**Step 4: Insert disfluencies** (imperfect side only)

|              |                                                                                                   |
| ------------ | ------------------------------------------------------------------------------------------------- |
| **Location** | `backend/prompts.py` — `DISFLUENCY_INSERT_SYSTEM`, `DISFLUENCY_INSERT_USER`                       |
| **Function** | LLM (`DISFLUENCY_INSERT_MODEL`, default `gpt-4o`) weaves the requested types into segments.       |
| **Code**     | `_choose_disfluencies()`                                                                          |

**Group segments into TTS units**

|              |                                                                                          |
| ------------ | ---------------------------------------------------------------------------------------- |
| **Location** | `backend/tts.py` — `group_segments_for_tts_units()`                                      |
| **Function** | Sew comma-clauses into units; sentence boundaries at `. ! ?`.                            |

**Step 5: Backchannels** — off unless `backchannel_speech = True` in `exp_control.py` (`_choose_backchannels()`).

**Step 6: ElevenLabs audio tags** — off unless `audio_tag_speech = True` in `exp_control.py` (`_apply_expression_tags()`).

**Step 7: TTS**

|              |                                                                                                                                                                      |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Location** | `backend/main.py` — `_start_turn_tts_background()` / `_synthesize_turn_audio()`; `backend/tts.py` — ElevenLabs client                                                |
| **Function** | One clip per TTS unit, synthesized in parallel. The browser polls `GET /turn_audio` and plays units in order; on the imperfect side it inserts a sampled pause between units. |

Pause distributions (imperfect side, `frontend/index.html`):

- Within a turn, between units: N(580 ms, 200 ms), clamped to [180, 980] ms.
- Between turns: N(200 ms, 50 ms), clamped to [100, 300] ms.

---

### Turn handling for latency

`SessionPipeline` prefetches upcoming turns while the current one plays.

- As soon as a turn's TTS is **started**, the worker begins building the next turn's text.
- `POST /next_turn` only hands over the already-built turn; it does not start generation.
- The browser asks for the next turn only after the current turn's audio finishes, so text is revealed in step with the audio.

OpenAI calls use short timeouts (15 s per turn, 20 s for personas) with retries, so a stalled connection is retried quickly instead of hanging.

Relevant code: `SessionPipeline.build_first_turn()`, `SessionPipeline._worker_loop()`, `SessionPipeline.take_next_turn()`.

---

## API

| Route                                | Purpose                                              |
| ------------------------------------ | ---------------------------------------------------- |
| `GET /key-config`                    | Whether server-key fallback is available             |
| `POST /validate-keys`                | Check the OpenAI and ElevenLabs keys in the headers  |
| `GET /topics`                        | Example topic                                        |
| `POST /generate-personas-from-topic` | Generate stances and personal stories                |
| `GET /personas`                      | Current personas                                     |
| `POST /start`                        | Start a session (`mode`: `baseline` or `exp`)        |
| `POST /next_turn`                    | Deliver the next prefetched turn                     |
| `GET /turn_audio`                    | Poll a turn's audio clips                            |
| `POST /cancel`                       | Stop a session's prefetching                         |

---

## Deploying publicly

- Serve over **HTTPS**; keys travel in request headers.
- Run without `--reload` and without `ALLOW_SERVER_KEYS`.
- `data/agent_personas.json` is shared by all visitors, so simultaneous users overwrite each other's generated profiles.
- There is no rate limiting.

## Project layout

```
audio-exp-nc/
├── backend/
│   ├── main.py         # API, key handling, pipeline, post-process, TTS orchestration
│   ├── prompts.py      # All LLM prompts
│   ├── tts.py          # Segment grouping, ElevenLabs
│   └── exp_control.py  # Experiment switches
├── frontend/
│   └── index.html      # UI and playback
├── data/
│   └── agent_personas.json
├── requirements.txt
├── .env.example
└── README.md
```
