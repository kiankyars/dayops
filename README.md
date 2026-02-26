# dayops

Turn raw voice memos into a perfectly orchestrated day—with *no typing*. Record your plans, errands, and goals on the go: dayops transcribes, schedules, syncs Strava run times, and pushes everything straight to Google Calendar.

## What it does

- Reads `.m4a` files from `VOICE_MEMOS_DIR`
- Transcribes memo audio (`STT_PROVIDER=gemini|venice`)
- Parses intent and generates a schedule (`MODEL_PROVIDER=google|venice`)
- Adds Strava run start/end as a calendar event
- Applies calendar updates (and keeps rollback snapshots)

## What it does not do

- No Obsidian writes
- No phone rollback endpoint

## Setup

1. Copy `.env.example` to `.env` and fill all required values.
2. Install deps: `uv sync`
3. Run once: `uv run dayops run`

## Provider config

### Google via OpenAI-compatible API

Use these values:

- `MODEL_PROVIDER=google`
- `MODEL_NAME=gemini-3-flash-preview` (or another Gemini model)
- `GEMINI_API_KEY=...`
- `GEMINI_OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`

### Venice inference

Use these values:

- `MODEL_PROVIDER=venice`
- `MODEL_NAME=<venice-chat-model>`
- `VENICE_INFERENCE_KEY=...`
- `VENICE_BASE_URL=https://api.venice.ai/api/v1`

### Venice STT (optional)

Use:

- `STT_PROVIDER=venice`
- `VENICE_STT_MODEL=nvidia/parakeet-tdt-0.6b-v3`

Alternative model:

- `VENICE_STT_MODEL=openai/whisper-large-v3`

If `STT_PROVIDER=gemini`, DayOps uses Gemini audio input through the Google OpenAI-compatible endpoint.

## CLI

- `dayops run`
- `dayops tui`
- `dayops plan generate --date YYYY-MM-DD`
- `dayops plan preview --date YYYY-MM-DD`
- `dayops plan apply --date YYYY-MM-DD`
- `dayops plan revise --from-audio /path/to/file.m4a`
- `dayops plan rollback --date YYYY-MM-DD`

