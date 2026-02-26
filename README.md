# dayops

Turn raw voice memos into a day plan, enrich with Strava timing, and write to Google Calendar.

## What it does

- Reads `.m4a` files from `VOICE_MEMOS_DIR`
- Transcribes memo audio
- Parses intent and generates a schedule
- Adds Strava run start/end as a calendar event
- Applies calendar updates (and keeps rollback snapshots)

## Setup

1. Copy `.env.example` to `.env` and fill all required values.
2. Install dependencies and CLI entrypoints:
   - `uv sync`
   - `uv pip install -e .`
3. Validate:
   - `dayops tui`
4. Run once:
   - `dayops run`

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

### Venice STT

Use:

- `STT_PROVIDER=venice`
- `VENICE_STT_MODEL=nvidia/parakeet-tdt-0.6b-v3`

If `STT_PROVIDER=gemini`, DayOps uses Gemini audio input through the Google OpenAI-compatible endpoint.

## CLI

- `dayops run`
- `dayops tui`
- `dayops plan generate --date YYYY-MM-DD`
- `dayops plan preview --date YYYY-MM-DD`
- `dayops plan apply --date YYYY-MM-DD`
- `dayops plan revise --from-audio /path/to/file.m4a`
- `dayops plan rollback --date YYYY-MM-DD`

## Backend API

- Start locally: `dayops-api`
- Health: `GET /healthz`
- Trigger processing: `POST /run`
- Plan routes:
  - `POST /plan/generate`
  - `POST /plan/preview`
  - `POST /plan/apply`
  - `POST /plan/revise`
  - `POST /plan/rollback`

If `BACKEND_API_KEY` is set in env, send `x-api-key: <value>` on all `POST` routes.

Example:

```bash
curl -X POST http://localhost:8000/run \
  -H "x-api-key: $BACKEND_API_KEY"
```

## Akash minimal deploy

1. Build and push container image (for example `ghcr.io/kiankyars/dayops:latest`).
2. Update image/env values in `akash/deploy.yaml`.
3. Deploy from Akash Console using that SDL.
