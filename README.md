# dayops

Turn raw voice memos into day plans and write to Google Calendar.

## What it does

- Ingests `.m4a` voice memos
- Transcribes and parses planning intent
- Generates structured daily schedules
- Applies plans to Google Calendar with rollback snapshots

## Setup

1. Copy `.env.example` to `.env` and fill required values.
2. Install dependencies and CLI entrypoints:
   - `uv sync`
   - `uv pip install -e .`
3. Validate:
   - `dayops tui`
4. Run once:
   - `dayops run`

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
- Upload + process now: `POST /ingest`
- Process pending single-user queue: `POST /run`
- Process pending queue for one multi-user profile: `POST /users/{user_id}/run`
- Plan routes:
  - `POST /plan/generate`
  - `POST /plan/preview`
  - `POST /plan/apply`
  - `POST /plan/revise`
  - `POST /plan/rollback`

### Single-user auth

If `BACKEND_API_KEY` is set, send `x-api-key` on protected routes.

```bash
curl -X POST http://localhost:8000/run \
  -H "x-api-key: $BACKEND_API_KEY"
```

### Multi-user auth and secret isolation

Set `USERS_CONFIG_PATH` to a server-side JSON file. Each user has:

- a dedicated `api_key`
- their own env block (`VOICE_MEMOS_DIR`, `DAYOPS_STATE_DIR`, Google OAuth path, model keys, etc.)

`/ingest` accepts `x-user-id` + `x-api-key` and processes using only that user's env.

```bash
curl -X POST http://localhost:8000/ingest \
  -H "x-user-id: mom" \
  -H "x-api-key: <mom-api-key>" \
  -F "file=@/path/to/memo.m4a"
```

Example users config (`USERS_CONFIG_PATH`):

```json
{
  "users": {
    "mom": {
      "api_key": "replace-me",
      "env": {
        "VOICE_MEMOS_DIR": "/data/users/mom/voice",
        "PLAN_TEMPLATE_PATH": "/app/plan.md",
        "DAYOPS_STATE_DIR": "/data/users/mom/state",
        "DAYOPS_SNAPSHOT_DIR": "/data/users/mom/state/snapshots",
        "MODEL_PROVIDER": "google",
        "MODEL_NAME": "gemini-3-flash-preview",
        "STT_PROVIDER": "gemini",
        "GEMINI_STT_MODEL": "gemini-3-flash-preview",
        "VENICE_STT_MODEL": "nvidia/parakeet-tdt-0.6b-v3",
        "GOOGLE_CALENDAR_ID": "primary",
        "GOOGLE_OAUTH_TOKEN_PATH": "/data/users/mom/google_oauth_token.json",
        "GEMINI_API_KEY": "replace-me",
        "GEMINI_OPENAI_BASE_URL": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "VENICE_INFERENCE_KEY": "",
        "VENICE_BASE_URL": "https://api.venice.ai/api/v1",
        "TIMEZONE": "America/Los_Angeles",
        "AUTO_APPLY": "true",
        "TRASH_PROCESSED_MEMOS": "false",
        "HYDRATE_MAX_RETRIES": "6",
        "HYDRATE_RETRY_SECONDS": "1.5"
      }
    }
  }
}
```

## Akash deploy

1. Build and push container image (for example `ghcr.io/kiankyars/dayops:latest`).
2. Update image/env values in `akash/deploy.yaml`.
3. Mount persistent storage for `/data`.
4. Store `users.json` on that volume and set `USERS_CONFIG_PATH=/data/users.json`.
5. Deploy from Akash Console.
