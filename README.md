# transcribe / coo

Local COO scheduler that ingests voice memos, generates a day plan with Gemini, enriches it with Strava run timing, writes Google Calendar events, and mirrors the plan into Obsidian.

## Setup

1. Create `.env` from `.env.example`.
2. Install dependencies.
3. Run `coo run`.

## Commands

- `coo run`
- `coo tui`
- `coo plan generate --date YYYY-MM-DD`
- `coo plan preview --date YYYY-MM-DD`
- `coo plan apply --date YYYY-MM-DD`
- `coo plan revise --from-audio /path/to/file.m4a`
- `coo plan rollback --date YYYY-MM-DD`

## Behavior

- `AUTO_APPLY=true` means calendar updates are published automatically.
- Rollback is computer-only via CLI/TUI (`coo plan rollback ...`).
- All filesystem paths and API keys are env-configured; no inline absolute paths in app logic.
