# Repository Guidelines

## Project Structure & Module Organization
This repository is a small Python backend service.

- `backend.py`: FastAPI app, routes, OAuth flow, API key auth, and request handling.
- `dayops_core.py`: Core calendar/LLM/domain logic (transcription, planning, snapshot restore, apply/rollback).
- `pyproject.toml`: Package metadata and runtime dependencies.
- `Dockerfile`: Container build and `uvicorn` runtime entrypoint.
- `deploy.yaml`: Deployment configuration.
- `README.md`, `uv.lock`: project docs and dependency lock.

There is currently no `tests/` directory or bundled frontend source in this repo.

## Build, Test, and Development Commands
- `pip install -e .`: install app dependencies in editable mode.
- `python backend.py`: run the API directly (uses `run_server` entry).
- `uvicorn backend:app --reload --host 0.0.0.0 --port 8000`: fast local dev loop.
- `dayops-api` (after install): run the app via package script from `pyproject.toml`.
- `docker build -t dayops-api .`: build container image.
- `docker run -p 8000:8000 --env-file .env dayops-api`: run container locally.

Use `.env` or shell exports for required vars (examples in README: `GEMINI_API_KEY`, `GOOGLE_OAUTH_CLIENT_ID`, etc.).

## Coding Style & Naming Conventions
- Python 3.10+; use type hints on new/changed functions.
- 4-space indentation, `snake_case` for functions/variables, `PascalCase` only for classes.
- Keep HTTP handler code thin in `backend.py`; place processing logic in `dayops_core.py`.
- Prefer explicit error messages and consistent API response shapes (`PlanResponse`).
- Formatting is closest to PEP 8; run your editor formatter before committing.

## Testing Guidelines
- No automated test framework is currently configured in the repo.
- For now, manually validate critical paths: `/healthz`, `/ingest`, `/revise`, `/rollback`.
- Add tests as `pytest` modules under `tests/` (e.g., `tests/test_core.py`) when touching parsing, planning, or rollback logic.

## Commit & Pull Request Guidelines
- Commit history is currently short and inconsistent (`button`, `remove main.py`, `update to latest image`), so this project does not enforce a strict format.
- Recommended convention for new work: `type(scope): summary` (e.g., `feat(ingest): validate date input`).
- PRs should include:
  - what changed and why,
  - env/config impact,
  - manual verification commands and sample request payloads,
  - screenshots for UI changes (`/` and `/app`) when applicable.

## Security & Configuration Tips
- Never commit `.env` or service credentials.
- Keep `DAYOPS_STORAGE_ROOT` outside repo during dev if persistence is important.
- Scope OAuth credentials to this app and verify redirect URI uses `/auth/google/callback`.
