# dayops

Minimal backend for voice-memo planning into Google Calendar.

## Flow

1. User signs in with Google and grants Calendar scope.
2. Backend stores that user's refresh token under `.dayops_state/users/<google-sub>/`.
3. Backend generates a unique API key for that user.
4. Client uploads `.m4a` to `/ingest` with `x-api-key`.
5. Day plan is generated and applied to that user's calendar.

## Web UI

- `GET /` sign-in landing
- `GET /app` dashboard
  - shows API key
  - has **Rotate API Key** button
  - timezone + calendar picker (fixed timezone list)

## API

- `POST /ingest` (`x-api-key`, multipart `file=@memo.m4a`)
- `POST /plan/revise`
- `POST /plan/rollback`

## GCP setup

Create one OAuth client of type **Web application**.

Required redirect URI must point to your backend callback, not GitHub Pages:

- local: `http://localhost:8000/auth/google/callback`
- prod: `https://<your-backend-domain>/auth/google/callback`

If you host static UI at `https://kiankyars.github.io/dayops`, that is fine, but OAuth callback still goes to backend.

Enable Google Calendar API in the same GCP project.

If you see `missing_google_oauth_client_env`:

Set `GOOGLE_OAUTH_CLIENT_ID` and `GOOGLE_OAUTH_CLIENT_SECRET` in `.env`.

## Env model

Global/server env vars (same for all users):

- `GEMINI_API_KEY`
- `GEMINI_MODELS` (comma-separated, tried left-to-right on rate limit)
- `DAYOPS_STORAGE_ROOT` (optional, default `.dayops_state`)
- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `GOOGLE_OAUTH_REDIRECT_URI`
- `DEFAULT_USER_TIMEZONE`

User-specific values are stored in `.dayops_state/users.json` and managed by the app:

- `api_key`
- `GOOGLE_OAUTH_TOKEN_PATH`
- `GOOGLE_CALENDAR_ID`
- `TIMEZONE`
- user state/snapshot dirs

## Notes

- `plan.md` is kept in repo but no longer used by runtime.
- Voice memo directory persistence was removed; uploads are processed from a temp file and deleted.
- Session secret is auto-generated once and persisted at `.dayops_state/session_secret`.
