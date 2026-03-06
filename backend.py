from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow
from pydantic import BaseModel, Field
from starlette.middleware.sessions import SessionMiddleware

from dayops_core import (
    apply_artifact,
    artifact_path,
    calendar_service,
    load_artifact,
    load_settings,
    load_state,
    preview_apply_diff,
    process_file,
    rollback_day,
    save_state,
)

load_dotenv()

APP_STATE_DIR = Path(os.getenv("DAYOPS_STORAGE_ROOT", ".dayops_state")).expanduser()
USERS_CONFIG_PATH = APP_STATE_DIR / "users.json"
USERS_DATA_DIR = APP_STATE_DIR / "users"
SESSION_SECRET_PATH = APP_STATE_DIR / "session_secret"
CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"
OAUTH_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    CALENDAR_SCOPE,
]

_ENV_LOCK = threading.Lock()
_USERS_LOCK = threading.Lock()


def _ensure_paths() -> None:
    APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
    USERS_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _session_secret() -> str:
    _ensure_paths()
    if SESSION_SECRET_PATH.exists():
        value = SESSION_SECRET_PATH.read_text().strip()
        if value:
            return value
    generated = secrets.token_urlsafe(48)
    SESSION_SECRET_PATH.write_text(generated)
    return generated


app = FastAPI(title="dayops-backend", version="0.6.0")
app.add_middleware(SessionMiddleware, secret_key=_session_secret())


def _sanitize_filename(name: str | None) -> str:
    base = (name or "").strip()
    if not base:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"memo_{stamp}.m4a"
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", Path(base).name)
    if not safe.lower().endswith(".m4a"):
        safe += ".m4a"
    return safe


def _read_users_config() -> dict[str, dict[str, Any]]:
    _ensure_paths()
    if not USERS_CONFIG_PATH.exists():
        return {}
    try:
        data = json.loads(USERS_CONFIG_PATH.read_text())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"users_config_invalid_json: {exc}") from exc

    if isinstance(data, dict) and "users" in data:
        data = data["users"]
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    raise HTTPException(status_code=500, detail="users_config_must_be_object")


def _write_users_config(users: dict[str, dict[str, Any]]) -> None:
    _ensure_paths()
    payload = {"users": users}
    with _USERS_LOCK:
        USERS_CONFIG_PATH.write_text(json.dumps(payload, indent=2))


def _new_api_key() -> str:
    return secrets.token_urlsafe(32)


def _require_api_profile(x_api_key: str | None) -> dict[str, Any]:
    key = (x_api_key or "").strip()
    if not key:
        raise HTTPException(status_code=401, detail="x_api_key_required")
    users = _read_users_config()
    for user_id, profile in users.items():
        expected = str(profile.get("api_key", "")).strip()
        if expected and secrets.compare_digest(expected, key):
            env_map = profile.get("env")
            if not isinstance(env_map, dict):
                raise HTTPException(status_code=400, detail="user_env_missing")
            return {
                "user_id": user_id,
                "api_key": expected,
                "env": {str(k): str(v) for k, v in env_map.items()},
                "email": str(profile.get("email", "")),
            }
    raise HTTPException(status_code=401, detail="invalid_api_key")


@contextmanager
def _env_overrides(overrides: dict[str, str]):
    with _ENV_LOCK:
        original: dict[str, str | None] = {k: os.environ.get(k) for k in overrides}
        try:
            for key, value in overrides.items():
                os.environ[key] = value
            yield
        finally:
            for key, old in original.items():
                if old is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = old


def _oauth_client_config() -> dict[str, Any]:
    client_id = os.getenv("GOOGLE_OAUTH_CLIENT_ID")
    client_secret = os.getenv("GOOGLE_OAUTH_CLIENT_SECRET")
    if client_id and client_secret:
        return {
            "web": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }
    raise HTTPException(status_code=500, detail="missing_google_oauth_client_env")


def _oauth_redirect_uri() -> str:
    uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI")
    if not uri:
        raise HTTPException(status_code=500, detail="missing_google_oauth_redirect_env")
    return uri


def _allow_local_insecure_oauth_transport(redirect_uri: str) -> None:
    if redirect_uri.startswith("http://localhost") or redirect_uri.startswith("http://127.0.0.1"):
        # Local dev only. Production OAuth callback must use HTTPS.
        os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"


def _fetch_google_userinfo(access_token: str) -> dict[str, Any]:
    req = urllib.request.Request(
        "https://www.googleapis.com/oauth2/v3/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:  # noqa: S310
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"google_userinfo_failed: {exc.code}") from exc


def _user_root(user_id: str) -> Path:
    _ensure_paths()
    return USERS_DATA_DIR / user_id


def _upsert_user_from_oauth(userinfo: dict[str, Any], token_json: str) -> dict[str, Any]:
    google_sub = str(userinfo.get("sub", "")).strip()
    email = str(userinfo.get("email", "")).strip().lower()
    if not google_sub:
        raise HTTPException(status_code=400, detail="google_sub_missing")

    users = _read_users_config()
    profile = users.get(google_sub, {})

    user_root = _user_root(google_sub)
    state_dir = user_root / "state"
    snapshots_dir = state_dir / "snapshots"
    token_path = user_root / "google_oauth_token.json"

    user_root.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    token_path.write_text(token_json)

    env_map = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    env_map = {str(k): str(v) for k, v in env_map.items()}

    env_map["DAYOPS_STATE_DIR"] = str(state_dir)
    env_map["DAYOPS_SNAPSHOT_DIR"] = str(snapshots_dir)
    env_map["GOOGLE_OAUTH_TOKEN_PATH"] = str(token_path)
    env_map.setdefault("GOOGLE_CALENDAR_ID", "primary")
    env_map.setdefault("TIMEZONE", os.getenv("DEFAULT_USER_TIMEZONE", "America/Los_Angeles"))

    api_key = str(profile.get("api_key", "")).strip() or _new_api_key()

    users[google_sub] = {
        "email": email,
        "api_key": api_key,
        "env": env_map,
    }
    _write_users_config(users)
    return {"user_id": google_sub, "email": email, "api_key": api_key, "env": env_map}


def _require_session_user(request: Request) -> dict[str, Any]:
    user_id = str(request.session.get("user_id", "")).strip()
    if not user_id:
        raise HTTPException(status_code=401, detail="not_logged_in")
    users = _read_users_config()
    profile = users.get(user_id)
    if not profile:
        raise HTTPException(status_code=401, detail="session_user_not_found")
    env_map = profile.get("env") if isinstance(profile.get("env"), dict) else {}
    return {
        "user_id": user_id,
        "email": str(profile.get("email", "")),
        "api_key": str(profile.get("api_key", "")),
        "env": {str(k): str(v) for k, v in env_map.items()},
    }


def _calendar_choices(profile: dict[str, Any]) -> list[tuple[str, str]]:
    with _env_overrides(profile["env"]):
        settings = load_settings()
        service = calendar_service(settings)
        items = service.calendarList().list(maxResults=100).execute().get("items", [])
    out: list[tuple[str, str]] = []
    for item in items:
        out.append((str(item.get("id", "")), str(item.get("summary", ""))))
    return out


def _process_uploaded_audio(profile: dict[str, Any], upload: UploadFile) -> tuple[dict[str, Any], dict[str, int] | None, Path]:
    with _env_overrides(profile["env"]):
        settings = load_settings()
        state = load_state(settings)

        filename = _sanitize_filename(upload.filename)
        suffix = Path(filename).suffix or ".m4a"
        inbox = settings.state_dir / "incoming"
        inbox.mkdir(parents=True, exist_ok=True)

        with tempfile.NamedTemporaryFile(prefix="memo_", suffix=suffix, dir=inbox, delete=False) as handle:
            data = upload.file.read()
            if not data:
                raise HTTPException(status_code=400, detail="empty_file")
            handle.write(data)
            temp_path = Path(handle.name)

        try:
            artifact, diff = process_file(settings, state, temp_path, forced_type=None, apply_override=None)
            save_state(settings, state)
            return artifact, diff, temp_path
        finally:
            temp_path.unlink(missing_ok=True)


class RunResponse(BaseModel):
    processed: int
    files: list[str]


class IngestResponse(BaseModel):
    user_id: str
    stored_file: str
    date: str
    memo_type: str
    artifact_path: str
    applied: bool
    diff: dict[str, int] | None


class GenerateRequest(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")
    from_audio: str | None = None


class DateRequest(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD")


class ReviseRequest(BaseModel):
    from_audio: str
    apply: bool = True


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def landing(request: Request) -> str:
    if request.session.get("user_id"):
        return (
            "<html><body style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;"
            "max-width:640px;margin:80px auto;padding:0 20px;'>"
            "<h1 style='font-size:34px;margin-bottom:8px;'>dayops</h1>"
            "<p style='color:#4b5563;margin-bottom:24px;'>You're signed in.</p>"
            "<a href='/app' style='display:inline-block;padding:10px 14px;border:1px solid #111827;"
            "border-radius:10px;text-decoration:none;color:#111827;font-weight:600;'>Open Dashboard</a>"
            "</body></html>"
        )
    return (
        "<html><body style='font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;"
        "max-width:640px;margin:80px auto;padding:0 20px;background:#fafafa;'>"
        "<h1 style='font-size:38px;margin:0 0 10px 0;'>dayops</h1>"
        "<p style='color:#4b5563;margin:0 0 26px 0;'>AI day planning with Google Calendar.</p>"
        "<a href='/auth/google/start' style='display:inline-flex;align-items:center;gap:10px;"
        "padding:12px 16px;border:1px solid #d1d5db;border-radius:12px;background:white;"
        "text-decoration:none;color:#111827;font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.06);'>"
        "<span style='font-size:18px;'>G</span> Sign in with Google</a>"
        "</body></html>"
    )


@app.get("/auth/google/start")
def auth_google_start(request: Request) -> RedirectResponse:
    flow = Flow.from_client_config(_oauth_client_config(), scopes=OAUTH_SCOPES)
    redirect_uri = _oauth_redirect_uri()
    _allow_local_insecure_oauth_transport(redirect_uri)
    flow.redirect_uri = redirect_uri
    auth_url, state = flow.authorization_url(access_type="offline", include_granted_scopes="true", prompt="consent")
    request.session["oauth_state"] = state
    request.session["oauth_code_verifier"] = flow.code_verifier
    return RedirectResponse(auth_url, status_code=302)


@app.get("/auth/google/callback")
def auth_google_callback(request: Request) -> RedirectResponse:
    state = str(request.session.get("oauth_state", ""))
    code_verifier = str(request.session.get("oauth_code_verifier", ""))
    if not state:
        raise HTTPException(status_code=400, detail="oauth_state_missing")
    if not code_verifier:
        raise HTTPException(status_code=400, detail="oauth_code_verifier_missing")

    flow = Flow.from_client_config(_oauth_client_config(), scopes=OAUTH_SCOPES, state=state)
    redirect_uri = _oauth_redirect_uri()
    _allow_local_insecure_oauth_transport(redirect_uri)
    flow.redirect_uri = redirect_uri
    flow.code_verifier = code_verifier
    flow.fetch_token(authorization_response=str(request.url))

    userinfo = _fetch_google_userinfo(flow.credentials.token)
    created = _upsert_user_from_oauth(userinfo, flow.credentials.to_json())
    request.session["user_id"] = created["user_id"]
    request.session.pop("oauth_state", None)
    request.session.pop("oauth_code_verifier", None)
    return RedirectResponse("/app", status_code=302)


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    request.session.clear()
    return RedirectResponse("/", status_code=302)


@app.get("/app", response_class=HTMLResponse)
def dashboard(request: Request) -> str:
    profile = _require_session_user(request)
    tz = escape(profile["env"].get("TIMEZONE", ""))
    cal_id = escape(profile["env"].get("GOOGLE_CALENDAR_ID", "primary"))
    api_key = escape(profile["api_key"])
    email = escape(profile.get("email", ""))

    calendar_options = ""
    try:
        for cid, name in _calendar_choices(profile):
            selected = " selected" if cid == profile["env"].get("GOOGLE_CALENDAR_ID", "") else ""
            calendar_options += f'<option value="{escape(cid)}"{selected}>{escape(name)} ({escape(cid)})</option>'
    except Exception:
        calendar_options = ""

    calendar_input = (
        f'<select name="google_calendar_id">{calendar_options}</select>'
        if calendar_options
        else f'<input type="text" name="google_calendar_id" value="{cal_id}" />'
    )

    common_timezones = [
        "America/Los_Angeles",
        "America/Denver",
        "America/Chicago",
        "America/New_York",
        "Europe/London",
        "Europe/Paris",
        "Asia/Dubai",
        "Asia/Kolkata",
        "Asia/Singapore",
        "Asia/Tokyo",
        "Australia/Sydney",
        "UTC",
    ]
    if tz and tz not in common_timezones:
        common_timezones.insert(0, tz)
    tz_options = "".join(
        f'<option value="{escape(name)}"' + (' selected' if name == tz else '') + f">{escape(name)}</option>"
        for name in common_timezones
    )
    timezone_input = f'<select name="timezone" style="width:100%;padding:10px;border:1px solid #d1d5db;border-radius:10px;margin-bottom:14px;">{tz_options}</select>'

    return f"""
<html>
  <body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;background:#fafafa;color:#111827;">
    <h1 style="font-size:32px;margin:0 0 8px 0;">dayops</h1>
    <p style="margin:0 0 22px 0;color:#4b5563;">{email or profile['user_id']}</p>

    <div style="background:white;border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;margin-bottom:14px;">
      <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">API Key</div>
      <code style="display:block;word-break:break-all;font-size:13px;">{api_key}</code>
      <form method="post" action="/app/rotate-key" style="margin-top:12px;">
        <button type="submit" style="padding:8px 12px;border:1px solid #111827;border-radius:10px;background:#111827;color:white;font-weight:600;">Rotate API Key</button>
      </form>
    </div>

    <div style="background:white;border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;">
      <form method="post" action="/app/config">
        <label style="display:block;font-size:13px;color:#6b7280;margin-bottom:6px;">Timezone</label>
        {timezone_input}
        <label style="display:block;font-size:13px;color:#6b7280;margin-bottom:6px;">Calendar</label>
        {calendar_input}
        <div style="margin-top:14px;">
          <button type="submit" style="padding:9px 13px;border:1px solid #111827;border-radius:10px;background:#111827;color:white;font-weight:600;">Save</button>
        </div>
      </form>
    </div>
    <p style="margin-top:16px;"><a href="/logout" style="color:#4b5563;">Logout</a></p>
  </body>
</html>
"""


@app.post("/app/rotate-key")
def rotate_api_key(request: Request) -> RedirectResponse:
    profile = _require_session_user(request)
    users = _read_users_config()
    current = users[profile["user_id"]]
    current["api_key"] = _new_api_key()
    users[profile["user_id"]] = current
    _write_users_config(users)
    return RedirectResponse("/app", status_code=302)


@app.post("/app/config")
def update_config(
    request: Request,
    timezone_value: str = Form(..., alias="timezone"),
    calendar_id: str = Form(..., alias="google_calendar_id"),
) -> RedirectResponse:
    profile = _require_session_user(request)
    users = _read_users_config()
    current = users[profile["user_id"]]
    env_map = current.get("env") if isinstance(current.get("env"), dict) else {}
    env_map = {str(k): str(v) for k, v in env_map.items()}
    env_map["TIMEZONE"] = timezone_value.strip() or env_map.get("TIMEZONE", "America/Los_Angeles")
    env_map["GOOGLE_CALENDAR_ID"] = calendar_id.strip() or "primary"
    current["env"] = env_map
    users[profile["user_id"]] = current
    _write_users_config(users)
    return RedirectResponse("/app", status_code=302)


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".m4a"):
        raise HTTPException(status_code=400, detail="file_must_be_m4a")

    profile = _require_api_profile(x_api_key)
    artifact, diff, temp_path = _process_uploaded_audio(profile, file)
    with _env_overrides(profile["env"]):
        settings = load_settings()
        return IngestResponse(
            user_id=profile["user_id"],
            stored_file=str(temp_path),
            date=artifact["date"],
            memo_type=artifact["memo_type"],
            artifact_path=str(artifact_path(settings, artifact["date"])),
            applied=diff is not None,
            diff=diff,
        )


@app.post("/run", response_model=RunResponse)
def run_all() -> RunResponse:
    return RunResponse(processed=0, files=[])


@app.post("/plan/generate")
def plan_generate(payload: GenerateRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    profile = _require_api_profile(x_api_key)
    if not payload.from_audio:
        raise HTTPException(status_code=400, detail="from_audio_required")

    with _env_overrides(profile["env"]):
        settings = load_settings()
        state = load_state(settings)
        audio = Path(payload.from_audio).expanduser()
        if not audio.exists():
            raise HTTPException(status_code=404, detail=f"audio_not_found: {audio}")
        artifact, _ = process_file(settings, state, audio, forced_type=None, apply_override=False)
        save_state(settings, state)
        return {"date": artifact["date"], "artifact_path": str(artifact_path(settings, artifact["date"]))}


@app.post("/plan/preview")
def plan_preview(payload: DateRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    profile = _require_api_profile(x_api_key)
    with _env_overrides(profile["env"]):
        settings = load_settings()
        artifact = load_artifact(settings, payload.date)
        future_only = artifact["memo_type"] == "revision"
        diff = preview_apply_diff(settings, artifact, future_only=future_only)
        return {"date": payload.date, "future_only": future_only, "diff": diff}


@app.post("/plan/apply")
def plan_apply(payload: DateRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    profile = _require_api_profile(x_api_key)
    with _env_overrides(profile["env"]):
        settings = load_settings()
        artifact = load_artifact(settings, payload.date)
        future_only = artifact["memo_type"] == "revision"
        diff = apply_artifact(settings, artifact, future_only=future_only)
        return {"date": payload.date, "future_only": future_only, "diff": diff}


@app.post("/plan/revise")
def plan_revise(payload: ReviseRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    profile = _require_api_profile(x_api_key)
    with _env_overrides(profile["env"]):
        settings = load_settings()
        state = load_state(settings)
        audio = Path(payload.from_audio).expanduser()
        if not audio.exists():
            raise HTTPException(status_code=404, detail=f"audio_not_found: {audio}")
        artifact, diff = process_file(settings, state, audio, forced_type="revision", apply_override=payload.apply)
        save_state(settings, state)
        return {
            "date": artifact["date"],
            "artifact_path": str(artifact_path(settings, artifact["date"])),
            "diff": diff,
        }


@app.post("/plan/rollback")
def plan_rollback(payload: DateRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
    profile = _require_api_profile(x_api_key)
    with _env_overrides(profile["env"]):
        settings = load_settings()
        diff = rollback_day(settings, payload.date)
        return {"date": payload.date, "diff": diff}


def run_server() -> None:
    uvicorn.run("backend:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    run_server()
