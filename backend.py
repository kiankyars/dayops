from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from typing import Any

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from dayops_core import (
    artifact_path,
    calendar_service,
    load_settings,
    load_state,
    process_file,
    rollback_day,
    save_state,
)

load_dotenv()

APP_STATE_DIR = Path(os.getenv("DAYOPS_STORAGE_ROOT", ".dayops_state")).expanduser()
USERS_CONFIG_PATH = APP_STATE_DIR / "users.json"
USERS_DATA_DIR = APP_STATE_DIR / "users"
SESSION_SECRET_PATH = APP_STATE_DIR / "session_secret"
TIMEZONE_OPTIONS = {
    "Pacific Time": "America/Los_Angeles",
    "Mountain Time": "America/Denver",
    "Central Time": "America/Chicago",
    "Eastern Time": "America/New_York",
    "UTC": "UTC",
}

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


def _oauth_entrypoint_url() -> str:
    return os.getenv("DAYOPS_OAUTH_ENTRYPOINT_URL", "").strip()


def _bootstrap_secret() -> str:
    return os.getenv("DAYOPS_OAUTH_BOOTSTRAP_SECRET", "").strip()


def _auth_ticket_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_session_secret(), salt="dayops-auth-ticket")


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
    default_tz = os.getenv("DEFAULT_USER_TIMEZONE", "America/Denver")
    if default_tz not in TIMEZONE_OPTIONS.values():
        default_tz = "America/Denver"
    env_map.setdefault("TIMEZONE", default_tz)

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


def _process_uploaded_audio(
    profile: dict[str, Any],
    upload: UploadFile,
    forced_type: str | None = None,
    date_override: str | None = None,
) -> tuple[dict[str, Any], dict[str, int] | None, Path]:
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
            artifact, diff = process_file(
                settings,
                state,
                temp_path,
                forced_type=forced_type,
                date_override=date_override,
            )
            save_state(settings, state)
            return artifact, diff, temp_path
        finally:
            temp_path.unlink(missing_ok=True)


class OAuthBootstrapRequest(BaseModel):
    userinfo: dict[str, Any]
    token_json: str


class PlanResponse(BaseModel):
    user_id: str
    date: str
    memo_type: str
    summary: str
    notes_markdown: str
    artifact_path: str
    applied: bool
    creates: int
    deletes: int
    locked: int


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def landing(request: Request) -> str:
    auth_href = _oauth_entrypoint_url() or "/auth/google/start"
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
        f"<a href='{escape(auth_href)}' style='display:inline-flex;align-items:center;gap:10px;"
        "padding:12px 16px;border:1px solid #d1d5db;border-radius:12px;background:white;"
        "text-decoration:none;color:#111827;font-weight:600;box-shadow:0 1px 3px rgba(0,0,0,.06);'>"
        "<span style='font-size:18px;'>G</span> Sign in with Google</a>"
        "</body></html>"
    )


@app.post("/auth/google/bootstrap")
def auth_google_bootstrap(
    payload: OAuthBootstrapRequest,
    x_bootstrap_secret: str | None = Header(default=None),
) -> dict[str, str]:
    expected_secret = _bootstrap_secret()
    if not expected_secret:
        raise HTTPException(status_code=500, detail="oauth_bootstrap_secret_missing")
    provided_secret = (x_bootstrap_secret or "").strip()
    if not provided_secret or not secrets.compare_digest(expected_secret, provided_secret):
        raise HTTPException(status_code=401, detail="invalid_bootstrap_secret")

    created = _upsert_user_from_oauth(payload.userinfo, payload.token_json)
    auth_token = _auth_ticket_serializer().dumps({"user_id": created["user_id"]})
    return {
        "user_id": created["user_id"],
        "email": created["email"],
        "api_key": created["api_key"],
        "auth_token": auth_token,
    }


@app.get("/auth/google/complete")
def auth_google_complete(token: str, request: Request) -> RedirectResponse:
    try:
        payload = _auth_ticket_serializer().loads(token, max_age=600)
    except SignatureExpired as exc:
        raise HTTPException(status_code=400, detail="auth_token_expired") from exc
    except BadSignature as exc:
        raise HTTPException(status_code=400, detail="auth_token_invalid") from exc

    user_id = str(payload.get("user_id", "")).strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="auth_token_invalid")
    request.session["user_id"] = user_id
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

    current_tz = tz if tz in TIMEZONE_OPTIONS.values() else "America/Denver"
    tz_options = "".join(
        f'<option value="{escape(value)}"' + (' selected' if value == current_tz else '') + f">{escape(label)}</option>"
        for label, value in TIMEZONE_OPTIONS.items()
    )
    timezone_input = f'<select name="timezone" style="width:100%;padding:10px;border:1px solid #d1d5db;border-radius:10px;margin-bottom:14px;">{tz_options}</select>'

    return f"""
<html>
  <body style="font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;background:#fafafa;color:#111827;">
    <h1 style="font-size:32px;margin:0 0 8px 0;">dayops</h1>
    <p style="margin:0 0 22px 0;color:#4b5563;">{email or profile['user_id']}</p>

    <div style="background:white;border:1px solid #e5e7eb;border-radius:14px;padding:16px 18px;margin-bottom:14px;">
      <div style="font-size:13px;color:#6b7280;margin-bottom:8px;">API Key</div>
      <div style="display:flex;align-items:center;gap:10px;background:#f3f4f6;border-radius:10px;padding:8px 10px;margin-bottom:8px;">
        <code id="api-key" data-full-key="{api_key}" style="display:block;flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:13px;background:transparent;">{api_key}</code>
        <button type="button" onclick="copyApiKey()" aria-label="Copy API Key" style="display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border:0;border-radius:8px;background:white;color:#111827;cursor:pointer;box-shadow:0 1px 2px rgba(0,0,0,.06);flex:0 0 auto;">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
        </button>
      </div>
      <span id="copy-badge" style="display:none;font-size:12px;color:#059669;font-weight:600;background:#ecfdf5;padding:4px 8px;border-radius:6px;border:1px solid #10b981;">Copied!</span>
      <form method="post" action="/app/rotate-key" style="margin-top:12px;">
        <button type="submit" style="padding:8px 12px;border:1px solid #111827;border-radius:10px;background:#111827;color:white;font-weight:600;cursor:pointer;">Rotate API Key</button>
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
    <script>
      function copyApiKey() {{
        const badge = document.getElementById('copy-badge');
        const value = document.getElementById('api-key').dataset.fullKey || '';
        const showCopied = () => {{
          badge.style.display = 'inline-block';
          setTimeout(() => {{ badge.style.display = 'none'; }}, 2000);
        }};

        if (navigator.clipboard && window.isSecureContext) {{
          navigator.clipboard.writeText(value).then(showCopied).catch(copyWithFallback);
          return;
        }}

        copyWithFallback();

        function copyWithFallback() {{
          const input = document.createElement('textarea');
          input.value = value;
          input.setAttribute('readonly', '');
          input.style.position = 'absolute';
          input.style.left = '-9999px';
          document.body.appendChild(input);
          input.select();
          document.execCommand('copy');
          document.body.removeChild(input);
          showCopied();
        }}
      }}
    </script>
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
    selected_timezone = timezone_value.strip()
    if selected_timezone not in TIMEZONE_OPTIONS.values():
        selected_timezone = "America/Denver"
    env_map["TIMEZONE"] = selected_timezone
    env_map["GOOGLE_CALENDAR_ID"] = calendar_id.strip() or "primary"
    current["env"] = env_map
    users[profile["user_id"]] = current
    _write_users_config(users)
    return RedirectResponse("/app", status_code=302)


@app.post("/revise", response_model=PlanResponse)
def plan_revise(
    file: UploadFile = File(...),
    date: str = Form(...),
    x_api_key: str | None = Header(default=None),
) -> PlanResponse:
    if not file.filename or not file.filename.lower().endswith(".m4a"):
        raise HTTPException(status_code=400, detail="file_must_be_m4a")
    date_value = date.strip()
    if not date_value:
        raise HTTPException(status_code=400, detail="date_required")
    profile = _require_api_profile(x_api_key)
    artifact, diff, _ = _process_uploaded_audio(
        profile,
        file,
        forced_type="revision",
        date_override=date_value,
    )
    with _env_overrides(profile["env"]):
        settings = load_settings()
        return PlanResponse(
            user_id=profile["user_id"],
            date=artifact["date"],
            memo_type=artifact["memo_type"],
            summary=artifact.get("summary", ""),
            notes_markdown=artifact.get("notes_markdown", ""),
            artifact_path=str(artifact_path(settings, artifact["date"])),
            applied=diff is not None,
            creates=diff.get("creates", 0) if diff else 0,
            deletes=diff.get("deletes", 0) if diff else 0,
            locked=diff.get("locked", 0) if diff else 0,
        )


def _plan_ingest(
    file: UploadFile = File(...),
    date: str = Form(...),
    x_api_key: str | None = Header(default=None),
) -> PlanResponse:
    if not file.filename or not file.filename.lower().endswith(".m4a"):
        raise HTTPException(status_code=400, detail="file_must_be_m4a")
    date_value = date.strip()
    if not date_value:
        raise HTTPException(status_code=400, detail="date_required")
    profile = _require_api_profile(x_api_key)
    artifact, diff, _ = _process_uploaded_audio(
        profile,
        file,
        forced_type="morning_plan",
        date_override=date_value,
    )
    with _env_overrides(profile["env"]):
        settings = load_settings()
        return PlanResponse(
            user_id=profile["user_id"],
            date=artifact["date"],
            memo_type=artifact["memo_type"],
            summary=artifact.get("summary", ""),
            notes_markdown=artifact.get("notes_markdown", ""),
            artifact_path=str(artifact_path(settings, artifact["date"])),
            applied=diff is not None,
            creates=diff.get("creates", 0) if diff else 0,
            deletes=diff.get("deletes", 0) if diff else 0,
            locked=diff.get("locked", 0) if diff else 0,
        )


@app.post("/plan", response_model=PlanResponse)
def plan(
    file: UploadFile = File(...),
    date: str = Form(...),
    x_api_key: str | None = Header(default=None),
) -> PlanResponse:
    return _plan_ingest(file=file, date=date, x_api_key=x_api_key)


@app.post("/ingest", response_model=PlanResponse)
def ingest(
    file: UploadFile = File(...),
    date: str = Form(...),
    x_api_key: str | None = Header(default=None),
) -> PlanResponse:
    return _plan_ingest(file=file, date=date, x_api_key=x_api_key)


@app.post("/rollback", response_model=PlanResponse)
def plan_rollback(
    date: str = Form(...),
    x_api_key: str | None = Header(default=None),
) -> PlanResponse:
    profile = _require_api_profile(x_api_key)
    with _env_overrides(profile["env"]):
        settings = load_settings()
        date_value = date.strip()
        if not date_value:
            raise HTTPException(status_code=400, detail="date_required")
        diff = rollback_day(settings, date_value)
        return PlanResponse(
            user_id=profile["user_id"],
            date=date_value,
            memo_type="rollback",
            summary=f"Rolled back {date_value}.",
            notes_markdown="",
            artifact_path="",
            applied=True,
            creates=diff.get("creates", 0) if diff else 0,
            deletes=diff.get("deletes", 0) if diff else 0,
            locked=diff.get("locked", 0) if diff else 0,
        )


def run_server() -> None:
    uvicorn.run("backend:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    run_server()
