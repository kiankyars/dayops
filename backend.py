from __future__ import annotations

import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from dayops_core import (
    apply_artifact,
    artifact_path,
    is_processed,
    latest_audio_for_date,
    list_audio_files,
    load_artifact,
    load_settings,
    load_state,
    preview_apply_diff,
    process_file,
    rollback_day,
    save_state,
)

app = FastAPI(title="dayops-backend", version="0.2.0")
_ENV_LOCK = threading.Lock()


def _sanitize_filename(name: str | None) -> str:
    base = name or ""
    base = base.strip()
    if not base:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"memo_{stamp}.m4a"
    base = Path(base).name
    safe = re.sub(r"[^A-Za-z0-9._ -]", "_", base)
    if not safe.lower().endswith(".m4a"):
        safe += ".m4a"
    return safe


def _read_users_config() -> dict[str, dict[str, Any]]:
    path = os.getenv("USERS_CONFIG_PATH")
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise HTTPException(status_code=500, detail=f"users_config_not_found: {p}")
    try:
        data = __import__("json").loads(p.read_text())
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=f"users_config_invalid_json: {exc}") from exc

    if isinstance(data, dict) and "users" in data:
        data = data["users"]
    if isinstance(data, list):
        out: dict[str, dict[str, Any]] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            uid = str(item.get("user_id", "")).strip()
            if uid:
                out[uid] = item
        return out
    if isinstance(data, dict):
        return {str(k): v for k, v in data.items() if isinstance(v, dict)}
    return {}


def _single_user_api_key_ok(x_api_key: str | None) -> bool:
    expected = os.getenv("BACKEND_API_KEY")
    if not expected:
        return True
    return x_api_key == expected


def _resolve_user_profile(user_id: str, x_api_key: str | None) -> dict[str, Any]:
    users = _read_users_config()
    if not users:
        raise HTTPException(status_code=404, detail="users_config_empty")
    profile = users.get(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="user_not_found")
    expected = str(profile.get("api_key", "")).strip()
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid_api_key")
    env_map = profile.get("env")
    if not isinstance(env_map, dict):
        raise HTTPException(status_code=400, detail="user_env_missing")
    return {"user_id": user_id, "env": {str(k): str(v) for k, v in env_map.items()}}


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


def _save_upload(upload: UploadFile, target_dir: Path) -> Path:
    filename = _sanitize_filename(upload.filename)
    target_dir.mkdir(parents=True, exist_ok=True)
    candidate = target_dir / filename
    if candidate.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        candidate = target_dir / f"{candidate.stem}_{stamp}{candidate.suffix}"
    data = upload.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty_file")
    candidate.write_bytes(data)
    return candidate


def _process_single_file(file_path: Path) -> tuple[dict[str, Any], dict[str, int] | None]:
    settings = load_settings()
    state = load_state(settings)
    artifact, diff = process_file(settings, state, file_path, forced_type=None, apply_override=None)
    save_state(settings, state)
    return artifact, diff


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    if not _single_user_api_key_ok(x_api_key):
        raise HTTPException(status_code=401, detail="invalid_api_key")


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


@app.post("/ingest", response_model=IngestResponse)
def ingest(
    file: UploadFile = File(...),
    user_id_form: str | None = Form(default=None, alias="user_id"),
    x_user_id: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> IngestResponse:
    if not file.filename or not file.filename.lower().endswith(".m4a"):
        raise HTTPException(status_code=400, detail="file_must_be_m4a")

    resolved_user = (user_id_form or x_user_id or "").strip()
    if resolved_user:
        profile = _resolve_user_profile(resolved_user, x_api_key)
        with _env_overrides(profile["env"]):
            settings = load_settings()
            stored = _save_upload(file, settings.voice_memos_dir)
            artifact, diff = _process_single_file(stored)
            return IngestResponse(
                user_id=resolved_user,
                stored_file=str(stored),
                date=artifact["date"],
                memo_type=artifact["memo_type"],
                artifact_path=str(artifact_path(settings, artifact["date"])),
                applied=diff is not None,
                diff=diff,
            )

    if not _single_user_api_key_ok(x_api_key):
        raise HTTPException(status_code=401, detail="invalid_api_key")

    settings = load_settings()
    stored = _save_upload(file, settings.voice_memos_dir)
    artifact, diff = _process_single_file(stored)
    return IngestResponse(
        user_id="default",
        stored_file=str(stored),
        date=artifact["date"],
        memo_type=artifact["memo_type"],
        artifact_path=str(artifact_path(settings, artifact["date"])),
        applied=diff is not None,
        diff=diff,
    )


@app.post("/run", response_model=RunResponse, dependencies=[Depends(require_api_key)])
def run_all() -> RunResponse:
    settings = load_settings()
    state = load_state(settings)
    pending = [f for f in list_audio_files(settings) if not is_processed(state, f)]
    processed: list[str] = []
    for file_path in pending:
        process_file(settings, state, file_path, forced_type=None, apply_override=None)
        processed.append(file_path.name)
    save_state(settings, state)
    return RunResponse(processed=len(processed), files=processed)


@app.post("/users/{user_id}/run", response_model=RunResponse)
def run_user(user_id: str, x_api_key: str | None = Header(default=None)) -> RunResponse:
    profile = _resolve_user_profile(user_id, x_api_key)
    with _env_overrides(profile["env"]):
        settings = load_settings()
        state = load_state(settings)
        pending = [f for f in list_audio_files(settings) if not is_processed(state, f)]
        processed: list[str] = []
        for file_path in pending:
            process_file(settings, state, file_path, forced_type=None, apply_override=None)
            processed.append(file_path.name)
        save_state(settings, state)
        return RunResponse(processed=len(processed), files=processed)


@app.post("/plan/generate", dependencies=[Depends(require_api_key)])
def plan_generate(payload: GenerateRequest) -> dict[str, Any]:
    settings = load_settings()
    state = load_state(settings)
    if payload.from_audio:
        audio = Path(payload.from_audio).expanduser()
        if not audio.exists():
            raise HTTPException(status_code=404, detail=f"audio_not_found: {audio}")
    else:
        audio = latest_audio_for_date(settings, payload.date)
    artifact, _ = process_file(settings, state, audio, forced_type=None, apply_override=False)
    save_state(settings, state)
    return {"date": artifact["date"], "artifact_path": str(artifact_path(settings, artifact["date"]))}


@app.post("/plan/preview", dependencies=[Depends(require_api_key)])
def plan_preview(payload: DateRequest) -> dict[str, Any]:
    settings = load_settings()
    artifact = load_artifact(settings, payload.date)
    future_only = artifact["memo_type"] == "revision"
    diff = preview_apply_diff(settings, artifact, future_only=future_only)
    return {"date": payload.date, "future_only": future_only, "diff": diff}


@app.post("/plan/apply", dependencies=[Depends(require_api_key)])
def plan_apply(payload: DateRequest) -> dict[str, Any]:
    settings = load_settings()
    artifact = load_artifact(settings, payload.date)
    future_only = artifact["memo_type"] == "revision"
    diff = apply_artifact(settings, artifact, future_only=future_only)
    return {"date": payload.date, "future_only": future_only, "diff": diff}


@app.post("/plan/revise", dependencies=[Depends(require_api_key)])
def plan_revise(payload: ReviseRequest) -> dict[str, Any]:
    settings = load_settings()
    state = load_state(settings)
    audio = Path(payload.from_audio).expanduser()
    if not audio.exists():
        raise HTTPException(status_code=404, detail=f"audio_not_found: {audio}")
    artifact, diff = process_file(
        settings,
        state,
        audio,
        forced_type="revision",
        apply_override=payload.apply,
    )
    save_state(settings, state)
    return {"date": artifact["date"], "artifact_path": str(artifact_path(settings, artifact["date"])), "diff": diff}


@app.post("/plan/rollback", dependencies=[Depends(require_api_key)])
def plan_rollback(payload: DateRequest) -> dict[str, Any]:
    settings = load_settings()
    diff = rollback_day(settings, payload.date)
    return {"date": payload.date, "diff": diff}


def run_server() -> None:
    uvicorn.run("backend:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))


if __name__ == "__main__":
    run_server()
