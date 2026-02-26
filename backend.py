from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
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

app = FastAPI(title="dayops-backend", version="0.1.0")


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.getenv("BACKEND_API_KEY")
    if not expected:
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid_api_key")


class RunResponse(BaseModel):
    processed: int
    files: list[str]


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
