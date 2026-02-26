from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI
from send2trash import send2trash

SCOPES = ["https://www.googleapis.com/auth/calendar"]
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
VENICE_BASE_URL = "https://api.venice.ai/api/v1"


@dataclass
class Settings:
    voice_memos_dir: Path
    plan_template_path: Path
    state_dir: Path
    rollback_snapshot_dir: Path

    model_provider: str
    model_name: str
    stt_provider: str

    gemini_api_key: str | None
    venice_inference_key: str | None
    gemini_base_url: str
    venice_base_url: str
    gemini_stt_model: str
    venice_stt_model: str

    google_calendar_id: str
    google_oauth_token_path: Path

    strava_client_id: str
    strava_client_secret: str
    strava_refresh_token: str

    timezone: str
    auto_apply: bool
    trash_processed_memos: bool
    hydrate_max_retries: int
    hydrate_retry_seconds: float


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _env_optional(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def _env_bool(name: str) -> bool:
    return _env(name).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int:
    return int(_env(name))


def _env_float(name: str) -> float:
    return float(_env(name))


def _require_key(value: str | None, key_name: str) -> str:
    if value:
        return value
    raise RuntimeError(f"Missing required env var: {key_name}")


def artifacts_root(settings: Settings) -> Path:
    return settings.state_dir / "artifacts"


def state_file(settings: Settings) -> Path:
    return settings.state_dir / "state.json"


def artifact_path(settings: Settings, date_str: str) -> Path:
    folder = artifacts_root(settings) / date_str
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "latest_plan.json"


def load_settings() -> Settings:
    load_dotenv()

    settings = Settings(
        voice_memos_dir=Path(_env("VOICE_MEMOS_DIR")).expanduser(),
        plan_template_path=Path(_env("PLAN_TEMPLATE_PATH")).expanduser(),
        state_dir=Path(_env("DAYOPS_STATE_DIR")).expanduser(),
        rollback_snapshot_dir=Path(_env("DAYOPS_SNAPSHOT_DIR")).expanduser(),
        model_provider=_env("MODEL_PROVIDER").strip().lower(),
        model_name=_env("MODEL_NAME").strip(),
        stt_provider=_env("STT_PROVIDER").strip().lower(),
        gemini_api_key=_env_optional("GEMINI_API_KEY"),
        venice_inference_key=_env_optional("VENICE_INFERENCE_KEY"),
        gemini_base_url=_env_optional("GEMINI_OPENAI_BASE_URL") or GOOGLE_OPENAI_BASE_URL,
        venice_base_url=_env_optional("VENICE_BASE_URL") or VENICE_BASE_URL,
        gemini_stt_model=_env("GEMINI_STT_MODEL"),
        venice_stt_model=_env("VENICE_STT_MODEL"),
        google_calendar_id=_env("GOOGLE_CALENDAR_ID"),
        google_oauth_token_path=Path(_env("GOOGLE_OAUTH_TOKEN_PATH")).expanduser(),
        strava_client_id=_env("STRAVA_CLIENT_ID"),
        strava_client_secret=_env("STRAVA_CLIENT_SECRET"),
        strava_refresh_token=_env("STRAVA_REFRESH_TOKEN"),
        timezone=_env("TIMEZONE"),
        auto_apply=_env_bool("AUTO_APPLY"),
        trash_processed_memos=_env_bool("TRASH_PROCESSED_MEMOS"),
        hydrate_max_retries=_env_int("HYDRATE_MAX_RETRIES"),
        hydrate_retry_seconds=_env_float("HYDRATE_RETRY_SECONDS"),
    )

    if settings.model_provider not in {"google", "venice"}:
        raise RuntimeError("MODEL_PROVIDER must be 'google' or 'venice'")
    if settings.stt_provider not in {"gemini", "venice"}:
        raise RuntimeError("STT_PROVIDER must be 'gemini' or 'venice'")

    if not settings.voice_memos_dir.exists():
        raise RuntimeError(f"VOICE_MEMOS_DIR not found: {settings.voice_memos_dir}")
    if not settings.plan_template_path.exists():
        raise RuntimeError(f"PLAN_TEMPLATE_PATH not found: {settings.plan_template_path}")
    if not settings.google_oauth_token_path.exists():
        raise RuntimeError(f"GOOGLE_OAUTH_TOKEN_PATH not found: {settings.google_oauth_token_path}")

    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.rollback_snapshot_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root(settings).mkdir(parents=True, exist_ok=True)
    return settings


def provider_client(settings: Settings, provider: str) -> OpenAI:
    if provider == "google":
        return OpenAI(
            api_key=_require_key(settings.gemini_api_key, "GEMINI_API_KEY"),
            base_url=settings.gemini_base_url,
        )
    if provider == "venice":
        return OpenAI(
            api_key=_require_key(settings.venice_inference_key, "VENICE_INFERENCE_KEY"),
            base_url=settings.venice_base_url,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def load_state(settings: Settings) -> dict[str, Any]:
    path = state_file(settings)
    if not path.exists():
        return {"processed_hashes": {}}
    return json.loads(path.read_text())


def save_state(settings: Settings, state: dict[str, Any]) -> None:
    state_file(settings).write_text(json.dumps(state, indent=2))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_processed(state: dict[str, Any], file_path: Path) -> bool:
    return sha256_file(file_path) in state.get("processed_hashes", {})


def mark_processed(state: dict[str, Any], file_path: Path, date_str: str) -> None:
    state.setdefault("processed_hashes", {})[sha256_file(file_path)] = {
        "file": str(file_path),
        "date": date_str,
        "processed_at": datetime.now(UTC).isoformat(),
    }


def file_flags(file_path: Path) -> str:
    result = subprocess.run(
        ["stat", "-f", "%Sf", str(file_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def ensure_local_file(file_path: Path, settings: Settings) -> None:
    for _ in range(settings.hydrate_max_retries):
        if "dataless" in file_flags(file_path):
            subprocess.run(["brctl", "download", str(file_path)], check=False)
            time.sleep(settings.hydrate_retry_seconds)
            continue
        return


def read_audio_bytes(file_path: Path, settings: Settings) -> bytes:
    last_error: OSError | None = None
    for _ in range(settings.hydrate_max_retries):
        ensure_local_file(file_path, settings)
        try:
            return file_path.read_bytes()
        except OSError as err:
            last_error = err
            if err.errno == 11:
                time.sleep(settings.hydrate_retry_seconds)
                continue
            raise
    if last_error:
        raise last_error
    raise RuntimeError(f"Unable to read audio file: {file_path}")


def extract_recorded_datetime(file_path: Path) -> datetime:
    match = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{2}\.\d{2}\.\d{2})", file_path.name)
    if not match:
        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
    parsed = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H.%M.%S")
    return parsed.replace(tzinfo=UTC)


def llm_json(settings: Settings, prompt: str) -> dict[str, Any]:
    client = provider_client(settings, settings.model_provider)
    response = client.chat.completions.create(
        model=settings.model_name,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "Return valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or "{}"
    return json.loads(content)


def llm_text(settings: Settings, prompt: str) -> str:
    client = provider_client(settings, settings.model_provider)
    response = client.chat.completions.create(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": "Return plain text only."},
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def _audio_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".wav", ".wave"}:
        return "wav"
    if ext in {".mp3"}:
        return "mp3"
    # m4a is common from iOS voice memos; mp3 format tag works with most compat layers.
    return "mp3"


def transcribe_audio_text(settings: Settings, audio_file: Path) -> str:
    if settings.stt_provider == "venice":
        client = provider_client(settings, "venice")
        with audio_file.open("rb") as handle:
            result = client.audio.transcriptions.create(
                file=handle,
                model=settings.venice_stt_model,
                response_format="json",
                timestamps=False,
            )
        text = getattr(result, "text", None) or ""
        if not text:
            raise RuntimeError("Venice STT returned empty text")
        return text.strip()

    # gemini STT via Google OpenAI-compatible chat endpoint with audio input
    client = provider_client(settings, "google")
    audio_b64 = base64.b64encode(read_audio_bytes(audio_file, settings)).decode("utf-8")
    response = client.chat.completions.create(
        model=settings.gemini_stt_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Transcribe this audio. Return plain text only."},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": audio_b64,
                            "format": _audio_format(audio_file),
                        },
                    },
                ],
            }
        ],
    )
    text = response.choices[0].message.content or ""
    if not text:
        raise RuntimeError("Gemini STT returned empty text")
    return text.strip()


def transcribe_intent(settings: Settings, audio_file: Path, forced_type: str | None = None) -> dict[str, Any]:
    transcript = transcribe_audio_text(settings, audio_file)
    forced_line = f"Set memo_type to '{forced_type}'." if forced_type else ""
    prompt = f"""
You parse a planning transcript into strict JSON.
{forced_line}
Return JSON with keys:
- memo_type: morning_plan or revision
- date: YYYY-MM-DD
- timezone: IANA timezone
- high_level_intent: string
- constraints: string[]
- tasks: string[]
- requested_blocks: object[]
- raw_summary: string
JSON only.

Transcript:
{transcript}
""".strip()
    data = llm_json(settings, prompt)
    if data.get("memo_type") not in {"morning_plan", "revision"}:
        data["memo_type"] = "morning_plan"
    return data


def google_creds(settings: Settings) -> Credentials:
    creds = Credentials.from_authorized_user_file(str(settings.google_oauth_token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        settings.google_oauth_token_path.write_text(creds.to_json())
    if not creds.valid:
        raise RuntimeError("Google OAuth credentials invalid")
    return creds


def calendar_service(settings: Settings):
    return build("calendar", "v3", credentials=google_creds(settings), cache_discovery=False)


def day_bounds(date_str: str, timezone: str) -> tuple[str, str]:
    tz = ZoneInfo(timezone)
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def calendar_events(service, settings: Settings, date_str: str) -> list[dict[str, Any]]:
    time_min, time_max = day_bounds(date_str, settings.timezone)
    return (
        service.events()
        .list(
            calendarId=settings.google_calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
            maxResults=250,
        )
        .execute()
        .get("items", [])
    )


def managed_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for event in events:
        props = event.get("extendedProperties", {}).get("private", {})
        if props.get("dayopsManaged") == "true":
            out.append(event)
    return out


def refresh_strava_token(settings: Settings, refresh_token: str) -> dict[str, Any]:
    response = requests.post(
        "https://www.strava.com/api/v3/oauth/token",
        data={
            "client_id": settings.strava_client_id,
            "client_secret": settings.strava_client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def fetch_strava_run(settings: Settings, date_str: str) -> dict[str, Any] | None:
    cache = settings.state_dir / "strava_token_cache.json"
    token = settings.strava_refresh_token
    if cache.exists():
        try:
            token = json.loads(cache.read_text()).get("refresh_token", token)
        except json.JSONDecodeError:
            pass

    token_payload = refresh_strava_token(settings, token)
    cache.write_text(
        json.dumps(
            {
                "refresh_token": token_payload.get("refresh_token", settings.strava_refresh_token),
                "saved_at": datetime.now(UTC).isoformat(),
            },
            indent=2,
        )
    )

    access_token = token_payload["access_token"]
    resp = requests.get(
        "https://www.strava.com/api/v3/athlete/activities",
        params={"per_page": 20, "page": 1},
        headers={"Authorization": f"Bearer {access_token}"},
        timeout=20,
    )
    resp.raise_for_status()

    for activity in resp.json():
        if activity.get("type") != "Run":
            continue
        if not str(activity.get("start_date_local", "")).startswith(date_str):
            continue

        start = datetime.fromisoformat(activity["start_date"].replace("Z", "+00:00"))
        end = start + timedelta(seconds=int(activity.get("elapsed_time", 0)))
        return {
            "activity_id": str(activity.get("id")),
            "distance_km": round(float(activity.get("distance", 0.0)) / 1000.0, 2),
            "start_iso": start.isoformat(),
            "end_iso": end.isoformat(),
            "elapsed_seconds": int(activity.get("elapsed_time", 0)),
            "moving_seconds": int(activity.get("moving_time", 0)),
        }
    return None


def normalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for event in sorted(events, key=lambda x: x["start_iso"]):
        key = (event["title"], event["start_iso"], event["end_iso"])
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "title": event["title"],
                "start_iso": event["start_iso"],
                "end_iso": event["end_iso"],
                "description": event.get("description", ""),
                "location": event.get("location", ""),
                "source": event.get("source", "dayops"),
            }
        )
    return cleaned


def add_strava_event(events: list[dict[str, Any]], run: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not run:
        return events
    filtered = [e for e in events if e.get("source") != "strava"]
    filtered.append(
        {
            "title": f"Run ({run['distance_km']} km)",
            "start_iso": run["start_iso"],
            "end_iso": run["end_iso"],
            "description": (
                f"Strava {run['activity_id']} | moving {run['moving_seconds']}s | "
                f"elapsed {run['elapsed_seconds']}s"
            ),
            "source": "strava",
        }
    )
    return filtered


def generate_plan(
    settings: Settings,
    intent: dict[str, Any],
    audio_file: Path,
    date_str: str,
    run: dict[str, Any] | None,
) -> dict[str, Any]:
    template = settings.plan_template_path.read_text()
    service = calendar_service(settings)
    busy = calendar_events(service, settings, date_str)

    prompt = f"""
Build a realistic day plan as JSON.
Rules:
- Respect constraints and timing.
- Use 2-hour deep work blocks with breaks/walks.
- Include transitions and meals.
- Respect existing busy events.
Return JSON with keys: summary, notes_markdown, events[]
where events[] have: title, start_iso, end_iso, description, location, source.

Date: {date_str}
Timezone: {intent.get('timezone', settings.timezone)}
Memo type: {intent.get('memo_type')}
Intent JSON: {json.dumps(intent)}
Template: {template}
Busy events: {json.dumps(busy)}
Strava run context: {json.dumps(run)}
Source audio: {audio_file.name}
""".strip()

    data = llm_json(settings, prompt)
    events = normalize_events(add_strava_event(data.get("events", []), run))
    if not events:
        raise RuntimeError("No events generated")

    return {
        "date": date_str,
        "memo_type": intent.get("memo_type", "morning_plan"),
        "source_audio": str(audio_file),
        "created_at": datetime.now(UTC).isoformat(),
        "summary": data.get("summary", ""),
        "notes_markdown": data.get("notes_markdown", ""),
        "events": events,
        "strava_run": run,
    }


def write_artifact(settings: Settings, artifact: dict[str, Any]) -> Path:
    latest = artifact_path(settings, artifact["date"])
    latest.write_text(json.dumps(artifact, indent=2))
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (latest.parent / f"plan_{stamp}.json").write_text(json.dumps(artifact, indent=2))
    return latest


def load_artifact(settings: Settings, date_str: str) -> dict[str, Any]:
    path = artifact_path(settings, date_str)
    if not path.exists():
        raise RuntimeError(f"No artifact for {date_str}. Run generate first.")
    return json.loads(path.read_text())


def snapshot_path(settings: Settings, date_str: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return settings.rollback_snapshot_dir / f"{date_str}_{stamp}.json"


def latest_snapshot(settings: Settings, date_str: str) -> Path:
    files = sorted(settings.rollback_snapshot_dir.glob(f"{date_str}_*.json"))
    if not files:
        raise RuntimeError(f"No snapshot for {date_str}")
    return files[-1]


def event_end(event: dict[str, Any]) -> datetime:
    val = event.get("end", {}).get("dateTime")
    if not val:
        return datetime.min.replace(tzinfo=UTC)
    return datetime.fromisoformat(val.replace("Z", "+00:00"))


def plan_end(event: dict[str, Any]) -> datetime:
    return datetime.fromisoformat(event["end_iso"].replace("Z", "+00:00"))


def calendar_payload(event: dict[str, Any], date_str: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "summary": event["title"],
        "description": event.get("description", ""),
        "start": {"dateTime": event["start_iso"]},
        "end": {"dateTime": event["end_iso"]},
        "extendedProperties": {
            "private": {
                "dayopsManaged": "true",
                "dayopsDate": date_str,
                "dayopsSource": event.get("source", "dayops"),
            }
        },
    }
    if event.get("location"):
        body["location"] = event["location"]
    return body


def preview_apply_diff(settings: Settings, artifact: dict[str, Any], future_only: bool) -> dict[str, int]:
    service = calendar_service(settings)
    existing = managed_events(calendar_events(service, settings, artifact["date"]))
    now = datetime.now(ZoneInfo(settings.timezone))

    deletes = 0
    locked = 0
    for event in existing:
        if future_only and event_end(event) <= now:
            locked += 1
            continue
        deletes += 1

    creates = 0
    for event in artifact["events"]:
        if future_only and plan_end(event) <= now:
            continue
        creates += 1

    return {"creates": creates, "deletes": deletes, "locked": locked}


def apply_artifact(settings: Settings, artifact: dict[str, Any], future_only: bool) -> dict[str, int]:
    service = calendar_service(settings)
    existing = managed_events(calendar_events(service, settings, artifact["date"]))
    now = datetime.now(ZoneInfo(settings.timezone))

    snapshot = {
        "date": artifact["date"],
        "captured_at": datetime.now(UTC).isoformat(),
        "future_only": future_only,
        "events": existing,
    }
    snapshot_path(settings, artifact["date"]).write_text(json.dumps(snapshot, indent=2))

    deletes = 0
    locked = 0
    for event in existing:
        if future_only and event_end(event) <= now:
            locked += 1
            continue
        service.events().delete(calendarId=settings.google_calendar_id, eventId=event["id"]).execute()
        deletes += 1

    creates = 0
    for event in artifact["events"]:
        if future_only and plan_end(event) <= now:
            continue
        service.events().insert(
            calendarId=settings.google_calendar_id,
            body=calendar_payload(event, artifact["date"]),
        ).execute()
        creates += 1

    return {"creates": creates, "deletes": deletes, "locked": locked}


def rollback_day(settings: Settings, date_str: str) -> dict[str, int]:
    service = calendar_service(settings)
    previous = json.loads(latest_snapshot(settings, date_str).read_text()).get("events", [])
    current = managed_events(calendar_events(service, settings, date_str))

    for event in current:
        service.events().delete(calendarId=settings.google_calendar_id, eventId=event["id"]).execute()

    restored = 0
    for event in previous:
        body = {
            "summary": event.get("summary", ""),
            "description": event.get("description", ""),
            "start": event.get("start", {}),
            "end": event.get("end", {}),
            "location": event.get("location", ""),
            "extendedProperties": event.get("extendedProperties", {}),
        }
        service.events().insert(calendarId=settings.google_calendar_id, body=body).execute()
        restored += 1

    return {"creates": restored, "deletes": len(current), "locked": 0}


def list_audio_files(settings: Settings) -> list[Path]:
    files = [p for p in settings.voice_memos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".m4a"]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def latest_audio_for_date(settings: Settings, date_str: str) -> Path:
    files = [p for p in list_audio_files(settings) if extract_recorded_datetime(p).strftime("%Y-%m-%d") == date_str]
    if not files:
        raise RuntimeError(f"No audio found for {date_str} in {settings.voice_memos_dir}")
    return files[-1]


def process_file(
    settings: Settings,
    state: dict[str, Any],
    audio_file: Path,
    forced_type: str | None,
    apply_override: bool | None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    intent = transcribe_intent(settings, audio_file, forced_type=forced_type)
    date_str = intent.get("date") or extract_recorded_datetime(audio_file).strftime("%Y-%m-%d")
    run = fetch_strava_run(settings, date_str)
    artifact = generate_plan(settings, intent, audio_file, date_str, run)
    write_artifact(settings, artifact)

    should_apply = settings.auto_apply if apply_override is None else apply_override
    diff = None
    if should_apply:
        diff = apply_artifact(settings, artifact, future_only=artifact["memo_type"] == "revision")

    mark_processed(state, audio_file, date_str)
    if settings.trash_processed_memos:
        send2trash(str(audio_file))
    return artifact, diff
