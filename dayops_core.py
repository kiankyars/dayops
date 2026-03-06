from __future__ import annotations

import base64
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from openai import OpenAI
from openai import APIStatusError
from openai import RateLimitError

SCOPES = ["https://www.googleapis.com/auth/calendar"]
GOOGLE_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


@dataclass
class Settings:
    state_dir: Path

    gemini_models: list[str]
    gemini_api_key: str

    google_calendar_id: str
    google_oauth_token_path: Path

    timezone: str


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value


def _parse_models(value: str) -> list[str]:
    models = [part.strip() for part in value.split(",") if part.strip()]
    if not models:
        raise RuntimeError("GEMINI_MODELS must include at least one model")
    return models


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
        state_dir=Path(_env("DAYOPS_STATE_DIR")).expanduser(),
        gemini_models=_parse_models(_env("GEMINI_MODELS")),
        gemini_api_key=_env("GEMINI_API_KEY"),
        google_calendar_id=_env("GOOGLE_CALENDAR_ID"),
        google_oauth_token_path=Path(_env("GOOGLE_OAUTH_TOKEN_PATH")).expanduser(),
        timezone=_env("TIMEZONE"),
    )

    if not settings.google_oauth_token_path.exists():
        raise RuntimeError(f"GOOGLE_OAUTH_TOKEN_PATH not found: {settings.google_oauth_token_path}")

    settings.state_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root(settings).mkdir(parents=True, exist_ok=True)
    return settings


def provider_client(settings: Settings) -> OpenAI:
    return OpenAI(api_key=settings.gemini_api_key, base_url=GOOGLE_OPENAI_BASE_URL)


def _is_rate_limit_error(exc: Exception) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if isinstance(exc, APIStatusError) and exc.status_code == 429:
        return True
    return False


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


def extract_recorded_datetime(file_path: Path) -> datetime:
    match = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{2}\.\d{2}\.\d{2})", file_path.name)
    if not match:
        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
    parsed = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H.%M.%S")
    return parsed.replace(tzinfo=UTC)


def llm_json(settings: Settings, prompt: str) -> dict[str, Any]:
    client = provider_client(settings)
    last_error: Exception | None = None
    for model in settings.gemini_models:
        try:
            response = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = response.choices[0].message.content or "{}"
            return json.loads(content)
        except Exception as exc:
            if _is_rate_limit_error(exc):
                last_error = exc
                continue
            raise
    raise RuntimeError(f"All GEMINI_MODELS exhausted by rate limiting: {settings.gemini_models}") from last_error


def _audio_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".wav", ".wave"}:
        return "wav"
    if ext == ".mp3":
        return "mp3"
    return "mp3"


def transcribe_audio_text(settings: Settings, audio_file: Path) -> str:
    client = provider_client(settings)
    audio_b64 = base64.b64encode(audio_file.read_bytes()).decode("utf-8")
    last_error: Exception | None = None
    for model in settings.gemini_models:
        try:
            response = client.chat.completions.create(
                model=model,
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
        except Exception as exc:
            if _is_rate_limit_error(exc):
                last_error = exc
                continue
            raise
    raise RuntimeError(f"All GEMINI_MODELS exhausted by rate limiting: {settings.gemini_models}") from last_error


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
    out: list[dict[str, Any]] = []
    for event in events:
        props = event.get("extendedProperties", {}).get("private", {})
        if props.get("dayopsManaged") == "true":
            out.append(event)
    return out


def _coerce_local_iso(value: str, timezone: str) -> str:
    tz = ZoneInfo(timezone)
    raw = value.strip()
    if not raw:
        raise RuntimeError("Event timestamp is empty")

    if raw.endswith("Z"):
        base = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return base.replace(tzinfo=tz).isoformat()

    parsed = datetime.fromisoformat(raw)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=tz).isoformat()
    return parsed.isoformat()


def normalize_events(events: list[dict[str, Any]], timezone: str) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for event in sorted(events, key=lambda x: x["start_iso"]):
        start_iso = _coerce_local_iso(str(event["start_iso"]), timezone)
        end_iso = _coerce_local_iso(str(event["end_iso"]), timezone)
        key = (event["title"], start_iso, end_iso)
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(
            {
                "title": event["title"],
                "start_iso": start_iso,
                "end_iso": end_iso,
                "description": event.get("description", ""),
                "location": event.get("location", ""),
                "source": event.get("source", "dayops"),
            }
        )
    return cleaned


def generate_plan(settings: Settings, intent: dict[str, Any], audio_file: Path, date_str: str) -> dict[str, Any]:
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
Busy events: {json.dumps(busy)}
Source audio: {audio_file.name}
""".strip()

    data = llm_json(settings, prompt)
    events = normalize_events(data.get("events", []), settings.timezone)
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
    raise RuntimeError("Rollback disabled: snapshots have been removed.")


def process_file(
    settings: Settings,
    state: dict[str, Any],
    audio_file: Path,
    forced_type: str | None,
    apply_override: bool | None,
) -> tuple[dict[str, Any], dict[str, int] | None]:
    intent = transcribe_intent(settings, audio_file, forced_type=forced_type)
    date_str = intent.get("date") or extract_recorded_datetime(audio_file).strftime("%Y-%m-%d")
    artifact = generate_plan(settings, intent, audio_file, date_str)
    write_artifact(settings, artifact)

    should_apply = True if apply_override is None else apply_override
    diff = None
    if should_apply:
        diff = apply_artifact(settings, artifact, future_only=artifact["memo_type"] == "revision")

    mark_processed(state, audio_file, date_str)
    return artifact, diff
