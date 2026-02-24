from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests
import typer
from dotenv import load_dotenv
from google import genai
from google.auth.transport.requests import Request
from google.genai import errors, types
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from send2trash import send2trash

app = typer.Typer(help="COO assistant CLI")
plan_app = typer.Typer(help="Planning commands")
app.add_typer(plan_app, name="plan")

SCOPES = ["https://www.googleapis.com/auth/calendar"]
MARKER_START = "<!-- COO:START -->"
MARKER_END = "<!-- COO:END -->"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    voice_memos_dir: Path = Field(..., alias="VOICE_MEMOS_DIR")
    plan_template_path: Path = Field(..., alias="PLAN_TEMPLATE_PATH")

    obsidian_vault_dir: Path = Field(..., alias="OBSIDIAN_VAULT_DIR")
    obsidian_daily_notes_subdir: str = Field("notes", alias="OBSIDIAN_DAILY_NOTES_SUBDIR")

    state_dir: Path = Field(Path(".coo_state"), alias="COO_STATE_DIR")
    rollback_snapshot_dir: Path = Field(Path(".coo_state/snapshots"), alias="ROLLBACK_SNAPSHOT_DIR")

    gemini_api_key: str = Field(..., alias="GEMINI_API_KEY")
    gemini_models: str = Field(
        "gemini-2.5-flash,gemini-2.5-flash-lite,gemini-1.5-pro",
        alias="GEMINI_MODELS",
    )

    google_calendar_id: str = Field(..., alias="GOOGLE_CALENDAR_ID")
    google_oauth_token_path: Path | None = Field(None, alias="GOOGLE_OAUTH_TOKEN_PATH")
    google_service_account_file: Path | None = Field(None, alias="GOOGLE_SERVICE_ACCOUNT_FILE")

    strava_client_id: str = Field(..., alias="STRAVA_CLIENT_ID")
    strava_client_secret: str = Field(..., alias="STRAVA_CLIENT_SECRET")
    strava_refresh_token: str = Field(..., alias="STRAVA_REFRESH_TOKEN")

    timezone: str = Field("America/Los_Angeles", alias="TIMEZONE")
    auto_apply: bool = Field(True, alias="AUTO_APPLY")
    trash_processed_memos: bool = Field(True, alias="TRASH_PROCESSED_MEMOS")

    hydrate_max_retries: int = Field(6, alias="HYDRATE_MAX_RETRIES")
    hydrate_retry_seconds: float = Field(1.5, alias="HYDRATE_RETRY_SECONDS")


class TranscriptIntent(BaseModel):
    memo_type: str = Field(description="morning_plan or revision")
    date: str = Field(description="ISO date YYYY-MM-DD")
    timezone: str
    high_level_intent: str
    constraints: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    requested_blocks: list[dict[str, Any]] = Field(default_factory=list)
    raw_summary: str = ""


class PlannedEvent(BaseModel):
    title: str
    start_iso: str
    end_iso: str
    description: str = ""
    location: str = ""
    source: str = "coo"


class PlanArtifact(BaseModel):
    date: str
    memo_type: str
    source_audio: str
    created_at: str
    summary: str
    notes_markdown: str
    events: list[PlannedEvent]
    strava_run: dict[str, Any] | None = None


class StateStore(BaseModel):
    processed_hashes: dict[str, dict[str, str]] = Field(default_factory=dict)


@dataclass
class CalendarDiff:
    creates: int
    deletes: int
    locked: int


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def parse_models(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def extract_recorded_datetime(file_path: Path) -> datetime:
    match = re.search(r"(\d{4}-\d{2}-\d{2}).*?(\d{2}\.\d{2}\.\d{2})", file_path.name)
    if not match:
        return datetime.fromtimestamp(file_path.stat().st_mtime, tz=UTC)
    value = datetime.strptime(f"{match.group(1)} {match.group(2)}", "%Y-%m-%d %H.%M.%S")
    return value.replace(tzinfo=ZoneInfo("UTC"))


def file_flags(file_path: Path) -> str:
    result = subprocess.run(
        ["stat", "-f", "%Sf", str(file_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return ""
    return result.stdout.strip()


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
    raise RuntimeError(f"Failed to read audio file: {file_path}")


def file_hash(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_dirs(settings: Settings) -> None:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    settings.rollback_snapshot_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir(settings).mkdir(parents=True, exist_ok=True)


def state_file(settings: Settings) -> Path:
    return settings.state_dir / "state.json"


def artifacts_dir(settings: Settings) -> Path:
    return settings.state_dir / "artifacts"


def load_state(settings: Settings) -> StateStore:
    path = state_file(settings)
    if not path.exists():
        return StateStore()
    return StateStore.model_validate_json(path.read_text())


def save_state(settings: Settings, state: StateStore) -> None:
    state_file(settings).write_text(state.model_dump_json(indent=2))


def is_processed(state: StateStore, file_path: Path) -> bool:
    return file_hash(file_path) in state.processed_hashes


def mark_processed(state: StateStore, file_path: Path, date_str: str) -> None:
    state.processed_hashes[file_hash(file_path)] = {
        "file": str(file_path),
        "processed_at": utc_now_iso(),
        "date": date_str,
    }


def gemini_client(settings: Settings) -> genai.Client:
    return genai.Client(api_key=settings.gemini_api_key)


def gemini_json_response(
    settings: Settings,
    contents: list[Any],
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    client = gemini_client(settings)
    cfg = None
    if response_schema:
        cfg = types.GenerateContentConfig(response_mime_type="application/json", response_schema=response_schema)

    errors_seen: list[str] = []
    for model_name in parse_models(settings.gemini_models):
        try:
            response = client.models.generate_content(model=model_name, contents=contents, config=cfg)
            text = (response.text or "").strip()
            if not text:
                raise RuntimeError("Empty model response")
            return json.loads(text)
        except (errors.APIError, json.JSONDecodeError, RuntimeError) as err:
            errors_seen.append(f"{model_name}: {err}")
    raise RuntimeError("All Gemini model attempts failed: " + " | ".join(errors_seen))


def transcribe_intent(settings: Settings, audio_file: Path, force_memo_type: str | None = None) -> TranscriptIntent:
    audio_bytes = read_audio_bytes(audio_file, settings)

    forced = ""
    if force_memo_type:
        forced = f"Set memo_type strictly to '{force_memo_type}'."

    prompt = f"""
You are a planning parser.
Convert this voice memo into strict JSON.
{forced}

Rules:
- memo_type must be morning_plan or revision
- date must be ISO YYYY-MM-DD
- timezone must be IANA timezone (for example America/Los_Angeles)
- constraints/tasks should be concise strings
- requested_blocks is a list of objects with any known fields (title/start/end/duration_minutes)
- raw_summary is short and factual
- return JSON only
""".strip()

    schema = {
        "type": "object",
        "properties": {
            "memo_type": {"type": "string"},
            "date": {"type": "string"},
            "timezone": {"type": "string"},
            "high_level_intent": {"type": "string"},
            "constraints": {"type": "array", "items": {"type": "string"}},
            "tasks": {"type": "array", "items": {"type": "string"}},
            "requested_blocks": {"type": "array", "items": {"type": "object"}},
            "raw_summary": {"type": "string"},
        },
        "required": ["memo_type", "date", "timezone", "high_level_intent"],
    }

    result = gemini_json_response(
        settings,
        contents=[prompt, types.Part.from_bytes(data=audio_bytes, mime_type="audio/mp4")],
        response_schema=schema,
    )

    intent = TranscriptIntent.model_validate(result)
    if intent.memo_type not in {"morning_plan", "revision"}:
        intent.memo_type = "morning_plan"
    return intent


def load_plan_template(settings: Settings) -> str:
    return settings.plan_template_path.read_text()


def get_google_credentials(settings: Settings) -> Credentials:
    if settings.google_service_account_file:
        return service_account.Credentials.from_service_account_file(
            str(settings.google_service_account_file),
            scopes=SCOPES,
        )

    if not settings.google_oauth_token_path:
        raise RuntimeError("Set GOOGLE_OAUTH_TOKEN_PATH or GOOGLE_SERVICE_ACCOUNT_FILE")

    token_path = settings.google_oauth_token_path
    if not token_path.exists():
        raise RuntimeError(f"OAuth token file not found: {token_path}")

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json())
    if not creds.valid:
        raise RuntimeError("Google credentials are invalid")
    return creds


def calendar_service(settings: Settings):
    return build("calendar", "v3", credentials=get_google_credentials(settings), cache_discovery=False)


def day_bounds(date_str: str, tz_name: str) -> tuple[str, str]:
    tz = ZoneInfo(tz_name)
    start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=tz)
    end = start + timedelta(days=1)
    return start.isoformat(), end.isoformat()


def list_coo_events(service, settings: Settings, date_str: str) -> list[dict[str, Any]]:
    time_min, time_max = day_bounds(date_str, settings.timezone)
    items = (
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

    result = []
    for event in items:
        props = event.get("extendedProperties", {}).get("private", {})
        if props.get("cooManaged") == "true":
            result.append(event)
    return result


def list_busy_events(service, settings: Settings, date_str: str) -> list[dict[str, Any]]:
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


def refresh_strava_token(settings: Settings, token_override: str | None = None) -> dict[str, Any]:
    refresh_token = token_override or settings.strava_refresh_token
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


def fetch_latest_run_context(settings: Settings, date_str: str) -> dict[str, Any] | None:
    token_cache_file = settings.state_dir / "strava_token_cache.json"
    cached_refresh = None
    if token_cache_file.exists():
        try:
            cached_refresh = json.loads(token_cache_file.read_text()).get("refresh_token")
        except json.JSONDecodeError:
            cached_refresh = None

    token_payload = refresh_strava_token(settings, token_override=cached_refresh)
    token_cache_file.write_text(
        json.dumps(
            {
                "refresh_token": token_payload.get("refresh_token", settings.strava_refresh_token),
                "saved_at": utc_now_iso(),
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
    items = resp.json()

    for activity in items:
        if activity.get("type") != "Run":
            continue
        start_local = activity.get("start_date_local")
        if not start_local:
            continue
        if not str(start_local).startswith(date_str):
            continue

        start_utc = datetime.fromisoformat(activity["start_date"].replace("Z", "+00:00"))
        end_utc = start_utc + timedelta(seconds=int(activity.get("elapsed_time", 0)))
        distance_km = float(activity.get("distance", 0.0)) / 1000.0

        return {
            "strava_activity_id": str(activity.get("id")),
            "name": activity.get("name", "Run"),
            "start_iso": start_utc.isoformat(),
            "end_iso": end_utc.isoformat(),
            "distance_km": round(distance_km, 2),
            "elapsed_seconds": int(activity.get("elapsed_time", 0)),
            "moving_seconds": int(activity.get("moving_time", 0)),
        }

    return None


def generate_plan(
    settings: Settings,
    intent: TranscriptIntent,
    audio_file: Path,
    date_str: str,
    strava_run: dict[str, Any] | None,
) -> PlanArtifact:
    template_text = load_plan_template(settings)
    service = calendar_service(settings)
    busy_events = list_busy_events(service, settings, date_str)

    prompt = f"""
You are a chief operating officer scheduler.
Generate a highly realistic day plan.

Rules:
- Respect the user's constraints and timings.
- Use 2-hour work blocks with short breaks/walks.
- Include meals/transitions.
- Keep existing non-COO busy events in mind.
- Return strict JSON only.

Date: {date_str}
Timezone: {intent.timezone or settings.timezone}
Memo type: {intent.memo_type}
Intent JSON: {intent.model_dump_json()}
Template: {template_text}
Busy events: {json.dumps(busy_events)}
Strava run context: {json.dumps(strava_run)}
""".strip()

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "notes_markdown": {"type": "string"},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "start_iso": {"type": "string"},
                        "end_iso": {"type": "string"},
                        "description": {"type": "string"},
                        "location": {"type": "string"},
                        "source": {"type": "string"},
                    },
                    "required": ["title", "start_iso", "end_iso"],
                },
            },
        },
        "required": ["summary", "events"],
    }

    data = gemini_json_response(settings, [prompt], response_schema=schema)

    events = [PlannedEvent.model_validate(item) for item in data.get("events", [])]
    if strava_run:
        events = upsert_strava_event(events, strava_run)

    events = sort_and_dedupe_events(events)
    if not events:
        raise RuntimeError("Model returned no events")

    return PlanArtifact(
        date=date_str,
        memo_type=intent.memo_type,
        source_audio=str(audio_file),
        created_at=utc_now_iso(),
        summary=data.get("summary", ""),
        notes_markdown=data.get("notes_markdown", ""),
        events=events,
        strava_run=strava_run,
    )


def upsert_strava_event(events: list[PlannedEvent], strava_run: dict[str, Any]) -> list[PlannedEvent]:
    remaining = [e for e in events if "strava" not in e.title.lower() and "run" != e.source.lower()]
    remaining.append(
        PlannedEvent(
            title=f"Run ({strava_run.get('distance_km', 0)} km)",
            start_iso=strava_run["start_iso"],
            end_iso=strava_run["end_iso"],
            description=(
                f"Strava activity {strava_run.get('strava_activity_id')} | "
                f"moving {strava_run.get('moving_seconds')}s | elapsed {strava_run.get('elapsed_seconds')}s"
            ),
            source="strava",
        )
    )
    return remaining


def sort_and_dedupe_events(events: list[PlannedEvent]) -> list[PlannedEvent]:
    seen = set()
    unique: list[PlannedEvent] = []
    for event in sorted(events, key=lambda e: e.start_iso):
        key = (event.title, event.start_iso, event.end_iso)
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def artifact_path(settings: Settings, date_str: str) -> Path:
    date_dir = artifacts_dir(settings) / date_str
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir / "latest_plan.json"


def write_artifact(settings: Settings, artifact: PlanArtifact) -> Path:
    path = artifact_path(settings, artifact.date)
    path.write_text(artifact.model_dump_json(indent=2))
    ts_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    (path.parent / f"plan_{ts_name}.json").write_text(artifact.model_dump_json(indent=2))
    return path


def load_artifact(settings: Settings, date_str: str) -> PlanArtifact:
    path = artifact_path(settings, date_str)
    if not path.exists():
        raise RuntimeError(f"No artifact found for date {date_str}. Run generate first.")
    return PlanArtifact.model_validate_json(path.read_text())


def event_time(event: dict[str, Any], key: str) -> datetime:
    val = event.get(key, {}).get("dateTime")
    if not val:
        return datetime.min.replace(tzinfo=UTC)
    return datetime.fromisoformat(val.replace("Z", "+00:00"))


def plan_event_time(event: PlannedEvent, key: str) -> datetime:
    val = getattr(event, key)
    return datetime.fromisoformat(val.replace("Z", "+00:00"))


def build_calendar_payload(event: PlannedEvent, date_str: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "summary": event.title,
        "description": event.description,
        "start": {"dateTime": event.start_iso},
        "end": {"dateTime": event.end_iso},
        "extendedProperties": {
            "private": {
                "cooManaged": "true",
                "cooDate": date_str,
                "cooSource": event.source,
            }
        },
    }
    if event.location:
        body["location"] = event.location
    return body


def snapshot_file(settings: Settings, date_str: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return settings.rollback_snapshot_dir / f"{date_str}_{ts}.json"


def latest_snapshot_for_date(settings: Settings, date_str: str) -> Path:
    candidates = sorted(settings.rollback_snapshot_dir.glob(f"{date_str}_*.json"))
    if not candidates:
        raise RuntimeError(f"No snapshot found for date {date_str}")
    return candidates[-1]


def preview_diff(settings: Settings, artifact: PlanArtifact, future_only: bool = False) -> CalendarDiff:
    service = calendar_service(settings)
    existing = list_coo_events(service, settings, artifact.date)
    now = datetime.now(ZoneInfo(settings.timezone))

    locked = 0
    deletions = 0
    for event in existing:
        if future_only and event_time(event, "end") <= now:
            locked += 1
            continue
        deletions += 1

    creates = 0
    for event in artifact.events:
        if future_only and plan_event_time(event, "end_iso") <= now:
            continue
        creates += 1

    return CalendarDiff(creates=creates, deletes=deletions, locked=locked)


def apply_plan(settings: Settings, artifact: PlanArtifact, future_only: bool = False) -> CalendarDiff:
    service = calendar_service(settings)
    existing = list_coo_events(service, settings, artifact.date)
    now = datetime.now(ZoneInfo(settings.timezone))

    pre_snapshot = {
        "date": artifact.date,
        "captured_at": utc_now_iso(),
        "future_only": future_only,
        "events": existing,
    }
    snapshot_path = snapshot_file(settings, artifact.date)
    snapshot_path.write_text(json.dumps(pre_snapshot, indent=2))

    deletions = 0
    locked = 0
    for event in existing:
        if future_only and event_time(event, "end") <= now:
            locked += 1
            continue
        service.events().delete(calendarId=settings.google_calendar_id, eventId=event["id"]).execute()
        deletions += 1

    creates = 0
    for item in artifact.events:
        if future_only and plan_event_time(item, "end_iso") <= now:
            continue
        body = build_calendar_payload(item, artifact.date)
        service.events().insert(calendarId=settings.google_calendar_id, body=body).execute()
        creates += 1

    return CalendarDiff(creates=creates, deletes=deletions, locked=locked)


def rollback_day(settings: Settings, date_str: str) -> CalendarDiff:
    service = calendar_service(settings)
    snapshot = json.loads(latest_snapshot_for_date(settings, date_str).read_text())
    old_events: list[dict[str, Any]] = snapshot.get("events", [])

    existing = list_coo_events(service, settings, date_str)
    for event in existing:
        service.events().delete(calendarId=settings.google_calendar_id, eventId=event["id"]).execute()

    recreated = 0
    for old in old_events:
        body = {
            "summary": old.get("summary", ""),
            "description": old.get("description", ""),
            "start": old.get("start", {}),
            "end": old.get("end", {}),
            "location": old.get("location", ""),
            "extendedProperties": old.get("extendedProperties", {}),
        }
        service.events().insert(calendarId=settings.google_calendar_id, body=body).execute()
        recreated += 1

    return CalendarDiff(creates=recreated, deletes=len(existing), locked=0)


def notes_file_for_date(settings: Settings, date_str: str) -> Path:
    return settings.obsidian_vault_dir / settings.obsidian_daily_notes_subdir / f"{date_str}.md"


def render_plan_block(artifact: PlanArtifact) -> str:
    lines = [
        MARKER_START,
        f"## COO Plan ({artifact.created_at})",
        "",
        f"- Summary: {artifact.summary}",
        f"- Memo type: `{artifact.memo_type}`",
        "",
        "### Timeline",
    ]

    for event in artifact.events:
        start = datetime.fromisoformat(event.start_iso.replace("Z", "+00:00")).strftime("%H:%M")
        end = datetime.fromisoformat(event.end_iso.replace("Z", "+00:00")).strftime("%H:%M")
        lines.append(f"- {start}-{end} {event.title}")

    if artifact.strava_run:
        lines.extend(
            [
                "",
                "### Strava",
                f"- Activity: `{artifact.strava_run.get('strava_activity_id')}`",
                f"- Distance: {artifact.strava_run.get('distance_km')} km",
            ]
        )

    if artifact.notes_markdown:
        lines.extend(["", "### Notes", artifact.notes_markdown])

    lines.append(MARKER_END)
    return "\n".join(lines) + "\n"


def upsert_obsidian_note(settings: Settings, artifact: PlanArtifact) -> None:
    path = notes_file_for_date(settings, artifact.date)
    path.parent.mkdir(parents=True, exist_ok=True)
    block = render_plan_block(artifact)

    if not path.exists():
        path.write_text(block)
        return

    content = path.read_text()
    start_idx = content.find(MARKER_START)
    end_idx = content.find(MARKER_END)
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        end_idx += len(MARKER_END)
        new_content = content[:start_idx] + block + content[end_idx:]
    else:
        if content and not content.endswith("\n"):
            content += "\n"
        new_content = content + "\n" + block

    path.write_text(new_content)


def find_audio_files(settings: Settings) -> list[Path]:
    if not settings.voice_memos_dir.exists():
        return []
    files = [p for p in settings.voice_memos_dir.iterdir() if p.is_file() and p.suffix.lower() == ".m4a"]
    return sorted(files, key=lambda p: p.stat().st_mtime)


def latest_audio_for_date(settings: Settings, date_str: str) -> Path:
    candidates = []
    for file_path in find_audio_files(settings):
        ts = extract_recorded_datetime(file_path)
        if ts.strftime("%Y-%m-%d") == date_str:
            candidates.append(file_path)
    if not candidates:
        raise RuntimeError(f"No audio memo found for {date_str} in {settings.voice_memos_dir}")
    return candidates[-1]


def process_memo(
    settings: Settings,
    state: StateStore,
    audio_file: Path,
    force_memo_type: str | None = None,
    apply_override: bool | None = None,
) -> tuple[PlanArtifact, CalendarDiff | None]:
    intent = transcribe_intent(settings, audio_file, force_memo_type=force_memo_type)
    date_str = intent.date or extract_recorded_datetime(audio_file).strftime("%Y-%m-%d")
    strava_run = fetch_latest_run_context(settings, date_str)

    artifact = generate_plan(settings, intent, audio_file, date_str, strava_run)
    write_artifact(settings, artifact)

    should_apply = settings.auto_apply if apply_override is None else apply_override
    diff = None
    if should_apply:
        diff = apply_plan(settings, artifact, future_only=intent.memo_type == "revision")

    upsert_obsidian_note(settings, artifact)
    mark_processed(state, audio_file, date_str)
    if settings.trash_processed_memos:
        send2trash(str(audio_file))
    return artifact, diff


@app.command()
def run(once: bool = typer.Option(True, help="Run one scan pass and exit.")) -> None:
    """Process all unprocessed memos from VOICE_MEMOS_DIR."""
    load_dotenv()
    settings = Settings()
    ensure_dirs(settings)
    state = load_state(settings)

    files = find_audio_files(settings)
    if not files:
        typer.echo("No audio files found.")
        return

    processed_any = False
    for file_path in files:
        if is_processed(state, file_path):
            continue
        artifact, diff = process_memo(settings, state, file_path)
        processed_any = True
        typer.echo(f"Processed {file_path.name} -> {artifact.date} ({artifact.memo_type})")
        if diff:
            typer.echo(f"Calendar writes: create={diff.creates} delete={diff.deletes} locked={diff.locked}")

    if processed_any:
        save_state(settings, state)
    else:
        typer.echo("No new files to process.")

    if not once:
        typer.echo("Continuous mode is not implemented yet. Use launchd watch paths for now.")


@plan_app.command("generate")
def plan_generate(
    date: str = typer.Option(..., help="Date in YYYY-MM-DD"),
    from_audio: Path | None = typer.Option(None, help="Optional explicit audio file path"),
) -> None:
    """Generate a plan artifact from a memo without writing calendar."""
    load_dotenv()
    settings = Settings()
    ensure_dirs(settings)
    state = load_state(settings)

    audio_file = from_audio or latest_audio_for_date(settings, date)
    artifact, _ = process_memo(settings, state, audio_file, apply_override=False)
    save_state(settings, state)

    typer.echo(f"Generated artifact for {artifact.date} at {artifact_path(settings, artifact.date)}")


@plan_app.command("preview")
def plan_preview(date: str = typer.Option(..., help="Date in YYYY-MM-DD")) -> None:
    """Preview calendar diff for latest generated artifact of a date."""
    load_dotenv()
    settings = Settings()
    ensure_dirs(settings)
    artifact = load_artifact(settings, date)
    future_only = artifact.memo_type == "revision"
    diff = preview_diff(settings, artifact, future_only=future_only)
    typer.echo(
        f"Preview {date}: create={diff.creates} delete={diff.deletes} locked={diff.locked} "
        f"(future_only={future_only})"
    )


@plan_app.command("apply")
def plan_apply(date: str = typer.Option(..., help="Date in YYYY-MM-DD")) -> None:
    """Apply latest generated artifact to Google Calendar."""
    load_dotenv()
    settings = Settings()
    ensure_dirs(settings)
    artifact = load_artifact(settings, date)
    future_only = artifact.memo_type == "revision"
    diff = apply_plan(settings, artifact, future_only=future_only)
    upsert_obsidian_note(settings, artifact)
    typer.echo(f"Applied {date}: create={diff.creates} delete={diff.deletes} locked={diff.locked}")


@plan_app.command("revise")
def plan_revise(
    from_audio: Path = typer.Option(..., help="Audio file path for a revision memo"),
    apply: bool = typer.Option(True, help="Apply revision to calendar after generating."),
) -> None:
    """Generate a revision plan from an audio memo and optionally apply."""
    load_dotenv()
    settings = Settings()
    ensure_dirs(settings)
    state = load_state(settings)

    artifact, diff = process_memo(
        settings,
        state,
        from_audio,
        force_memo_type="revision",
        apply_override=apply,
    )
    save_state(settings, state)

    typer.echo(f"Revision artifact ready for {artifact.date}: {artifact_path(settings, artifact.date)}")
    if diff:
        typer.echo(f"Revision applied: create={diff.creates} delete={diff.deletes} locked={diff.locked}")


@plan_app.command("rollback")
def plan_rollback(date: str = typer.Option(..., help="Date in YYYY-MM-DD")) -> None:
    """Rollback COO-managed events for a day to latest snapshot."""
    load_dotenv()
    settings = Settings()
    ensure_dirs(settings)
    diff = rollback_day(settings, date)
    typer.echo(f"Rollback done for {date}: restored={diff.creates}, removed_current={diff.deletes}")


@app.command()
def tui() -> None:
    """Local computer-only status view (rollback remains CLI/TUI only)."""
    load_dotenv()
    settings = Settings()
    ensure_dirs(settings)
    state = load_state(settings)
    files = find_audio_files(settings)
    typer.echo("COO Status")
    typer.echo(f"- Voice memos dir: {settings.voice_memos_dir}")
    typer.echo(f"- Pending memos: {len([f for f in files if not is_processed(state, f)])}")
    typer.echo(f"- Auto apply: {settings.auto_apply}")
    typer.echo(f"- Artifacts: {artifacts_dir(settings)}")
    typer.echo(f"- Snapshots: {settings.rollback_snapshot_dir}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
