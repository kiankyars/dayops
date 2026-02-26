from pathlib import Path

import typer

from dayops_core import (
    apply_artifact,
    artifact_path,
    artifacts_root,
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

app = typer.Typer(help="DayOps CLI")
plan_app = typer.Typer(help="Planning commands")
app.add_typer(plan_app, name="plan")


@app.command()
def run() -> None:
    settings = load_settings()
    state = load_state(settings)

    pending = [f for f in list_audio_files(settings) if not is_processed(state, f)]
    if not pending:
        typer.echo("No new files.")
        return

    for file_path in pending:
        artifact, diff = process_file(settings, state, file_path, forced_type=None, apply_override=None)
        typer.echo(f"Processed {file_path.name} -> {artifact['date']} ({artifact['memo_type']})")
        if diff:
            typer.echo(f"Calendar: create={diff['creates']} delete={diff['deletes']} locked={diff['locked']}")

    save_state(settings, state)


@plan_app.command("generate")
def plan_generate(
    date: str = typer.Option(..., help="Date in YYYY-MM-DD"),
    from_audio: Path | None = typer.Option(None, help="Optional explicit audio path"),
) -> None:
    settings = load_settings()
    state = load_state(settings)
    audio = from_audio or latest_audio_for_date(settings, date)
    artifact, _ = process_file(settings, state, audio, forced_type=None, apply_override=False)
    save_state(settings, state)
    typer.echo(f"Generated {artifact['date']} -> {artifact_path(settings, artifact['date'])}")


@plan_app.command("preview")
def plan_preview(date: str = typer.Option(..., help="Date in YYYY-MM-DD")) -> None:
    settings = load_settings()
    artifact = load_artifact(settings, date)
    future_only = artifact["memo_type"] == "revision"
    diff = preview_apply_diff(settings, artifact, future_only=future_only)
    typer.echo(
        f"Preview {date}: create={diff['creates']} delete={diff['deletes']} "
        f"locked={diff['locked']} future_only={future_only}"
    )


@plan_app.command("apply")
def plan_apply(date: str = typer.Option(..., help="Date in YYYY-MM-DD")) -> None:
    settings = load_settings()
    artifact = load_artifact(settings, date)
    future_only = artifact["memo_type"] == "revision"
    diff = apply_artifact(settings, artifact, future_only=future_only)
    typer.echo(f"Applied {date}: create={diff['creates']} delete={diff['deletes']} locked={diff['locked']}")


@plan_app.command("revise")
def plan_revise(
    from_audio: Path = typer.Option(..., help="Revision audio path"),
    apply: bool = typer.Option(True, help="Apply revision immediately"),
) -> None:
    settings = load_settings()
    state = load_state(settings)
    artifact, diff = process_file(settings, state, from_audio, forced_type="revision", apply_override=apply)
    save_state(settings, state)
    typer.echo(f"Revision ready {artifact['date']} -> {artifact_path(settings, artifact['date'])}")
    if diff:
        typer.echo(f"Revision applied: create={diff['creates']} delete={diff['deletes']} locked={diff['locked']}")


@plan_app.command("rollback")
def plan_rollback(date: str = typer.Option(..., help="Date in YYYY-MM-DD")) -> None:
    settings = load_settings()
    diff = rollback_day(settings, date)
    typer.echo(f"Rollback {date}: restored={diff['creates']} removed_current={diff['deletes']}")


@app.command()
def tui() -> None:
    settings = load_settings()
    state = load_state(settings)
    pending = len([f for f in list_audio_files(settings) if not is_processed(state, f)])
    typer.echo("DayOps status")
    typer.echo(f"- Voice dir: {settings.voice_memos_dir}")
    typer.echo(f"- Pending memos: {pending}")
    typer.echo(f"- Auto apply: {settings.auto_apply}")
    typer.echo(f"- Artifacts: {artifacts_root(settings)}")
    typer.echo(f"- Snapshots: {settings.rollback_snapshot_dir}")


if __name__ == "__main__":
    app()
