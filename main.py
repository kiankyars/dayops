from pathlib import Path

import typer

from dayops_core import (
    apply_artifact,
    artifact_path,
    artifacts_root,
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
def run(
    from_audio: Path = typer.Option(..., help="Audio path (.m4a)"),
    date: str = typer.Option(..., help="Date in YYYY-MM-DD"),
) -> None:
    settings = load_settings()
    state = load_state(settings)

    artifact, diff = process_file(
        settings,
        state,
        from_audio,
        forced_type=None,
        apply_override=None,
        date_override=date,
    )
    typer.echo(f"Processed {from_audio.name} -> {artifact['date']} ({artifact['memo_type']})")
    if diff:
        typer.echo(f"Calendar: create={diff['creates']} delete={diff['deletes']} locked={diff['locked']}")

    save_state(settings, state)


@plan_app.command("generate")
def plan_generate(
    from_audio: Path = typer.Option(..., help="Audio path (.m4a)"),
    date: str = typer.Option(..., help="Date in YYYY-MM-DD"),
) -> None:
    settings = load_settings()
    state = load_state(settings)
    artifact, _ = process_file(
        settings,
        state,
        from_audio,
        forced_type=None,
        apply_override=False,
        date_override=date,
    )
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
    date: str = typer.Option(..., help="Date in YYYY-MM-DD"),
    apply: bool = typer.Option(True, help="Apply revision immediately"),
) -> None:
    settings = load_settings()
    state = load_state(settings)
    artifact, diff = process_file(
        settings,
        state,
        from_audio,
        forced_type="revision",
        apply_override=apply,
        date_override=date,
    )
    save_state(settings, state)
    typer.echo(f"Revision ready {artifact['date']} -> {artifact_path(settings, artifact['date'])}")
    if diff:
        typer.echo(f"Revision applied: create={diff['creates']} delete={diff['deletes']} locked={diff['locked']}")


@plan_app.command("rollback")
def plan_rollback(date: str = typer.Option(..., help="Date in YYYY-MM-DD")) -> None:
    settings = load_settings()
    try:
        diff = rollback_day(settings, date)
        typer.echo(f"Rollback {date}: restored={diff['creates']} removed_current={diff['deletes']}")
    except RuntimeError as exc:
        typer.echo(str(exc))


@app.command()
def tui() -> None:
    settings = load_settings()
    typer.echo("DayOps status")
    typer.echo(f"- Artifacts: {artifacts_root(settings)}")


if __name__ == "__main__":
    app()
