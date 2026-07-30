"""Command line entry point: `archlab ablate <name>`."""

from __future__ import annotations

from pathlib import Path

import typer

from archlab.ablations import qb_scale

app = typer.Typer(add_completion=False, help=__doc__)

RUNNERS = {"qb-scale": qb_scale.run}


@app.callback()
def main() -> None:
    """Keeps `ablate` a named subcommand — Typer otherwise promotes a lone command to root."""


@app.command()
def ablate(
    name: str = typer.Argument(..., help=f"One of: {', '.join(RUNNERS)}"),
    config_dir: Path = typer.Option(Path("configs/ablations"), help="Where the specs live"),
    out_root: Path = typer.Option(Path("experiments"), help="Where results.jsonl is written"),
) -> None:
    if name not in RUNNERS:
        raise typer.BadParameter(f"no runner for {name!r}; have: {', '.join(RUNNERS)}")
    rows = RUNNERS[name](config_dir / f"{name}.yaml", out_root / name)
    typer.echo(f"{len(rows)} rows -> {out_root / name / 'results.jsonl'}")


if __name__ == "__main__":
    app()
