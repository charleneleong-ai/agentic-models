"""Command line entry point: `archlab ablate <name>`."""

from __future__ import annotations

import inspect
from pathlib import Path

import typer

from archlab import corpus_gate
from archlab.ablations import activation_bound, attn_res, depth_sweep, qb_scale

app = typer.Typer(add_completion=False, help=__doc__)

RUNNERS = {
    "qb-scale": qb_scale.run,
    "attn-res": attn_res.run,
    "attn-res-depth": depth_sweep.run,
    "activation-bound": activation_bound.run,
    "attn-res-dyck": depth_sweep.run,
}


@app.callback()
def main() -> None:
    """Keeps `ablate` a named subcommand — Typer otherwise promotes a lone command to root."""


@app.command()
def ablate(
    name: str = typer.Argument(..., help=f"One of: {', '.join(RUNNERS)}"),
    config_dir: Path = typer.Option(Path("configs/ablations"), help="Where the specs live"),
    out_root: Path = typer.Option(Path("experiments"), help="Where results.jsonl is written"),
    device: str = typer.Option("cpu", help="torch device for training-based ablations"),
) -> None:
    if name not in RUNNERS:
        raise typer.BadParameter(f"no runner for {name!r}; have: {', '.join(RUNNERS)}")
    runner = RUNNERS[name]
    kwargs = {"device": device} if "device" in inspect.signature(runner).parameters else {}
    rows = runner(config_dir / f"{name}.yaml", out_root / name, **kwargs)
    typer.echo(f"{len(rows)} rows -> {out_root / name / 'results.jsonl'}")


@app.command()
def gate(
    check: str = typer.Argument(..., help="envelope | depth"),
    chain_len: int = typer.Option(4, help="chain length for the depth check"),
    steps: int = typer.Option(600, help="training steps per run"),
) -> None:
    """Validate a corpus before spending sweep time on it.

    `attn-res-depth` burned two GPU-hours to discover its corpus was depth-saturated. These
    checks cost minutes and answer the only question that makes a depth ablation meaningful:
    is there a regime where the task is hard, learnable, and sensitive to depth?
    """
    if check == "envelope":
        corpus_gate.chain_length_envelope()
    elif check == "depth":
        corpus_gate.depth_at_the_cliff(chain_len=chain_len, steps=steps)
    elif check == "dyck":
        corpus_gate.dyck_depth_gate(depth=chain_len, steps=steps)
    elif check == "dyck-nesting":
        corpus_gate.dyck_nesting_envelope(steps=steps)
    elif check == "dyck-stability":
        corpus_gate.dyck_stability(depth=chain_len)
    elif check == "dyck-converged":
        corpus_gate.dyck_converged_envelope(steps=steps)
    else:
        raise typer.BadParameter("check must be 'envelope', 'depth', 'dyck' or 'dyck-nesting'")


if __name__ == "__main__":
    app()
