"""Top-level CLI for the skewerchess ML package.

Subcommands will be added across Weeks 1-2:
  skewer ingest    — pull public games + Stockfish-annotate them
  skewer tag       — run the concept tagger on a dataset
  skewer teacher   — generate teacher annotations via Gemini
  skewer train     — QLoRA fine-tune via mlx-lm
  skewer eval      — run the blind eval harness
  skewer serve     — local inference server
"""

from __future__ import annotations

import typer
from rich import print

app = typer.Typer(
    name="skewer",
    help="skewerchess: train and run a small chess-coach LLM on Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)


@app.command()
def hello() -> None:
    """Print a banner to confirm the CLI is wired up."""
    print("[bold green]skewer[/bold green] CLI is alive. Subcommands land in Day 2.")


if __name__ == "__main__":
    app()
