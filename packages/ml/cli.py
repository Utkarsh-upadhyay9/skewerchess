"""Top-level CLI for the skewerchess ML package.

Entry point is registered in ``pyproject.toml`` as ``skewer``.
"""

from __future__ import annotations

import typer
from rich import print
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="skewer",
    help="skewerchess: train and run a small chess-coach LLM on Apple Silicon.",
    no_args_is_help=True,
    add_completion=False,
)

ingest_app = typer.Typer(
    name="ingest",
    help="Pull games from public chess data sources into the local DuckDB.",
    no_args_is_help=True,
)
app.add_typer(ingest_app, name="ingest")

db_app = typer.Typer(
    name="db",
    help="Inspect the local data warehouse.",
    no_args_is_help=True,
)
app.add_typer(db_app, name="db")


@app.command()
def hello() -> None:
    """Print a banner to confirm the CLI is wired up."""
    print("[bold green]skewer[/bold green] CLI is alive.")


# ---- ingest subcommands ----------------------------------------------------


@ingest_app.command("lichess")
def ingest_lichess(
    month: str = typer.Option(
        ...,
        "--month",
        "-m",
        help="Lichess month to pull, format 'YYYY-MM' (e.g. 2024-11).",
    ),
    max_games: int = typer.Option(
        500, "--max-games", "-n", help="Stop after this many filtered games are inserted."
    ),
    rating_min: int = typer.Option(1500, help="Minimum Elo for both players."),
    rating_max: int = typer.Option(2400, help="Maximum Elo for both players."),
    time_classes: str = typer.Option(
        "rapid,classical",
        help="Comma-separated time classes to keep (rapid, classical, blitz, bullet).",
    ),
    require_eval: bool = typer.Option(
        True,
        "--require-eval/--allow-no-eval",
        help="Only keep games that have Lichess engine %eval comments.",
    ),
    min_ply: int = typer.Option(20, help="Drop games shorter than this many half-moves."),
) -> None:
    """Stream a Lichess monthly archive and ingest filtered games into DuckDB."""
    from packages.ml.data.ingest_lichess import IngestFilter, ingest_month

    try:
        year_s, month_s = month.split("-")
        year_i, month_i = int(year_s), int(month_s)
    except ValueError:
        raise typer.BadParameter(
            f"--month must be 'YYYY-MM', got {month!r}", param_hint="--month"
        )

    filt = IngestFilter(
        rating_min=rating_min,
        rating_max=rating_max,
        time_classes=tuple(s.strip() for s in time_classes.split(",") if s.strip()),
        require_engine_eval=require_eval,
        min_ply_count=min_ply,
    )

    print(f"[bold]Lichess {year_i}-{month_i:02d}[/bold] — target: {max_games} games")
    print(f"  filter: elo {filt.rating_min}-{filt.rating_max}, {filt.time_classes}, eval={filt.require_engine_eval}")

    stats = ingest_month(year_i, month_i, max_games=max_games, filt=filt)

    table = Table(title="Ingest results")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="bold")
    for k, v in stats.items():
        table.add_row(k, str(v))
    Console().print(table)


# ---- db subcommands --------------------------------------------------------


@db_app.command("stats")
def db_stats() -> None:
    """Show row counts per table and a quick rating histogram."""
    from packages.ml.data.store import connect

    with connect() as con:
        tables = ["games", "annotations", "teacher"]
        table = Table(title="Row counts")
        table.add_column("table", style="cyan")
        table.add_column("rows", justify="right", style="bold")
        for t in tables:
            n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
            table.add_row(t, f"{n:,}")
        Console().print(table)

        n_games = con.execute("SELECT count(*) FROM games").fetchone()[0]
        if n_games:
            rows = con.execute(
                """
                SELECT time_class, count(*) AS n,
                       round(avg((white_elo + black_elo) / 2.0)) AS avg_elo
                FROM games
                GROUP BY time_class
                ORDER BY n DESC
                """
            ).fetchall()
            tt = Table(title="Games by time class")
            tt.add_column("time_class")
            tt.add_column("count", justify="right")
            tt.add_column("avg elo", justify="right")
            for tc, n, avg_elo in rows:
                tt.add_row(str(tc), f"{n:,}", str(int(avg_elo)) if avg_elo else "?")
            Console().print(tt)


@db_app.command("sample")
def db_sample(
    limit: int = typer.Option(5, "--limit", "-n", help="Rows to show."),
) -> None:
    """Print a few sample rows from the games table."""
    from packages.ml.data.store import connect

    with connect() as con:
        rows = con.execute(
            """
            SELECT id, white, white_elo, black, black_elo, time_class, eco, ply_count
            FROM games
            ORDER BY ingested_at DESC
            LIMIT ?
            """,
            [limit],
        ).fetchall()
        if not rows:
            print("[yellow]no games yet — run `skewer ingest lichess --month YYYY-MM`[/yellow]")
            return
        t = Table(title=f"Latest {len(rows)} games")
        for col in ("id", "white", "elo_w", "black", "elo_b", "tc", "eco", "ply"):
            t.add_column(col)
        for r in rows:
            t.add_row(*(str(c) if c is not None else "—" for c in r))
        Console().print(t)


if __name__ == "__main__":
    app()
