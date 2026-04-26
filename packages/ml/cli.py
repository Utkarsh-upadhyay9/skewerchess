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

annotate_app = typer.Typer(
    name="annotate",
    help="Run Stockfish + concept tagger over ingested games.",
    no_args_is_help=True,
)
app.add_typer(annotate_app, name="annotate")

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


# ---- annotate subcommands -------------------------------------------------


@annotate_app.command("run")
def annotate_run(
    max_games: int = typer.Option(
        None, "--max-games", "-n", help="Cap on games to annotate (default: all unannotated)."
    ),
    depth: int = typer.Option(
        16, "--depth", help="Stockfish search depth (used only if --time is not set)."
    ),
    time_per_move: float = typer.Option(
        0.1, "--time", help="Seconds Stockfish spends per position (overrides --depth if > 0)."
    ),
    multipv: int = typer.Option(3, "--multipv", help="Number of top engine lines to record."),
    workers: int = typer.Option(
        0,
        "--workers",
        "-w",
        help="Parallel Stockfish processes. 0 = auto (CPU-2, capped at 6).",
    ),
) -> None:
    """Annotate all unannotated games with Stockfish evaluations + concept tags."""
    from packages.ml.data.annotate import annotate_games, default_workers

    if workers <= 0:
        workers = default_workers()
    tpm = time_per_move if time_per_move and time_per_move > 0 else None

    print(
        f"[bold]Annotating[/bold]: depth={depth} time={tpm}s multipv={multipv} "
        f"workers={workers} max_games={max_games or 'all'}"
    )

    stats = annotate_games(
        max_games=max_games,
        depth=depth,
        time_per_move=tpm,
        multipv=multipv,
        workers=workers,
    )

    table = Table(title="Annotation results")
    table.add_column("metric", style="cyan")
    table.add_column("value", style="bold")
    for k, v in stats.items():
        table.add_row(k, str(v))
    Console().print(table)


@annotate_app.command("stats")
def annotate_stats() -> None:
    """Show annotation coverage and move-classification distribution."""
    from packages.ml.data.store import connect

    with connect() as con:
        coverage = con.execute(
            """
            SELECT
              (SELECT count(*) FROM games) AS total_games,
              (SELECT count(DISTINCT game_id) FROM annotations) AS annotated_games,
              (SELECT count(*) FROM annotations) AS total_positions
            """
        ).fetchone()
        ct = Table(title="Coverage")
        ct.add_column("metric", style="cyan")
        ct.add_column("value", justify="right", style="bold")
        ct.add_row("total games", f"{coverage[0]:,}")
        ct.add_row("annotated games", f"{coverage[1]:,}")
        ct.add_row("annotated positions", f"{coverage[2]:,}")
        Console().print(ct)

        if coverage[2] == 0:
            return

        rows = con.execute(
            """
            SELECT classification, count(*) AS n,
                   round(avg(eval_drop_cp)) AS avg_drop_cp
            FROM annotations
            GROUP BY classification
            ORDER BY array_position(
                ['best','great','good','inaccuracy','mistake','blunder']::VARCHAR[],
                classification
            )
            """
        ).fetchall()
        ct2 = Table(title="Move classification distribution")
        ct2.add_column("classification")
        ct2.add_column("count", justify="right")
        ct2.add_column("avg drop (cp)", justify="right")
        for cls, n, drop in rows:
            ct2.add_row(str(cls), f"{n:,}", str(int(drop)) if drop else "0")
        Console().print(ct2)

        tag_rows = con.execute(
            """
            SELECT tag, count(*) AS n
            FROM (SELECT unnest(concept_tags) AS tag FROM annotations)
            GROUP BY tag
            ORDER BY n DESC
            LIMIT 15
            """
        ).fetchall()
        ct3 = Table(title="Top 15 concept tags")
        ct3.add_column("tag")
        ct3.add_column("positions", justify="right")
        for tag, n in tag_rows:
            ct3.add_row(str(tag), f"{n:,}")
        Console().print(ct3)


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
