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

teach_app = typer.Typer(
    name="teach",
    help="Generate teacher-LLM coaching explanations for sampled positions.",
    no_args_is_help=True,
)
app.add_typer(teach_app, name="teach")

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


# ---- teach subcommands ----------------------------------------------------


@teach_app.command("sample")
def teach_sample(
    n: int = typer.Option(20, "--n", "-n", help="How many samples to preview."),
    teacher: str = typer.Option(
        "gemini-2.5-flash",
        "--teacher",
        help="Teacher name (controls already-done filter).",
    ),
    seed: int = typer.Option(42, "--seed", help="Sampling seed."),
) -> None:
    """Preview which positions would be selected for teacher generation."""
    from packages.ml.data.teacher import sample_positions

    samples = sample_positions(n_samples=n, teacher_model=teacher, seed=seed)
    if not samples:
        print("[yellow]no samples found — annotations table is empty?[/yellow]")
        return
    t = Table(title=f"Sampled {len(samples)} positions")
    for col in ("game", "ply", "phase", "class", "drop", "move", "best", "tags"):
        t.add_column(col)
    for s in samples[:n]:
        tags = ",".join(s.concept_tags[:3])
        if len(s.concept_tags) > 3:
            tags += f"+{len(s.concept_tags)-3}"
        t.add_row(
            s.game_id.split(":")[-1][:8],
            str(s.ply),
            s.phase,
            s.classification,
            str(s.eval_drop_cp) if s.eval_drop_cp is not None else "?",
            s.move_san,
            s.best_pv_san or "—",
            tags or "—",
        )
    Console().print(t)


@teach_app.command("inspect")
def teach_inspect(
    game_id: str = typer.Argument(..., help="Game id, e.g. lichess:abc12345."),
    ply: int = typer.Argument(..., help="Half-move number."),
    teacher: str = typer.Option(
        "gemini-2.5-flash",
        "--teacher",
        help="Teacher whose explanation to display (if any exists).",
    ),
) -> None:
    """Print the prompt that would be sent + any existing teacher explanation."""
    from packages.ml.data.store import connect
    from packages.ml.data.teacher import (
        PROMPT_VERSION,
        _row_to_sample,
        build_prompt,
    )

    with connect() as con:
        row = con.execute(
            """
            SELECT
                a.game_id, a.ply, a.fen, a.side_to_move,
                a.move_san, a.move_uci, a.eval_cp, a.eval_mate,
                a.best_pv_san, a.multipv2_san, a.multipv3_san,
                a.eval_drop_cp, a.classification, a.concept_tags,
                g.white_elo, g.black_elo, g.result, g.pgn
            FROM annotations a JOIN games g ON a.game_id = g.id
            WHERE a.game_id = ? AND a.ply = ?
            """,
            [game_id, ply],
        ).fetchone()
        if not row:
            print(f"[red]no annotation for {game_id} ply {ply}[/red]")
            return
        sample = _row_to_sample(row)
        if sample is None:
            print("[red]could not reconstruct fen_before from PGN[/red]")
            return

        existing = con.execute(
            """
            SELECT explanation FROM teacher
            WHERE game_id = ? AND ply = ? AND teacher_model = ? AND prompt_version = ?
            """,
            [game_id, ply, teacher, PROMPT_VERSION],
        ).fetchone()

    print("[bold cyan]PROMPT[/bold cyan]")
    print(build_prompt(sample))
    print("\n[bold cyan]EXISTING EXPLANATION[/bold cyan]")
    if existing:
        print(existing[0])
    else:
        print("[dim](none — run `skewer teach run` first)[/dim]")


@teach_app.command("run")
def teach_run(
    n: int = typer.Option(100, "--n", "-n", help="Number of samples to generate."),
    teacher: str = typer.Option(
        "gemini-2.5-flash",
        "--teacher",
        help="One of: gemini, gemini-2.5-flash, groq, llama-3.3-70b-versatile.",
    ),
    rpm: int = typer.Option(
        0,
        "--rpm",
        help="Override requests-per-minute. 0 = default for the chosen teacher.",
    ),
    seed: int = typer.Option(42, "--seed"),
    skip_done: bool = typer.Option(
        True,
        "--skip-done/--include-done",
        help="Skip positions already explained by this teacher (default true).",
    ),
) -> None:
    """Generate teacher explanations for ``n`` sampled positions."""
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from packages.ml.data.teacher import (
        GeminiTeacher,
        GroqTeacher,
        generate_explanations,
        sample_positions,
    )

    n_lower = teacher.lower()
    if n_lower.startswith("gemini"):
        client = GeminiTeacher(model=teacher, rpm=rpm or 10)
    elif "llama" in n_lower or n_lower in ("groq",):
        model = "llama-3.3-70b-versatile" if n_lower == "groq" else teacher
        client = GroqTeacher(model=model, rpm=rpm or 30)
    else:
        print(f"[red]unknown teacher: {teacher}[/red]")
        raise typer.Exit(1)

    samples = sample_positions(
        n_samples=n,
        teacher_model=client.name if skip_done else None,
        exclude_already_done=skip_done,
        seed=seed,
    )
    if not samples:
        print("[yellow]no samples to generate — annotations empty or all done.[/yellow]")
        return

    print(
        f"[bold]Generating[/bold] {len(samples)} explanations via "
        f"[bold]{client.name}[/bold]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold]teacher[/bold]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as bar:
        task = bar.add_task("samples", total=len(samples))

        def _on_progress(idx: int, total: int, _stats) -> None:
            bar.update(task, completed=idx)

        stats = generate_explanations(samples, client, on_progress=_on_progress)

    t = Table(title="Teacher generation results")
    t.add_column("metric", style="cyan")
    t.add_column("value", style="bold")
    t.add_row("requested", str(stats.requested))
    t.add_row("succeeded", str(stats.succeeded))
    t.add_row("failed (validation)", str(stats.failed_validation))
    t.add_row("failed (API)", str(stats.failed_api))
    t.add_row("tokens in", f"{stats.tokens_in:,}")
    t.add_row("tokens out", f"{stats.tokens_out:,}")
    t.add_row("seconds", f"{stats.seconds:.1f}")
    Console().print(t)

    if stats.failures_sample:
        print("\n[yellow]first failures:[/yellow]")
        for f in stats.failures_sample:
            print(f"  • {f}")


@teach_app.command("stats")
def teach_stats() -> None:
    """Show teacher coverage by model + a few example explanations."""
    from packages.ml.data.store import connect

    with connect() as con:
        coverage = con.execute(
            """
            SELECT teacher_model, prompt_version, count(*) AS n,
                   sum(cost_tokens_in) AS in_tok, sum(cost_tokens_out) AS out_tok
            FROM teacher
            GROUP BY teacher_model, prompt_version
            ORDER BY n DESC
            """
        ).fetchall()
        if not coverage:
            print("[yellow]teacher table is empty — run `skewer teach run` first[/yellow]")
            return
        t = Table(title="Teacher coverage")
        for col in ("model", "prompt", "rows", "tokens_in", "tokens_out"):
            t.add_column(col)
        for model, ver, n, ti, to in coverage:
            t.add_row(model, ver, f"{n:,}", f"{ti or 0:,}", f"{to or 0:,}")
        Console().print(t)

        ex = con.execute(
            """
            SELECT t.teacher_model, t.game_id, t.ply, a.classification, a.move_san, t.explanation
            FROM teacher t JOIN annotations a USING (game_id, ply)
            ORDER BY random()
            LIMIT 3
            """
        ).fetchall()
        for model, gid, ply, cls, san, expl in ex:
            print(f"\n[bold cyan]{model}[/bold cyan] {gid} ply {ply} ({cls}, {san}):")
            print(expl)


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
