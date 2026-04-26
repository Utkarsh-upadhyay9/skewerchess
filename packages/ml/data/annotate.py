"""Multi-process Stockfish annotator.

For each game in the ``games`` table that hasn't been annotated yet, this:

  1. Replays the mainline move-by-move
  2. Runs Stockfish at a configurable depth/time, with multipv=3
  3. Classifies each move (best/great/good/inaccuracy/mistake/blunder)
  4. Tags each resulting position with rule-based concepts
  5. Inserts one row per ply into the ``annotations`` table

The pipeline is fully resumable — already-annotated games are skipped — and
runs N Stockfish subprocesses in parallel (one per worker process). Workers
return per-game annotation rows; the parent aggregates and writes to DuckDB
in batches so we keep a single writer.
"""

from __future__ import annotations

import io
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Iterable

import chess
import chess.engine
import chess.pgn
from loguru import logger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from packages.ml.config import settings
from packages.ml.data.concepts import tag_position
from packages.ml.data.store import connect


# ---- classification thresholds (in centipawns of eval drop, player POV) ----

# These are tighter than chess.com's defaults but match what most coaches use.
CP_BEST = 0
CP_GREAT = 5
CP_GOOD = 25
CP_INACCURACY = 75
CP_MISTAKE = 200
# >= 200 is mistake; >= 400 is blunder


def classify_move(eval_drop_cp: int) -> str:
    """Return a coaching-grade label for a move given its centipawn drop vs. best."""
    if eval_drop_cp <= CP_BEST:
        return "best"
    if eval_drop_cp < CP_GREAT:
        return "great"
    if eval_drop_cp < CP_GOOD:
        return "good"
    if eval_drop_cp < CP_INACCURACY:
        return "inaccuracy"
    if eval_drop_cp < CP_MISTAKE:
        return "mistake"
    return "blunder"


# ---- per-ply row ----------------------------------------------------------


@dataclass
class AnnotationRow:
    game_id: str
    ply: int
    fen: str
    side_to_move: str
    move_san: str
    move_uci: str
    eval_cp: int | None
    eval_mate: int | None
    best_pv_san: str | None
    multipv2_san: str | None
    multipv3_san: str | None
    eval_drop_cp: int | None
    classification: str
    concept_tags: list[str] = field(default_factory=list)


# ---- worker function ------------------------------------------------------


# Module-level so it can be pickled by ProcessPoolExecutor.
def _annotate_one_game(args: tuple[str, str, str, int, float | None, int]) -> tuple[str, list[AnnotationRow], str | None]:
    """Worker: spawn Stockfish, replay PGN, return annotation rows.

    Args tuple: (game_id, pgn_text, stockfish_path, depth, time_per_move, multipv)

    Returns (game_id, rows, error_msg).
    """
    game_id, pgn_text, sf_path, depth, time_per_move, multipv = args

    try:
        engine = chess.engine.SimpleEngine.popen_uci(sf_path)
    except Exception as e:
        return game_id, [], f"failed to spawn stockfish: {e}"

    try:
        engine.configure({"Threads": 1, "Hash": 64})
    except Exception:
        pass

    rows: list[AnnotationRow] = []
    try:
        game = chess.pgn.read_game(io.StringIO(pgn_text))
        if game is None:
            return game_id, [], "could not parse PGN"

        board = game.board()

        # Pre-evaluate the starting position once to bootstrap the eval-delta loop
        if time_per_move is not None:
            limit = chess.engine.Limit(time=time_per_move)
        else:
            limit = chess.engine.Limit(depth=depth)

        prev_info = engine.analyse(board, limit, multipv=multipv)
        prev_pov_score = prev_info[0]["score"]  # PovScore

        ply = 0
        for move in game.mainline_moves():
            ply += 1
            move_san = board.san(move)
            move_uci = move.uci()
            mover_color = board.turn

            # The pre-move analysis (multipv list) tells us what the BEST move was.
            best_pv_san = (
                board.san(prev_info[0]["pv"][0]) if prev_info[0].get("pv") else None
            )
            multipv2_san = (
                board.san(prev_info[1]["pv"][0])
                if len(prev_info) > 1 and prev_info[1].get("pv")
                else None
            )
            multipv3_san = (
                board.san(prev_info[2]["pv"][0])
                if len(prev_info) > 2 and prev_info[2].get("pv")
                else None
            )

            board.push(move)

            # Post-move analysis
            new_info = engine.analyse(board, limit, multipv=multipv)
            new_pov_score = new_info[0]["score"]  # PovScore

            # eval drop, from the mover's POV (PovScore.pov returns a Score)
            prev_cp = prev_pov_score.pov(mover_color).score(mate_score=10000)
            new_cp = new_pov_score.pov(mover_color).score(mate_score=10000)
            eval_drop = (
                max(0, prev_cp - new_cp)
                if prev_cp is not None and new_cp is not None
                else 0
            )

            # eval (white POV) of position after the move
            white_score = new_pov_score.white()
            eval_cp = white_score.score()  # None when mate
            eval_mate = white_score.mate()  # None when not mate

            classification = classify_move(eval_drop)

            tags = tag_position(board)

            rows.append(
                AnnotationRow(
                    game_id=game_id,
                    ply=ply,
                    fen=board.fen(),
                    side_to_move=tags.side_to_move,
                    move_san=move_san,
                    move_uci=move_uci,
                    eval_cp=eval_cp,
                    eval_mate=eval_mate,
                    best_pv_san=best_pv_san,
                    multipv2_san=multipv2_san,
                    multipv3_san=multipv3_san,
                    eval_drop_cp=eval_drop,
                    classification=classification,
                    concept_tags=tags.tags,
                )
            )

            prev_info = new_info
            prev_pov_score = new_pov_score

        return game_id, rows, None
    except Exception as e:
        return game_id, rows, f"worker error: {e}"
    finally:
        try:
            engine.quit()
        except Exception:
            pass


# ---- public API -----------------------------------------------------------


def list_unannotated_game_ids(limit: int | None = None) -> list[str]:
    """Return ids of games with at least one move and no rows in annotations."""
    sql = """
        SELECT g.id FROM games g
        LEFT JOIN annotations a ON a.game_id = g.id
        WHERE a.game_id IS NULL AND g.ply_count > 0
        ORDER BY g.ingested_at
    """
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    with connect() as con:
        return [row[0] for row in con.execute(sql).fetchall()]


def fetch_pgns(game_ids: Iterable[str]) -> dict[str, str]:
    ids = list(game_ids)
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    sql = f"SELECT id, pgn FROM games WHERE id IN ({placeholders})"
    with connect() as con:
        return {row[0]: row[1] for row in con.execute(sql, ids).fetchall()}


def insert_annotations(rows: list[AnnotationRow]) -> None:
    if not rows:
        return
    with connect() as con:
        con.executemany(
            """
            INSERT INTO annotations (
                game_id, ply, fen, side_to_move, move_san, move_uci,
                eval_cp, eval_mate, best_pv_san, multipv2_san, multipv3_san,
                eval_drop_cp, classification, concept_tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.game_id, r.ply, r.fen, r.side_to_move, r.move_san, r.move_uci,
                    r.eval_cp, r.eval_mate, r.best_pv_san, r.multipv2_san, r.multipv3_san,
                    r.eval_drop_cp, r.classification, r.concept_tags,
                )
                for r in rows
            ],
        )


def annotate_games(
    max_games: int | None = None,
    depth: int = 16,
    time_per_move: float | None = 0.1,
    multipv: int = 3,
    workers: int = 4,
    show_progress: bool = True,
) -> dict[str, int]:
    """Annotate all unannotated games, ``max_games`` at most.

    Either ``time_per_move`` (seconds) or ``depth`` is honored — if
    ``time_per_move`` is not None we use a wall-clock limit per position;
    otherwise depth.
    """
    game_ids = list_unannotated_game_ids(limit=max_games)
    pgns = fetch_pgns(game_ids)

    stats = {
        "games_total": len(game_ids),
        "games_done": 0,
        "games_failed": 0,
        "rows_inserted": 0,
    }

    if not game_ids:
        return stats

    sf_path = settings.stockfish_path
    work_args = [
        (gid, pgns[gid], sf_path, depth, time_per_move, multipv) for gid in game_ids
    ]

    progress_cols = (
        SpinnerColumn(),
        TextColumn("[bold]annotating[/bold]"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )

    started = time.monotonic()

    def _record(gid: str, rows: list[AnnotationRow], err: str | None, bar, task) -> None:
        if err:
            logger.warning(f"{gid}: {err}")
            stats["games_failed"] += 1
        else:
            insert_annotations(rows)
            stats["rows_inserted"] += len(rows)
            stats["games_done"] += 1
        bar.advance(task)

    with Progress(*progress_cols, transient=False, disable=not show_progress) as bar:
        task = bar.add_task("games", total=len(game_ids))

        if workers <= 1:
            # In-process path: simpler, easier to debug, works in restricted
            # sandboxes that block multiprocessing semaphores.
            for args in work_args:
                game_id, rows, err = _annotate_one_game(args)
                _record(game_id, rows, err, bar, task)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_annotate_one_game, args): args[0] for args in work_args
                }
                for fut in as_completed(futures):
                    gid = futures[fut]
                    try:
                        game_id, rows, err = fut.result()
                    except Exception as e:
                        _record(gid, [], f"future raised: {e}", bar, task)
                        continue
                    _record(game_id, rows, err, bar, task)

    stats["seconds"] = round(time.monotonic() - started, 1)
    return stats


def default_workers() -> int:
    """Recommended worker count for this machine.

    Each Stockfish + python process uses ~150-300 MB RAM. We leave headroom for
    the OS. On a 16GB Mac, 4 workers gives strong throughput without thrashing.
    """
    physical = os.cpu_count() or 4
    return max(2, min(6, physical - 2))
