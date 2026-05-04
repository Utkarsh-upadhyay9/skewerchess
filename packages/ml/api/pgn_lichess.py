"""Parse Lichess-style PGN with ``[%eval …]`` comments and estimate move quality.

Used by the HTTP API so game analysis works with **only** a PGN export (no
local DuckDB required on the server). When eval comments are missing, rows are
marked ``classification=unknown`` and skipped for coaching unless we add a
Stockfish fallback later.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import chess
import chess.pgn

from packages.ml.data.annotate import classify_move

_EVAL_RE = re.compile(r"\[%eval\s+([^\]]+?)\s*\]")


def _parse_eval_token(raw: str) -> float | None:
    """Return white POV score in **pawns** (Lichess convention), or None.

    Supports ``0.25``, ``-0.17``, ``#3`` (mate in 3 for white), ``#-2``.
    Mate values are clamped to a large synthetic pawn score for thresholds.
    """
    t = raw.strip()
    if not t:
        return None
    if t.startswith("#"):
        sign = 1 if not t.startswith("#-") else -1
        digits = t[1:].lstrip("-") or "0"
        try:
            n = int(digits)
        except ValueError:
            return None
        # Map mate to ±9.9 pawns so cp-loss math still orders blunders last.
        return sign * 9.9 if n else None
    try:
        return float(t)
    except ValueError:
        return None


def extract_white_eval_pawns(comment: str) -> float | None:
    m = _EVAL_RE.search(comment or "")
    if not m:
        return None
    return _parse_eval_token(m.group(1))


@dataclass
class AnalyzedMove:
    ply: int
    fen_before: str
    move_san: str
    mover_color: str  # "w" | "b"
    classification: str
    eval_drop_cp: int
    white_eval_pawns_after: float | None


def analyze_pgn_mainline(
    pgn: str,
    *,
    max_plies: int = 120,
) -> list[AnalyzedMove]:
    """Replay mainline, derive per-move class from adjacent Lichess evals."""
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return []

    board = game.board()
    out: list[AnalyzedMove] = []
    last_white_pawns: float | None = 0.0
    ply = 0

    node = game
    while node.variations and ply < max_plies:
        node = node.variations[0]
        move = node.move
        fen_before = board.fen()
        mover_white = board.turn == chess.WHITE
        mover_color = "w" if mover_white else "b"
        san = board.san(move)

        board.push(move)
        ply += 1

        w_after = extract_white_eval_pawns(node.comment or "")
        drop_cp = 0
        cls = "unknown"

        if last_white_pawns is not None and w_after is not None:
            if mover_white:
                drop_cp = int(max(0.0, (last_white_pawns - w_after) * 100))
            else:
                drop_cp = int(max(0.0, (w_after - last_white_pawns) * 100))
            cls = classify_move(drop_cp)
            last_white_pawns = w_after
        elif w_after is not None:
            last_white_pawns = w_after

        out.append(
            AnalyzedMove(
                ply=ply,
                fen_before=fen_before,
                move_san=san,
                mover_color=mover_color,
                classification=cls,
                eval_drop_cp=drop_cp,
                white_eval_pawns_after=w_after,
            )
        )

    return out
