"""Annotation-pipeline tests.

The Stockfish-driven worker is exercised in a single small integration test
(skipped by default — needs the binary). Pure-Python helpers are unit-tested.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from packages.ml.data.annotate import (
    AnnotationRow,
    classify_move,
    insert_annotations,
    list_unannotated_game_ids,
)


def test_classify_move_thresholds() -> None:
    assert classify_move(0) == "best"
    assert classify_move(2) == "great"
    assert classify_move(20) == "good"
    assert classify_move(50) == "inaccuracy"
    assert classify_move(150) == "mistake"
    assert classify_move(500) == "blunder"


def test_list_unannotated_when_empty(tmp_db) -> None:
    assert list_unannotated_game_ids() == []


def test_insert_annotations_roundtrip(tmp_db) -> None:
    """Insert one synthetic row, read it back, confirm everything sticks."""
    from packages.ml.data.store import connect

    with connect() as con:
        con.execute(
            """
            INSERT INTO games (id, source, ply_count, pgn)
            VALUES ('lichess:fakeabcd', 'lichess', 1, '[Event "?"] *')
            """
        )

    insert_annotations(
        [
            AnnotationRow(
                game_id="lichess:fakeabcd",
                ply=1,
                fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                side_to_move="b",
                move_san="e4",
                move_uci="e2e4",
                eval_cp=25,
                eval_mate=None,
                best_pv_san="e4",
                multipv2_san="d4",
                multipv3_san="Nf3",
                eval_drop_cp=0,
                classification="best",
                concept_tags=["w_castled_kingside"],
            )
        ]
    )

    with connect() as con:
        n = con.execute("SELECT count(*) FROM annotations").fetchone()[0]
        assert n == 1
        row = con.execute(
            "SELECT classification, concept_tags FROM annotations"
        ).fetchone()
        assert row[0] == "best"
        assert "w_castled_kingside" in list(row[1])


@pytest.mark.skipif(
    not Path("/opt/homebrew/bin/stockfish").exists(),
    reason="stockfish binary not found; install with `brew install stockfish`.",
)
def test_annotate_tiny_pgn_end_to_end(tmp_db) -> None:
    """Run the worker once on a 4-move game. Confirms Stockfish + concept tagger
    + DuckDB write all line up. Fast (~2 seconds at time=0.05)."""
    from packages.ml.data.annotate import annotate_games
    from packages.ml.data.store import connect

    pgn = (
        '[Event "?"]\n'
        '[Site "https://lichess.org/teststeam"]\n'
        '[White "a"]\n[Black "b"]\n[Result "*"]\n'
        '[WhiteElo "1800"]\n[BlackElo "1800"]\n'
        '[TimeControl "600+0"]\n\n'
        "1. e4 e5 2. Nf3 Nc6 *"
    )
    with connect() as con:
        con.execute(
            "INSERT INTO games (id, source, ply_count, pgn) VALUES (?, 'lichess', 4, ?)",
            ["lichess:teststeam", pgn],
        )

    stats = annotate_games(
        max_games=1, depth=10, time_per_move=0.05, multipv=2, workers=1, show_progress=False
    )

    assert stats["games_done"] == 1
    assert stats["rows_inserted"] == 4
