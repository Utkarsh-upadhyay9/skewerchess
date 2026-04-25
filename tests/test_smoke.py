"""Lightweight pytest version of the smoke tests — fast, no network."""

from __future__ import annotations


def test_python_chess_imports() -> None:
    import chess
    import chess.pgn

    board = chess.Board()
    assert board.is_valid()


def test_mlx_imports_and_runs() -> None:
    import mlx.core as mx

    a = mx.array([1.0, 2.0, 3.0])
    assert float(mx.sum(a)) == 6.0


def test_config_loads() -> None:
    from packages.ml.config import settings

    assert settings.base_model.startswith("mlx-community/")
    assert settings.stockfish_depth >= 10
