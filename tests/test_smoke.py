"""Lightweight pytest version of the smoke tests — fast, no network."""

from __future__ import annotations

import pytest


def test_python_chess_imports() -> None:
    import chess
    import chess.pgn

    board = chess.Board()
    assert board.is_valid()


def test_mlx_imports_and_runs() -> None:
    import mlx.core as mx

    try:
        a = mx.array([1.0, 2.0, 3.0])
        assert float(mx.sum(a)) == 6.0
    except RuntimeError as e:
        if "Metal device" in str(e):
            pytest.skip("Metal GPU not exposed (likely sandboxed test runner).")
        raise


def test_config_loads() -> None:
    from packages.ml.config import settings

    assert settings.base_model.startswith("mlx-community/")
    assert settings.stockfish_depth >= 10
