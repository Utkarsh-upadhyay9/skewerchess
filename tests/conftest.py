"""Pytest fixtures shared across the suite."""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pytest


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """Point the DuckDB store at a temp file for the duration of the test."""
    monkeypatch.setattr(
        "packages.ml.config.settings.data_dir",
        tmp_path,
    )
    return tmp_path


@pytest.fixture
def sample_pgn() -> str:
    """A handful of synthetic Lichess-style games covering the filtering paths."""
    return """[Event "Rated rapid game"]
[Site "https://lichess.org/abcd1234"]
[White "alice"]
[Black "bob"]
[WhiteElo "2000"]
[BlackElo "1980"]
[Result "1-0"]
[UTCDate "2024.11.01"]
[TimeControl "600+5"]
[ECO "B12"]
[Opening "Caro-Kann Defense: Advance Variation"]
[Termination "Normal"]

1. e4 { [%eval 0.25] } c6 { [%eval 0.30] } 2. d4 { [%eval 0.28] } d5 { [%eval 0.32] } 3. e5 { [%eval 0.27] } Bf5 { [%eval 0.31] } 4. Nf3 { [%eval 0.29] } e6 { [%eval 0.33] } 5. Be2 { [%eval 0.26] } Nd7 { [%eval 0.30] } 6. O-O { [%eval 0.24] } h6 { [%eval 0.40] } 7. Nbd2 { [%eval 0.20] } Ne7 { [%eval 0.55] } 8. c4 { [%eval 0.50] } dxc4 { [%eval 0.45] } 9. Nxc4 { [%eval 0.48] } Bg6 { [%eval 0.50] } 10. Be3 { [%eval 0.55] } Nf5 1-0

[Event "Rated bullet game"]
[Site "https://lichess.org/zzzz0000"]
[White "carol"]
[Black "dave"]
[WhiteElo "2100"]
[BlackElo "2050"]
[Result "0-1"]
[UTCDate "2024.11.01"]
[TimeControl "60+0"]
[ECO "C20"]
[Opening "King's Pawn Game"]

1. e4 e5 2. Bc4 Nf6 3. Qf3 Bc5 4. Qxf7# 0-1

[Event "Rated rapid game"]
[Site "https://lichess.org/qqqq9999"]
[White "eve"]
[Black "frank"]
[WhiteElo "1200"]
[BlackElo "1180"]
[Result "1/2-1/2"]
[UTCDate "2024.11.01"]
[TimeControl "600+0"]
[ECO "C50"]
[Opening "Italian Game"]

1. e4 { [%eval 0.20] } e5 { [%eval 0.25] } 2. Nf3 { [%eval 0.22] } Nc6 { [%eval 0.27] } 3. Bc4 { [%eval 0.20] } Bc5 { [%eval 0.23] } 4. d3 { [%eval 0.18] } d6 { [%eval 0.22] } 5. O-O { [%eval 0.15] } O-O { [%eval 0.20] } 6. Nc3 { [%eval 0.18] } Nf6 { [%eval 0.25] } 7. Bg5 { [%eval 0.20] } h6 { [%eval 0.30] } 8. Bxf6 { [%eval 0.28] } Qxf6 { [%eval 0.32] } 9. Nd5 { [%eval 0.30] } Qd8 { [%eval 0.35] } 10. c3 { [%eval 0.32] } 1/2-1/2
"""
