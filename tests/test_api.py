"""Tests for Lichess PGN eval parsing and FastAPI routes."""

from __future__ import annotations

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from httpx import Response

from packages.ml.api.main import app
from packages.ml.api.pgn_lichess import (
    analyze_pgn_mainline,
    extract_white_eval_pawns,
)


def test_extract_white_eval_pawns_handles_float_and_mate() -> None:
    assert extract_white_eval_pawns("foo [%eval 0.25] bar") == pytest.approx(0.25)
    assert extract_white_eval_pawns("[%eval #-2]") == pytest.approx(-9.9)
    assert extract_white_eval_pawns("[%eval #3]") == pytest.approx(9.9)
    assert extract_white_eval_pawns("") is None


def test_analyze_pgn_mainline_counts_plies(sample_pgn: str) -> None:
    # ``read_game`` parses the first game in the multi-game PGN string.
    rows = analyze_pgn_mainline(sample_pgn, max_plies=200)
    assert len(rows) == 20  # Caro-Kann sample line: 10 full moves each side
    assert rows[0].move_san == "e4"
    assert rows[0].classification != "unknown"
    assert rows[0].mover_color == "w"


def test_health_endpoint() -> None:
    c = TestClient(app)
    r = c.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_coach_position_mock(monkeypatch) -> None:
    from packages.ml import config

    monkeypatch.setattr(config.settings, "coach_backend", "mock")

    c = TestClient(app)
    r = c.post(
        "/api/coach/position",
        json={
            "fen_before": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "move_san": "e4",
            "mover_color": "w",
            "classification": "best",
            "fullmove_number": 1,
        },
    )
    assert r.status_code == 200
    data = r.json()
    assert "e4" in data["explanation"]


def test_coach_pgn_smoke(sample_pgn: str, monkeypatch) -> None:
    from packages.ml import config

    monkeypatch.setattr(config.settings, "coach_backend", "mock")

    c = TestClient(app)
    r = c.post(
        "/api/coach/pgn",
        json={"pgn": sample_pgn, "max_plies": 6, "explain_each_move": True},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["turns"]) == 6
    assert all("explanation" in t for t in data["turns"])


def test_lichess_games_uses_httpx(monkeypatch) -> None:
    sample_line = json.dumps(
        {
            "id": "abc12345",
            "pgn": "[White \"x\"]\n1. e4 e5 *",
            "rated": True,
            "speed": "rapid",
        }
    )

    class _MockClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, *args, **kwargs):
            return Response(200, text=sample_line + "\n")

    monkeypatch.setattr(httpx, "AsyncClient", lambda *a, **kw: _MockClient())

    c = TestClient(app)
    r = c.get("/api/lichess/games/testuser", params={"max_games": 1})
    assert r.status_code == 200
    payload = r.json()
    assert payload["username"] == "testuser"
    assert len(payload["games"]) == 1
    assert payload["games"][0]["id"] == "abc12345"