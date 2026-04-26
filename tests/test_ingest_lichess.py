"""Offline tests for the Lichess ingest module.

We don't hit the Lichess CDN here — instead we feed a synthetic PGN through
the same parsing/filtering/storage path used by ``ingest_month``. The
streaming/zstd plumbing is exercised separately in an integration test
(skipped by default, opt in with `RUN_NETWORK_TESTS=1`).
"""

from __future__ import annotations

import io
import os

import chess.pgn
import pytest

from packages.ml.data.ingest_lichess import (
    IngestFilter,
    _extract_lichess_id,
    _has_engine_eval,
    _ply_count,
    _serialize_game,
    classify_time_control,
)
from packages.ml.data.store import connect


def _read_all(text: str) -> list[chess.pgn.Game]:
    stream = io.StringIO(text)
    games = []
    while True:
        g = chess.pgn.read_game(stream)
        if g is None:
            break
        games.append(g)
    return games


def test_classify_time_control() -> None:
    assert classify_time_control("60+0")[0] == "bullet"
    assert classify_time_control("180+2")[0] == "blitz"
    assert classify_time_control("600+5")[0] == "rapid"
    assert classify_time_control("1800+30")[0] == "classical"
    assert classify_time_control("-")[0] == "correspondence"
    assert classify_time_control("garbage")[0] == "unknown"


def test_extract_lichess_id() -> None:
    assert _extract_lichess_id({"Site": "https://lichess.org/abcd1234"}) == "abcd1234"
    assert _extract_lichess_id({"Site": "https://chess.com/foo"}) is None
    assert _extract_lichess_id({}) is None


def test_filter_matches_rapid_with_evals(sample_pgn: str) -> None:
    games = _read_all(sample_pgn)
    filt = IngestFilter()  # defaults: 1500-2400 elo, rapid+classical, require eval
    results = [
        filt.matches(dict(g.headers), _has_engine_eval(g), _ply_count(g))
        for g in games
    ]
    # game 1: rated rapid 2000 vs 1980 with evals  → keep
    # game 2: bullet 2100 vs 2050 short with no evals → drop (bullet + no eval + short)
    # game 3: rapid 1200 vs 1180 with evals → drop (below rating floor)
    assert results == [True, False, False]


def test_filter_allow_no_eval_and_lower_floor(sample_pgn: str) -> None:
    games = _read_all(sample_pgn)
    filt = IngestFilter(
        rating_min=1000,
        rating_max=2400,
        time_classes=("rapid", "classical", "bullet"),
        require_engine_eval=False,
        min_ply_count=1,
    )
    keeps = [
        filt.matches(dict(g.headers), _has_engine_eval(g), _ply_count(g))
        for g in games
    ]
    assert all(keeps)


def test_serialize_roundtrip(sample_pgn: str) -> None:
    """A serialized then re-parsed game should keep the same Site header."""
    g = _read_all(sample_pgn)[0]
    text = _serialize_game(g)
    g2 = chess.pgn.read_game(io.StringIO(text))
    assert g2 is not None
    assert g2.headers.get("Site") == g.headers.get("Site")


def test_store_and_count(sample_pgn: str, tmp_db) -> None:
    """End-to-end: insert one game via the same path the ingester uses."""
    games = _read_all(sample_pgn)
    g = games[0]
    headers = dict(g.headers)
    tc, base, inc = classify_time_control(headers["TimeControl"])
    lichess_id = _extract_lichess_id(headers)
    assert lichess_id is not None

    with connect() as con:
        con.execute(
            """
            INSERT INTO games (
                id, source, site,
                white, black, white_elo, black_elo, white_title, black_title,
                result, termination,
                eco, opening,
                time_control, time_class, base_seconds, increment,
                ply_count, has_engine_eval, utc_date, pgn
            ) VALUES (
                ?, 'lichess', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
            [
                f"lichess:{lichess_id}",
                headers["Site"],
                headers["White"],
                headers["Black"],
                int(headers["WhiteElo"]),
                int(headers["BlackElo"]),
                None,
                None,
                headers["Result"],
                headers.get("Termination"),
                headers.get("ECO"),
                headers.get("Opening"),
                headers["TimeControl"],
                tc,
                base,
                inc,
                _ply_count(g),
                _has_engine_eval(g),
                headers.get("UTCDate"),
                _serialize_game(g),
            ],
        )

    with connect(read_only=True) as con:
        count = con.execute("SELECT count(*) FROM games").fetchone()[0]
        assert count == 1
        row = con.execute(
            "SELECT white, white_elo, time_class, ply_count, has_engine_eval FROM games"
        ).fetchone()
        assert row[0] == "alice"
        assert row[1] == 2000
        assert row[2] == "rapid"
        assert row[3] >= 19
        assert row[4] is True


@pytest.mark.skipif(
    os.environ.get("RUN_NETWORK_TESTS") != "1",
    reason="Set RUN_NETWORK_TESTS=1 to hit the live Lichess CDN.",
)
def test_live_lichess_three_games(tmp_db) -> None:
    """Hits the real Lichess CDN. Pulls 3 games. Slow + network-dependent."""
    from packages.ml.data.ingest_lichess import ingest_month

    stats = ingest_month(2024, 11, max_games=3, show_progress=False)
    assert stats["inserted"] == 3
