"""Unit tests for the teacher LLM data-generation pipeline.

These tests exercise sampler, prompt builder, validator, and the orchestrator
end-to-end with a mock teacher client. No network, no real LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from packages.ml.data.teacher import (
    DEFAULT_CLASS_WEIGHTS,
    PROMPT_VERSION,
    GenStats,
    TeacherSample,
    TeacherUsage,
    _format_eval,
    _replay_to_ply,
    build_prompt,
    generate_explanations,
    humanize_tags,
    sample_positions,
    validate_explanation,
)


# ---- helpers --------------------------------------------------------------


SCHOLARS_PGN = (
    '[Event "?"]\n'
    '[Site "https://lichess.org/scholar01"]\n'
    '[White "alice"]\n[Black "bob"]\n[Result "1-0"]\n'
    '[WhiteElo "1800"]\n[BlackElo "1750"]\n'
    '[TimeControl "600+0"]\n\n'
    "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0"
)


def _seed_one_game(con) -> None:
    """Insert one game + four annotation rows into the temp DB."""
    con.execute(
        """
        INSERT INTO games (
            id, source, white, black, white_elo, black_elo, result,
            time_class, ply_count, has_engine_eval, pgn
        ) VALUES (?, 'lichess', 'alice', 'bob', 1800, 1750, '1-0',
                 'rapid', 7, false, ?)
        """,
        ["lichess:scholar01", SCHOLARS_PGN],
    )

    rows = [
        # ply, fen-after, side-to-move-next, san, uci, cp, drop, classification, tags
        (1, "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
         "w", "e5", "e7e5", 25, 0, "best", []),
        (2, "rnbqkbnr/pppp1ppp/8/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR b KQkq - 1 2",
         "b", "Bc4", "f1c4", 30, 0, "best", []),
        (3, "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
         "w", "Nc6", "b8c6", 22, 0, "best", []),
        # White plays the trap-y Qh5 — let's mark it as "great" so we hit
        # the great bucket with one row.
        (4, "r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
         "b", "Qh5", "d1h5", 50, 0, "great", ["w_bishop_pair"]),
        # Black's losing move Nf6 (allows mate). Big drop.
        (5, "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
         "w", "Nf6", "g8f6", 1500, 0, "blunder", ["opponent_has_hanging_piece"]),
        # White delivers mate.
        (6, "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
         "w", "Qxf7#", "h5f7", 0, 0, "best", ["checkmate", "in_check"]),
    ]

    for ply, fen, stm, san, uci, eval_cp, drop, cls, tags in rows:
        con.execute(
            """
            INSERT INTO annotations (
                game_id, ply, fen, side_to_move, move_san, move_uci,
                eval_cp, eval_mate, best_pv_san, multipv2_san, multipv3_san,
                eval_drop_cp, classification, concept_tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?, ?)
            """,
            ["lichess:scholar01", ply, fen, stm, san, uci, eval_cp, san, drop, cls, tags],
        )


# ---- replay ---------------------------------------------------------------


def test_replay_to_ply_returns_correct_fen_before_each_move() -> None:
    fen_before, full = _replay_to_ply(SCHOLARS_PGN, 1)
    assert fen_before is not None
    assert fen_before.startswith("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w")
    assert full == 1

    # PGN has 7 plies (e4 e5 Bc4 Nc6 Qh5 Nf6 Qxf7#) — ply 8 is past the end.
    fen_before, full = _replay_to_ply(SCHOLARS_PGN, 8)
    assert fen_before is None


def test_replay_to_ply_at_move_4_position() -> None:
    """Position before white's 3rd move (ply 5) is after 1.e4 e5 2.Bc4 Nc6."""
    fen_before, full = _replay_to_ply(SCHOLARS_PGN, 5)
    assert fen_before is not None
    assert "Q3" not in fen_before  # queen still on d1
    assert full == 3


# ---- humanize_tags --------------------------------------------------------


def test_humanize_tags_known_and_unknown() -> None:
    out = humanize_tags(["w_iqp", "checkmate", "totally_fake_tag"])
    assert out[0] == "white has an isolated d-pawn (IQP)"
    assert out[1] == "checkmate"
    assert out[2] == "totally_fake_tag"  # unknown passes through


# ---- format_eval ----------------------------------------------------------


def test_format_eval_handles_mate_cp_and_unclear() -> None:
    assert "mate" in _format_eval(None, 3).lower()
    assert "mate" in _format_eval(None, -5).lower()
    assert "equal" in _format_eval(0, None).lower()
    assert "white" in _format_eval(180, None)
    assert "black" in _format_eval(-180, None)
    assert _format_eval(None, None) == "unclear"


# ---- prompt builder -------------------------------------------------------


def _stub_sample(**overrides) -> TeacherSample:
    base = dict(
        game_id="lichess:test",
        ply=10,
        fen_before="rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        fen_after="rnbqkbnr/pp1ppppp/8/2p5/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2",
        move_san="d4",
        move_uci="d2d4",
        mover_color="w",
        classification="good",
        eval_drop_cp=10,
        eval_cp_after=30,
        eval_mate_after=None,
        best_pv_san="d4",
        multipv2_san="Nf3",
        multipv3_san="Nc3",
        concept_tags=["w_bishop_pair"],
        white_elo=1800,
        black_elo=1820,
        result="1-0",
        fullmove_number=2,
    )
    base.update(overrides)
    return TeacherSample(**base)


def test_build_prompt_contains_required_fields() -> None:
    sample = _stub_sample()
    p = build_prompt(sample)
    assert sample.fen_before in p
    assert "d4" in p
    assert "Nf3" in p
    assert "bishop pair" in p
    assert "White" in p
    assert "1800" in p and "1820" in p
    # The prompt must instruct against eval numbers
    assert "Do NOT cite eval numbers" in p
    assert "candidate" in p.lower()


def test_build_prompt_for_black_says_so() -> None:
    sample = _stub_sample(mover_color="b", move_san="Nf6", best_pv_san="Nc6")
    p = build_prompt(sample)
    assert "Black" in p
    # Black-move move number prefix uses ".." like '5.. Nf6'
    assert ".. Nf6" in p


# ---- validator ------------------------------------------------------------


def _scholar_blunder_sample() -> TeacherSample:
    """Position right before black plays the losing 4...Nf6."""
    return TeacherSample(
        game_id="lichess:scholar01",
        ply=8,
        fen_before="r1bqkbnr/pppp1ppp/2n5/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 3 3",
        fen_after="r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        move_san="Nf6",
        move_uci="g8f6",
        mover_color="b",
        classification="blunder",
        eval_drop_cp=2000,
        eval_cp_after=None,
        eval_mate_after=1,
        best_pv_san="Qe7",
        multipv2_san="g6",
        multipv3_san="Qf6",
        concept_tags=["opponent_has_hanging_piece"],
        white_elo=1800,
        black_elo=1750,
        result="1-0",
        fullmove_number=4,
    )


def test_validator_accepts_clean_explanation() -> None:
    sample = _scholar_blunder_sample()
    text = (
        "After Nf6, you fail to defend f7. White already had a hanging-piece "
        "threat with the queen and bishop pointed at f7, and Qxf7 is now mate. "
        "The right idea was Qe7 or g6 to physically block the queen's path "
        "and trade off your most exposed defender."
    )
    ok, reason = validate_explanation(text, sample)
    assert ok, reason


def test_validator_rejects_too_short() -> None:
    sample = _scholar_blunder_sample()
    ok, reason = validate_explanation("Bad move.", sample)
    assert not ok
    assert "short" in reason


def test_validator_rejects_eval_numbers() -> None:
    sample = _scholar_blunder_sample()
    text = (
        "You blundered: the engine drops 2000 cp here because Qxf7 is mate. "
        "The correct move was Qe7 to defend f7 cleanly and stay in the game."
    )
    ok, reason = validate_explanation(text, sample)
    assert not ok
    assert "eval" in reason


def test_validator_rejects_invented_moves() -> None:
    sample = _scholar_blunder_sample()
    text = (
        "After Nf6 you walk into mate. The right move was Bb4+ followed by "
        "Nd5xc3, picking up a piece and saving the king from Qxf7. Always "
        "look at the opponent's next-move threats before developing."
    )
    ok, reason = validate_explanation(text, sample)
    assert not ok
    assert "invented" in reason


def test_validator_accepts_legal_moves_not_in_pv() -> None:
    """Player can mention any legal move from the actual position, even if
    it wasn't one of the engine's top-3 candidates."""
    sample = _scholar_blunder_sample()
    text = (
        "After Nf6 you fail to spot the threat on f7. A safer try was "
        "Qf6 attacking the white queen and forcing a trade — same color "
        "pieces should not be left undefended near the king like that."
    )
    # Qf6 IS legal in the before-position so this should pass.
    ok, reason = validate_explanation(text, sample)
    assert ok, reason


# ---- sampler --------------------------------------------------------------


def test_sample_positions_returns_empty_when_db_empty(tmp_db) -> None:
    assert sample_positions(n_samples=10) == []


def test_sample_positions_pulls_from_seeded_data(tmp_db) -> None:
    from packages.ml.data.store import connect

    with connect() as con:
        _seed_one_game(con)

    # Ask for at least one of each class we seeded — there's only 1 'great'
    # and 1 'blunder' in the data.
    samples = sample_positions(
        n_samples=20,
        classification_weights={"blunder": 0.5, "great": 0.5},
        min_ply=1,
    )
    classes = {s.classification for s in samples}
    assert "blunder" in classes
    assert "great" in classes
    # Each sample should have a valid fen_before and a populated mover_color
    for s in samples:
        assert s.fen_before
        assert s.mover_color in ("w", "b")


def test_sample_positions_respects_min_ply(tmp_db) -> None:
    from packages.ml.data.store import connect

    with connect() as con:
        _seed_one_game(con)

    samples = sample_positions(n_samples=20, min_ply=5)
    assert all(s.ply >= 5 for s in samples)


def test_default_class_weights_sum_to_one() -> None:
    # Allow a tiny floating-point margin.
    assert abs(sum(DEFAULT_CLASS_WEIGHTS.values()) - 1.0) < 1e-9


# ---- orchestrator with mock client ----------------------------------------


@dataclass
class _MockClient:
    name: str = "mock-teacher"
    canned: str = (
        "After your move, you allowed a tactic on the kingside. "
        "The right idea was the engine's top recommendation, which "
        "would have prevented the threat by removing the attacker."
    )

    def generate(self, prompt: str) -> tuple[str, TeacherUsage]:
        return self.canned, TeacherUsage(tokens_in=200, tokens_out=60, seconds=0.0)


@dataclass
class _BadClient:
    name: str = "mock-bad"

    def generate(self, prompt: str) -> tuple[str, TeacherUsage]:
        return "no", TeacherUsage(tokens_in=10, tokens_out=2, seconds=0.0)


def test_generate_explanations_writes_to_teacher_table(tmp_db) -> None:
    from packages.ml.data.store import connect

    with connect() as con:
        _seed_one_game(con)

    samples = sample_positions(
        n_samples=2, classification_weights={"blunder": 0.5, "great": 0.5}, min_ply=1
    )
    assert samples
    client = _MockClient()
    stats = generate_explanations(samples, client)

    assert stats.requested == len(samples)
    assert stats.succeeded == len(samples)
    assert stats.failed_validation == 0

    with connect() as con:
        rows = con.execute(
            "SELECT teacher_model, prompt_version, explanation FROM teacher"
        ).fetchall()
    assert len(rows) == len(samples)
    assert all(r[0] == "mock-teacher" for r in rows)
    assert all(r[1] == PROMPT_VERSION for r in rows)


def test_generate_explanations_records_validation_failures(tmp_db) -> None:
    from packages.ml.data.store import connect

    with connect() as con:
        _seed_one_game(con)

    samples = sample_positions(
        n_samples=2, classification_weights={"blunder": 0.5, "great": 0.5}, min_ply=1
    )
    client = _BadClient()
    stats = generate_explanations(samples, client, max_retries=1)

    assert stats.succeeded == 0
    assert stats.failed_validation == len(samples)
    assert stats.failures_sample  # some failures were captured


def test_generate_explanations_idempotent_overwrite(tmp_db) -> None:
    """Re-running the generator on the same sample replaces the row, doesn't double."""
    from packages.ml.data.store import connect

    with connect() as con:
        _seed_one_game(con)

    samples = sample_positions(
        n_samples=1, classification_weights={"blunder": 1.0}, min_ply=1
    )
    assert samples
    generate_explanations(samples, _MockClient())
    generate_explanations(samples, _MockClient(canned=_MockClient().canned + " " * 0))

    with connect() as con:
        n = con.execute("SELECT count(*) FROM teacher").fetchone()[0]
    assert n == 1
