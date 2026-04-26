"""Concept-tagger correctness tests using known FEN positions."""

from __future__ import annotations

import chess

from packages.ml.data.concepts import (
    _bad_bishop,
    _doubled_pawns,
    _hanging_pieces,
    _has_bishop_pair,
    _has_castled_kingside,
    _has_castled_queenside,
    _isolated_pawns,
    _knight_outposts,
    _passed_pawns,
    _rooks_on_open_files,
    _rooks_on_seventh,
    _weak_back_rank,
    tag_position,
)


# ---- pawn structure tests -------------------------------------------------


def test_iqp_position_is_tagged() -> None:
    """Classic IQP — white d4 isolani, no c- or e-pawn (Tarrasch-style)."""
    fen = "r1bq1rk1/pp3ppp/2n1pn2/3p4/3P4/2NB1N2/PP3PPP/R1BQ1RK1 w - - 0 9"
    board = chess.Board(fen)
    iso = _isolated_pawns(board, chess.WHITE)
    assert any(chess.square_file(sq) == 3 for sq in iso)
    assert "w_iqp" in tag_position(board).tags


def test_doubled_pawns_detected() -> None:
    fen = "8/8/8/8/8/3P4/3P4/4K2k w - - 0 1"
    board = chess.Board(fen)
    doubled = _doubled_pawns(board, chess.WHITE)
    assert len(doubled) == 2


def test_passed_pawn_detected() -> None:
    """White pawn on a5 with no enemy pawns on a/b files in front."""
    fen = "8/8/8/P7/8/8/k7/4K3 w - - 0 1"
    board = chess.Board(fen)
    passed = _passed_pawns(board, chess.WHITE)
    assert chess.A5 in passed


def test_no_passed_pawn_when_blocked() -> None:
    fen = "8/p7/8/P7/8/8/k7/4K3 w - - 0 1"
    board = chess.Board(fen)
    passed = _passed_pawns(board, chess.WHITE)
    assert chess.A5 not in passed


# ---- piece activity tests -------------------------------------------------


def test_rook_on_open_file() -> None:
    """Rook on the d-file with no pawns on it."""
    fen = "4k3/p1p1p1p1/8/8/8/8/P1P1P1P1/3RK3 w - - 0 1"
    board = chess.Board(fen)
    on_open = _rooks_on_open_files(board, chess.WHITE)
    assert chess.D1 in on_open


def test_rook_on_seventh() -> None:
    fen = "4k3/3R4/8/8/8/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    on7 = _rooks_on_seventh(board, chess.WHITE)
    assert chess.D7 in on7


def test_bishop_pair() -> None:
    fen = "4k3/8/8/8/8/8/8/2BBK3 w - - 0 1"
    board = chess.Board(fen)
    assert _has_bishop_pair(board, chess.WHITE)
    assert not _has_bishop_pair(board, chess.BLACK)


def test_knight_outpost_supported_and_unattackable() -> None:
    """Ne5 supported by f4 with no enemy d/f pawns to challenge it."""
    fen = "4k3/8/8/4N3/5P2/8/8/4K3 w - - 0 1"
    board = chess.Board(fen)
    outposts = _knight_outposts(board, chess.WHITE)
    assert chess.E5 in outposts


# ---- king safety tests ----------------------------------------------------


def test_castled_kingside() -> None:
    board = chess.Board()
    board.push_san("e4")
    board.push_san("e5")
    board.push_san("Nf3")
    board.push_san("Nc6")
    board.push_san("Bc4")
    board.push_san("Bc5")
    board.push_san("O-O")
    assert _has_castled_kingside(board, chess.WHITE)
    assert not _has_castled_queenside(board, chess.WHITE)


def test_weak_back_rank_with_open_file_threat() -> None:
    """White king on g1, pawns on f2/g2/h2, black rook on open d-file? The
    classic weak-back-rank pattern requires the f/g/h pawns to NOT have moved."""
    fen = "3r2k1/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1"
    board = chess.Board(fen)
    assert _weak_back_rank(board, chess.WHITE)


# ---- tactical tests -------------------------------------------------------


def test_hanging_piece_detected() -> None:
    """Black knight on e5 attacked by white queen on e1 along the file, undefended."""
    fen = "4k3/8/8/4n3/8/8/8/4Q1K1 w - - 0 1"
    board = chess.Board(fen)
    hanging = _hanging_pieces(board, chess.BLACK)
    assert chess.E5 in hanging


def test_no_hanging_when_defended() -> None:
    """Same setup but a black pawn on d6 defends the knight."""
    fen = "4k3/8/3p4/4n3/8/8/8/4Q1K1 w - - 0 1"
    board = chess.Board(fen)
    hanging = _hanging_pieces(board, chess.BLACK)
    assert chess.E5 not in hanging


# ---- top-level tag_position -----------------------------------------------


def test_tag_position_starting_position_is_minimal() -> None:
    board = chess.Board()
    result = tag_position(board)
    assert result.side_to_move == "w"
    # opening position should produce no positional flags
    assert "w_iqp" not in result.tags
    assert "b_iqp" not in result.tags
    assert "in_check" not in result.tags


def test_tag_position_returns_sorted_unique() -> None:
    board = chess.Board()
    result = tag_position(board)
    assert list(result.tags) == sorted(set(result.tags))


def test_tag_position_check_and_mate() -> None:
    fool = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    board = chess.Board(fool)
    tags = tag_position(board).tags
    assert "in_check" in tags
    assert "checkmate" in tags
