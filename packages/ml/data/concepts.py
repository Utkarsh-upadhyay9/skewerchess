"""Rule-based positional concept tagger.

Given a ``chess.Board``, return a list of human-readable concept tags that
describe the structural and tactical features of the position. The tagger is
deliberately conservative: each rule is hand-written using primitives from
``python-chess`` and produces no false positives for the textbook definitions.
The teacher LLM consumes these tags to ground its explanations in concrete
chess facts, which dramatically reduces hallucination of strategic claims.

The set of tags here covers the most instructive themes for 1500-2400-rated
players. We keep it under 25 tags so it remains a dense, useful signal rather
than a sea of low-signal labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import chess

# ---- helpers --------------------------------------------------------------


PIECE_VALUE_CP: dict[chess.PieceType, int] = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def _pawns_by_file(board: chess.Board, color: chess.Color) -> dict[int, list[int]]:
    """{file_index -> [rank_indices]} for pawns of ``color``."""
    out: dict[int, list[int]] = {f: [] for f in range(8)}
    for sq in board.pieces(chess.PAWN, color):
        out[chess.square_file(sq)].append(chess.square_rank(sq))
    return out


def _material_cp(board: chess.Board, color: chess.Color) -> int:
    s = 0
    for pt, val in PIECE_VALUE_CP.items():
        s += val * len(board.pieces(pt, color))
    return s


def _is_open_file(board: chess.Board, file_idx: int) -> bool:
    """No pawns of either color on this file."""
    for color in (chess.WHITE, chess.BLACK):
        for sq in board.pieces(chess.PAWN, color):
            if chess.square_file(sq) == file_idx:
                return False
    return True


def _is_semi_open_file(board: chess.Board, file_idx: int, color: chess.Color) -> bool:
    """No pawns of ``color`` on this file (but possibly enemy pawns)."""
    for sq in board.pieces(chess.PAWN, color):
        if chess.square_file(sq) == file_idx:
            return False
    return True


def _seventh_rank_for(color: chess.Color) -> int:
    return 6 if color == chess.WHITE else 1


def _back_rank_for(color: chess.Color) -> int:
    return 0 if color == chess.WHITE else 7


# ---- pawn structure -------------------------------------------------------


def _isolated_pawns(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    files = _pawns_by_file(board, color)
    out: list[chess.Square] = []
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        left = files[f - 1] if f - 1 >= 0 else []
        right = files[f + 1] if f + 1 <= 7 else []
        if not left and not right:
            out.append(sq)
    return out


def _doubled_pawns(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    files = _pawns_by_file(board, color)
    out: list[chess.Square] = []
    for f, ranks in files.items():
        if len(ranks) >= 2:
            for r in ranks:
                out.append(chess.square(f, r))
    return out


def _passed_pawns(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    enemy = not color
    enemy_pawns = _pawns_by_file(board, enemy)
    out: list[chess.Square] = []
    for sq in board.pieces(chess.PAWN, color):
        f = chess.square_file(sq)
        r = chess.square_rank(sq)
        is_passed = True
        for adj_f in (f - 1, f, f + 1):
            if not (0 <= adj_f <= 7):
                continue
            for er in enemy_pawns.get(adj_f, []):
                if color == chess.WHITE and er > r:
                    is_passed = False
                    break
                if color == chess.BLACK and er < r:
                    is_passed = False
                    break
            if not is_passed:
                break
        if is_passed:
            out.append(sq)
    return out


# ---- piece activity -------------------------------------------------------


def _rooks_on_open_files(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    out: list[chess.Square] = []
    for sq in board.pieces(chess.ROOK, color):
        if _is_open_file(board, chess.square_file(sq)):
            out.append(sq)
    return out


def _rooks_on_seventh(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    target_rank = _seventh_rank_for(color)
    return [sq for sq in board.pieces(chess.ROOK, color) if chess.square_rank(sq) == target_rank]


def _has_bishop_pair(board: chess.Board, color: chess.Color) -> bool:
    bishops = list(board.pieces(chess.BISHOP, color))
    if len(bishops) < 2:
        return False
    light = sum(1 for sq in bishops if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0)
    dark = len(bishops) - light
    return light >= 1 and dark >= 1


def _knight_outposts(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    """Knight on enemy half of board, supported by own pawn,
    not attackable by an enemy pawn now or by any future pawn advance."""
    out: list[chess.Square] = []
    enemy = not color
    enemy_pawn_files = {chess.square_file(sq) for sq in board.pieces(chess.PAWN, enemy)}

    for sq in board.pieces(chess.KNIGHT, color):
        rank = chess.square_rank(sq)
        # White: ranks 4-6 (0-indexed); Black: ranks 1-3
        if color == chess.WHITE and rank < 3:
            continue
        if color == chess.BLACK and rank > 4:
            continue

        # supported by own pawn
        f = chess.square_file(sq)
        support_rank = rank - 1 if color == chess.WHITE else rank + 1
        supported = False
        for df in (-1, 1):
            sf = f + df
            if 0 <= sf <= 7 and 0 <= support_rank <= 7:
                pawn_at = board.piece_at(chess.square(sf, support_rank))
                if pawn_at and pawn_at.piece_type == chess.PAWN and pawn_at.color == color:
                    supported = True
                    break
        if not supported:
            continue

        # cannot be attacked by an enemy pawn ever
        attackable = False
        for df in (-1, 1):
            af = f + df
            if 0 <= af <= 7 and af in enemy_pawn_files:
                # is there an enemy pawn on file ``af`` that could advance to attack?
                for ep_sq in board.pieces(chess.PAWN, enemy):
                    if chess.square_file(ep_sq) != af:
                        continue
                    ep_rank = chess.square_rank(ep_sq)
                    if color == chess.WHITE and ep_rank > rank:
                        attackable = True
                        break
                    if color == chess.BLACK and ep_rank < rank:
                        attackable = True
                        break
                if attackable:
                    break
        if attackable:
            continue
        out.append(sq)
    return out


def _bad_bishop(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    """A bishop is 'bad' if 60%+ of own pawns sit on its color squares."""
    pawns = list(board.pieces(chess.PAWN, color))
    if not pawns:
        return []
    light_pawns = sum(1 for sq in pawns if (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0)
    dark_pawns = len(pawns) - light_pawns

    out: list[chess.Square] = []
    for sq in board.pieces(chess.BISHOP, color):
        is_light_sq = (chess.square_file(sq) + chess.square_rank(sq)) % 2 == 0
        same_color = light_pawns if is_light_sq else dark_pawns
        if len(pawns) >= 4 and same_color / len(pawns) >= 0.6:
            out.append(sq)
    return out


# ---- king safety ----------------------------------------------------------


def _king_in_center(board: chess.Board, color: chess.Color, fullmove: int) -> bool:
    """King still in the e/d files past move 10 — usually a danger sign."""
    if fullmove < 10:
        return False
    king_sq = board.king(color)
    if king_sq is None:
        return False
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)
    if color == chess.WHITE and r != 0:
        return False
    if color == chess.BLACK and r != 7:
        return False
    return f in (3, 4)


def _has_castled_kingside(board: chess.Board, color: chess.Color) -> bool:
    king_sq = board.king(color)
    if king_sq is None:
        return False
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)
    expected_rank = _back_rank_for(color)
    return r == expected_rank and f >= 6


def _has_castled_queenside(board: chess.Board, color: chess.Color) -> bool:
    king_sq = board.king(color)
    if king_sq is None:
        return False
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)
    expected_rank = _back_rank_for(color)
    return r == expected_rank and f <= 2


def _weak_back_rank(board: chess.Board, color: chess.Color) -> bool:
    """King on back rank with no luft and an enemy heavy piece on an open file."""
    king_sq = board.king(color)
    if king_sq is None:
        return False
    if chess.square_rank(king_sq) != _back_rank_for(color):
        return False
    f = chess.square_file(king_sq)
    r = chess.square_rank(king_sq)

    # check luft: any pawn directly in front of king on second rank?
    front_rank = 1 if color == chess.WHITE else 6
    has_luft = False
    for df in (-1, 0, 1):
        sf = f + df
        if 0 <= sf <= 7:
            piece = board.piece_at(chess.square(sf, front_rank))
            if piece and piece.piece_type == chess.PAWN and piece.color == color:
                # pawn is still there → could be a back-rank weakness IF it's NOT the f/g/h pawn...
                # actually any pawn on the second rank in front of the king blocks luft
                continue
            else:
                has_luft = True
    if has_luft:
        return False

    # is there an enemy heavy piece on a file that's open or semi-open from our POV?
    enemy = not color
    for pt in (chess.ROOK, chess.QUEEN):
        for sq in board.pieces(pt, enemy):
            sf = chess.square_file(sq)
            if _is_semi_open_file(board, sf, color):
                return True
    return False


def _opposite_side_castling(board: chess.Board) -> bool:
    w_ks = _has_castled_kingside(board, chess.WHITE)
    w_qs = _has_castled_queenside(board, chess.WHITE)
    b_ks = _has_castled_kingside(board, chess.BLACK)
    b_qs = _has_castled_queenside(board, chess.BLACK)
    return (w_ks and b_qs) or (w_qs and b_ks)


# ---- tactical signals -----------------------------------------------------


def _hanging_pieces(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    """Pieces of ``color`` attacked by enemy and not defended."""
    out: list[chess.Square] = []
    enemy = not color
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if not piece or piece.color != color or piece.piece_type == chess.KING:
            continue
        if board.attackers(enemy, sq) and not board.attackers(color, sq):
            out.append(sq)
    return out


def _pinned_pieces(board: chess.Board, color: chess.Color) -> list[chess.Square]:
    """Pieces of ``color`` absolutely pinned to their king."""
    return [sq for sq in chess.SQUARES if board.is_pinned(color, sq) and board.piece_at(sq)
            and board.piece_at(sq).color == color and board.piece_at(sq).piece_type != chess.KING]


# ---- public API -----------------------------------------------------------


@dataclass(frozen=True)
class ConceptResult:
    """Lightweight container — the ``tags`` list is what we store in DuckDB."""

    tags: list[str]
    side_to_move: str  # 'w' | 'b'

    def as_set(self) -> set[str]:
        return set(self.tags)


def tag_position(board: chess.Board) -> ConceptResult:
    """Return all concept tags that fire for the current position.

    Tags are prefixed with the side they apply to (``w_`` or ``b_``) where
    asymmetric, or unprefixed for game-state-wide concepts.
    """
    tags: list[str] = []
    fullmove = board.fullmove_number
    side_to_move = "w" if board.turn == chess.WHITE else "b"

    # --- material imbalance
    delta = _material_cp(board, chess.WHITE) - _material_cp(board, chess.BLACK)
    if abs(delta) >= 200:
        tags.append("material_advantage_white" if delta > 0 else "material_advantage_black")

    # --- pawn structure
    for color, pfx in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        for sq in _isolated_pawns(board, color):
            f = chess.square_file(sq)
            if f == 3:  # IQP specifically: isolated d-pawn
                tags.append(f"{pfx}_iqp")
            else:
                tags.append(f"{pfx}_isolated_pawn")
                break  # don't flood with one tag per pawn
        if _doubled_pawns(board, color):
            tags.append(f"{pfx}_doubled_pawns")
        passed = _passed_pawns(board, color)
        if passed:
            tags.append(f"{pfx}_passed_pawn")
            for sq in passed:
                if chess.square_rank(sq) >= (5 if color == chess.WHITE else 2):
                    tags.append(f"{pfx}_advanced_passed_pawn")
                    break

    # --- piece activity
    for color, pfx in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        if _rooks_on_open_files(board, color):
            tags.append(f"{pfx}_rook_on_open_file")
        if _rooks_on_seventh(board, color):
            tags.append(f"{pfx}_rook_on_seventh")
        if _has_bishop_pair(board, color):
            tags.append(f"{pfx}_bishop_pair")
        if _knight_outposts(board, color):
            tags.append(f"{pfx}_knight_outpost")
        if _bad_bishop(board, color):
            tags.append(f"{pfx}_bad_bishop")

    # --- king safety
    for color, pfx in ((chess.WHITE, "w"), (chess.BLACK, "b")):
        if _has_castled_kingside(board, color):
            tags.append(f"{pfx}_castled_kingside")
        elif _has_castled_queenside(board, color):
            tags.append(f"{pfx}_castled_queenside")
        elif _king_in_center(board, color, fullmove):
            tags.append(f"{pfx}_king_in_center")
        if _weak_back_rank(board, color):
            tags.append(f"{pfx}_weak_back_rank")
    if _opposite_side_castling(board):
        tags.append("opposite_side_castling")

    # --- tactical signals (from the perspective of side to move)
    stm = board.turn
    enemy = not stm
    if _hanging_pieces(board, enemy):
        tags.append("opponent_has_hanging_piece")
    if _hanging_pieces(board, stm):
        tags.append("own_piece_hanging")
    if _pinned_pieces(board, enemy):
        tags.append("opponent_has_pinned_piece")
    if _pinned_pieces(board, stm):
        tags.append("own_piece_pinned")

    # --- check / mate
    if board.is_check():
        tags.append("in_check")
    if board.is_checkmate():
        tags.append("checkmate")
    if board.is_stalemate():
        tags.append("stalemate")

    return ConceptResult(tags=sorted(set(tags)), side_to_move=side_to_move)


# Convenience: list all tag names this module can produce, useful for evals
ALL_TAGS: tuple[str, ...] = (
    "material_advantage_white",
    "material_advantage_black",
    "w_iqp", "b_iqp",
    "w_isolated_pawn", "b_isolated_pawn",
    "w_doubled_pawns", "b_doubled_pawns",
    "w_passed_pawn", "b_passed_pawn",
    "w_advanced_passed_pawn", "b_advanced_passed_pawn",
    "w_rook_on_open_file", "b_rook_on_open_file",
    "w_rook_on_seventh", "b_rook_on_seventh",
    "w_bishop_pair", "b_bishop_pair",
    "w_knight_outpost", "b_knight_outpost",
    "w_bad_bishop", "b_bad_bishop",
    "w_castled_kingside", "b_castled_kingside",
    "w_castled_queenside", "b_castled_queenside",
    "w_king_in_center", "b_king_in_center",
    "w_weak_back_rank", "b_weak_back_rank",
    "opposite_side_castling",
    "opponent_has_hanging_piece",
    "own_piece_hanging",
    "opponent_has_pinned_piece",
    "own_piece_pinned",
    "in_check",
    "checkmate",
    "stalemate",
)
