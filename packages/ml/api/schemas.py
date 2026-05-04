"""Pydantic models for the FastAPI layer."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PositionCoachRequest(BaseModel):
    fen_before: str = Field(..., description="FEN of the position before the move.")
    move_san: str = Field(..., description="Move played, SAN.")
    mover_color: Literal["w", "b"] = Field(
        ..., description="Side that played the move (not side to move in FEN)."
    )
    classification: str = Field(
        default="good",
        description="One of best|great|good|inaccuracy|mistake|blunder|unknown.",
    )
    fullmove_number: int = Field(default=1, ge=1, description="Full-move number for SAN prefix.")


class PositionCoachResponse(BaseModel):
    explanation: str
    coach_backend: str


class AnalyzedTurn(BaseModel):
    ply: int
    move_san: str
    mover_color: str
    classification: str
    eval_drop_cp: int
    explanation: str


class PgnAnalyzeRequest(BaseModel):
    pgn: str = Field(..., description="PGN text; Lichess exports with evals are best.")
    max_plies: int = Field(default=80, ge=1, le=400)
    explain_each_move: bool = Field(
        default=True,
        description="If true, run the coach for every ply with a known classification.",
    )


class PgnAnalyzeResponse(BaseModel):
    turns: list[AnalyzedTurn]
    coach_backend: str
    skipped_unknown: int


class LichessGameItem(BaseModel):
    id: str
    pgn: str
    rated: bool | None = None
    speed: str | None = None


class LichessGamesResponse(BaseModel):
    username: str
    games: list[LichessGameItem]
