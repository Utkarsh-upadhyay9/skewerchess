"""Produce a coaching line for the HTTP API (mock or MLX student)."""

from __future__ import annotations

import asyncio
from functools import lru_cache

from loguru import logger

from packages.ml.config import settings
from packages.ml.training.dataset import build_student_prompt


def mock_explanation(
    *,
    fen_before: str,
    move_san: str,
    mover_color: str,
    classification: str,
    fullmove_number: int,
) -> str:
    """Deterministic stub when ``COACH_BACKEND=mock`` or MLX load fails."""
    color = "White" if mover_color == "w" else "Black"
    return (
        f"{color} played {move_san} — tagged {classification}. "
        f"Once the student model is trained, this slot will be a full "
        f"coach-style explanation of plans, weaknesses, and better ideas from "
        f"the same input. FEN preview: {fen_before.split()[0]}…"
    )


@lru_cache(maxsize=1)
def _mlx_available() -> bool:
    try:
        import mlx.core  # noqa: F401

        return True
    except Exception:
        return False


async def generate_coaching(
    *,
    fen_before: str,
    move_san: str,
    mover_color: str,
    classification: str,
    fullmove_number: int,
) -> str:
    """Async wrapper; MLX runs in a thread pool."""
    backend = (settings.coach_backend or "mock").lower()

    if backend != "mlx":
        return mock_explanation(
            fen_before=fen_before,
            move_san=move_san,
            mover_color=mover_color,
            classification=classification,
            fullmove_number=fullmove_number,
        )

    if not _mlx_available():
        logger.warning("COACH_BACKEND=mlx but MLX not available — using mock")
        return mock_explanation(
            fen_before=fen_before,
            move_san=move_san,
            mover_color=mover_color,
            classification=classification,
            fullmove_number=fullmove_number,
        )

    def _run() -> str:
        from packages.ml.inference.coach import coach

        try:
            resp = coach(
                fen_before=fen_before,
                move_san=move_san,
                mover_color=mover_color,
                classification=classification,
                fullmove_number=fullmove_number,
            )
            return resp.explanation
        except Exception as e:
            logger.warning(f"MLX coach failed: {e}; falling back to mock")
            return mock_explanation(
                fen_before=fen_before,
                move_san=move_san,
                mover_color=mover_color,
                classification=classification,
                fullmove_number=fullmove_number,
            )

    return await asyncio.to_thread(_run)


def student_prompt_preview(
    *,
    fen_before: str,
    move_san: str,
    mover_color: str,
    classification: str,
    fullmove_number: int,
) -> str:
    """Expose the exact student-side prompt for debugging."""
    return build_student_prompt(
        fen_before=fen_before,
        move_san=move_san,
        mover_color=mover_color,
        classification=classification,
        fullmove_number=fullmove_number,
    )
