"""Thin inference helper for the chess-coach student model.

Loads the 4-bit base + our QLoRA adapter via ``mlx_lm.load`` and exposes a
single ``coach`` function: given a position and the move played, return a
2-4 sentence coaching explanation.

This is the on-device entry point that the FastAPI backend (and a future
Mac/iOS app) will call.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from packages.ml.config import settings
from packages.ml.data.teacher import PROMPT_VERSION
from packages.ml.training.dataset import build_student_prompt


@dataclass
class CoachResponse:
    explanation: str
    tokens_in: int
    tokens_out: int
    seconds: float


def _resolve_adapter_path(adapter_path: str | Path | None) -> str | None:
    if adapter_path is not None:
        return str(adapter_path)
    default = settings.models_dir / "lora" / f"coach-{PROMPT_VERSION}"
    return str(default) if default.exists() else None


_LOADED: dict[str, tuple] = {}


def _load(model: str, adapter_path: str | None) -> tuple:
    """Cache (model, tokenizer) per (model, adapter) tuple."""
    key = f"{model}::{adapter_path or ''}"
    if key in _LOADED:
        return _LOADED[key]

    from mlx_lm import load  # lazy: avoids import-time MLX requirement

    if adapter_path:
        model_obj, tokenizer = load(model, adapter_path=adapter_path)
    else:
        model_obj, tokenizer = load(model)
    _LOADED[key] = (model_obj, tokenizer)
    return model_obj, tokenizer


def coach(
    *,
    fen_before: str,
    move_san: str,
    mover_color: str,
    classification: str,
    fullmove_number: int,
    model: str | None = None,
    adapter_path: str | Path | None = None,
    max_tokens: int = 220,
    temperature: float = 0.7,
) -> CoachResponse:
    """Run the student model on a single position. Imports MLX lazily."""
    import time

    from mlx_lm import generate

    model_name = model or settings.base_model
    resolved_adapter = _resolve_adapter_path(adapter_path)
    model_obj, tokenizer = _load(model_name, resolved_adapter)

    prompt = build_student_prompt(
        fen_before=fen_before,
        move_san=move_san,
        mover_color=mover_color,
        classification=classification,
        fullmove_number=fullmove_number,
    )

    t0 = time.monotonic()
    text = generate(
        model_obj,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temp=temperature,
        verbose=False,
    )
    dt = time.monotonic() - t0

    completion = text[len(prompt):] if text.startswith(prompt) else text
    return CoachResponse(
        explanation=completion.strip(),
        tokens_in=len(tokenizer.encode(prompt)),
        tokens_out=len(tokenizer.encode(completion)),
        seconds=dt,
    )
