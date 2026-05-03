"""Build the instruction-tuning dataset for the student model.

This module joins the ``annotations`` table with the ``teacher`` table, drops
any rows whose teacher response failed validation post-hoc, and emits MLX-LM
compatible JSONL (one ``{"prompt": ..., "completion": ...}`` per line).

Design choice: **the student prompt is intentionally smaller than the teacher
prompt.** The teacher needed Stockfish PV moves and concept tags as
hallucination guardrails. The student should learn to produce coaching from
raw position + move + classification alone — that's what users will provide
at inference time. This is the key bit of distillation: the student
internalizes the engine reasoning.

Train/val/test split is deterministic by ``hash(game_id) % 100`` so the same
position always lands in the same split, even as we add more teacher data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import chess
from loguru import logger

from packages.ml.config import settings
from packages.ml.data.store import connect
from packages.ml.data.teacher import (
    PROMPT_VERSION,
    _replay_to_ply,
    humanize_tags,
    validate_explanation,
    TeacherSample,
)


# ---- prompt formats -------------------------------------------------------


STUDENT_SYSTEM = (
    "You are a chess coach. Given a position and the move just played, "
    "explain in 2-4 sentences why the move was good or bad. Speak directly "
    "to the player. Name the concept or pattern at play. Keep it concise."
)


def build_student_prompt(
    *,
    fen_before: str,
    move_san: str,
    mover_color: str,
    classification: str,
    fullmove_number: int,
) -> str:
    """The prompt the student sees at training and inference time.

    Deliberately omits engine PV moves and concept tags — those were
    teacher-side scaffolding only. At inference we want the student to
    accept raw {position, move, quality-tag} from a UI/engine wrapper.
    """
    color = "White" if mover_color == "w" else "Black"
    move_label = (
        f"{fullmove_number}.{'..' if mover_color == 'b' else ''} {move_san}"
    )
    return (
        f"{STUDENT_SYSTEM}\n\n"
        f"Position (FEN): {fen_before}\n"
        f"Move played by {color}: {move_label}\n"
        f"Move quality: {classification}\n\n"
        f"Coaching explanation:"
    )


# ---- joined-row dataclass -------------------------------------------------


@dataclass
class TrainExample:
    game_id: str
    ply: int
    teacher_model: str
    fen_before: str
    move_san: str
    mover_color: str
    classification: str
    fullmove_number: int
    explanation: str
    split: str  # 'train' | 'val' | 'test'

    def to_jsonl(self) -> str:
        return json.dumps(
            {
                "prompt": build_student_prompt(
                    fen_before=self.fen_before,
                    move_san=self.move_san,
                    mover_color=self.mover_color,
                    classification=self.classification,
                    fullmove_number=self.fullmove_number,
                ),
                "completion": " " + self.explanation.strip(),
                # Metadata for analysis / debugging — MLX-LM ignores extra keys.
                "meta": {
                    "game_id": self.game_id,
                    "ply": self.ply,
                    "teacher": self.teacher_model,
                    "split": self.split,
                },
            },
            ensure_ascii=False,
        )


# ---- split ----------------------------------------------------------------


def assign_split(game_id: str, *, val_pct: int = 10, test_pct: int = 10) -> str:
    """Stable hash-based split. Same game always lands in the same bucket."""
    bucket = abs(hash(game_id)) % 100
    if bucket < val_pct:
        return "val"
    if bucket < val_pct + test_pct:
        return "test"
    return "train"


# ---- DB → in-memory iterator ----------------------------------------------


def iter_train_examples(
    *,
    teacher_model: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    revalidate: bool = True,
) -> Iterator[TrainExample]:
    """Yield TrainExamples joined from teacher + annotations + games.

    If ``teacher_model`` is None, takes whichever teacher row exists per
    (game_id, ply) — useful when you've generated through multiple teachers.
    If both exist for a position, the first by alphabetical order wins
    (stable, no double-counting).

    ``revalidate`` re-runs the explanation validator before yielding; rows
    that no longer pass (e.g. after a stricter validator update) are dropped.
    """
    teacher_filter = "AND t.teacher_model = ?" if teacher_model else ""
    params: list = [prompt_version]
    if teacher_model:
        params.append(teacher_model)

    sql = f"""
        SELECT
            t.game_id, t.ply, t.teacher_model, t.explanation,
            a.fen, a.side_to_move, a.move_san, a.move_uci,
            a.eval_cp, a.eval_mate,
            a.best_pv_san, a.multipv2_san, a.multipv3_san,
            a.eval_drop_cp, a.classification, a.concept_tags,
            g.white_elo, g.black_elo, g.result, g.pgn
        FROM teacher t
        JOIN annotations a USING (game_id, ply)
        JOIN games g ON g.id = t.game_id
        WHERE t.prompt_version = ?
          {teacher_filter}
        QUALIFY row_number() OVER (
            PARTITION BY t.game_id, t.ply ORDER BY t.teacher_model
        ) = 1
        ORDER BY t.game_id, t.ply
    """

    with connect() as con:
        rows = con.execute(sql, params).fetchall()

    dropped = 0
    for row in rows:
        (
            game_id, ply, tmodel, explanation,
            fen_after, side_to_move, move_san, move_uci,
            eval_cp, eval_mate,
            best_pv_san, multipv2_san, multipv3_san,
            eval_drop_cp, classification, concept_tags,
            white_elo, black_elo, result, pgn,
        ) = row

        fen_before, fullmove_number = _replay_to_ply(pgn, ply)
        if fen_before is None:
            dropped += 1
            continue
        mover_color = "b" if side_to_move == "w" else "w"

        if revalidate:
            sample = TeacherSample(
                game_id=game_id,
                ply=ply,
                fen_before=fen_before,
                fen_after=fen_after,
                move_san=move_san,
                move_uci=move_uci,
                mover_color=mover_color,
                classification=classification,
                eval_drop_cp=eval_drop_cp,
                eval_cp_after=eval_cp,
                eval_mate_after=eval_mate,
                best_pv_san=best_pv_san,
                multipv2_san=multipv2_san,
                multipv3_san=multipv3_san,
                concept_tags=list(concept_tags) if concept_tags else [],
                white_elo=white_elo,
                black_elo=black_elo,
                result=result,
                fullmove_number=fullmove_number,
            )
            ok, _ = validate_explanation(explanation, sample)
            if not ok:
                dropped += 1
                continue

        yield TrainExample(
            game_id=game_id,
            ply=ply,
            teacher_model=tmodel,
            fen_before=fen_before,
            move_san=move_san,
            mover_color=mover_color,
            classification=classification,
            fullmove_number=fullmove_number,
            explanation=explanation,
            split=assign_split(game_id),
        )

    if dropped:
        logger.info(f"dropped {dropped} rows during dataset build")


# ---- writer ---------------------------------------------------------------


@dataclass
class WriteStats:
    train: int = 0
    val: int = 0
    test: int = 0

    @property
    def total(self) -> int:
        return self.train + self.val + self.test


def write_dataset(
    out_dir: Path | None = None,
    *,
    teacher_model: str | None = None,
    revalidate: bool = True,
) -> WriteStats:
    """Materialize train.jsonl / val.jsonl / test.jsonl on disk.

    MLX-LM's lora trainer expects exactly those filenames inside one
    directory. We default to ``data/datasets/coach-{prompt_version}/``.
    """
    out_dir = out_dir or (settings.data_dir / "datasets" / f"coach-{PROMPT_VERSION}")
    out_dir.mkdir(parents=True, exist_ok=True)

    handles = {
        "train": (out_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (out_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (out_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    stats = WriteStats()
    try:
        for ex in iter_train_examples(
            teacher_model=teacher_model, revalidate=revalidate
        ):
            handles[ex.split].write(ex.to_jsonl() + "\n")
            setattr(stats, ex.split, getattr(stats, ex.split) + 1)
    finally:
        for h in handles.values():
            h.close()

    return stats
