"""Unit tests for the training-dataset builder.

We avoid exercising mlx_lm here — that's covered by the existing smoke test.
This module focuses on the pure-Python data join: teacher table + annotations
+ games → instruction-tuning JSONL.
"""

from __future__ import annotations

import json

import pytest

from packages.ml.training.dataset import (
    STUDENT_SYSTEM,
    TrainExample,
    assign_split,
    build_student_prompt,
    iter_train_examples,
    write_dataset,
)


# ---- prompt construction --------------------------------------------------


def test_build_student_prompt_has_no_concept_tags_or_pv_moves() -> None:
    """The student prompt is intentionally minimal — no engine PVs, no tags.

    That's the whole point of distillation: the student has internalized the
    teacher's reasoning.
    """
    p = build_student_prompt(
        fen_before="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        move_san="e4",
        mover_color="w",
        classification="best",
        fullmove_number=1,
    )
    # Required pieces
    assert STUDENT_SYSTEM[:30] in p  # the system blurb leads
    assert "Position (FEN)" in p
    assert "Move played by White" in p
    assert "Move quality: best" in p
    assert p.rstrip().endswith("Coaching explanation:")
    # Forbidden pieces (those were teacher-side scaffolding only). We allow
    # the word "concept" because the system blurb itself uses it as a goal,
    # but no PV / Stockfish / engine-eval terminology should leak through.
    assert "candidate" not in p.lower()
    assert "engine" not in p.lower()
    assert "stockfish" not in p.lower()
    assert "centipawn" not in p.lower()
    assert "eval" not in p.lower()


def test_build_student_prompt_for_black_uses_dotdot_notation() -> None:
    p = build_student_prompt(
        fen_before="rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
        move_san="e5",
        mover_color="b",
        classification="best",
        fullmove_number=1,
    )
    assert "Move played by Black" in p
    assert ".. e5" in p


# ---- split ----------------------------------------------------------------


def test_assign_split_is_deterministic_and_three_buckets() -> None:
    seen: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for i in range(5_000):
        gid = f"lichess:gid_{i:06x}"
        s1 = assign_split(gid)
        s2 = assign_split(gid)
        assert s1 == s2  # deterministic
        assert s1 in seen
        seen[s1].add(gid)

    n_train = len(seen["train"])
    n_val = len(seen["val"])
    n_test = len(seen["test"])
    # 80/10/10 default split — allow a generous margin since the hash
    # distribution at 5000 samples is approximate.
    assert 0.7 <= n_train / 5_000 <= 0.9
    assert 0.05 <= n_val / 5_000 <= 0.16
    assert 0.05 <= n_test / 5_000 <= 0.16


# ---- DB-backed integration ------------------------------------------------


_PGN = (
    '[Event "?"]\n'
    '[Site "https://lichess.org/coachtst"]\n'
    '[White "a"]\n[Black "b"]\n[Result "1-0"]\n'
    '[WhiteElo "1800"]\n[BlackElo "1800"]\n'
    '[TimeControl "600+0"]\n\n'
    "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 *"
)

_VALID_EXPLANATION = (
    "After Bb5, you pin the knight on c6 to the king-side and prepare to "
    "trade off Black's most active developer. This is the heart of the "
    "Ruy Lopez idea — apply pressure to e5 by removing its defender. "
    "The follow-up plan is to castle quickly and then play c3 and d4."
)


@pytest.fixture
def seeded_db(tmp_db):
    """Insert one game, three annotation rows, and three teacher rows."""
    from packages.ml.data.store import connect

    with connect() as con:
        con.execute(
            """
            INSERT INTO games (id, source, white, black, white_elo, black_elo,
                              result, time_class, ply_count, has_engine_eval, pgn)
            VALUES ('lichess:coachtst','lichess','a','b',1800,1800,'1-0',
                    'rapid', 6, false, ?)
            """,
            [_PGN],
        )
        rows = [
            (1, "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
             "w", "e5", "e7e5", 25, 0, "best"),
            (3, "rnbqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
             "w", "Nc6", "b8c6", 22, 0, "best"),
            (5, "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4",
             "w", "Bb5", "f1b5", 30, 0, "best"),
        ]
        for ply, fen, stm, san, uci, eval_cp, drop, cls in rows:
            con.execute(
                """
                INSERT INTO annotations (
                    game_id, ply, fen, side_to_move, move_san, move_uci,
                    eval_cp, eval_mate, best_pv_san, multipv2_san, multipv3_san,
                    eval_drop_cp, classification, concept_tags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, NULL, ?, ?, [])
                """,
                ["lichess:coachtst", ply, fen, stm, san, uci, eval_cp, san, drop, cls],
            )
        # Two teacher rows (one per ply for plies 1 and 5) to test join.
        for ply in (1, 5):
            con.execute(
                """
                INSERT INTO teacher (
                    game_id, ply, teacher_model, prompt_version,
                    explanation, raw_response, cost_tokens_in, cost_tokens_out
                ) VALUES (?, ?, 'llama-3.3-70b-versatile', 'v1', ?, ?, 200, 60)
                """,
                ["lichess:coachtst", ply, _VALID_EXPLANATION, _VALID_EXPLANATION],
            )

    return tmp_db


def test_iter_train_examples_yields_only_rows_with_teacher_explanations(
    seeded_db,
) -> None:
    examples = list(iter_train_examples())
    plies = sorted(e.ply for e in examples)
    assert plies == [1, 5]  # ply 3 had no teacher row → not yielded


def test_iter_train_examples_drops_invalid_explanations(seeded_db) -> None:
    """Stuff a truncated stub into the teacher table — the join should drop it."""
    from packages.ml.data.store import connect

    with connect() as con:
        # Ply 1 explanation gets corrupted to a truncated stub.
        con.execute(
            """
            UPDATE teacher SET explanation = 'You played e5 which was a great'
            WHERE game_id = 'lichess:coachtst' AND ply = 1
            """
        )

    examples = list(iter_train_examples(revalidate=True))
    assert [e.ply for e in examples] == [5]


def test_iter_train_examples_revalidate_off_keeps_invalid(seeded_db) -> None:
    from packages.ml.data.store import connect

    with connect() as con:
        con.execute(
            "UPDATE teacher SET explanation = 'too short' "
            "WHERE game_id = 'lichess:coachtst' AND ply = 1"
        )

    examples = list(iter_train_examples(revalidate=False))
    assert sorted(e.ply for e in examples) == [1, 5]


def test_to_jsonl_round_trips_and_metadata_present(seeded_db) -> None:
    examples = list(iter_train_examples())
    line = examples[0].to_jsonl()
    obj = json.loads(line)
    assert "prompt" in obj
    assert "completion" in obj
    assert obj["completion"].startswith(" ")  # leading space conventional
    assert obj["meta"]["teacher"] == "llama-3.3-70b-versatile"
    assert obj["meta"]["split"] in ("train", "val", "test")


def test_write_dataset_creates_three_files(seeded_db, tmp_path) -> None:
    out = tmp_path / "ds"
    stats = write_dataset(out_dir=out)
    assert (out / "train.jsonl").exists()
    assert (out / "val.jsonl").exists()
    assert (out / "test.jsonl").exists()
    assert stats.total == 2  # the two valid teacher rows


def test_write_dataset_filters_by_teacher_model(seeded_db, tmp_path) -> None:
    out = tmp_path / "ds"
    stats = write_dataset(out_dir=out, teacher_model="some-other-model")
    assert stats.total == 0
    out2 = tmp_path / "ds2"
    stats2 = write_dataset(out_dir=out2, teacher_model="llama-3.3-70b-versatile")
    assert stats2.total == 2
