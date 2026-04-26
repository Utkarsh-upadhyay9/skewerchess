"""DuckDB schema for the skewerchess local data warehouse.

Tables:
  games        — raw ingested games (PGN + headers)
  annotations  — per-move Stockfish evals + concept tags (filled in Day 3)
  teacher      — per-position natural-language coaching annotations (filled in Day 4)

We keep everything in a single DuckDB file at `data/cache/skewerchess.duckdb`.
DuckDB gives us cheap SQL over the raw data plus zero-config columnar storage.
"""

from __future__ import annotations

GAMES_DDL = """
CREATE TABLE IF NOT EXISTS games (
    id              VARCHAR PRIMARY KEY,           -- 'lichess:abc12345' or 'chesscom:...'
    source          VARCHAR NOT NULL,              -- 'lichess' | 'chesscom'
    site            VARCHAR,                       -- e.g. 'https://lichess.org/abc12345'

    white           VARCHAR,
    black           VARCHAR,
    white_elo       INTEGER,
    black_elo       INTEGER,
    white_title     VARCHAR,                       -- GM/IM/FM/etc., often null
    black_title     VARCHAR,

    result          VARCHAR,                       -- '1-0' | '0-1' | '1/2-1/2' | '*'
    termination     VARCHAR,                       -- 'Normal' | 'Time forfeit' | etc.

    eco             VARCHAR,                       -- e.g. 'B12'
    opening         VARCHAR,                       -- e.g. 'Caro-Kann Defense: Advance Variation'

    time_control    VARCHAR,                       -- raw header e.g. '600+5'
    time_class      VARCHAR,                       -- 'bullet'|'blitz'|'rapid'|'classical'|'correspondence'
    base_seconds    INTEGER,
    increment       INTEGER,

    ply_count       INTEGER,                       -- number of half-moves
    has_engine_eval BOOLEAN,                       -- true if Lichess included %eval comments
    utc_date        VARCHAR,                       -- e.g. '2024.11.01'
    pgn             TEXT NOT NULL,

    ingested_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_games_elo        ON games(white_elo, black_elo);
CREATE INDEX IF NOT EXISTS idx_games_time_class ON games(time_class);
CREATE INDEX IF NOT EXISTS idx_games_source     ON games(source);
"""

# Per-move Stockfish evaluations + rule-based concept tags.
# One row = one position reached after a half-move.
ANNOTATIONS_DDL = """
CREATE TABLE IF NOT EXISTS annotations (
    game_id          VARCHAR NOT NULL,
    ply              INTEGER NOT NULL,             -- 1-indexed half-move number
    fen              VARCHAR NOT NULL,             -- position AFTER the move
    side_to_move     VARCHAR NOT NULL,             -- 'w' | 'b' for the side about to move next
    move_san         VARCHAR NOT NULL,             -- the move just played, in SAN
    move_uci         VARCHAR NOT NULL,
    eval_cp          INTEGER,                      -- engine eval after the move, white POV, centipawns; NULL for mate
    eval_mate        INTEGER,                      -- mate-in-N, signed; NULL when not mate
    best_pv_san      VARCHAR,                      -- engine's preferred continuation, SAN
    multipv2_san     VARCHAR,
    multipv3_san     VARCHAR,
    eval_drop_cp     INTEGER,                      -- centipawn drop vs. engine best (player POV)
    classification   VARCHAR,                      -- 'best'|'great'|'good'|'inaccuracy'|'mistake'|'blunder'
    concept_tags     VARCHAR[],                    -- e.g. ['IQP','open_file','hanging_piece']
    annotated_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, ply)
);

CREATE INDEX IF NOT EXISTS idx_annot_classification ON annotations(classification);
"""

# Teacher-LLM-generated coaching explanations for the most instructive positions.
TEACHER_DDL = """
CREATE TABLE IF NOT EXISTS teacher (
    game_id        VARCHAR NOT NULL,
    ply            INTEGER NOT NULL,
    teacher_model  VARCHAR NOT NULL,               -- 'gemini-2.5-flash' | 'llama-3.3-70b-versatile'
    prompt_version VARCHAR NOT NULL,               -- bumped when we change the prompt
    explanation    TEXT NOT NULL,
    raw_response   TEXT,
    cost_tokens_in  INTEGER,
    cost_tokens_out INTEGER,
    generated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (game_id, ply, teacher_model, prompt_version)
);
"""

ALL_DDL = [GAMES_DDL, ANNOTATIONS_DDL, TEACHER_DDL]
