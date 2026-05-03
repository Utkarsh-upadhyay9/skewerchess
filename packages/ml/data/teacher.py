"""Teacher-LLM data generation pipeline.

Given the ``annotations`` table (Stockfish + concept tags), produce
high-quality natural-language coaching explanations for a stratified sample
of positions, written by a frontier free-tier LLM (Gemini 2.5 Flash / Groq
Llama-3.3-70B). Those explanations become the supervision target when we
distill into the on-device 3B student model.

The module is split into four layers:

  1. :class:`TeacherSample`  — what one position's worth of context looks like
  2. :func:`sample_positions` — stratified sampler over the annotations table
  3. :func:`build_prompt` + :func:`humanize_tags` — prompt construction
  4. :class:`TeacherClient` (+ Gemini/Groq impls)  — bounded API access
  5. :func:`generate_explanations` — orchestrator with retries, rate-limiting,
     idempotent writes into the ``teacher`` table

Only the orchestrator touches DuckDB; samplers and clients are pure functions
that are individually unit-testable.
"""

from __future__ import annotations

import io
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Iterable, Protocol

import chess
import chess.pgn
from loguru import logger

from packages.ml.config import settings
from packages.ml.data.store import connect


PROMPT_VERSION = "v1"


# ---- sample dataclass -----------------------------------------------------


@dataclass
class TeacherSample:
    """All context needed to write one coaching explanation."""

    game_id: str
    ply: int
    fen_before: str
    fen_after: str
    move_san: str
    move_uci: str
    mover_color: str  # 'w' or 'b' — the side that just MOVED
    classification: str
    eval_drop_cp: int | None
    eval_cp_after: int | None
    eval_mate_after: int | None
    best_pv_san: str | None
    multipv2_san: str | None
    multipv3_san: str | None
    concept_tags: list[str]
    white_elo: int | None
    black_elo: int | None
    result: str | None
    fullmove_number: int

    @property
    def phase(self) -> str:
        if self.ply <= 16:
            return "opening"
        if self.ply <= 60:
            return "middlegame"
        return "endgame"


# ---- sampling -------------------------------------------------------------


DEFAULT_CLASS_WEIGHTS: dict[str, float] = {
    "blunder": 0.25,
    "mistake": 0.35,
    "inaccuracy": 0.20,
    "great": 0.10,
    "best": 0.10,
}


def sample_positions(
    n_samples: int,
    *,
    classification_weights: dict[str, float] | None = None,
    min_ply: int = 8,
    seed: int = 42,
    exclude_already_done: bool = True,
    teacher_model: str | None = None,
) -> list[TeacherSample]:
    """Return ``n_samples`` annotation rows with full game context.

    Stratified by ``classification`` (heavy on mistakes/blunders), excludes
    book-theory plies, optionally skips positions that already have a teacher
    annotation for ``teacher_model``.
    """
    weights = classification_weights or DEFAULT_CLASS_WEIGHTS
    rng = random.Random(seed)

    out: list[TeacherSample] = []
    with connect() as con:
        for cls, frac in weights.items():
            target = int(round(n_samples * frac))
            if target == 0:
                continue

            params: list = [cls, min_ply]
            already_clause = ""
            if exclude_already_done and teacher_model:
                already_clause = """
                AND NOT EXISTS (
                    SELECT 1 FROM teacher t
                    WHERE t.game_id = a.game_id
                      AND t.ply = a.ply
                      AND t.teacher_model = ?
                      AND t.prompt_version = ?
                )
                """
                params.extend([teacher_model, PROMPT_VERSION])

            sql = f"""
                SELECT
                    a.game_id, a.ply, a.fen, a.side_to_move,
                    a.move_san, a.move_uci,
                    a.eval_cp, a.eval_mate,
                    a.best_pv_san, a.multipv2_san, a.multipv3_san,
                    a.eval_drop_cp, a.classification, a.concept_tags,
                    g.white_elo, g.black_elo, g.result, g.pgn
                FROM annotations a
                JOIN games g ON a.game_id = g.id
                WHERE a.classification = ?
                  AND a.ply >= ?
                  {already_clause}
                ORDER BY hash(a.game_id || cast(a.ply AS VARCHAR))
                LIMIT ?
            """
            params.append(target)

            rows = con.execute(sql, params).fetchall()
            for r in rows:
                sample = _row_to_sample(r)
                if sample is not None:
                    out.append(sample)

    rng.shuffle(out)
    return out


def _row_to_sample(row: tuple) -> TeacherSample | None:
    """Convert a (annotations JOIN games) row into a TeacherSample.

    Replays the PGN to reconstruct the position BEFORE the move was played.
    Returns ``None`` if the PGN can't be parsed past the target ply.
    """
    (
        game_id,
        ply,
        fen_after,
        side_to_move,
        move_san,
        move_uci,
        eval_cp,
        eval_mate,
        best_pv_san,
        multipv2_san,
        multipv3_san,
        eval_drop_cp,
        classification,
        concept_tags,
        white_elo,
        black_elo,
        result,
        pgn,
    ) = row

    fen_before, fullmove_number = _replay_to_ply(pgn, ply)
    if fen_before is None:
        return None

    # The "side that just moved" is the OPPOSITE of the side now to move.
    mover_color = "b" if side_to_move == "w" else "w"

    return TeacherSample(
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


def _replay_to_ply(pgn: str, target_ply: int) -> tuple[str | None, int]:
    """Return (fen_before, fullmove_number_of_the_move) for the move at ``target_ply``.

    ``target_ply`` is 1-indexed half-moves. ``fen_before`` is the FEN of the
    position BEFORE the move was played. Returns (None, 0) on parse failure.
    """
    game = chess.pgn.read_game(io.StringIO(pgn))
    if game is None:
        return None, 0
    board = game.board()
    ply = 0
    for move in game.mainline_moves():
        ply += 1
        if ply == target_ply:
            return board.fen(), board.fullmove_number
        board.push(move)
    return None, 0


# ---- prompt construction --------------------------------------------------


# Map machine tags to short human phrases. Unknown tags pass through verbatim.
TAG_HUMAN: dict[str, str] = {
    "material_advantage_white": "white is up material",
    "material_advantage_black": "black is up material",
    "w_iqp": "white has an isolated d-pawn (IQP)",
    "b_iqp": "black has an isolated d-pawn (IQP)",
    "w_isolated_pawn": "white has an isolated pawn",
    "b_isolated_pawn": "black has an isolated pawn",
    "w_doubled_pawns": "white has doubled pawns",
    "b_doubled_pawns": "black has doubled pawns",
    "w_passed_pawn": "white has a passed pawn",
    "b_passed_pawn": "black has a passed pawn",
    "w_advanced_passed_pawn": "white has an advanced passed pawn",
    "b_advanced_passed_pawn": "black has an advanced passed pawn",
    "w_rook_on_open_file": "white has a rook on an open file",
    "b_rook_on_open_file": "black has a rook on an open file",
    "w_rook_on_seventh": "white has a rook on the 7th rank",
    "b_rook_on_seventh": "black has a rook on the 2nd rank",
    "w_bishop_pair": "white has the bishop pair",
    "b_bishop_pair": "black has the bishop pair",
    "w_knight_outpost": "white has a knight on a strong outpost",
    "b_knight_outpost": "black has a knight on a strong outpost",
    "w_bad_bishop": "white has a bad bishop (own pawns on its color)",
    "b_bad_bishop": "black has a bad bishop (own pawns on its color)",
    "w_castled_kingside": "white is castled kingside",
    "b_castled_kingside": "black is castled kingside",
    "w_castled_queenside": "white is castled queenside",
    "b_castled_queenside": "black is castled queenside",
    "w_king_in_center": "white's king is stuck in the center",
    "b_king_in_center": "black's king is stuck in the center",
    "w_weak_back_rank": "white has a weak back rank",
    "b_weak_back_rank": "black has a weak back rank",
    "opposite_side_castling": "the kings are castled on opposite sides",
    "opponent_has_hanging_piece": "the opponent has an undefended attacked piece",
    "own_piece_hanging": "the side to move has an undefended attacked piece",
    "opponent_has_pinned_piece": "the opponent has a pinned piece",
    "own_piece_pinned": "the side to move has a pinned piece",
    "in_check": "the side to move is in check",
    "checkmate": "checkmate",
    "stalemate": "stalemate",
}


def humanize_tags(tags: Iterable[str]) -> list[str]:
    return [TAG_HUMAN.get(t, t) for t in tags]


def _format_eval(cp: int | None, mate: int | None) -> str:
    """Convert engine eval to a player-readable phrase."""
    if mate is not None:
        sign = "white" if mate > 0 else "black"
        return f"forced mate for {sign} in {abs(mate)}"
    if cp is None:
        return "unclear"
    if abs(cp) < 30:
        return "roughly equal"
    side = "white" if cp > 0 else "black"
    pawns = abs(cp) / 100
    if pawns < 0.7:
        return f"slight edge for {side}"
    if pawns < 1.5:
        return f"clear edge for {side} (≈{pawns:.1f} pawns)"
    if pawns < 3.0:
        return f"large advantage for {side} (≈{pawns:.1f} pawns)"
    return f"winning for {side} (≈{pawns:.1f} pawns)"


def _eval_before(sample: TeacherSample) -> str:
    """Approximate the eval before the move from drop and post-move eval."""
    if sample.eval_mate_after is not None:
        return "(see post-move eval)"
    if sample.eval_cp_after is None:
        return "unclear"
    if sample.eval_drop_cp is None:
        return _format_eval(sample.eval_cp_after, None)

    # eval_drop is in the mover's POV; eval_cp_after is white POV.
    # Reconstruct white-POV before-eval.
    drop_white_pov = sample.eval_drop_cp if sample.mover_color == "w" else -sample.eval_drop_cp
    cp_before = sample.eval_cp_after + drop_white_pov
    return _format_eval(cp_before, None)


SYSTEM_PREAMBLE = (
    "You are a strong chess coach (2500+ FIDE) explaining a single move to a 1700-rated "
    "student. Your job is to make the student understand WHY the move was good or bad, in "
    "plain coaching language. Be concrete, specific, and concise."
)


def build_prompt(sample: TeacherSample) -> str:
    """Construct the full prompt fed to the teacher LLM."""
    color = "White" if sample.mover_color == "w" else "Black"
    move_label = f"{sample.fullmove_number}.{'..' if sample.mover_color == 'b' else ''} {sample.move_san}"

    alts = [a for a in (sample.best_pv_san, sample.multipv2_san, sample.multipv3_san) if a]
    alts_str = ", ".join(alts) if alts else "(none recorded)"

    tags_human = humanize_tags(sample.concept_tags)
    if tags_human:
        tags_str = "; ".join(tags_human)
    else:
        tags_str = "(no notable structural features)"

    eval_after = _format_eval(sample.eval_cp_after, sample.eval_mate_after)
    eval_before = _eval_before(sample)

    elo_str = ""
    if sample.white_elo and sample.black_elo:
        elo_str = f"(White {sample.white_elo}, Black {sample.black_elo})"

    return f"""{SYSTEM_PREAMBLE}

GAME CONTEXT {elo_str}
Phase: {sample.phase}.

POSITION BEFORE THE MOVE (it is {color}'s turn):
FEN: {sample.fen_before}

THE MOVE PLAYED: {move_label}

ENGINE ASSESSMENT
- Position before: {eval_before}
- Position after the move: {eval_after}
- Centipawn loss for {color}: {sample.eval_drop_cp if sample.eval_drop_cp is not None else "?"}
- Move quality: {sample.classification}
- Engine's top candidate moves in this position: {alts_str}

POSITIONAL FEATURES (after the move):
{tags_str}

WRITE A 2-4 SENTENCE COACHING EXPLANATION addressed to {color}.

Strict rules:
- Speak directly to the player ("You played...", "After {sample.move_san}...").
- Name the concept or pattern at play in plain English (open file, weak square, hanging piece, exchange sacrifice, etc.).
- If this is a mistake or blunder, briefly say what {sample.best_pv_san or "the engine's top move"} would have accomplished.
- Do NOT invent moves not listed in the candidate list above.
- Do NOT cite eval numbers — translate them into ideas.
- Do NOT use generic praise/criticism without saying WHY.
- Keep it under 90 words. Plain prose only, no bullet points, no headings.
"""


# ---- output validation ----------------------------------------------------


_SAN_RE = re.compile(
    r"\b(?:O-O-O|O-O|"
    r"(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8](?:=[QRBN])?[+#]?|"
    r"[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|"
    r"[a-h][1-8](?:=[QRBN])?[+#]?))\b"
)


def _legal_san_set(fen: str) -> set[str]:
    board = chess.Board(fen)
    out = set()
    for m in board.legal_moves:
        out.add(board.san(m).rstrip("+#"))
        out.add(board.san(m))
    return out


SENTENCE_TERMINATORS = (".", "!", "?", '"', "”", "’", ".\"", ".'")


def _looks_truncated(text: str) -> bool:
    """Heuristic: the LLM ran out of output tokens mid-sentence."""
    last = text.rstrip()
    if not last:
        return False
    if last.endswith(SENTENCE_TERMINATORS):
        return False
    return True


def validate_explanation(text: str, sample: TeacherSample) -> tuple[bool, str]:
    """Return (ok, reason). Cheap heuristic checks; doesn't try to be perfect."""
    text = text.strip()
    if not text:
        return False, "empty"
    if len(text) < 80:
        return False, "too short"
    if len(text) > 1200:
        return False, "too long"
    if _looks_truncated(text):
        return False, "truncated mid-sentence"
    # Reject if it embeds large eval numbers (we asked it not to).
    if re.search(r"\b\d+\s*cp\b|\b\d+\s*centipawns\b|\(\+?-?\d+\.\d+\)", text):
        return False, "cites raw eval numbers"

    # Verify any cited SAN move is either the played move, in the candidate
    # list, or actually legal in the BEFORE position. This catches halluc-
    # ination of fake tactics.
    cited = set(_SAN_RE.findall(text))
    # Filter out things that look like move-numbers e.g. "e4" at position 0
    if not cited:
        return True, "ok"

    legal = _legal_san_set(sample.fen_before)
    legal_after = _legal_san_set(sample.fen_after)
    allowed = (
        legal
        | legal_after
        | {sample.move_san}
        | {a for a in (sample.best_pv_san, sample.multipv2_san, sample.multipv3_san) if a}
    )
    allowed_normalized = {s.rstrip("+#") for s in allowed} | allowed

    invented = []
    bare_square_re = re.compile(r"^[a-h][1-8]$")
    for m in cited:
        stripped = m.rstrip("+#")
        if m in allowed_normalized or stripped in allowed_normalized:
            continue
        # Bare-pawn SAN like "e4" might appear as e.g. "with e4" in narrative;
        # accept if legal in either FEN.
        if m in legal or m in legal_after:
            continue
        # In coaching prose, bare square names (e2, f7, h5, ...) overwhelmingly
        # refer to the square itself rather than a hypothetical pawn move.
        # Only flag bare-square citations as invented if the regex caught a
        # disambiguator like a piece letter or capture marker (i.e. NOT bare).
        if bare_square_re.match(stripped):
            continue
        invented.append(m)

    if invented:
        return False, f"invented moves: {','.join(sorted(set(invented))[:5])}"
    return True, "ok"


# ---- teacher clients ------------------------------------------------------


@dataclass
class TeacherUsage:
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0


class TeacherClient(Protocol):
    name: str

    def generate(self, prompt: str) -> tuple[str, TeacherUsage]: ...


class _RateLimiter:
    """Simple sliding-window rate limiter on requests-per-minute."""

    def __init__(self, rpm: int):
        self.rpm = rpm
        self._lock = threading.Lock()
        self._stamps: deque[float] = deque()

    def acquire(self) -> None:
        if self.rpm <= 0:
            return
        with self._lock:
            now = time.monotonic()
            cutoff = now - 60.0
            while self._stamps and self._stamps[0] < cutoff:
                self._stamps.popleft()
            if len(self._stamps) >= self.rpm:
                sleep_for = 60.0 - (now - self._stamps[0]) + 0.05
                time.sleep(max(sleep_for, 0))
                now = time.monotonic()
                cutoff = now - 60.0
                while self._stamps and self._stamps[0] < cutoff:
                    self._stamps.popleft()
            self._stamps.append(now)


class GeminiTeacher:
    """Google Gemini 2.5 Flash via the public free tier.

    Important detail for 2.5-series models: by default Gemini reasons in
    "thinking" tokens that count against ``max_output_tokens`` *before* any
    visible response is emitted, which can leave the actual answer
    truncated mid-sentence. For coaching annotations the engine has already
    done the analysis — the LLM only needs to translate facts into prose —
    so we explicitly disable thinking by setting ``thinking_budget=0``.
    """

    def __init__(
        self,
        model: str | None = None,
        rpm: int = 10,
        max_output_tokens: int = 512,
        thinking_budget: int = 0,
    ):
        from google import genai  # lazy import — keeps module importable in tests

        self.name = model or settings.teacher_model or "gemini-2.5-flash"
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._rl = _RateLimiter(rpm)
        self._max_output_tokens = max_output_tokens
        self._thinking_budget = thinking_budget

    def _build_config(self):
        from google.genai import types as genai_types

        kwargs: dict = dict(
            temperature=0.7,
            max_output_tokens=self._max_output_tokens,
            response_mime_type="text/plain",
        )
        # ``thinking_config`` is supported on 2.5-series models; older models
        # silently ignore it. Setting budget=0 disables thinking entirely.
        try:
            kwargs["thinking_config"] = genai_types.ThinkingConfig(
                thinking_budget=self._thinking_budget,
            )
        except Exception:
            pass
        return genai_types.GenerateContentConfig(**kwargs)

    def generate(self, prompt: str) -> tuple[str, TeacherUsage]:
        self._rl.acquire()
        t0 = time.monotonic()
        resp = self._client.models.generate_content(
            model=self.name,
            contents=prompt,
            config=self._build_config(),
        )
        dt = time.monotonic() - t0

        text = (resp.text or "").strip()
        usage_meta = getattr(resp, "usage_metadata", None)
        tokens_out = getattr(usage_meta, "candidates_token_count", 0) or 0
        usage = TeacherUsage(
            tokens_in=getattr(usage_meta, "prompt_token_count", 0) or 0,
            tokens_out=tokens_out,
            seconds=dt,
        )

        # Surface MAX_TOKENS truncation early — this is exactly the failure
        # mode 2.5 Flash exhibits when thinking_budget gobbles up the cap.
        finish = None
        try:
            finish = resp.candidates[0].finish_reason
            if hasattr(finish, "name"):
                finish = finish.name
            finish = str(finish or "").upper()
        except Exception:
            pass
        if finish and "MAX_TOKEN" in finish:
            logger.warning(
                f"[{self.name}] response truncated by max_output_tokens "
                f"(out={tokens_out}); raise --max-output-tokens or check "
                f"thinking_budget"
            )

        return text, usage


class GroqTeacher:
    """Groq-hosted Llama-3.3-70B-Versatile via the free tier."""

    def __init__(self, model: str = "llama-3.3-70b-versatile", rpm: int = 30):
        from groq import Groq  # lazy import

        self.name = model
        self._client = Groq(api_key=settings.groq_api_key)
        self._rl = _RateLimiter(rpm)

    def generate(self, prompt: str) -> tuple[str, TeacherUsage]:
        self._rl.acquire()
        t0 = time.monotonic()
        resp = self._client.chat.completions.create(
            model=self.name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=320,
        )
        dt = time.monotonic() - t0

        text = (resp.choices[0].message.content or "").strip()
        usage = TeacherUsage(
            tokens_in=getattr(resp.usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(resp.usage, "completion_tokens", 0) or 0,
            seconds=dt,
        )
        return text, usage


def make_client(name: str) -> TeacherClient:
    n = name.lower()
    if n.startswith("gemini"):
        return GeminiTeacher(model=name)
    if "llama" in n or n.startswith("groq"):
        model = "llama-3.3-70b-versatile" if n in ("groq", "llama") else name
        return GroqTeacher(model=model)
    raise ValueError(f"unknown teacher: {name!r}")


# ---- orchestrator ---------------------------------------------------------


@dataclass
class GenStats:
    requested: int = 0
    succeeded: int = 0
    failed_validation: int = 0
    failed_api: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    failures_sample: list[str] = field(default_factory=list)


def insert_teacher_row(
    *,
    sample: TeacherSample,
    teacher_model: str,
    explanation: str,
    raw_response: str | None,
    usage: TeacherUsage,
) -> None:
    with connect() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO teacher (
                game_id, ply, teacher_model, prompt_version,
                explanation, raw_response,
                cost_tokens_in, cost_tokens_out
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                sample.game_id,
                sample.ply,
                teacher_model,
                PROMPT_VERSION,
                explanation,
                raw_response,
                usage.tokens_in,
                usage.tokens_out,
            ],
        )


def generate_explanations(
    samples: list[TeacherSample],
    client: TeacherClient,
    *,
    max_retries: int = 2,
    on_progress: Callable[[int, int, GenStats], None] | None = None,
) -> GenStats:
    """Iterate ``samples``, call ``client``, validate, write to ``teacher`` table.

    Idempotent (PK = game_id, ply, teacher_model, prompt_version) — re-running
    on already-done rows simply overwrites them. The orchestrator does NOT
    skip already-done rows; the sampler does that.
    """
    stats = GenStats(requested=len(samples))
    started = time.monotonic()

    for idx, sample in enumerate(samples, start=1):
        prompt = build_prompt(sample)

        last_err: str | None = None
        for attempt in range(max_retries + 1):
            try:
                text, usage = client.generate(prompt)
            except Exception as e:
                last_err = f"api: {type(e).__name__}: {e}"
                wait = 2 ** attempt
                logger.warning(
                    f"[{client.name}] {sample.game_id}#{sample.ply} api error "
                    f"(attempt {attempt + 1}): {last_err}, sleeping {wait}s"
                )
                time.sleep(wait)
                continue

            ok, reason = validate_explanation(text, sample)
            if ok:
                insert_teacher_row(
                    sample=sample,
                    teacher_model=client.name,
                    explanation=text,
                    raw_response=text,
                    usage=usage,
                )
                stats.succeeded += 1
                stats.tokens_in += usage.tokens_in
                stats.tokens_out += usage.tokens_out
                last_err = None
                break
            else:
                last_err = f"validation: {reason}"
                logger.debug(
                    f"[{client.name}] {sample.game_id}#{sample.ply} validation "
                    f"failed (attempt {attempt + 1}): {reason}"
                )

        if last_err is not None:
            if last_err.startswith("validation"):
                stats.failed_validation += 1
            else:
                stats.failed_api += 1
            if len(stats.failures_sample) < 10:
                stats.failures_sample.append(f"{sample.game_id}#{sample.ply}: {last_err}")

        if on_progress is not None:
            on_progress(idx, len(samples), stats)

    stats.seconds = time.monotonic() - started
    return stats
