"""Stream the Lichess open database and ingest filtered games into DuckDB.

The Lichess open database (https://database.lichess.org) publishes one
zstandard-compressed PGN per month. Files are huge (~30-200 GB compressed) but
we never download more than we need: we stream the response, pipe it through a
streaming zstd decoder, parse PGNs one at a time with python-chess, filter by
rating + time control + presence of engine evaluations, and stop as soon as we
have the requested number of games.

Memory footprint stays bounded (a few MB) regardless of N.

Public entry point: :func:`ingest_month`.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterator

import chess.pgn
import httpx
import zstandard as zstd
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from packages.ml.data.store import connect

LICHESS_DB_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_{year:04d}-{month:02d}.pgn.zst"
)

USER_AGENT = "skewerchess/0.1 (+https://github.com/utkarsh-95/skewerchess)"

# Time-control classification follows the Lichess convention.
# https://lichess.org/faq#time-controls
def classify_time_control(tc: str) -> tuple[str, int | None, int | None]:
    """Return ``(class, base_seconds, increment)`` for a 'base+inc' TC string.

    Rules (estimated game length = base + 40 * increment):
      < 30 s  → bullet
      < 180 s → blitz
      < 480 s → rapid
      else    → classical
    """
    if not tc or tc == "-":
        return "correspondence", None, None
    m = re.fullmatch(r"(\d+)\+(\d+)", tc)
    if not m:
        return "unknown", None, None
    base, inc = int(m.group(1)), int(m.group(2))
    estimated = base + 40 * inc
    if estimated < 30:
        cls = "ultra_bullet"
    elif estimated < 180:
        cls = "bullet"
    elif estimated < 480:
        cls = "blitz"
    elif estimated < 1500:
        cls = "rapid"
    else:
        cls = "classical"
    return cls, base, inc


@dataclass(frozen=True)
class IngestFilter:
    rating_min: int = 1500
    rating_max: int = 2400
    time_classes: tuple[str, ...] = ("rapid", "classical")
    require_engine_eval: bool = True
    min_ply_count: int = 20

    def matches(self, headers: dict[str, str], has_eval: bool, ply_count: int) -> bool:
        try:
            we = int(headers.get("WhiteElo", "0") or 0)
            be = int(headers.get("BlackElo", "0") or 0)
        except ValueError:
            return False
        if we < self.rating_min or we > self.rating_max:
            return False
        if be < self.rating_min or be > self.rating_max:
            return False

        cls, _, _ = classify_time_control(headers.get("TimeControl", ""))
        if cls not in self.time_classes:
            return False

        if self.require_engine_eval and not has_eval:
            return False

        if ply_count < self.min_ply_count:
            return False
        return True


# ---- streaming plumbing ---------------------------------------------------


class _HttpxByteReader(io.RawIOBase):
    """Adapt an httpx streaming response to the file-like API zstd wants."""

    def __init__(self, response: httpx.Response, chunk_bytes: int = 1 << 20) -> None:
        self._iter = response.iter_bytes(chunk_size=chunk_bytes)
        self._buf = b""
        self._eof = False
        self._bytes_read = 0

    def readable(self) -> bool:
        return True

    @property
    def bytes_read(self) -> int:
        return self._bytes_read

    def readinto(self, b) -> int:
        n = len(b)
        while len(self._buf) < n and not self._eof:
            try:
                self._buf += next(self._iter)
            except StopIteration:
                self._eof = True
                break
        chunk = self._buf[:n]
        self._buf = self._buf[len(chunk):]
        b[: len(chunk)] = chunk
        self._bytes_read += len(chunk)
        return len(chunk)


def _stream_pgn_text(url: str, http_timeout: float = 60.0) -> Iterator[str]:
    """Yield a single ``TextIOWrapper`` over the streaming-decompressed PGN text.

    Yields exactly one item — wrap is so callers can ``with`` the returned
    object via :func:`open_lichess_stream` below.
    """
    raise NotImplementedError("use open_lichess_stream")


class _LichessStream:
    """Context manager that exposes a text stream over the Lichess monthly file."""

    def __init__(self, year: int, month: int, http_timeout: float = 60.0) -> None:
        self.url = LICHESS_DB_URL.format(year=year, month=month)
        self._timeout = http_timeout
        self._client: httpx.Client | None = None
        self._response: httpx.Response | None = None
        self._byte_reader: _HttpxByteReader | None = None
        self._zstd_reader = None
        self._text_stream: io.TextIOWrapper | None = None

    def __enter__(self) -> io.TextIOWrapper:
        self._client = httpx.Client(
            timeout=self._timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        ctx = self._client.stream("GET", self.url)
        self._response = ctx.__enter__()
        self._stream_ctx = ctx
        if self._response.status_code == 404:
            raise FileNotFoundError(
                f"Lichess database not yet available for that month: {self.url}"
            )
        self._response.raise_for_status()

        self._byte_reader = _HttpxByteReader(self._response)
        self._zstd_reader = zstd.ZstdDecompressor().stream_reader(
            self._byte_reader, read_size=1 << 16
        )
        self._text_stream = io.TextIOWrapper(self._zstd_reader, encoding="utf-8", errors="replace")
        return self._text_stream

    def __exit__(self, *exc) -> None:
        try:
            if self._text_stream is not None:
                self._text_stream.detach()
        except Exception:
            pass
        try:
            if self._zstd_reader is not None:
                self._zstd_reader.close()
        except Exception:
            pass
        try:
            if hasattr(self, "_stream_ctx"):
                self._stream_ctx.__exit__(*exc)
        except Exception:
            pass
        try:
            if self._client is not None:
                self._client.close()
        except Exception:
            pass

    @property
    def bytes_downloaded(self) -> int:
        return self._byte_reader.bytes_read if self._byte_reader else 0


# ---- public API -----------------------------------------------------------


_LICHESS_GAME_ID_RE = re.compile(r"lichess\.org/(?P<id>[A-Za-z0-9]{8})")


def _extract_lichess_id(headers: dict[str, str]) -> str | None:
    site = headers.get("Site", "")
    m = _LICHESS_GAME_ID_RE.search(site)
    return m.group("id") if m else None


def _has_engine_eval(game: chess.pgn.Game) -> bool:
    node = game.next()
    return bool(node and "%eval" in (node.comment or ""))


def _ply_count(game: chess.pgn.Game) -> int:
    n = 0
    for _ in game.mainline_moves():
        n += 1
    return n


def _serialize_game(game: chess.pgn.Game) -> str:
    """Return the canonical PGN text for a single game."""
    exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=True)
    return game.accept(exporter)


def ingest_month(
    year: int,
    month: int,
    max_games: int = 5000,
    filt: IngestFilter | None = None,
    show_progress: bool = True,
) -> dict[str, int]:
    """Stream the given Lichess month, filter, and insert into ``games``.

    Returns a stats dict. Idempotent — games already in the table are skipped.
    """
    filt = filt or IngestFilter()

    stats = {"scanned": 0, "filtered_in": 0, "inserted": 0, "skipped_duplicate": 0}

    with connect() as con, _LichessStream(year, month) as text_stream:
        existing = {
            row[0]
            for row in con.execute("SELECT id FROM games WHERE source = 'lichess'").fetchall()
        }

        progress_cols = (
            SpinnerColumn(),
            TextColumn("[bold]ingesting[/bold] {task.description}"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
        )

        with Progress(*progress_cols, transient=False, disable=not show_progress) as bar:
            task = bar.add_task(
                f"lichess {year}-{month:02d}", total=max_games
            )

            while stats["inserted"] < max_games:
                try:
                    game = chess.pgn.read_game(text_stream)
                except Exception as e:
                    logger.warning(f"PGN parse error after {stats['scanned']}: {e}")
                    continue
                if game is None:
                    break

                stats["scanned"] += 1
                headers = dict(game.headers)
                ply = _ply_count(game)
                has_eval = _has_engine_eval(game)

                if not filt.matches(headers, has_eval, ply):
                    continue
                stats["filtered_in"] += 1

                lichess_id = _extract_lichess_id(headers)
                if not lichess_id:
                    continue
                game_id = f"lichess:{lichess_id}"

                if game_id in existing:
                    stats["skipped_duplicate"] += 1
                    continue

                tc = headers.get("TimeControl", "")
                cls, base_s, inc = classify_time_control(tc)

                con.execute(
                    """
                    INSERT INTO games (
                        id, source, site,
                        white, black, white_elo, black_elo, white_title, black_title,
                        result, termination,
                        eco, opening,
                        time_control, time_class, base_seconds, increment,
                        ply_count, has_engine_eval, utc_date, pgn
                    ) VALUES (
                        ?, 'lichess', ?,
                        ?, ?, ?, ?, ?, ?,
                        ?, ?,
                        ?, ?,
                        ?, ?, ?, ?,
                        ?, ?, ?, ?
                    )
                    """,
                    [
                        game_id,
                        headers.get("Site"),
                        headers.get("White"),
                        headers.get("Black"),
                        int(headers.get("WhiteElo") or 0) or None,
                        int(headers.get("BlackElo") or 0) or None,
                        headers.get("WhiteTitle"),
                        headers.get("BlackTitle"),
                        headers.get("Result"),
                        headers.get("Termination"),
                        headers.get("ECO"),
                        headers.get("Opening"),
                        tc,
                        cls,
                        base_s,
                        inc,
                        ply,
                        has_eval,
                        headers.get("UTCDate"),
                        _serialize_game(game),
                    ],
                )
                existing.add(game_id)
                stats["inserted"] += 1
                bar.update(task, completed=stats["inserted"])

    return stats
