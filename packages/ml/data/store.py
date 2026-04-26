"""DuckDB connection management.

A single DuckDB file at ``data/cache/skewerchess.duckdb`` holds every table.
Use :func:`connect` for short-lived connections; the schema is created lazily.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from packages.ml.config import settings
from packages.ml.data.schema import ALL_DDL


def db_path() -> Path:
    p = settings.data_dir / "cache" / "skewerchess.duckdb"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_schema(con: duckdb.DuckDBPyConnection) -> None:
    """Idempotently create all tables and indexes."""
    for ddl in ALL_DDL:
        con.execute(ddl)


@contextmanager
def connect(read_only: bool = False) -> Iterator[duckdb.DuckDBPyConnection]:
    """Yield a connection with the schema initialized.

    Usage::

        with connect() as con:
            con.execute("SELECT count(*) FROM games").fetchone()
    """
    con = duckdb.connect(str(db_path()), read_only=read_only)
    try:
        if not read_only:
            init_schema(con)
        yield con
    finally:
        con.close()


def count_games(source: str | None = None) -> int:
    with connect(read_only=True) as con:
        if source:
            row = con.execute(
                "SELECT count(*) FROM games WHERE source = ?", [source]
            ).fetchone()
        else:
            row = con.execute("SELECT count(*) FROM games").fetchone()
        return int(row[0]) if row else 0
