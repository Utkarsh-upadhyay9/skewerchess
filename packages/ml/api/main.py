"""FastAPI application entrypoint."""

from __future__ import annotations

import json

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from packages.ml.config import settings
from packages.ml.api.coach_service import generate_coaching
from packages.ml.api.pgn_lichess import analyze_pgn_mainline
from packages.ml.api.schemas import (
    AnalyzedTurn,
    LichessGameItem,
    LichessGamesResponse,
    PgnAnalyzeRequest,
    PgnAnalyzeResponse,
    PositionCoachRequest,
    PositionCoachResponse,
)


def _cors_origins() -> list[str]:
    raw = (settings.cors_origins or "*").strip()
    if raw == "*":
        return ["*"]
    return [o.strip() for o in raw.split(",") if o.strip()]


app = FastAPI(
    title="skewerchess API",
    description="Chess coach explanations + Lichess export helpers.",
    version="0.1.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "coach_backend": settings.coach_backend}


@app.post("/api/coach/position", response_model=PositionCoachResponse)
async def coach_position(body: PositionCoachRequest) -> PositionCoachResponse:
    text = await generate_coaching(
        fen_before=body.fen_before,
        move_san=body.move_san,
        mover_color=body.mover_color,
        classification=body.classification,
        fullmove_number=body.fullmove_number,
    )
    return PositionCoachResponse(explanation=text, coach_backend=settings.coach_backend)


@app.post("/api/coach/pgn", response_model=PgnAnalyzeResponse)
async def coach_pgn(body: PgnAnalyzeRequest) -> PgnAnalyzeResponse:
    analyzed = analyze_pgn_mainline(body.pgn, max_plies=body.max_plies)
    skipped = 0
    turns: list[AnalyzedTurn] = []
    for row in analyzed:
        if row.classification == "unknown":
            skipped += 1
            if not body.explain_each_move:
                continue
            turns.append(
                AnalyzedTurn(
                    ply=row.ply,
                    move_san=row.move_san,
                    mover_color=row.mover_color,
                    classification=row.classification,
                    eval_drop_cp=row.eval_drop_cp,
                    explanation="(no Lichess eval in PGN — export with evals=true)",
                )
            )
            continue

        expl = ""
        if body.explain_each_move:
            # Full-move number: white uses ceil(ply/2) roughly
            fm = (row.ply + 1) // 2 if row.mover_color == "w" else row.ply // 2
            expl = await generate_coaching(
                fen_before=row.fen_before,
                move_san=row.move_san,
                mover_color=row.mover_color,
                classification=row.classification,
                fullmove_number=max(1, fm),
            )

        turns.append(
            AnalyzedTurn(
                ply=row.ply,
                move_san=row.move_san,
                mover_color=row.mover_color,
                classification=row.classification,
                eval_drop_cp=row.eval_drop_cp,
                explanation=expl,
            )
        )

    return PgnAnalyzeResponse(
        turns=turns,
        coach_backend=settings.coach_backend,
        skipped_unknown=skipped,
    )


@app.get("/api/lichess/games/{username}", response_model=LichessGamesResponse)
async def lichess_user_games(
    username: str,
    max_games: int = 5,
    perf_type: str = "rapid,blitz",
) -> LichessGamesResponse:
    """Fetch recent games (PGN inside JSON) from the public Lichess API."""
    url = f"https://lichess.org/api/games/user/{username}"
    params = {
        "max": min(max(max_games, 1), 30),
        "pgnInJson": "true",
        "moves": "true",
        "tags": "true",
        "opening": "true",
        "evals": "true",
        "perfType": perf_type,
    }
    headers = {"Accept": "application/x-ndjson"}

    async with httpx.AsyncClient(timeout=45.0) as client:
        r = await client.get(url, params=params, headers=headers)
        if r.status_code == 404:
            raise HTTPException(status_code=404, detail="user or games not found")
        if r.status_code >= 400:
            raise HTTPException(
                status_code=r.status_code,
                detail=r.text[:500] if r.text else "lichess API error",
            )

    games: list[LichessGameItem] = []
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        gid = obj.get("id")
        pgn = obj.get("pgn")
        if not gid or not pgn:
            continue
        games.append(
            LichessGameItem(
                id=str(gid),
                pgn=str(pgn),
                rated=obj.get("rated"),
                speed=obj.get("speed"),
            )
        )

    return LichessGamesResponse(username=username, games=games)


def run() -> None:
    """CLI entrypoint: ``uv run skewer-api``."""
    import uvicorn

    uvicorn.run(
        "packages.ml.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
