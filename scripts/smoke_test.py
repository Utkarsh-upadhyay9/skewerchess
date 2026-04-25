"""
Day-1 smoke test for skewerchess.

Verifies that every critical dependency is installed, importable, and reachable:
  1. python-chess can parse a PGN
  2. Stockfish binary is callable and returns a move
  3. MLX is installed and can do tensor math on the Apple GPU
  4. mlx-lm can be imported (model download is deferred to Day 2)
  5. Gemini API key works (only if GEMINI_API_KEY is set)
  6. Groq API key works (only if GROQ_API_KEY is set)

Run:  uv run python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

GREEN = "\033[0;32m"
RED = "\033[0;31m"
YELLOW = "\033[1;33m"
BOLD = "\033[1m"
NC = "\033[0m"


class Check:
    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = False
        self.detail = ""

    def __enter__(self) -> "Check":
        print(f"  {BOLD}{self.name}{NC} ... ", end="", flush=True)
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed = time.time() - self.start
        if exc is None:
            print(f"{GREEN}OK{NC} ({elapsed:.2f}s) {self.detail}")
            self.passed = True
        else:
            print(f"{RED}FAIL{NC} ({elapsed:.2f}s)\n      {exc}")
            self.passed = False
        return True

    def info(self, msg: str) -> None:
        self.detail = f"— {msg}"


CHECKS: list[Check] = []


def section(title: str) -> None:
    print(f"\n{BOLD}{title}{NC}")


def main() -> int:
    print(f"{BOLD}skewerchess Day-1 smoke test{NC}")
    print(f"  repo: {REPO_ROOT}")

    section("1. Core chess libs")
    chk = Check("python-chess parses PGN")
    CHECKS.append(chk)
    with chk:
        import chess
        import chess.pgn
        import io
        sample_pgn = (
            '[Event "Smoke test"]\n[White "A"]\n[Black "B"]\n[Result "*"]\n\n'
            "1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 *"
        )
        game = chess.pgn.read_game(io.StringIO(sample_pgn))
        moves = list(game.mainline_moves()) if game else []
        chk.info(f"parsed {len(moves)} moves, version {chess.__version__}")

    chk = Check("Stockfish binary at $STOCKFISH_PATH")
    CHECKS.append(chk)
    with chk:
        from stockfish import Stockfish
        sf_path = os.getenv("STOCKFISH_PATH", "/opt/homebrew/bin/stockfish")
        sf = Stockfish(path=sf_path, depth=10)
        sf.set_position([])
        best = sf.get_best_move()
        chk.info(f"opening best move: {best}")

    section("2. Apple Silicon ML stack")
    chk = Check("MLX tensor math on GPU")
    CHECKS.append(chk)
    with chk:
        import mlx.core as mx
        a = mx.random.normal(shape=(1024, 1024))
        b = mx.random.normal(shape=(1024, 1024))
        c = a @ b
        mx.eval(c)
        chk.info(f"matmul ok, mlx {mx.__version__ if hasattr(mx, '__version__') else 'installed'}, device: {mx.default_device()}")

    chk = Check("mlx-lm importable")
    CHECKS.append(chk)
    with chk:
        import mlx_lm
        chk.info(f"version {getattr(mlx_lm, '__version__', 'installed')}")

    section("3. Teacher LLM APIs (only checked if key is set)")
    if os.getenv("GEMINI_API_KEY"):
        chk = Check("Gemini API reachable")
        CHECKS.append(chk)
        with chk:
            from google import genai
            client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
            resp = client.models.generate_content(
                model=os.getenv("TEACHER_MODEL", "gemini-2.5-flash"),
                contents="Reply with the single word: ready",
            )
            chk.info(f"reply: {resp.text.strip()[:40]!r}")
    else:
        print(f"  {YELLOW}Gemini API skipped — set GEMINI_API_KEY in .env{NC}")

    if os.getenv("GROQ_API_KEY"):
        chk = Check("Groq API reachable")
        CHECKS.append(chk)
        with chk:
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "Reply with the single word: ready"}],
                max_tokens=10,
            )
            chk.info(f"reply: {resp.choices[0].message.content.strip()[:40]!r}")
    else:
        print(f"  {YELLOW}Groq API skipped — set GROQ_API_KEY in .env{NC}")

    section("4. Hugging Face")
    if os.getenv("HF_TOKEN"):
        chk = Check("Hugging Face Hub auth")
        CHECKS.append(chk)
        with chk:
            from huggingface_hub import whoami
            info = whoami(token=os.environ["HF_TOKEN"])
            chk.info(f"logged in as {info['name']}")
    else:
        print(f"  {YELLOW}HF check skipped — set HF_TOKEN in .env{NC}")

    print()
    failed = [c for c in CHECKS if not c.passed]
    if failed:
        print(f"{RED}{BOLD}{len(failed)} check(s) failed.{NC}")
        return 1
    print(f"{GREEN}{BOLD}All {len(CHECKS)} checks passed. Day 1 unblocked.{NC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
