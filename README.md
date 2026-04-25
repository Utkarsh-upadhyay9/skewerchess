# skewerchess

> An AI chess coach that explains your games like a 2500-rated trainer would. A small open model, distilled from a frontier LLM teacher, running entirely on Apple Silicon.

## What it does

Paste a PGN, get back a full post-game breakdown — not "blunder on move 23" like a raw engine, but **why** it was a mistake, what the right idea was, and what concept you're missing (e.g. *"You're not respecting open file control after the e4–e5 trade"*). Across multiple games, it surfaces your recurring weaknesses and proposes a personalized study plan.

## Why this is technically interesting

- **Distilled frontier knowledge into a 3B local model.** Gemini 2.5 Flash generates ~15K coaching annotations on real positions; a Qwen2.5-3B base is QLoRA fine-tuned on Apple Silicon (MLX-LM) to match the teacher on a blind chess-instruction eval.
- **Engine-grounded explanations.** Every output is conditioned on Stockfish 16 multipv-3 evaluations + a hand-built positional concept tagger (40+ themes: IQP, minority attack, weak back rank, etc.) so the model never hallucinates the chess facts — only the prose around them.
- **Cross-game weakness clustering.** Mistake positions across a user's history are embedded and aggregated by concept tags + eval delta to produce a personalized "where you bleed centipawns" report.
- **Rigorous human eval.** 50 positions × 3 sources (Stockfish-only, Gemini, our model) blind-rated by 2000+ rated players on a 1–5 instructional-quality scale.

## Architecture

```
PGN  →  python-chess parser  →  Stockfish (depth 20, multipv 3)
                                       ↓
                           rule-based concept tagger
                                       ↓
                       (FEN, eval, top-3 lines, tags)
                                       ↓
                     Distilled 3B coach LLM (MLX, 4-bit)
                                       ↓
                              Coaching explanation
```

## Stack

- **ML/training:** MLX, MLX-LM, transformers, peft (interop), Qwen2.5-3B base
- **Teacher:** Gemini 2.5 Flash (primary), Llama-3.3-70B via Groq (backup)
- **Engine:** Stockfish 16, python-chess
- **Backend:** FastAPI on Fly.io
- **Frontend:** Next.js 15 + Tailwind + shadcn/ui + react-chessground
- **Inference in prod:** Modal (free tier GPU)

## Project layout

```
skewerchess/
├── packages/ml/        # data pipeline, training, inference
├── apps/api/           # FastAPI backend (Week 3)
├── apps/web/           # Next.js frontend (Week 3)
├── data/               # gitignored datasets, checkpoints, caches
├── notebooks/          # exploratory analysis
├── scripts/            # one-off CLI helpers
└── tests/
```

## Setup

See [`scripts/setup.sh`](./scripts/setup.sh). One command, one Mac password prompt.

```bash
./scripts/setup.sh
```

After it finishes, copy the env template and fill in your keys:

```bash
cp .env.example .env
# edit .env with your GEMINI_API_KEY, GROQ_API_KEY, HF_TOKEN, etc.
```

Then verify everything is wired up:

```bash
uv run python scripts/smoke_test.py
```

## 4-week MVP timeline

| Week | Focus |
|---|---|
| 1 | Data ingestion + Stockfish pipeline + concept tagger + 15K teacher dataset |
| 2 | QLoRA fine-tuning on MLX + iteration + blind eval |
| 3 | FastAPI backend + Next.js frontend + weakness profiler |
| 4 | Deploy + launch + writeup |

## License

MIT
