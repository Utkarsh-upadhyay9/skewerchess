"""QLoRA fine-tuning driver for the on-device chess-coach student.

Delegates the actual training loop to ``mlx_lm`` — it's already tuned for
Apple Silicon, supports 4-bit base models, and writes adapter checkpoints
in a format we can later fuse and ship. We just orchestrate paths, hyper-
parameters and logging.

The pipeline:

  1. ``packages.ml.training.dataset.write_dataset`` materializes the JSONL
     splits expected by ``mlx_lm.lora`` (train.jsonl, val.jsonl, test.jsonl).
  2. We invoke ``mlx_lm.lora`` via subprocess with the QLoRA hyperparameters
     defined in ``settings`` (rank 8, alpha 16, lr 2e-4, etc.). Subprocess
     gives us clean isolation, real-time stdout, and Ctrl-C safety.
  3. Adapters land at ``data/models/lora/coach-{prompt_version}/`` so
     subsequent fuse / inference can find them deterministically.

This module is import-safe even on machines without MLX (it doesn't actually
``import mlx`` at module load time).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from packages.ml.config import settings
from packages.ml.data.teacher import PROMPT_VERSION


# ---- training config ------------------------------------------------------


@dataclass
class TrainConfig:
    base_model: str
    data_dir: Path
    adapters_dir: Path
    iters: int
    batch_size: int
    learning_rate: float
    lora_rank: int
    lora_alpha: int
    max_seq_len: int
    grad_checkpoint: bool
    val_batches: int
    seed: int
    save_every: int

    @classmethod
    def default(
        cls,
        *,
        prompt_version: str = PROMPT_VERSION,
        iters: int | None = None,
    ) -> "TrainConfig":
        return cls(
            base_model=settings.base_model,
            data_dir=settings.data_dir / "datasets" / f"coach-{prompt_version}",
            adapters_dir=settings.models_dir / "lora" / f"coach-{prompt_version}",
            iters=iters or settings.train_iters,
            # Bare-minimum batch + seq for 16GB Macs running a 3B QLoRA model.
            batch_size=1,
            learning_rate=settings.learning_rate,
            lora_rank=settings.lora_rank,
            lora_alpha=settings.lora_alpha,
            max_seq_len=settings.max_seq_len,
            # Trades compute for memory — required on 16GB.
            grad_checkpoint=True,
            val_batches=20,
            seed=42,
            save_every=200,
        )


# ---- assertions -----------------------------------------------------------


def assert_ready(cfg: TrainConfig) -> None:
    """Fail fast with actionable messages before launching mlx_lm."""
    if not cfg.data_dir.exists():
        raise SystemExit(
            f"dataset directory missing: {cfg.data_dir}\n"
            f"run `skewer train build-dataset` first."
        )
    train = cfg.data_dir / "train.jsonl"
    val = cfg.data_dir / "val.jsonl"
    if not train.exists() or train.stat().st_size == 0:
        raise SystemExit(f"empty train file at {train}")
    if not val.exists() or val.stat().st_size == 0:
        raise SystemExit(f"empty val file at {val}")
    if shutil.which("python") is None:
        raise SystemExit("python not on PATH (?)")


def _count_lines(p: Path) -> int:
    with p.open("rb") as f:
        return sum(1 for _ in f)


def dataset_summary(cfg: TrainConfig) -> dict[str, int]:
    return {
        "train": _count_lines(cfg.data_dir / "train.jsonl"),
        "val": _count_lines(cfg.data_dir / "val.jsonl"),
        "test": _count_lines(cfg.data_dir / "test.jsonl"),
    }


# ---- launch ---------------------------------------------------------------


def build_command(cfg: TrainConfig) -> list[str]:
    """Render the ``mlx_lm.lora`` CLI invocation for this run."""
    cfg.adapters_dir.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        cfg.base_model,
        "--train",
        "--data",
        str(cfg.data_dir),
        "--adapter-path",
        str(cfg.adapters_dir),
        "--iters",
        str(cfg.iters),
        "--batch-size",
        str(cfg.batch_size),
        "--learning-rate",
        f"{cfg.learning_rate:g}",
        "--lora-layers",
        "16",
        "--num-layers",
        "16",
        "--seed",
        str(cfg.seed),
        "--save-every",
        str(cfg.save_every),
        "--val-batches",
        str(cfg.val_batches),
        "--max-seq-length",
        str(cfg.max_seq_len),
        *(["--grad-checkpoint"] if cfg.grad_checkpoint else []),
    ]


def run_training(cfg: TrainConfig) -> int:
    """Spawn mlx_lm and stream its output. Returns the subprocess exit code."""
    assert_ready(cfg)
    cmd = build_command(cfg)
    logger.info("launching: " + " ".join(cmd))

    env = os.environ.copy()
    # Ensure HF cache lives where the user has space.
    env.setdefault("HF_HOME", str(settings.hf_home))
    env.setdefault("MLX_METAL_DEBUG", "0")

    started = time.monotonic()
    proc = subprocess.Popen(
        cmd, env=env, stdout=sys.stdout, stderr=sys.stderr, text=True
    )
    rc = proc.wait()
    logger.info(
        f"mlx_lm.lora exited with code {rc} after {time.monotonic() - started:.1f}s"
    )
    return rc
