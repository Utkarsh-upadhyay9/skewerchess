"""Centralized config loaded from environment / .env."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime config. Reads from `.env` at the repo root."""

    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # === API keys ===
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    modal_token_id: str = Field(default="", alias="MODAL_TOKEN_ID")
    modal_token_secret: str = Field(default="", alias="MODAL_TOKEN_SECRET")

    # === Chess accounts (optional, used only as personal test set) ===
    lichess_username: str = Field(default="", alias="LICHESS_USERNAME")
    chesscom_username: str = Field(default="", alias="CHESSCOM_USERNAME")

    # === Local paths ===
    stockfish_path: str = Field(default="/opt/homebrew/bin/stockfish", alias="STOCKFISH_PATH")
    data_dir: Path = Field(default=REPO_ROOT / "data", alias="DATA_DIR")
    models_dir: Path = Field(default=REPO_ROOT / "data" / "models", alias="MODELS_DIR")
    hf_home: Path = Field(default=REPO_ROOT / ".hf-cache", alias="HF_HOME")

    # === Pipeline defaults ===
    stockfish_depth: int = Field(default=20, alias="STOCKFISH_DEPTH")
    stockfish_multipv: int = Field(default=3, alias="STOCKFISH_MULTIPV")
    teacher_model: str = Field(default="gemini-2.5-flash", alias="TEACHER_MODEL")
    base_model: str = Field(
        default="mlx-community/Qwen2.5-3B-Instruct-4bit",
        alias="BASE_MODEL",
    )
    max_seq_len: int = Field(default=1024, alias="MAX_SEQ_LEN")
    lora_rank: int = Field(default=8, alias="LORA_RANK")
    lora_alpha: int = Field(default=16, alias="LORA_ALPHA")
    learning_rate: float = Field(default=2e-4, alias="LEARNING_RATE")
    train_iters: int = Field(default=2000, alias="TRAIN_ITERS")


settings = Settings()
