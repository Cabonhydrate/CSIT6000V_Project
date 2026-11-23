"""Runtime settings. All model endpoints point to a self-hosted service by default."""
from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.environ.get("APP_DATA_DIR", "./data")).resolve()
    owner_id: str = "local-user"  # Explicitly single-user development mode.
    model_base_url: str = os.environ.get("MODEL_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
    model_name: str = os.environ.get("MODEL_NAME", "qwen3:4b")
    model_api_key: str = os.environ.get("MODEL_API_KEY", "local")
    model_timeout: float = float(os.environ.get("MODEL_TIMEOUT", "180"))
    # auto: the standard local :11434/v1 endpoint uses Ollama's native /api/chat,
    # while other endpoints retain OpenAI-compatible /v1/chat/completions.
    model_provider: str = os.environ.get("MODEL_PROVIDER", "auto")
    model_reasoning_effort: str = os.environ.get("MODEL_REASONING_EFFORT", "")
    embedding_model: str = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    enable_rerank: bool = os.environ.get("ENABLE_RERANK", "false").lower() == "true"
    max_upload_mb: int = int(os.environ.get("MAX_UPLOAD_MB", "25"))
    cors_origins: tuple[str, ...] = tuple(os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(","))


settings = Settings()


def prepare_storage() -> None:
    for name in ("", "pdfs", "indexes"):
        (settings.data_dir / name).mkdir(parents=True, exist_ok=True)
