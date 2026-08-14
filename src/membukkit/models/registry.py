"""Model path resolution.

Resolution order for the default fine-tuned models:
1. Explicit ``ModelConfig.model_dir`` / ``MEMBUKKIT_MODEL_DIR`` env var
2. An explicit existing filesystem path in ``config.encoder`` / ``config.reranker``
3. A repo-checkout ``models/`` directory (developer setups)
4. Auto-download from the HuggingFace Hub (cached under ``$MEMBUKKIT_HOME/models``,
   default ``~/.membukkit/models``)
5. Off-the-shelf base models (``all-mpnet-base-v2`` / ``ms-marco-MiniLM-L-6-v2``)
   so MemBukkit works offline-first-run or before the Hub repos are reachable.
"""
from __future__ import annotations

import os
from pathlib import Path

from membukkit.config import ModelConfig

_ENV_VAR = "MEMBUKKIT_MODEL_DIR"
_DEFAULT_ENCODER = "biencoder_v1"
_DEFAULT_RERANKER = "reranker_v2/model"

_HUB_ENCODER_REPO = "MemseekAI/membukkit-biencoder-v1"
_HUB_RERANKER_REPO = "MemseekAI/membukkit-reranker-v2"
_FALLBACK_ENCODER = "sentence-transformers/all-mpnet-base-v2"
_FALLBACK_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"

def _hub_cache() -> Path:
    home = os.environ.get("MEMBUKKIT_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".membukkit"
    return root / "models"


def _explicit_base(config: ModelConfig) -> Path | None:
    if config.model_dir:
        return Path(config.model_dir).expanduser()
    env = os.environ.get(_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return None


def _checkout_model_dir() -> Path | None:
    root = Path(__file__).resolve().parents[3]
    models = root / "models"
    return models if models.exists() else None


def _existing_path(value: str) -> Path | None:
    path = Path(value).expanduser()
    return path if path.exists() else None


def _prefer_model_subdir(path: Path) -> Path:
    model_dir = path / "model"
    if path.exists() and model_dir.exists():
        return model_dir
    return path


def _hub_download(repo_id: str) -> str | None:
    """Snapshot-download a model repo, cached locally. None on any failure."""
    target = _hub_cache() / repo_id.replace("/", "__")
    if (target / "config.json").exists():
        return str(target)
    try:
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=repo_id, local_dir=str(target))
        return str(path)
    except Exception:
        return None


def resolve_encoder_path(config: ModelConfig) -> str:
    base = _explicit_base(config)
    if base:
        return str(base / config.encoder)

    explicit_path = _existing_path(config.encoder)
    if explicit_path:
        return str(explicit_path)

    if config.encoder != _DEFAULT_ENCODER:
        # A non-default name that isn't a local path: assume it's a HF model id
        # and let sentence-transformers resolve it.
        return config.encoder

    checkout = _checkout_model_dir()
    if checkout and (checkout / _DEFAULT_ENCODER).exists():
        return str(checkout / _DEFAULT_ENCODER)

    hub = _hub_download(_HUB_ENCODER_REPO)
    if hub:
        return hub

    return _FALLBACK_ENCODER


def resolve_reranker_path(config: ModelConfig) -> str:
    base = _explicit_base(config)
    if base:
        return str(_prefer_model_subdir(base / config.reranker))

    explicit_path = _existing_path(config.reranker)
    if explicit_path:
        return str(_prefer_model_subdir(explicit_path))

    if config.reranker != _DEFAULT_RERANKER:
        return config.reranker

    checkout = _checkout_model_dir()
    if checkout and (checkout / _DEFAULT_RERANKER).exists():
        return str(checkout / _DEFAULT_RERANKER)

    hub = _hub_download(_HUB_RERANKER_REPO)
    if hub:
        return _prefer_model_subdir(Path(hub)).as_posix()

    return _FALLBACK_RERANKER
