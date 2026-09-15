"""Shared Microsoft Foundry Local model setup."""

from __future__ import annotations

import os
from typing import Any

EMBEDDING_MODEL_ALIAS = os.getenv("FOUNDRY_EMBEDDING_MODEL", "qwen3-embedding-0.6b")
CHAT_MODEL_ALIAS = os.getenv("FOUNDRY_CHAT_MODEL", "qwen2.5-0.5b")
_manager: Any | None = None
_loaded_models: list[Any] = []


def get_manager() -> Any:
    """Initialize the Foundry Local SDK once and return its manager."""
    global _manager
    if _manager is not None:
        return _manager
    try:
        from foundry_local_sdk import Configuration, FoundryLocalManager
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Foundry Local's Python SDK is not installed. Activate a virtual environment "
            "and run: python -m pip install -r requirements.txt"
        ) from exc
    FoundryLocalManager.initialize(Configuration(app_name="local_rag_assistant"))
    _manager = FoundryLocalManager.instance
    return _manager


def _progress(label: str):
    def callback(percent: float) -> None:
        print(f"\r{label}: {percent:.1f}%", end="", flush=True)
    return callback


def _load_model(alias: str, label: str) -> Any:
    model = get_manager().catalog.get_model(alias)
    if model is None:
        raise RuntimeError(f"Foundry Local does not list '{alias}'. Run 'foundry model list' to see aliases.")
    model.download(_progress(f"Downloading {label}"))
    print()
    model.load()
    _loaded_models.append(model)
    print(f"{label.capitalize()} model '{alias}' is ready.")
    return model


def load_embedding_client() -> Any:
    return _load_model(EMBEDDING_MODEL_ALIAS, "embedding").get_embedding_client()


def load_chat_client() -> Any:
    manager = get_manager()
    manager.download_and_register_eps(progress_callback=lambda _name, _percent: None)
    return _load_model(CHAT_MODEL_ALIAS, "chat").get_chat_client()


def unload_models() -> None:
    """Release model memory; cached downloads are kept for later runs."""
    while _loaded_models:
        try:
            _loaded_models.pop().unload()
        except Exception:
            pass
