from __future__ import annotations

import os

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_model_provider() -> str:
    provider = os.getenv("MODEL_PROVIDER", "openai").strip().lower()
    return provider or "openai"


def get_chat_llm(*, temperature: float = 0.7):
    provider = get_model_provider()

    if provider == "ollama":
        return ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "qwen2.5:32b"),
            temperature=temperature,
        )

    if provider == "openai":
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
        )

    raise ValueError(
        "Unsupported MODEL_PROVIDER. Expected 'openai' or 'ollama'."
    )
