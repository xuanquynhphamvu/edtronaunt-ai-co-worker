from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

load_dotenv()

DEFAULT_OLLAMA_MODEL = "qwen2.5:32b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
TOOL_CAPABLE_OLLAMA_MODEL_PREFIXES = ("qwen", "mistral", "smollm", "gemma", "deepseek")


def get_model_provider() -> str:
    return os.getenv("MODEL_PROVIDER", "ollama").strip().lower()


def create_chat_model(*, temperature: float = 0.7):
    provider = get_model_provider()
    if provider == "openai":
        kwargs: dict[str, object] = {
            "model": os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
            "temperature": temperature,
        }
        openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
        if openai_base_url:
            kwargs["base_url"] = openai_base_url

        openai_organization = os.getenv("OPENAI_ORG_ID", "").strip()
        if openai_organization:
            kwargs["organization"] = openai_organization

        return ChatOpenAI(**kwargs)

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        temperature=temperature,
    )


def get_model_name(llm: object) -> str:
    model_name = getattr(llm, "model", "") or getattr(llm, "model_name", "")
    return str(model_name).strip().lower()


def model_supports_tools(llm: object) -> bool:
    if get_model_provider() == "openai":
        return True

    return get_model_name(llm).startswith(TOOL_CAPABLE_OLLAMA_MODEL_PREFIXES)
