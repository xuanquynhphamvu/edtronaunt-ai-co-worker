from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from coworker_engine.utils import model_provider  # noqa: E402


class ModelProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_env = {
            key: os.environ.get(key)
            for key in [
                "MODEL_PROVIDER",
                "OPENAI_MODEL",
                "OPENAI_BASE_URL",
                "OPENAI_ORG_ID",
                "OLLAMA_MODEL",
            ]
        }

    def tearDown(self) -> None:
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    @patch("coworker_engine.utils.model_provider.ChatOpenAI")
    def test_create_chat_model_uses_openai_when_provider_is_openai(
        self, mock_chat_openai: object
    ) -> None:
        os.environ["MODEL_PROVIDER"] = "openai"
        os.environ["OPENAI_MODEL"] = "gpt-4.1-mini"
        os.environ["OPENAI_BASE_URL"] = "https://example.test/v1"
        os.environ["OPENAI_ORG_ID"] = "org_test"

        model_provider.create_chat_model(temperature=0.2)

        mock_chat_openai.assert_called_once_with(
            model="gpt-4.1-mini",
            temperature=0.2,
            base_url="https://example.test/v1",
            organization="org_test",
        )

    @patch("coworker_engine.utils.model_provider.ChatOllama")
    def test_create_chat_model_uses_ollama_by_default(self, mock_chat_ollama: object) -> None:
        os.environ.pop("MODEL_PROVIDER", None)
        os.environ["OLLAMA_MODEL"] = "qwen2.5:14b"

        model_provider.create_chat_model(temperature=0.4)

        mock_chat_ollama.assert_called_once_with(
            model="qwen2.5:14b",
            temperature=0.4,
        )

    def test_openai_provider_is_always_marked_tool_capable(self) -> None:
        os.environ["MODEL_PROVIDER"] = "openai"

        self.assertTrue(model_provider.model_supports_tools(object()))


if __name__ == "__main__":
    unittest.main()
