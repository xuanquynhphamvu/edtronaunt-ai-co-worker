from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from coworker_engine.simulation import ACTIVE_SIMULATION  # noqa: E402
from coworker_engine.utils.agent_memory import (  # noqa: E402
    append_persona_knowledge,
    ensure_persona_files,
    ensure_simulation_agent_files,
    load_persona_memory,
)
from coworker_engine.utils.knowledge import retrieve_knowledge  # noqa: E402


class AgentMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.previous_root = os.environ.get("AGENT_MEMORY_ROOT")
        os.environ["AGENT_MEMORY_ROOT"] = self.tmp_dir.name
        self.addCleanup(self._restore_env)

    def _restore_env(self) -> None:
        if self.previous_root is None:
            os.environ.pop("AGENT_MEMORY_ROOT", None)
        else:
            os.environ["AGENT_MEMORY_ROOT"] = self.previous_root

    def test_persona_files_are_created_as_markdown(self) -> None:
        persona = ACTIVE_SIMULATION.personas[0]

        files = ensure_persona_files(persona)

        self.assertTrue(files.soul_path.exists())
        self.assertTrue(files.knowledge_path.exists())
        self.assertIn("# SOUL", files.soul_path.read_text(encoding="utf-8"))
        self.assertIn("# Knowledge", files.knowledge_path.read_text(encoding="utf-8"))

    def test_task_updates_append_to_knowledge_markdown(self) -> None:
        persona = ACTIVE_SIMULATION.personas[1]
        unique_phrase = "people-adoption-signal-2049"

        ensure_persona_files(persona)
        append_persona_knowledge(
            persona,
            title="Task update",
            user_message=f"Please address {unique_phrase}",
            agent_response="Use manager coaching during the pilot to lower onboarding risk.",
            mode="direct_reply",
        )

        _, knowledge_markdown = load_persona_memory(persona, max_knowledge_chars=None)
        self.assertIn("Task update", knowledge_markdown)
        self.assertIn(unique_phrase, knowledge_markdown)

    def test_retrieve_knowledge_includes_markdown_backed_agent_memory(self) -> None:
        ensure_simulation_agent_files(ACTIVE_SIMULATION)
        persona = next(
            persona for persona in ACTIVE_SIMULATION.personas if persona.route == "operations"
        )
        unique_phrase = "ops-capacity-beacon-731"

        append_persona_knowledge(
            persona,
            title="Task update",
            user_message=f"Track {unique_phrase} before the regional launch.",
            agent_response="Gate rollout on local staffing readiness and sequence the pilot first.",
            mode="meeting",
        )

        chunks = retrieve_knowledge(unique_phrase, namespaces=[persona.route], top_k=5)

        self.assertTrue(
            any(unique_phrase in chunk.content for chunk in chunks),
            "Expected markdown-backed Knowledge.md content to be retrievable.",
        )


if __name__ == "__main__":
    unittest.main()
