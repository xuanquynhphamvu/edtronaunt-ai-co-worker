from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from coworker_engine.agent import supervisor_plan_node  # noqa: E402
from coworker_engine.simulation import ACTIVE_SIMULATION  # noqa: E402
from coworker_engine.utils import nodes as nodes_module  # noqa: E402


APP_SPEC = importlib.util.spec_from_file_location("app", ROOT / "app.py")
assert APP_SPEC and APP_SPEC.loader
app_module = importlib.util.module_from_spec(APP_SPEC)
APP_SPEC.loader.exec_module(app_module)


class FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.model = "stub-model"

    def invoke(self, _messages: list[object]) -> AIMessage:
        if not self.responses:
            raise AssertionError("No fake responses left for llm.invoke().")
        return AIMessage(content=self.responses.pop(0))


def _base_state(user_text: str) -> dict[str, object]:
    return {
        "messages": [HumanMessage(content=user_text)],
        "session_id": "test-session",
        "reputation": 0.5,
        "alignment_score": 0.0,
        "persona_reputation": {},
        "persona_alignment": {},
        "reputation_updated_for_turn": [],
        "meeting_queue": [],
        "meeting_notes": [],
        "visible_responses": [],
        "supervisor_hint": "",
        "mode": "direct_reply",
    }


def _merge_state(state: dict[str, object], update: dict[str, object]) -> dict[str, object]:
    merged = dict(state)
    for key, value in update.items():
        if key == "messages":
            merged[key] = list(state.get("messages", [])) + list(value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


class VisibleResponseNodeTests(unittest.TestCase):
    @patch.object(nodes_module, "append_persona_knowledge")
    @patch.object(nodes_module, "retrieve_knowledge", return_value=[])
    @patch.object(nodes_module, "load_persona_memory", return_value=("soul", "knowledge"))
    def test_direct_reply_has_one_final_visible_response(
        self,
        _load_persona_memory: object,
        _retrieve_knowledge: object,
        _append_persona_knowledge: object,
    ) -> None:
        persona = ACTIVE_SIMULATION.personas[0]
        state = _base_state("@executive What should we prioritize first?")

        with patch.object(nodes_module, "llm", FakeLLM(["Prioritize a narrow pilot first."])):
            node = nodes_module.build_npc_node(persona)
            result = node(state)

        visible_responses = result["visible_responses"]
        self.assertEqual(len(visible_responses), 1)
        self.assertEqual(visible_responses[0]["speaker"], persona.name)
        self.assertTrue(visible_responses[0]["is_final"])

    @patch("coworker_engine.agent.append_supervisor_knowledge")
    @patch("coworker_engine.agent.load_supervisor_memory", return_value=("soul", "knowledge"))
    @patch.object(nodes_module, "append_persona_knowledge")
    @patch.object(nodes_module, "retrieve_knowledge", return_value=[])
    @patch.object(nodes_module, "load_persona_memory", return_value=("soul", "knowledge"))
    def test_meeting_flow_returns_three_visible_responses_in_fixed_order(
        self,
        _load_persona_memory: object,
        _retrieve_knowledge: object,
        _append_persona_knowledge: object,
        _load_supervisor_memory: object,
        _append_supervisor_knowledge: object,
    ) -> None:
        state = _base_state("Design a rollout plan that balances strategy and adoption.")
        state = _merge_state(state, supervisor_plan_node(state))

        responses = [
            "Focus the scope and sequence the pilot.",
            "Back the pilot with manager coaching and practical enablement.",
            "Conclude with a regional rollout paced to local readiness.",
        ]

        for persona, response_text in zip(ACTIVE_SIMULATION.personas, responses):
            with patch.object(nodes_module, "llm", FakeLLM([response_text])):
                node = nodes_module.build_npc_node(persona)
                state = _merge_state(state, node(state))

        visible_responses = state["visible_responses"]
        self.assertEqual(len(visible_responses), 3)
        self.assertEqual(
            [response["speaker"] for response in visible_responses],
            list(ACTIVE_SIMULATION.persona_names),
        )
        self.assertEqual(
            [response["is_final"] for response in visible_responses],
            [False, False, True],
        )
        self.assertNotIn("Supervisor", [response["speaker"] for response in visible_responses])


class AppHelperTests(unittest.TestCase):
    def test_assistant_messages_from_state_preserves_order_and_finality(self) -> None:
        final_state = {
            "visible_responses": [
                {"speaker": "CEO", "content": "Scope the pilot tightly.", "is_final": False},
                {"speaker": "CHRO", "content": "Support adoption with coaching.", "is_final": False},
                {
                    "speaker": "Employer Branding & Internal Communications Regional Manager",
                    "content": "Roll out region by region once managers are ready.",
                    "is_final": True,
                },
            ]
        }

        messages = app_module._assistant_messages_from_state(final_state)

        self.assertEqual([message["contributor"] for message in messages], ["CEO", "CHRO", "Employer Branding & Internal Communications Regional Manager"])
        self.assertEqual([message["is_final"] for message in messages], [False, False, True])
        self.assertEqual(messages[-1]["content"], "Roll out region by region once managers are ready.")

    def test_assistant_messages_from_state_skips_empty_contributors(self) -> None:
        final_state = {
            "visible_responses": [
                {"speaker": "CEO", "content": "Start with a narrow pilot.", "is_final": False},
                {"speaker": "CHRO", "content": "   ", "is_final": False},
                {
                    "speaker": "Employer Branding & Internal Communications Regional Manager",
                    "content": "Sequence rollout by local readiness.",
                    "is_final": True,
                },
            ]
        }

        messages = app_module._assistant_messages_from_state(final_state)

        self.assertEqual([message["contributor"] for message in messages], ["CEO", "Employer Branding & Internal Communications Regional Manager"])
        self.assertEqual([message["is_final"] for message in messages], [False, True])

    def test_to_langchain_messages_ignores_contributor_metadata(self) -> None:
        chat_history = [
            {"role": "user", "content": "Need a rollout plan."},
            {
                "role": "assistant",
                "content": "Start with a small pilot.",
                "contributor": "CEO",
                "is_final": False,
            },
            {
                "role": "assistant",
                "content": "Support it with manager coaching.",
                "contributor": "CHRO",
                "is_final": True,
            },
        ]

        messages = app_module._to_langchain_messages(chat_history)

        self.assertEqual(len(messages), 3)
        self.assertIsInstance(messages[0], HumanMessage)
        self.assertIsInstance(messages[1], AIMessage)
        self.assertEqual(messages[2].content, "Support it with manager coaching.")

    def test_message_is_final_only_for_final_assistant_messages(self) -> None:
        self.assertFalse(
            app_module._message_is_final(
                {"role": "assistant", "content": "Interim note", "is_final": False}
            )
        )
        self.assertTrue(
            app_module._message_is_final(
                {"role": "assistant", "content": "Final note", "is_final": True}
            )
        )


if __name__ == "__main__":
    unittest.main()
