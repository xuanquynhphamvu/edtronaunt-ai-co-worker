from __future__ import annotations

from datetime import datetime
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from coworker_engine.utils.portfolio import (  # noqa: E402
    PORTFOLIO_EXPORT_SESSION_KEY,
    PortfolioError,
    export_portfolio_pack,
    get_portfolio_status,
    save_portfolio_artifact,
)


HAS_REPORTLAB = importlib.util.find_spec("reportlab") is not None


class PortfolioRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session_state: dict[str, object] = {}

    def _seed_complete_portfolio(self, executive_update_text: str = "Executive summary for leadership.") -> None:
        save_portfolio_artifact(
            self.session_state,
            artifact_type="final_plan",
            title="Final plan",
            body_markdown="A measured rollout plan with brand-specific calibration.",
        )
        save_portfolio_artifact(
            self.session_state,
            artifact_type="internal_comm",
            title="Regional launch note",
            body_markdown="Email draft announcing the pilot launch.",
            metadata={"comm_type": "email", "audience": "Regional leaders"},
        )
        save_portfolio_artifact(
            self.session_state,
            artifact_type="executive_update",
            title="Executive update",
            body_markdown=executive_update_text,
        )

    def test_final_plan_save_overwrites_latest_singleton(self) -> None:
        save_portfolio_artifact(
            self.session_state,
            artifact_type="final_plan",
            title="Plan v1",
            body_markdown="First draft.",
        )
        latest = save_portfolio_artifact(
            self.session_state,
            artifact_type="final_plan",
            title="Plan v2",
            body_markdown="Second draft with rollout trade-offs.",
        )

        status = get_portfolio_status(self.session_state)
        self.assertEqual(status["counts"]["final_plan"], 1)
        self.assertEqual(status["titles"]["final_plan"], "Plan v2")
        self.assertEqual(latest.body_markdown, "Second draft with rollout trade-offs.")

    def test_internal_comms_append_in_order(self) -> None:
        first = save_portfolio_artifact(
            self.session_state,
            artifact_type="internal_comm",
            title="Launch email",
            body_markdown="Email to stakeholders.",
            metadata={"comm_type": "email"},
        )
        second = save_portfolio_artifact(
            self.session_state,
            artifact_type="internal_comm",
            title="Leadership post",
            body_markdown="Post for internal feed.",
            metadata={"comm_type": "post"},
        )

        status = get_portfolio_status(self.session_state)
        self.assertEqual(status["counts"]["internal_comm"], 2)
        self.assertEqual(status["titles"]["internal_comm"], [first.title, second.title])

    def test_readiness_gating_requires_all_sections(self) -> None:
        save_portfolio_artifact(
            self.session_state,
            artifact_type="final_plan",
            title="Final plan",
            body_markdown="Rollout plan.",
        )

        status = get_portfolio_status(self.session_state)
        self.assertFalse(status["is_ready"])
        self.assertEqual(status["missing"], ["internal_comm", "executive_update"])

        with self.assertRaises(PortfolioError):
            export_portfolio_pack(
                self.session_state,
                thread_id="session-a",
                export_root=ROOT / "exports" / "test-artifacts",
                created_at=datetime(2026, 3, 20, 8, 15, 30),
            )

    def test_export_rejects_forbidden_language(self) -> None:
        self._seed_complete_portfolio(executive_update_text="We should bet on this rollout.")

        with self.assertRaises(PortfolioError) as caught:
            export_portfolio_pack(
                self.session_state,
                thread_id="session-b",
                export_root=ROOT / "exports" / "test-artifacts",
                created_at=datetime(2026, 3, 20, 8, 15, 30),
            )

        self.assertIn("Executive update", str(caught.exception))
        self.assertIn("Forbidden keyword detected: 'bet'", str(caught.exception))

    @unittest.skipUnless(HAS_REPORTLAB, "reportlab not installed")
    def test_export_writes_pdf_and_uses_deterministic_name(self) -> None:
        self._seed_complete_portfolio()
        created_at = datetime(2026, 3, 20, 8, 15, 30)

        with tempfile.TemporaryDirectory() as tmp_dir:
            exported = export_portfolio_pack(
                self.session_state,
                thread_id="thread-123",
                export_root=Path(tmp_dir),
                created_at=created_at,
            )

            self.assertEqual(exported.file_name, "portfolio-pack-thread-123-20260320-081530.pdf")
            self.assertTrue(exported.file_path.exists())
            self.assertEqual(exported.file_path.read_bytes(), exported.pdf_bytes)
            self.assertTrue(exported.pdf_bytes.startswith(b"%PDF"))
            self.assertIn(b"Final Plan", exported.pdf_bytes)
            self.assertIn(b"Internal Communications", exported.pdf_bytes)
            self.assertIn(b"Executive Update", exported.pdf_bytes)
            self.assertEqual(
                self.session_state[PORTFOLIO_EXPORT_SESSION_KEY]["file_name"],
                exported.file_name,
            )


if __name__ == "__main__":
    unittest.main()
