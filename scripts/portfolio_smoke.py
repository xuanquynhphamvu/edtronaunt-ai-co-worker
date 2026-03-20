from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from coworker_engine.utils.portfolio import export_portfolio_pack, get_portfolio_status, save_portfolio_artifact  # noqa: E402


def main() -> int:
    session_state: dict[str, object] = {}

    save_portfolio_artifact(
        session_state,
        artifact_type="final_plan",
        title="Cross-functional rollout plan",
        body_markdown=(
            "Create a rollout plan with a clear scope, staged launch sequence, "
            "and room for local adaptation."
        ),
        source_notes="Scenario brief; simulation design brief",
    )
    save_portfolio_artifact(
        session_state,
        artifact_type="internal_comm",
        title="Pilot launch email",
        body_markdown=(
            "Subject: Pilot launch\n\n"
            "We will launch the pilot with role-based enablement, local readiness checks, and manager support."
        ),
        metadata={"comm_type": "email", "audience": "Regional leads"},
    )
    save_portfolio_artifact(
        session_state,
        artifact_type="executive_update",
        title="Executive leadership update",
        body_markdown=(
            "The proposal clarifies scope, improves adoption readiness, and stages execution risk "
            "through a pilot before wider expansion."
        ),
    )

    status = get_portfolio_status(session_state)
    if not status["is_ready"]:
        print("Portfolio Pack is not ready.")
        return 1

    exported = export_portfolio_pack(
        session_state,
        thread_id="smoke-session",
        export_root=ROOT / "exports" / "portfolio-packs",
    )
    print(f"Exported: {exported.file_path}")
    print(f"Bytes: {len(exported.pdf_bytes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
