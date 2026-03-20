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
        title="Gucci Group leadership system",
        body_markdown=(
            "Create a group-level leadership system with a shared DNA spine, "
            "brand-specific calibration, and a phased rollout."
        ),
        source_notes="Scenario brief; simulation design brief",
    )
    save_portfolio_artifact(
        session_state,
        artifact_type="internal_comm",
        title="Regional pilot email",
        body_markdown=(
            "Subject: Regional pilot launch\n\n"
            "We will launch the leadership pilot with local adaptation workshops and manager coaching."
        ),
        metadata={"comm_type": "email", "audience": "Regional HR leads"},
    )
    save_portfolio_artifact(
        session_state,
        artifact_type="executive_update",
        title="Executive leadership update",
        body_markdown=(
            "The proposal protects brand DNA, improves talent visibility, and stages rollout risk "
            "through a pilot before global expansion."
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
