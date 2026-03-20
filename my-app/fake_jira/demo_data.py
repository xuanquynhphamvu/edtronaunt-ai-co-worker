from __future__ import annotations

from .database import get_db, init_db
from .repository import add_comment, count_tasks, create_task


def seed_demo_data(database_path: str) -> bool:
    init_db(database_path)

    with get_db(database_path) as connection:
        if count_tasks(connection) > 0:
            return False

        pilot = create_task(
            connection,
            title="Pilot training pack",
            description="Prepare manager enablement material before the first regional pilot.",
            status="in_progress",
        )
        create_task(
            connection,
            title="Store readiness checklist",
            description="Finalize the rollout checklist for understaffed locations.",
            status="todo",
        )
        reporting = create_task(
            connection,
            title="Leadership update draft",
            description="Draft the weekly leadership update for the change rollout.",
            status="todo",
        )

        add_comment(
            connection,
            task_id=int(pilot["id"]),
            agent_id="people",
            body="Manager coaching is the main dependency before launch.",
        )
        add_comment(
            connection,
            task_id=int(reporting["id"]),
            agent_id="executive",
            body="Call out adoption risk and budget protection explicitly.",
        )

    return True
