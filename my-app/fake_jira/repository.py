from __future__ import annotations

import sqlite3
from typing import Any


def create_task(
    connection: sqlite3.Connection,
    title: str,
    description: str,
    status: str,
) -> dict[str, Any]:
    cursor = connection.execute(
        """
        INSERT INTO tasks (title, description, status)
        VALUES (?, ?, ?)
        """,
        (title, description, status),
    )
    return get_task(connection, cursor.lastrowid)


def list_tasks(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    task_rows = connection.execute(
        """
        SELECT id, title, description, status
        FROM tasks
        ORDER BY id ASC
        """
    ).fetchall()

    comments_by_task = _load_comments(connection)
    return [_serialize_task(row, comments_by_task.get(row["id"], [])) for row in task_rows]


def update_task(
    connection: sqlite3.Connection,
    task_id: int,
    title: str | None = None,
    description: str | None = None,
    status: str | None = None,
) -> dict[str, Any] | None:
    existing = connection.execute(
        "SELECT id, title, description, status FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if existing is None:
        return None

    connection.execute(
        """
        UPDATE tasks
        SET title = ?, description = ?, status = ?
        WHERE id = ?
        """,
        (
            title if title is not None else existing["title"],
            description if description is not None else existing["description"],
            status if status is not None else existing["status"],
            task_id,
        ),
    )
    return get_task(connection, task_id)


def add_comment(
    connection: sqlite3.Connection,
    task_id: int,
    agent_id: str,
    body: str,
) -> dict[str, Any] | None:
    task_exists = connection.execute(
        "SELECT 1 FROM tasks WHERE id = ?",
        (task_id,),
    ).fetchone()
    if task_exists is None:
        return None

    connection.execute(
        """
        INSERT INTO comments (task_id, agent_id, body)
        VALUES (?, ?, ?)
        """,
        (task_id, agent_id, body),
    )
    return get_task(connection, task_id)


def get_task(connection: sqlite3.Connection, task_id: int) -> dict[str, Any] | None:
    task_row = connection.execute(
        """
        SELECT id, title, description, status
        FROM tasks
        WHERE id = ?
        """,
        (task_id,),
    ).fetchone()
    if task_row is None:
        return None

    comments = connection.execute(
        """
        SELECT id, agent_id, body, created_at
        FROM comments
        WHERE task_id = ?
        ORDER BY id ASC
        """,
        (task_id,),
    ).fetchall()
    return _serialize_task(task_row, [_serialize_comment(row) for row in comments])


def count_tasks(connection: sqlite3.Connection) -> int:
    row = connection.execute("SELECT COUNT(*) AS count FROM tasks").fetchone()
    return int(row["count"]) if row is not None else 0


def _load_comments(connection: sqlite3.Connection) -> dict[int, list[dict[str, Any]]]:
    comment_rows = connection.execute(
        """
        SELECT id, task_id, agent_id, body, created_at
        FROM comments
        ORDER BY id ASC
        """
    ).fetchall()

    comments_by_task: dict[int, list[dict[str, Any]]] = {}
    for row in comment_rows:
        comments_by_task.setdefault(row["task_id"], []).append(_serialize_comment(row))
    return comments_by_task


def _serialize_task(task_row: sqlite3.Row, comments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": task_row["id"],
        "title": task_row["title"],
        "description": task_row["description"],
        "status": task_row["status"],
        "comments": comments,
    }


def _serialize_comment(comment_row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": comment_row["id"],
        "agent_id": comment_row["agent_id"],
        "body": comment_row["body"],
        "created_at": comment_row["created_at"],
    }
