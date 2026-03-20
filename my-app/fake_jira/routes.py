from __future__ import annotations

from http import HTTPStatus

from flask import Flask, jsonify, render_template_string, request

from .database import get_db
from .repository import add_comment, create_task, list_tasks, update_task

ALLOWED_STATUSES = ("todo", "in_progress", "done")

INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Fake Jira</title>
  </head>
  <body>
    <h1>Fake Jira</h1>
    <p>Available statuses: {{ statuses | join(", ") }}</p>
    <ul>
      <li><code>GET /health</code></li>
      <li><code>GET /tasks</code></li>
      <li><code>POST /tasks</code></li>
      <li><code>PATCH /tasks/&lt;task_id&gt;</code></li>
      <li><code>POST /tasks/&lt;task_id&gt;/comments</code></li>
    </ul>
  </body>
</html>
"""


def register_routes(app: Flask) -> None:
    @app.get("/")
    def index():
        return render_template_string(INDEX_TEMPLATE, statuses=ALLOWED_STATUSES)

    @app.get("/health")
    def healthcheck():
        return jsonify({"status": "ok"})

    @app.post("/tasks")
    def create_task_route():
        payload = request.get_json(silent=True) or {}

        title = _clean_text(payload.get("title"))
        description = _clean_text(payload.get("description"))
        status = _clean_text(payload.get("status")) or "todo"

        if not title or not description:
            return _error("title and description are required", HTTPStatus.BAD_REQUEST)
        if status not in ALLOWED_STATUSES:
            return _error(
                "status must be one of: todo, in_progress, done",
                HTTPStatus.BAD_REQUEST,
            )

        with get_db(app.config["DATABASE"]) as connection:
            task = create_task(connection, title=title, description=description, status=status)
        return jsonify(task), HTTPStatus.CREATED

    @app.get("/tasks")
    def list_tasks_route():
        with get_db(app.config["DATABASE"]) as connection:
            tasks = list_tasks(connection)
        return jsonify(tasks)

    @app.patch("/tasks/<int:task_id>")
    def update_task_route(task_id: int):
        payload = request.get_json(silent=True) or {}

        title = _clean_optional_text(payload.get("title"))
        description = _clean_optional_text(payload.get("description"))
        status = _clean_optional_text(payload.get("status"))

        if title is None and description is None and status is None:
            return _error(
                "at least one of title, description, or status is required",
                HTTPStatus.BAD_REQUEST,
            )
        if status is not None and status not in ALLOWED_STATUSES:
            return _error(
                "status must be one of: todo, in_progress, done",
                HTTPStatus.BAD_REQUEST,
            )

        with get_db(app.config["DATABASE"]) as connection:
            task = update_task(
                connection,
                task_id=task_id,
                title=title,
                description=description,
                status=status,
            )

        if task is None:
            return _error("task not found", HTTPStatus.NOT_FOUND)
        return jsonify(task)

    @app.post("/tasks/<int:task_id>/comments")
    def add_comment_route(task_id: int):
        payload = request.get_json(silent=True) or {}

        agent_id = _clean_text(payload.get("agent_id"))
        body = _clean_text(payload.get("body"))

        if not agent_id or not body:
            return _error("agent_id and body are required", HTTPStatus.BAD_REQUEST)

        with get_db(app.config["DATABASE"]) as connection:
            task = add_comment(connection, task_id=task_id, agent_id=agent_id, body=body)

        if task is None:
            return _error("task not found", HTTPStatus.NOT_FOUND)
        return jsonify(task), HTTPStatus.CREATED


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _clean_optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _clean_text(value)


def _error(message: str, status: HTTPStatus):
    return jsonify({"error": message}), status
