from langchain_core.tools import tool
import os
import re
from typing import Any

import requests

from .knowledge import format_knowledge_context, retrieve_knowledge


FAKE_JIRA_BASE_URL = os.getenv("FAKE_JIRA_BASE_URL", "http://127.0.0.1:5000")
TASK_ID_RE = re.compile(r"\btask\s*#?(\d+)\b|\bticket\s*#?(\d+)\b", re.IGNORECASE)
STATUS_RE = re.compile(r"\b(todo|in_progress|done)\b", re.IGNORECASE)


def _jira_url(path: str) -> str:
    return f"{FAKE_JIRA_BASE_URL.rstrip('/')}{path}"


def _fetch_tasks() -> list[dict[str, Any]]:
    response = requests.get(_jira_url("/tasks"), timeout=5)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, list):
        raise ValueError("Unexpected Jira response")
    return data


def fetch_jira_tasks() -> list[dict[str, Any]]:
    """Fetch all fake Jira tasks as Python data."""
    return _fetch_tasks()


def find_jira_tasks(query: str, limit: int = 3) -> list[dict[str, Any]]:
    tasks = _fetch_tasks()

    task_id_match = TASK_ID_RE.search(query)
    if task_id_match:
        task_id = int(next(group for group in task_id_match.groups() if group))
        return [task for task in tasks if task.get("id") == task_id][:limit]

    query_terms = {
        token
        for token in re.findall(r"[a-z0-9']+", query.lower())
        if len(token) > 2 and token not in {"jira", "task", "ticket", "issue"}
    }
    if not query_terms:
        return tasks[:limit]

    scored: list[tuple[int, dict[str, Any]]] = []
    for task in tasks:
        haystack = " ".join(
            [
                str(task.get("title", "")),
                str(task.get("description", "")),
                str(task.get("status", "")),
                " ".join(comment.get("body", "") for comment in task.get("comments", [])),
            ]
        ).lower()
        score = sum(1 for term in query_terms if term in haystack)
        if score > 0:
            scored.append((score, task))

    scored.sort(key=lambda item: (item[0], -int(item[1].get("id", 0))), reverse=True)
    return [task for _, task in scored[:limit]]


def _format_task(task: dict[str, Any]) -> str:
    comments = task.get("comments", [])
    comment_lines = [
        f"    - {comment.get('agent_id', 'unknown')}: {comment.get('body', '')}"
        for comment in comments[:3]
    ]
    comments_block = "\n".join(comment_lines) if comment_lines else "    - none"
    return (
        f"Task #{task.get('id')} [{task.get('status')}]: {task.get('title')}\n"
        f"  Description: {task.get('description')}\n"
        f"  Comments:\n{comments_block}"
    )


def get_jira_context(query: str, limit: int = 3) -> str:
    try:
        selected = find_jira_tasks(query, limit=limit)
    except Exception as exc:
        return (
            "Live Jira data is unavailable right now. "
            f"Could not reach {FAKE_JIRA_BASE_URL}. Error: {exc}"
        )

    if not selected:
        return f"No Jira tasks matched '{query}'."
    return "\n\n".join(_format_task(task) for task in selected)


def update_jira_task(task_id: int, *, title: str | None = None, description: str | None = None, status: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if title is not None:
        payload["title"] = title
    if description is not None:
        payload["description"] = description
    if status is not None:
        payload["status"] = status
    response = requests.patch(_jira_url(f"/tasks/{task_id}"), json=payload, timeout=5)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected Jira update response")
    return data


def add_jira_comment_entry(task_id: int, agent_id: str, body: str) -> dict[str, Any]:
    response = requests.post(
        _jira_url(f"/tasks/{task_id}/comments"),
        json={"agent_id": agent_id, "body": body},
        timeout=5,
    )
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected Jira comment response")
    return data


def summarize_jira_query(query: str) -> str | None:
    lowered = query.lower()
    asks_status = any(term in lowered for term in ["status", "progress", "done", "in progress"])
    asks_comment = any(term in lowered for term in ["comment", "comments", "update", "updates", "note", "notes"])
    asks_task = bool(TASK_ID_RE.search(query)) or any(term in lowered for term in ["task", "ticket", "issue"])
    asks_list = any(term in lowered for term in ["list", "show", "all tasks", "tasks pls", "tasks please"])

    if not asks_task:
        return None

    try:
        if asks_list:
            matches = fetch_jira_tasks()
        else:
            matches = find_jira_tasks(query, limit=3)
    except Exception:
        return None

    if not matches:
        return "I couldn't find a matching Jira task."

    first = matches[0]
    if len(matches) == 1 and TASK_ID_RE.search(query):
        parts: list[str] = [f"Task {first['id']} is {first['status']}."]
        if asks_comment:
            comments = first.get("comments", [])
            if comments:
                latest = comments[-1]
                parts.append(
                    f"The latest comment is from {latest.get('agent_id', 'unknown')}: "
                    f"\"{latest.get('body', '')}\"."
                )
            else:
                parts.append("There are no comments on it yet.")
        return " ".join(parts)

    if asks_comment:
        commented_tasks: list[str] = []
        for task in matches:
            comments = task.get("comments", [])
            if not comments:
                continue
            latest = comments[-1]
            commented_tasks.append(
                f"task {task['id']} has comments; the latest is from {latest.get('agent_id', 'unknown')}: "
                f"\"{latest.get('body', '')}\""
            )
        if commented_tasks:
            return "Here’s what I found in Jira: " + "; ".join(commented_tasks[:3]) + "."
        return "I found matching tasks, but none of them have comments yet."

    if asks_list:
        lines = [
            f"- task {task['id']}: {task['title']} [{task['status']}]"
            for task in matches[:10]
        ]
        return "Here are the current Jira tasks:\n" + "\n".join(lines)

    status_summary = ", ".join(f"task {task['id']} is {task['status']}" for task in matches[:3])
    return f"Here’s what I found in Jira: {status_summary}."


def maybe_apply_jira_action(query: str, agent_id: str) -> str | None:
    lowered = query.lower()
    task_id_match = TASK_ID_RE.search(query)
    if not task_id_match:
        return None

    task_id = int(next(group for group in task_id_match.groups() if group))

    comment_command_patterns = [
        r"(?i)^\s*@?\w*\s*add a comment to (?:task|ticket)\s*#?\d+\s+saying\s+.+$",
        r"(?i)^\s*@?\w*\s*leave a comment on (?:task|ticket)\s*#?\d+\s+saying\s+.+$",
        r"(?i)^\s*@?\w*\s*post a comment on (?:task|ticket)\s*#?\d+\s+saying\s+.+$",
        r"(?i)^\s*@?\w*\s*comment on (?:task|ticket)\s*#?\d+\s+that\s+.+$",
    ]
    if any(re.match(pattern, query.strip()) for pattern in comment_command_patterns):
        comment_text = query
        patterns = [
            r"(?i)add a comment to (?:task|ticket)\s*#?\d+\s+saying\s+(.+)$",
            r"(?i)comment on (?:task|ticket)\s*#?\d+\s+that\s+(.+)$",
            r"(?i)leave a comment on (?:task|ticket)\s*#?\d+\s+saying\s+(.+)$",
            r"(?i)post a comment on (?:task|ticket)\s*#?\d+\s+saying\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, query.strip())
            if match:
                comment_text = match.group(1).strip().strip("\"'")
                break
        task = add_jira_comment_entry(task_id, agent_id=agent_id, body=comment_text)
        return (
            f"I added a comment to task {task_id}. "
            f"The latest comment is now from {agent_id}: \"{comment_text}\"."
        )

    status_command_patterns = [
        r"(?i)^\s*@?\w*\s*move (?:task|ticket)\s*#?\d+\s+to\s+\w+\.?$",
        r"(?i)^\s*@?\w*\s*update (?:task|ticket)\s*#?\d+\s+to\s+\w+\.?$",
        r"(?i)^\s*@?\w*\s*set (?:task|ticket)\s*#?\d+\s+to\s+\w+\.?$",
        r"(?i)^\s*@?\w*\s*mark (?:task|ticket)\s*#?\d+\s+as\s+\w+\.?$",
    ]
    if any(re.match(pattern, query.strip()) for pattern in status_command_patterns):
        status_match = STATUS_RE.search(lowered)
        if not status_match:
            return None
        status = status_match.group(1).lower()
        task = update_jira_task(task_id, status=status)
        return f"I updated task {task_id} to {task['status']}."

    return None

@tool
def calculate_kpi(brand_name: str, metric: str) -> str:
    """Mock business function to calculate KPI for a given brand."""
    # Insert real logic here
    return f"{metric} for {brand_name} is performing at 110% of target."

@tool
def retrieve_brand_data(namespace: str, query: str) -> str:
    """Retrieve simulation context scoped to the given namespace."""
    chunks = retrieve_knowledge(query, namespaces=[namespace], top_k=3)
    if not chunks:
        return (
            f"No grounded context found for '{query}' in namespace '{namespace}'. "
            "Respond conservatively and state assumptions."
        )
    return format_knowledge_context(chunks)


@tool
def list_jira_tasks(status: str = "") -> str:
    """List fake Jira tasks, optionally filtered by status."""
    try:
        tasks = _fetch_tasks()
    except Exception as exc:
        return f"Jira unavailable: {exc}"

    status = status.strip().lower()
    if status:
        tasks = [task for task in tasks if str(task.get("status", "")).lower() == status]
    if not tasks:
        return "No Jira tasks found."
    return "\n\n".join(_format_task(task) for task in tasks[:10])


@tool
def search_jira_tasks(query: str) -> str:
    """Search fake Jira tasks using title, description, status, and comments."""
    return get_jira_context(query)


@tool
def add_jira_comment(task_id: int, agent_id: str, body: str) -> str:
    """Add a comment to a fake Jira task."""
    try:
        task = add_jira_comment_entry(task_id, agent_id=agent_id, body=body)
    except Exception as exc:
        return f"Failed to add Jira comment: {exc}"
    return _format_task(task)


@tool
def update_jira_status(task_id: int, status: str) -> str:
    """Update a fake Jira task status."""
    try:
        task = update_jira_task(task_id, status=status)
    except Exception as exc:
        return f"Failed to update Jira task: {exc}"
    return _format_task(task)
