from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from fake_jira import create_app  # noqa: E402
from fake_jira.demo_data import seed_demo_data  # noqa: E402


class TaskApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        database_path = Path(self.temp_dir.name) / "test.db"
        app = create_app(
            {
                "TESTING": True,
                "DATABASE": str(database_path),
            }
        )
        self.database_path = str(database_path)
        self.client = app.test_client()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_create_list_update_and_comment_flow(self) -> None:
        create_response = self.client.post(
            "/tasks",
            json={
                "title": "First task",
                "description": "Track implementation details",
            },
        )
        self.assertEqual(create_response.status_code, 201)
        created_task = create_response.get_json()
        self.assertEqual(created_task["status"], "todo")
        self.assertEqual(created_task["comments"], [])

        update_response = self.client.patch(
            f"/tasks/{created_task['id']}",
            json={"status": "in_progress"},
        )
        self.assertEqual(update_response.status_code, 200)
        updated_task = update_response.get_json()
        self.assertEqual(updated_task["status"], "in_progress")

        comment_response = self.client.post(
            f"/tasks/{created_task['id']}/comments",
            json={"agent_id": "AI1", "body": "Initial implementation started"},
        )
        self.assertEqual(comment_response.status_code, 201)
        commented_task = comment_response.get_json()
        self.assertEqual(len(commented_task["comments"]), 1)
        self.assertEqual(commented_task["comments"][0]["agent_id"], "AI1")

        list_response = self.client.get("/tasks")
        self.assertEqual(list_response.status_code, 200)
        tasks = list_response.get_json()
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["comments"][0]["body"], "Initial implementation started")

    def test_index_page_loads(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Fake Jira", response.data)

    def test_rejects_invalid_status(self) -> None:
        response = self.client.post(
            "/tasks",
            json={
                "title": "Bad task",
                "description": "Invalid status should fail",
                "status": "blocked",
            },
        )
        self.assertEqual(response.status_code, 400)

    def test_comment_requires_agent_id(self) -> None:
        create_response = self.client.post(
            "/tasks",
            json={
                "title": "Task",
                "description": "Needs a comment",
            },
        )
        task_id = create_response.get_json()["id"]

        response = self.client.post(
            f"/tasks/{task_id}/comments",
            json={"body": "Missing agent"},
        )
        self.assertEqual(response.status_code, 400)

    def test_demo_seed_only_runs_once(self) -> None:
        self.assertTrue(seed_demo_data(self.database_path))
        self.assertFalse(seed_demo_data(self.database_path))

        list_response = self.client.get("/tasks")
        tasks = list_response.get_json()
        self.assertEqual(len(tasks), 3)


if __name__ == "__main__":
    unittest.main()
