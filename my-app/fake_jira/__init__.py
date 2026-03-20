from __future__ import annotations

import os
from pathlib import Path

from flask import Flask

from .database import init_db
from .routes import register_routes


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "fake_jira" / "tasks.db"


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_mapping(
        DATABASE=os.environ.get("FAKE_JIRA_DB_PATH", str(DEFAULT_DB_PATH)),
        TESTING=False,
    )

    if test_config:
        app.config.update(test_config)

    init_db(app.config["DATABASE"])
    register_routes(app)
    return app
