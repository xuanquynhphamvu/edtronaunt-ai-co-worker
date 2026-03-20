from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from fake_jira import create_app  # noqa: E402
from fake_jira.demo_data import seed_demo_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bundled fake Jira service.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--db",
        default=str(ROOT / "data" / "fake_jira" / "tasks.db"),
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Do not seed the database with demo tasks when it is empty.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.skip_seed:
        seed_demo_data(args.db)

    app = create_app({"DATABASE": args.db})
    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
