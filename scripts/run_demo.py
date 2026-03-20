from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen

from werkzeug.serving import make_server


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "my-app"))

from fake_jira import create_app  # noqa: E402
from fake_jira.demo_data import seed_demo_data  # noqa: E402


class ServerThread(threading.Thread):
    def __init__(self, host: str, port: int, database_path: str) -> None:
        super().__init__(daemon=True)
        self.app = create_app({"DATABASE": database_path})
        self.server = make_server(host, port, self.app)
        self.context = self.app.app_context()
        self.context.push()

    def run(self) -> None:
        self.server.serve_forever()

    def shutdown(self) -> None:
        self.server.shutdown()
        self.context.pop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bundled fake Jira service and Streamlit demo together.",
    )
    parser.add_argument("--jira-host", default="127.0.0.1")
    parser.add_argument("--jira-port", type=int, default=5000)
    parser.add_argument("--streamlit-port", type=int, default=8501)
    parser.add_argument(
        "--jira-db",
        default=str(ROOT / "data" / "fake_jira" / "tasks.db"),
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Do not seed the bundled fake Jira database with demo tasks.",
    )
    return parser.parse_args()


def wait_for_healthcheck(base_url: str, timeout_seconds: float = 5.0) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(f"{base_url}/health", timeout=0.5) as response:
                if response.status == 200:
                    return
        except URLError as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"Fake Jira did not become healthy: {last_error}")


def is_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) == 0


def main() -> int:
    args = parse_args()
    jira_base_url = f"http://{args.jira_host}:{args.jira_port}"

    if is_port_open(args.jira_host, args.jira_port):
        raise SystemExit(
            f"Port {args.jira_port} is already in use. Stop the existing service or pick --jira-port."
        )

    Path(args.jira_db).parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_seed:
        seed_demo_data(args.jira_db)

    server = ServerThread(args.jira_host, args.jira_port, args.jira_db)
    server.start()
    wait_for_healthcheck(jira_base_url)

    env = os.environ.copy()
    env.setdefault("FAKE_JIRA_BASE_URL", jira_base_url)
    env.setdefault("FAKE_JIRA_DB_PATH", args.jira_db)

    streamlit_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ROOT / "app.py"),
        "--server.port",
        str(args.streamlit_port),
    ]

    process: subprocess.Popen[str] | None = None

    def stop_process(*_: object) -> None:
        if process is not None and process.poll() is None:
            process.terminate()

    signal.signal(signal.SIGINT, stop_process)
    signal.signal(signal.SIGTERM, stop_process)

    try:
        print(f"Fake Jira running at {jira_base_url}")
        print(f"Streamlit will open on http://127.0.0.1:{args.streamlit_port}")
        process = subprocess.Popen(streamlit_cmd, cwd=ROOT, env=env)
        return process.wait()
    finally:
        stop_process()
        server.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
