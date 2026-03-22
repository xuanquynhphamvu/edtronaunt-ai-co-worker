#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$ROOT_DIR/.venv"
RUN_DIR="$ROOT_DIR/.run"
STREAMLIT_LOG="$RUN_DIR/streamlit.log"
LANGGRAPH_LOG="$RUN_DIR/langgraph.log"
STREAMLIT_PID_FILE="$RUN_DIR/streamlit.pid"
LANGGRAPH_PID_FILE="$RUN_DIR/langgraph.pid"

TAILSCALE_IP="${TAILSCALE_IP:-100.72.212.8}"
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
LANGGRAPH_PORT="${LANGGRAPH_PORT:-2024}"
LANGGRAPH_CONFIG="${LANGGRAPH_CONFIG:-$ROOT_DIR/langgraph.json}"
LANGGRAPH_STUDIO_ORIGIN="${LANGGRAPH_STUDIO_ORIGIN:-https://smith.langchain.com}"
PUBLIC_BASE_URL="${PUBLIC_BASE_URL:-http://$TAILSCALE_IP:$STREAMLIT_PORT}"

require_bin() {
  local path="$1"
  local label="$2"
  if [[ ! -x "$path" ]]; then
    echo "Missing $label at $path"
    exit 1
  fi
}

kill_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "${pid:-}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$pid_file"
  fi
}

kill_matching_processes() {
  local pattern="$1"
  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 1
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
}

kill_port_listener() {
  local port="$1"
  local pids
  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 1
    echo "$pids" | xargs kill -9 2>/dev/null || true
  fi
}

mkdir -p "$RUN_DIR"

require_bin "$VENV_DIR/bin/python" "project Python"
require_bin "$VENV_DIR/bin/streamlit" "Streamlit"
require_bin "$VENV_DIR/bin/langgraph" "LangGraph CLI"

echo "Cleaning session state..."
rm -rf "$ROOT_DIR"/agent_memory/sessions/streamlit-*
rm -f "$STREAMLIT_LOG" "$LANGGRAPH_LOG"

echo "Stopping existing project services..."
kill_pid_file "$STREAMLIT_PID_FILE"
kill_pid_file "$LANGGRAPH_PID_FILE"
kill_matching_processes "$ROOT_DIR/app.py"
kill_matching_processes "$LANGGRAPH_CONFIG"
kill_port_listener "$STREAMLIT_PORT"
kill_port_listener "$LANGGRAPH_PORT"

echo "Starting LangGraph on 0.0.0.0:$LANGGRAPH_PORT..."
nohup "$VENV_DIR/bin/langgraph" dev \
  --host 0.0.0.0 \
  --port "$LANGGRAPH_PORT" \
  --config "$LANGGRAPH_CONFIG" \
  --allow-blocking \
  --no-browser \
  >"$LANGGRAPH_LOG" 2>&1 &
echo $! > "$LANGGRAPH_PID_FILE"

echo "Starting Streamlit on 0.0.0.0:$STREAMLIT_PORT..."
PUBLIC_BASE_URL="$PUBLIC_BASE_URL" nohup "$VENV_DIR/bin/streamlit" run "$ROOT_DIR/app.py" \
  --server.address 0.0.0.0 \
  --server.port "$STREAMLIT_PORT" \
  --browser.serverAddress "$TAILSCALE_IP" \
  --browser.serverPort "$STREAMLIT_PORT" \
  >"$STREAMLIT_LOG" 2>&1 &
echo $! > "$STREAMLIT_PID_FILE"

sleep 3

echo
echo "Services started."
echo "Streamlit: http://$TAILSCALE_IP:$STREAMLIT_PORT"
echo "LangGraph: http://$TAILSCALE_IP:$LANGGRAPH_PORT"
echo "LangGraph Studio: $LANGGRAPH_STUDIO_ORIGIN/studio/?baseUrl=http://$TAILSCALE_IP:$LANGGRAPH_PORT"
echo "Logs:"
echo "  $STREAMLIT_LOG"
echo "  $LANGGRAPH_LOG"
