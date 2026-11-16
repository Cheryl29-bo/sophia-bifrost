#!/bin/sh
set -eu

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the deployment environment}"

BIFROST_APP_DIR="${BIFROST_APP_DIR:-/app/bifrost-data}"
mkdir -p "$BIFROST_APP_DIR"

cat > "$BIFROST_APP_DIR/config.json" <<'EOF'
{
  "$schema": "https://www.getbifrost.ai/schema",
  "providers": {
    "gemini": {
      "keys": [
        {
          "name": "gemini-primary",
          "value": "env.GEMINI_API_KEY",
          "models": ["*"],
          "weight": 1.0
        }
      ]
    }
  }
}
EOF

bifrost -host 127.0.0.1 -port 8080 -app-dir "$BIFROST_APP_DIR" &
BIFROST_PID=$!

cleanup() {
    kill "$BIFROST_PID" 2>/dev/null || true
    wait "$BIFROST_PID" 2>/dev/null || true
}

trap cleanup EXIT

python - "$BIFROST_PID" <<'PY'
import os
import socket
import sys
import time

pid = int(sys.argv[1])
deadline = time.monotonic() + 60

while time.monotonic() < deadline:
    try:
        with socket.create_connection(("127.0.0.1", 8080), timeout=1):
            break
    except OSError:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            raise SystemExit("Bifrost exited before opening port 8080")
        time.sleep(0.5)
else:
    raise SystemExit("Bifrost did not open port 8080 within 60 seconds")
PY

python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
