#!/bin/sh
set -eu

export GOGC=50
export GOMEMLIMIT=240MiB

: "${GEMINI_API_KEY:?Set GEMINI_API_KEY in the deployment environment}"

BIFROST_APP_DIR="${BIFROST_APP_DIR:-/app/bifrost-data}"
mkdir -p "$BIFROST_APP_DIR"

cat > "$BIFROST_APP_DIR/config.json" <<'EOF'
{
  "$schema": "https://www.getbifrost.ai/schema",
  "client": {
    "enforce_auth_on_inference": false
  },
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

python - <<'PY'
import sys
import time
import urllib.error
import urllib.request

deadline = time.monotonic() + 45

while True:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        print("Bifrost /health is not ready after 45 seconds; starting Sophia anyway", file=sys.stderr)
        break
    try:
        with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=min(1, remaining)) as response:
            if 200 <= response.status < 300:
                print("Bifrost health check passed")
                break
    except (urllib.error.URLError, TimeoutError, OSError):
        pass
    time.sleep(min(1, max(0, deadline - time.monotonic())))
PY

exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
