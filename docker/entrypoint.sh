#!/bin/sh
# LZ-TTS container bootstrap.
# The image ships only the base OS toolchain and the source tree; the Python
# environment (uv sync) and model artifacts (S3) are provisioned on first run.
set -eu

cd /app

# ---------------------------------------------------------------------------
# 1. Model artifacts + runtime wheels from Wasabi S3.
#    Must run BEFORE uv sync: uv.lock resolves flash-attn from
#    data/runtime-wheels/. Per-file ETag checks make re-runs cheap.
# ---------------------------------------------------------------------------
if [ "${LZ_TTS_SKIP_DATA_DOWNLOAD:-0}" = "1" ]; then
    echo "[bootstrap] LZ_TTS_SKIP_DATA_DOWNLOAD=1 — skipping S3 data sync"
else
    echo "[bootstrap] Syncing model artifacts from S3..."
    uv run --no-project --with boto3 --with python-dotenv \
        python scripts/download_data.py
fi

# ---------------------------------------------------------------------------
# 2. Locked Python environment. Re-synced only when uv.lock changes.
# ---------------------------------------------------------------------------
SYNC_HASH="$(md5sum uv.lock | awk '{print $1}')"
if [ -f .venv/.sync-hash ] && [ "$(cat .venv/.sync-hash)" = "$SYNC_HASH" ]; then
    echo "[bootstrap] .venv matches uv.lock (${SYNC_HASH}*) — skipping uv sync"
else
    echo "[bootstrap] Running uv sync (first run downloads ~5GB of wheels)..."
    uv sync --frozen --no-dev
    echo "$SYNC_HASH" > .venv/.sync-hash
    echo "[bootstrap] uv sync complete"
fi

# ---------------------------------------------------------------------------
# 3. Hand off to the service command (default: lz-tts-worker, like pm2).
#    The worker serves /health itself on PORT; no separate health server needed.
# ---------------------------------------------------------------------------
exec "$@"
