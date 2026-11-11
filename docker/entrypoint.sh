#!/bin/sh
# LZAI container bootstrap.
# The image ships only the base OS toolchain and the source tree; the Python
# environment (uv sync) and model artifacts (S3) are provisioned on first run.
set -eu

cd /app

# ---------------------------------------------------------------------------
# 1. Model artifacts + runtime wheels from Wasabi S3.
#    Must run BEFORE uv sync: uv.lock resolves flash-attn from
#    data/runtime-wheels/. Per-file ETag checks make re-runs cheap.
# ---------------------------------------------------------------------------
if [ "${LZAI_SKIP_DATA_DOWNLOAD:-0}" = "1" ]; then
    echo "[bootstrap] LZAI_SKIP_DATA_DOWNLOAD=1 — skipping S3 data sync"
else
    echo "[bootstrap] Syncing model artifacts from S3..."
    # Media weights (~60 GB in all) come down only for the features listed in
    # LZAI_MEDIA_FEATURES ("music,videogen,lipsync"; unset = none).
    uv run --no-project --with boto3 --with python-dotenv \
        python scripts/download_data.py
fi

# ---------------------------------------------------------------------------
# 2. Locked Python environment. Re-synced only when uv.lock or pyproject.toml
#    changes (pyproject holds the console scripts, which uv.lock doesn't record).
# ---------------------------------------------------------------------------
SYNC_HASH="$(cat uv.lock pyproject.toml | md5sum | awk '{print $1}')"
if [ -f .venv/.sync-hash ] && [ "$(cat .venv/.sync-hash)" = "$SYNC_HASH" ]; then
    echo "[bootstrap] .venv matches uv.lock + pyproject.toml (${SYNC_HASH}*) — skipping uv sync"
else
    echo "[bootstrap] Running uv sync (first run downloads ~5GB of wheels)..."
    uv sync --frozen --no-dev
    echo "$SYNC_HASH" > .venv/.sync-hash
    echo "[bootstrap] uv sync complete"
fi

# ---------------------------------------------------------------------------
# 2b. Media runtime venv (src/media_runtime: YuE2 + YingMusic-SVC + ComfyUI/H3,
#     Python 3.11, its own torch cu130). Separate lockfile and interpreter; the
#     image's uv is system-python-only, so this sync lets uv fetch a managed
#     3.11. The prebuilt pyworld / yue2-infer wheels come from
#     data/runtime-wheels (S3 step above). Synced only when LZAI_MEDIA_FEATURES
#     whitelists at least one feature (music, videogen, lipsync).
# ---------------------------------------------------------------------------
if [ -z "$(echo "${LZAI_MEDIA_FEATURES:-}" | tr -d ' ,')" ]; then
    echo "[bootstrap] LZAI_MEDIA_FEATURES is empty — no media runtime venv"
else
    MEDIA_DIR=src/media_runtime
    MEDIA_HASH="$(cat $MEDIA_DIR/uv.lock $MEDIA_DIR/pyproject.toml | md5sum | awk '{print $1}')"
    if [ -f $MEDIA_DIR/.venv/.sync-hash ] && [ "$(cat $MEDIA_DIR/.venv/.sync-hash)" = "$MEDIA_HASH" ]; then
        echo "[bootstrap] media .venv matches its uv.lock + pyproject.toml (${MEDIA_HASH}*) — skipping uv sync"
    else
        echo "[bootstrap] Running uv sync for the media runtime..."
        (cd $MEDIA_DIR && UV_PYTHON_PREFERENCE=managed \
            UV_PYTHON_INSTALL_DIR=/app/cache/uv-python uv sync --frozen --no-dev)
        echo "$MEDIA_HASH" > $MEDIA_DIR/.venv/.sync-hash
        echo "[bootstrap] media uv sync complete"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Hand off to the service command (default: lzai-worker, like pm2).
#    The worker serves /health itself on PORT; no separate health server needed.
# ---------------------------------------------------------------------------
exec "$@"
