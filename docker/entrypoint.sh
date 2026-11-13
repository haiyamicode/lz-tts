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
#    Model sets (src/model_sets.py): LZAI_MODELS picks what downloads and is
#    served (unset = what server.json enables, "all" = everything);
#    LZAI_REQUIRED_MODELS (unset = all of LZAI_MODELS) is what must be on disk
#    before the worker starts and joins Taskflow. The rest is marked in
#    data/.pending/ and downloads in the background (step 4); the worker loads
#    each engine and starts pulling its task types once its sets are in.
# ---------------------------------------------------------------------------
rm -rf data/.pending
if [ "${LZAI_SKIP_DATA_DOWNLOAD:-0}" = "1" ]; then
    echo "[bootstrap] LZAI_SKIP_DATA_DOWNLOAD=1 — skipping S3 data sync"
else
    echo "[bootstrap] Syncing core + required model sets from S3..."
    uv run --no-project --with boto3 --with python-dotenv \
        python scripts/download_data.py --stage required
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
#     data/runtime-wheels (S3 step above). Synced only when LZAI_MODELS has a
#     media set (music, videogen, lipsync, imagegen); in the background
#     (data/.pending/media-venv) when none of them is required.
# ---------------------------------------------------------------------------
sync_media_venv() {
    MEDIA_DIR=src/media_runtime
    MEDIA_HASH="$(cat $MEDIA_DIR/uv.lock $MEDIA_DIR/pyproject.toml | md5sum | awk '{print $1}')"
    if [ -f $MEDIA_DIR/.venv/.sync-hash ] && [ "$(cat $MEDIA_DIR/.venv/.sync-hash)" = "$MEDIA_HASH" ]; then
        echo "[bootstrap] media .venv matches its uv.lock + pyproject.toml (${MEDIA_HASH}*) — skipping uv sync"
    else
        echo "[bootstrap] Running uv sync for the media runtime..."
        (cd $MEDIA_DIR && UV_PYTHON_PREFERENCE=managed \
            UV_PYTHON_INSTALL_DIR=/app/cache/uv-python uv sync --frozen --no-dev) || return 1
        echo "$MEDIA_HASH" > $MEDIA_DIR/.venv/.sync-hash
        echo "[bootstrap] media uv sync complete"
    fi
}

MEDIA_STAGE="$(uv run --no-project python -m src.model_sets media-stage)"
case "$MEDIA_STAGE" in
    none) echo "[bootstrap] no media model set — no media runtime venv" ;;
    required) sync_media_venv ;;
    deferred)
        mkdir -p data/.pending && touch data/.pending/media-venv
        (until sync_media_venv; do echo "[bootstrap] media uv sync failed; retrying in 30s"; sleep 30; done
         rm -f data/.pending/media-venv) &
        ;;
esac

# ---------------------------------------------------------------------------
# 2c. The model sets not required at startup, in the background: each set's
#     data/.pending marker is removed once it is complete. Failures retry.
# ---------------------------------------------------------------------------
if [ "${LZAI_SKIP_DATA_DOWNLOAD:-0}" != "1" ] && [ -n "$(ls data/.pending 2>/dev/null | grep -v '^media-venv$')" ]; then
    echo "[bootstrap] Downloading deferred model sets in the background: $(ls data/.pending | grep -v '^media-venv$' | tr '\n' ' ')"
    (until uv run --no-project --with boto3 --with python-dotenv \
            python scripts/download_data.py --stage deferred; do
        echo "[bootstrap] deferred download failed; retrying in 30s"; sleep 30
     done) &
fi

# ---------------------------------------------------------------------------
# 3. Hand off to the service command (default: lzai-worker, like pm2).
#    The worker serves /health itself on PORT; no separate health server needed.
# ---------------------------------------------------------------------------
exec "$@"
