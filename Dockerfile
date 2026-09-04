# LZ-TTS production image — minimal: runtime base + source only (<1GB).
#
# NOT baked in (provisioned at bootstrap by docker/entrypoint.sh):
#   - Python deps (uv sync --frozen): torch 2.9.1+cu128 pulls the full CUDA 12.8
#     runtime as nvidia-* pip packages, so no CUDA base image is needed here.
#   - Model artifacts + prebuilt runtime wheels (Wasabi S3 via
#     scripts/download_data.py; flash-attn/pyicu/pycld2/monotonic-align wheels
#     have no compilers available here, they are prebuilt in data/runtime-wheels).
#
# Host requirements: NVIDIA driver + nvidia-container-toolkit, run with --gpus.

# --- Static ffmpeg/ffprobe (~1/4 the size of Debian's ffmpeg + codec stack) ---
FROM debian:bookworm-slim AS ffmpeg-static
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
    && curl -fsSL https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz -o /tmp/ffmpeg.tar.xz \
    && tar -xJf /tmp/ffmpeg.tar.xz -C /usr/local/bin --strip-components=2 \
        ffmpeg-master-latest-linux64-gpl/bin/ffmpeg \
        ffmpeg-master-latest-linux64-gpl/bin/ffprobe \
    && chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe \
    && rm -rf /tmp/ffmpeg.tar.xz /var/lib/apt/lists/*

FROM python:3.10.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # uv
    UV_LINK_MODE=copy \
    UV_CACHE_DIR=/app/cache/uv \
    UV_TOOL_DIR=/app/cache/uv-tools \
    UV_PYTHON_PREFERENCE=only-system \
    # HF model downloads at runtime
    HF_HOME=/app/cache/huggingface

# Runtime libraries:
#   libicu72 — PyICU runtime; libsndfile1/libgomp1 — soundfile/torch runtime
#   sox — audio post-processing used by the worker (ffmpeg comes from ffmpeg-static)
#   gcc — runtime C compiler: torch inductor + triton JIT-compile host code at first model load
#   g++ + libicu-dev — build toolchain so uv sync can compile sdist-only C
#     extensions (pyicu, webrtcvad, pesq, monotonic-align) against THIS
#     image's libicu — no prebuilt-wheel/system-library mismatch
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        sox \
        gcc g++ \
        pkg-config \
        libicu-dev \
        libicu72 \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv binary (uvx is a symlink; `uv tool run` shells out to it)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN ln -s uv /usr/local/bin/uvx
COPY --from=ffmpeg-static /usr/local/bin/ffmpeg /usr/local/bin/ffprobe /usr/local/bin/

WORKDIR /app

# Source only. local/server.json is baked in (no secrets); credentials and
# runtime settings come from the deployment platform's env vars.
COPY pyproject.toml uv.lock .python-version README.md ./
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY local/server.json ./local/server.json

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh \
    && mkdir -p /app/cache /app/data /app/output

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["entrypoint.sh"]
# Mirrors pm2 (local/ecosystem.config.js): .venv/bin/lz-tts-worker
CMD ["lz-tts-worker"]
