#!/usr/bin/env bash
set -euo pipefail

# Train a non-BERT Piper/VITS model for lzspeech-multilingual-plus.
# Defaults are intentionally phoneme-only: no semantic encoder and no BERT input.

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_SRC_DIR="${DATASET_SRC_DIR:-$ROOT_DIR/local/datasets/lzspeech-multilingual-plus}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT_DIR/local/data/exp/lzspeech-multilingual-plus-piper}"
PUBLISH_DIR="${PUBLISH_DIR:-$ROOT_DIR/data/lzspeech-multilingual}"

SAMPLE_RATE="${SAMPLE_RATE:-22050}"
LANGUAGE="${LANGUAGE:-multilingual}"
PRIMARY_VOICE="${PRIMARY_VOICE:-en-us}"
QUALITY="${QUALITY:-high}"
BATCH_SIZE="${BATCH_SIZE:-16}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-500}"
MAX_PHONEME_IDS="${MAX_PHONEME_IDS:-400}"
PRECISION="${PRECISION:-16-mixed}"
CHECKPOINT_EPOCHS="${CHECKPOINT_EPOCHS:-1}"
VALIDATION_SPLIT="${VALIDATION_SPLIT:-0.0}"
NUM_TEST_EXAMPLES="${NUM_TEST_EXAMPLES:-0}"
MAX_WORKERS="${MAX_WORKERS:-}"

# Optional non-BERT warm start. BERT checkpoints are rejected by src.piper.train.
DEFAULT_INIT_CKPT="/mnt/data/piper/local/models/lzspeech-multilingual-299.ckpt"
INIT_CKPT="${INIT_CKPT:-}"
if [[ -z "$INIT_CKPT" && -f "$DEFAULT_INIT_CKPT" ]]; then
  INIT_CKPT="$DEFAULT_INIT_CKPT"
fi

if [[ "${FORCE_PREPROCESS:-0}" == "1" ]]; then
  rm -f "$TRAIN_DIR/config.json" "$TRAIN_DIR/dataset.jsonl"
fi

if [[ ! -f "$TRAIN_DIR/config.json" || ! -f "$TRAIN_DIR/dataset.jsonl" ]]; then
  mkdir -p "$TRAIN_DIR"
  echo "Preprocessing $DATASET_SRC_DIR -> $TRAIN_DIR"
  PREPROCESS_CMD=(uv run python -m src.piper.train_preprocess
    --language "$LANGUAGE"
    --primary-voice "$PRIMARY_VOICE"
    --input-dir "$DATASET_SRC_DIR"
    --output-dir "$TRAIN_DIR"
    --dataset-format ljspeech
    --sample-rate "$SAMPLE_RATE")
  if [[ -n "$MAX_WORKERS" ]]; then
    PREPROCESS_CMD+=(--max-workers "$MAX_WORKERS")
  fi
  if [[ -n "${ESPEAK_DATA_PATH:-}" ]]; then
    PREPROCESS_CMD+=(--espeak-data "$ESPEAK_DATA_PATH")
  fi
  "${PREPROCESS_CMD[@]}"
fi

if [[ "${SKIP_TRAIN:-0}" == "1" ]]; then
  echo "SKIP_TRAIN=1; preprocessed files are in $TRAIN_DIR"
  exit 0
fi

ACCEL="${ACCEL:-}"
if [[ -z "$ACCEL" ]]; then
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    ACCEL="gpu"
  else
    ACCEL="cpu"
  fi
fi

if [[ "$ACCEL" == "cpu" && "$PRECISION" != "32" ]]; then
  PRECISION="32"
fi

DEVICES="${DEVICES:-1}"
if [[ "$ACCEL" == "gpu" && -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
  DEVICES="$(echo "$GPU_ID" | tr ',' '\n' | wc -l)"
fi

RESUME_ARGS=()
LATEST_CKPT=""
if [[ -d "$TRAIN_DIR/lightning_logs" ]]; then
  LATEST_CKPT="$(find "$TRAIN_DIR/lightning_logs" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')" || true
fi

if [[ "${SKIP_RESUME:-0}" != "1" && -n "$LATEST_CKPT" && -f "$LATEST_CKPT" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$LATEST_CKPT")
elif [[ -n "$INIT_CKPT" && -f "$INIT_CKPT" ]]; then
  RESUME_ARGS=(--init_from_checkpoint "$INIT_CKPT")
fi

echo "Training non-BERT Piper/VITS model"
echo "  dataset:   $DATASET_SRC_DIR"
echo "  train dir: $TRAIN_DIR"
echo "  quality:   $QUALITY"
echo "  accel:     $ACCEL devices=$DEVICES precision=$PRECISION"
echo "  workers:   preprocess=${MAX_WORKERS:-auto} train=$TRAIN_NUM_WORKERS"
echo "  resume:    ${RESUME_ARGS[*]:-(none)}"

uv run python -m src.piper.train \
  --dataset-dir "$TRAIN_DIR" \
  --default_root_dir "$TRAIN_DIR" \
  --accelerator "$ACCEL" \
  --devices "$DEVICES" \
  --precision "$PRECISION" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$TRAIN_NUM_WORKERS" \
  --validation-split "$VALIDATION_SPLIT" \
  --num-test-examples "$NUM_TEST_EXAMPLES" \
  --max_epochs "$MAX_EPOCHS" \
  --checkpoint-epochs "$CHECKPOINT_EPOCHS" \
  --quality "$QUALITY" \
  --max-phoneme-ids "$MAX_PHONEME_IDS" \
  "${RESUME_ARGS[@]}" \
  ${EXTRA_ARGS:-}

if [[ "${PUBLISH:-0}" == "1" ]]; then
  mkdir -p "$PUBLISH_DIR"
  FINAL_CKPT="$(find "$TRAIN_DIR/lightning_logs" -type f -name '*.ckpt' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -1 | awk '{print $2}')"
  if [[ -z "$FINAL_CKPT" || ! -f "$FINAL_CKPT" ]]; then
    echo "No checkpoint found to publish" >&2
    exit 1
  fi
  cp "$TRAIN_DIR/config.json" "$PUBLISH_DIR/config.json"
  cp "$FINAL_CKPT" "$PUBLISH_DIR/model.ckpt"
  echo "Published $PUBLISH_DIR/config.json and $PUBLISH_DIR/model.ckpt"
fi
