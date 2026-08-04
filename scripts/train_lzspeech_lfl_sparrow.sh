#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_SRC_DIR="${DATASET_SRC_DIR:-/home/haiyami/Projects/gen-tts/data/lzspeech-lfl}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT_DIR/local/data/exp/lzspeech-lfl-sparrow-24k}"
SPEAKER_CONFIG="${SPEAKER_CONFIG:-$ROOT_DIR/local/configs/piper/lzspeech_lfl_speakers.json}"
EXPERIMENT="${EXPERIMENT:-lzspeech_sparrow_lfl_multilingual}"
MAX_WORKERS="${MAX_WORKERS:-}"
BERT_DEVICE="${BERT_DEVICE:-auto}"
BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-32}"
MAX_PHONEME_IDS="${MAX_PHONEME_IDS:-900}"

if [[ ! -f "$TRAIN_DIR/config.json" || ( ! -f "$TRAIN_DIR/dataset.parquet" && ! -f "$TRAIN_DIR/dataset.jsonl" ) ]]; then
  mkdir -p "$TRAIN_DIR"
  PREPROCESS_CMD=(uv run python -m src.piper.train_preprocess
    --language multilingual
    --primary-voice en-us
    --input-dir "$DATASET_SRC_DIR"
    --output-dir "$TRAIN_DIR"
    --dataset-format ljspeech
    --sample-rate 24000
    --speaker-config "$SPEAKER_CONFIG")
  if [[ -n "$MAX_WORKERS" ]]; then
    PREPROCESS_CMD+=(--max-workers "$MAX_WORKERS")
  fi
  if [[ -n "${ESPEAK_DATA_PATH:-}" ]]; then
    PREPROCESS_CMD+=(--espeak-data "$ESPEAK_DATA_PATH")
  fi
  "${PREPROCESS_CMD[@]}"
fi

if [[ ! -f "$TRAIN_DIR/.bert_features_complete" ]]; then
  uv run python scripts/precompute_piper_bert_features.py \
    --dataset-dir "$TRAIN_DIR" \
    --model-name distilbert-base-multilingual-cased \
    --device "$BERT_DEVICE" \
    --batch-size "$BERT_BATCH_SIZE" \
    --max-phoneme-ids "$MAX_PHONEME_IDS"
  touch "$TRAIN_DIR/.bert_features_complete"
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared Sparrow training data in $TRAIN_DIR"
  exit 0
fi

uv run python -m src.piper.train_hydra \
  trainer=gpu \
  experiment="$EXPERIMENT" \
  "$@"
