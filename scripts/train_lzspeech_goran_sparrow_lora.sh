#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

DATASET_SRC_DIR="${DATASET_SRC_DIR:-/home/haiyami/Projects/gen-tts/data/lzspeech-lfl}"
SLICE_DIR="${SLICE_DIR:-$ROOT_DIR/local/data/source/lzspeech-goran}"
TRAIN_DIR="${TRAIN_DIR:-$ROOT_DIR/local/data/exp/lzspeech-goran-sparrow-24k}"
SPEAKER_CONFIG="${SPEAKER_CONFIG:-$ROOT_DIR/data/lzspeech-sparrow/config.json}"
EXPERIMENT="${EXPERIMENT:-lzspeech_sparrow_goran_voice_lora}"
GORAN_FIRST_ID="${GORAN_FIRST_ID:-2080}"
EXPECTED_GORAN_ITEMS="${EXPECTED_GORAN_ITEMS:-1556}"
MAX_WORKERS="${MAX_WORKERS:-16}"
BERT_DEVICE="${BERT_DEVICE:-auto}"
BERT_BATCH_SIZE="${BERT_BATCH_SIZE:-32}"

mkdir -p "$SLICE_DIR"
ln -sfn "$DATASET_SRC_DIR/wav" "$SLICE_DIR/wav"

awk -F'|' -v first_id="$GORAN_FIRST_ID" '
  $2 == "bs-BA" {
    split($1, parts, "_")
    if ((parts[length(parts)] + 0) >= first_id) print $0
  }
' "$DATASET_SRC_DIR/metadata.csv" > "$SLICE_DIR/metadata.csv.tmp"
mv "$SLICE_DIR/metadata.csv.tmp" "$SLICE_DIR/metadata.csv"

item_count="$(wc -l < "$SLICE_DIR/metadata.csv")"
if [[ "$item_count" -ne "$EXPECTED_GORAN_ITEMS" ]]; then
  echo "Expected $EXPECTED_GORAN_ITEMS Goran rows, found $item_count" >&2
  exit 1
fi

if [[ ! -f "$TRAIN_DIR/config.json" || ! -f "$TRAIN_DIR/dataset.parquet" ]]; then
  uv run python -m src.piper.train_preprocess \
    --language multilingual \
    --primary-voice en-us \
    --input-dir "$SLICE_DIR" \
    --output-dir "$TRAIN_DIR" \
    --dataset-format ljspeech \
    --sample-rate 24000 \
    --speaker-config "$SPEAKER_CONFIG" \
    --max-workers "$MAX_WORKERS"
fi

if [[ ! -f "$TRAIN_DIR/.bert_features_complete" ]]; then
  uv run python scripts/precompute_piper_bert_features.py \
    --dataset-dir "$TRAIN_DIR" \
    --model-name distilbert-base-multilingual-cased \
    --device "$BERT_DEVICE" \
    --batch-size "$BERT_BATCH_SIZE" \
    --max-phoneme-ids 900
  touch "$TRAIN_DIR/.bert_features_complete"
fi

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "Prepared Goran Sparrow adapter data in $TRAIN_DIR"
  exit 0
fi

uv run python -m src.piper.train_hydra \
  trainer=gpu \
  experiment="$EXPERIMENT" \
  "$@"
