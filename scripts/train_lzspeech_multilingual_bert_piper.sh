#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

if [[ -n "${GPU_ID:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

TRAINER="${TRAINER:-gpu}"
EXPERIMENT="${EXPERIMENT:-lzspeech_multilingual_bert_87m}"

echo "Training Piper/VITS BERT model with Hydra"
echo "  trainer:    $TRAINER"
echo "  experiment: $EXPERIMENT"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  cuda:       $CUDA_VISIBLE_DEVICES"
fi

uv run python -m src.piper.train_hydra \
  trainer="$TRAINER" \
  experiment="$EXPERIMENT" \
  "$@"
