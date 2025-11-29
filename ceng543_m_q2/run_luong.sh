#!/usr/bin/env bash
set -euo pipefail

TRAIN_PY="./train_iwslt14_attention.py"
EVAL_PY="./eval_metrics.py"

if [ ! -f "$TRAIN_PY" ]; then
  echo "ERROR: $TRAIN_PY bulunamadı. Çalışma dizininde olduğundan emin ol."
  exit 1
fi

ATTN="luong"
BASE_CHECKPOINT_DIR="./checkpoints"
BASE_OUTPUT_DIR="./outputs"
BASE_LOG_DIR="./run_logs"

mkdir -p "${BASE_CHECKPOINT_DIR}/${ATTN}" "${BASE_OUTPUT_DIR}/${ATTN}" "${BASE_LOG_DIR}/${ATTN}"

# Set attention choice in the script
sed -i.bak "s/^ATTN_CHOICE = .*/ATTN_CHOICE = \"${ATTN}\"/" "$TRAIN_PY"

LOGFILE="${BASE_LOG_DIR}/train_${ATTN}.log"
echo "=== Running training for attention: $ATTN ==="
echo "Starting python $TRAIN_PY (output -> $LOGFILE)..."
python "$TRAIN_PY" 2>&1 | tee "$LOGFILE"
echo "Training for $ATTN finished. Logs: $LOGFILE"

# Restore original if backup exists
if [ -f "${TRAIN_PY}.bak" ]; then
  mv -f "${TRAIN_PY}.bak" "$TRAIN_PY"
fi

# Run evaluation
CKPT="${BASE_CHECKPOINT_DIR}/${ATTN}/best.pt"
if [ -f "$CKPT" ]; then
  OUTDIR="${BASE_OUTPUT_DIR}/${ATTN}"
  mkdir -p "$OUTDIR"
  EVAL_LOG="${BASE_LOG_DIR}/eval_${ATTN}.log"
  echo "=== Evaluating ${ATTN} -> checkpoint ${CKPT} ==="
  python "$EVAL_PY" --ckpt "$CKPT" --attn "$ATTN" --batch 64 2>&1 | tee "$EVAL_LOG"
  echo "Eval for ${ATTN} done. Log: ${EVAL_LOG}"
else
  echo "WARNING: checkpoint not found for ${ATTN} at ${CKPT}, skipping eval."
fi

echo "Done for ${ATTN}!"

