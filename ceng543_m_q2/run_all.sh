#!/usr/bin/env bash
set -euo pipefail

# Paths
TRAIN_PY="./train_iwslt14_attention.py"
EVAL_PY="./eval_metrics.py"

# sanity check
if [ ! -f "$TRAIN_PY" ]; then
  echo "ERROR: $TRAIN_PY not found. Make sure you are in the correct directory."
  exit 1
fi
if [ ! -f "$EVAL_PY" ]; then
  echo "ERROR: $EVAL_PY not found. Make sure you are in the correct directory."
  exit 1
fi

# Attn list
ATTS=("bahdanau" "luong" "scaleddot")

# Directories
BASE_CHECKPOINT_DIR="./checkpoints"
BASE_OUTPUT_DIR="./outputs"
BASE_LOG_DIR="./run_logs"
mkdir -p "$BASE_CHECKPOINT_DIR" "$BASE_OUTPUT_DIR" "$BASE_LOG_DIR"

# backup original train script
TRAIN_BAK="${TRAIN_PY}.bak"
if [ ! -f "$TRAIN_BAK" ]; then
  cp "$TRAIN_PY" "$TRAIN_BAK"
fi

# run training for each attention
for A in "${ATTS[@]}"; do
  echo "=== Running training for attention: $A ==="
  LOGFILE="${BASE_LOG_DIR}/train_${A}.log"
  # replace the ATTN_CHOICE line in the script temporarily
  # assumes a line like: ATTN_CHOICE = "bahdanau"
  sed -E "s/^ATTN_CHOICE\s*=.*/ATTN_CHOICE = \"${A}\"/" "$TRAIN_BAK" > "$TRAIN_PY"
  # create per-attention output folders (train script already creates some, but ensure)
  mkdir -p "${BASE_CHECKPOINT_DIR}/${A}" "${BASE_OUTPUT_DIR}/${A}" "${BASE_LOG_DIR}/${A}"
  # run training (stdout+stderr to logfile)
  echo "Starting python $TRAIN_PY (output -> $LOGFILE)..."
  python "$TRAIN_PY" 2>&1 | tee "$LOGFILE"
  echo "Training for $A finished. Logs: $LOGFILE"
done

# restore original train script
mv -f "$TRAIN_BAK" "$TRAIN_PY"

echo "=== All trainings finished ==="

# Run evaluation for each attention (uses eval_metrics.py)
for A in "${ATTS[@]}"; do
  CKPT="${BASE_CHECKPOINT_DIR}/${A}/best.pt"
  if [ ! -f "$CKPT" ]; then
    echo "WARNING: checkpoint not found for ${A} at ${CKPT}, skipping eval."
    continue
  fi
  OUTDIR="${BASE_OUTPUT_DIR}/${A}"
  mkdir -p "$OUTDIR"
  EVAL_LOG="${BASE_LOG_DIR}/eval_${A}.log"
  echo "=== Evaluating ${A} -> checkpoint ${CKPT} ==="
  python "$EVAL_PY" --ckpt "$CKPT" --attn "$A" --batch 64 2>&1 | tee "$EVAL_LOG"
  # move/save any test visualizations if eval script writes them; keep logs
  echo "Eval for ${A} done. Log: ${EVAL_LOG}"
done

echo "All done. Check $BASE_CHECKPOINT_DIR, $BASE_OUTPUT_DIR, and $BASE_LOG_DIR for outputs."
