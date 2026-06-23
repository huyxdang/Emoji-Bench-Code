#!/usr/bin/env bash
set -euo pipefail
# bash scripts/Exp2_Local_Model/run.sh artifacts/emoji-bench-dataset-100-boxed-ver
# ── Edit these to match your setup ──────────────────────────────────────────
GPUS=(0 1 2 3)
TP_SIZE=4
GPU_MEM_UTIL=0.9
DATASET="artifacts/emoji-bench-dataset-100"
PYTHON_BIN="python"

MODELS=(
  # "Qwen/Qwen2.5-3B-Instruct"
  "scripts/Exp2_Local_Model/grpo_train/finetuned_models/Qwen2.5-3B-Instruct/emoji_grpo"
  # "google/gemma-3-12b-it"
  # "meta-llama/Llama-3.1-8B-Instruct"
  # "Qwen/Qwen3-8B"
  # "tiiuae/Falcon3-10B-Instruct"
)
LEVELS=("0" "1")
MODE="prefill"
# ────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#LEVELS[@]} ))
RUN_INDEX=0
SUCCESS_COUNT=0
FAILED_RUNS=()
SUCCESSFUL_OUTPUT_DIRS=()
SCORE_SUCCESS_COUNT=0
SCORE_FAILED_RUNS=()

echo "GPUs: ${GPUS[*]}  tp=$TP_SIZE  mem=$GPU_MEM_UTIL"
echo "Dataset: $DATASET"
echo "Runs: $TOTAL_RUNS  (${#MODELS[@]} models x ${#LEVELS[@]} levels)"
echo

for level in "${LEVELS[@]}"; do
  for model in "${MODELS[@]}"; do
    RUN_INDEX=$(( RUN_INDEX + 1 ))
    slug=$(echo "$model" | tr '/' '-')
    output_dir="$ARTIFACTS_DIR/evals/${slug}-B-L${level}"

    echo "[$RUN_INDEX/$TOTAL_RUNS] model=$model  level=$level"
    echo "  output -> $output_dir"

    if env "TORCHINDUCTOR_CACHE_DIR=/data/long/hai/tmp/torchinductor" \
        "$PYTHON_BIN" scripts/Exp2_Local_Model/evaluation_continuation_vllm.py \
        "$DATASET" \
        --model "$model" --mode "$MODE" --turn-2-prompt-level "$level" \
        --output-dir "$output_dir" \
        --gpus "${GPUS[@]}" --tensor-parallel-size "$TP_SIZE" \
        --gpu-memory-utilization "$GPU_MEM_UTIL"; then
      SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
      SUCCESSFUL_OUTPUT_DIRS+=("$output_dir")
    else
      FAILED_RUNS+=("model=$model level=$level")
      echo "FAILED: model=$model level=$level" >&2
    fi
    echo
  done
done

echo "Eval: $SUCCESS_COUNT/$TOTAL_RUNS successful."
echo

for output_dir in "${SUCCESSFUL_OUTPUT_DIRS[@]+"${SUCCESSFUL_OUTPUT_DIRS[@]}"}"; do
  echo "Scoring: $output_dir"
  if "$PYTHON_BIN" scripts/Exp2_Local_Model/score_continuation.py "$output_dir"; then
    SCORE_SUCCESS_COUNT=$(( SCORE_SUCCESS_COUNT + 1 ))
  else
    SCORE_FAILED_RUNS+=("$output_dir")
    echo "SCORE FAILED: $output_dir" >&2
  fi
done

echo
echo "Score: $SCORE_SUCCESS_COUNT/${#SUCCESSFUL_OUTPUT_DIRS[@]} successful."

PLOT_FAILED=0
echo
echo "Generating plots..."
if "$PYTHON_BIN" scripts/plot_b_final_answer.py \
     --evals-dir "$ARTIFACTS_DIR/evals" \
     --output-dir "$ARTIFACTS_DIR/plots"; then
  echo "Plots written to $ARTIFACTS_DIR/plots/"
else
  echo "PLOT FAILED" >&2
  PLOT_FAILED=1
fi

echo
if [[ ${#FAILED_RUNS[@]} -gt 0 ]]; then
  echo "Failed eval runs:"
  printf '  - %s\n' "${FAILED_RUNS[@]}"
fi
if [[ ${#SCORE_FAILED_RUNS[@]} -gt 0 ]]; then
  echo "Failed score runs:"
  printf '  - %s\n' "${SCORE_FAILED_RUNS[@]}"
fi

if (( ${#FAILED_RUNS[@]} > 0 || ${#SCORE_FAILED_RUNS[@]} > 0 || PLOT_FAILED > 0 )); then
  exit 1
fi

echo "All steps completed successfully."
