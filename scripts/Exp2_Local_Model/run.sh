#!/usr/bin/env bash
# Run E-CONTINUE benchmark on local models via vLLM.
#
# Usage:
#   ./scripts/Exp2_Local_Model/run.sh [dataset_path] [-- extra evaluate args...]
#
# Environment variables:
#   GPUS             Space-separated GPU IDs to use (default: "0")
#   TP_SIZE          Tensor parallel size (default: number of GPUs in GPUS)
#   PYTHON_BIN       Python executable (default: python)
#   GPU_MEM_UTIL     vLLM GPU memory utilization (default: 0.9)
#
# Examples:
#   GPUS="0 1 2 3" TP_SIZE=4 ./scripts/Exp2_Local_Model/run.sh artifacts/emoji-bench-dataset-100
#   ./scripts/Exp2_Local_Model/run.sh artifacts/emoji-bench-dataset-100 -- --batch-size 64
# GPUS="0 1 2 3" TP_SIZE=4 GPU_MEM_UTIL=0.85 bash scripts/Exp2_Local_Model/run.sh artifacts/emoji-bench-dataset-100



set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./scripts/Exp2_Local_Model/run.sh [dataset_path] [-- extra evaluate args...]

Environment:
  GPUS          Space-separated GPU IDs (default: "0")
  TP_SIZE       Tensor parallel size (default: len(GPUS))
  PYTHON_BIN    Python binary (default: python)
  GPU_MEM_UTIL  vLLM GPU memory fraction (default: 0.9)

Notes:
  - Runs B-L0 and B-L1 (prefill x L0/L1) for all 5 local models.
  - Scores each successful eval and generates B-variant plots.
  - Continues past failed cells and prints a final failure summary.
  - Artifact dirs: scripts/Exp2_Local_Model/artifacts/evals/<model-slug>-B-L{0,1}/
  - Plots:         scripts/Exp2_Local_Model/artifacts/plots/
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-artifacts/emoji-bench-dataset-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.9}"

# All artifacts (evals + plots) live under this script's own directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARTIFACTS_DIR="$SCRIPT_DIR/artifacts"

# Parse GPU config
read -ra GPU_ARRAY <<< "${GPUS:-0}"
TP_SIZE="${TP_SIZE:-${#GPU_ARRAY[@]}}"

# Parse -- extra args forwarded to evaluation_continuation_vllm.py
declare -a EXTRA_ARGS=()
if [[ $# -ge 2 ]]; then
  shift
  if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
  else
    EXTRA_ARGS=("$@")
  fi
fi

MODELS=(
  "google/gemma-3-12b-it"
  # "meta-llama/Llama-3.1-8B-Instruct"
  "Qwen/Qwen3-8B"
  # "openai/gpt-oss-20b"
  # "openai/gpt-oss-120b"
  # "tiiuae/Falcon3-10B-Instruct"
)
LEVELS=("0" "1")
MODE="prefill"
TOTAL_RUNS=$(( ${#MODELS[@]} * ${#LEVELS[@]} ))
RUN_INDEX=0
SUCCESS_COUNT=0
FAILED_RUNS=()
SUCCESSFUL_OUTPUT_DIRS=()
SCORE_SUCCESS_COUNT=0
SCORE_FAILED_RUNS=()

echo "Using GPUs: ${GPU_ARRAY[*]}  tensor-parallel-size: $TP_SIZE"
echo "Dataset:    $DATASET"
echo "Runs:       $TOTAL_RUNS  (${#MODELS[@]} models x ${#LEVELS[@]} levels)"
echo

for level in "${LEVELS[@]}"; do
  for model in "${MODELS[@]}"; do
    rm -rf /home/long_2/hai/tmp
    RUN_INDEX=$(( RUN_INDEX + 1 ))
    slug=$(echo "$model" | tr '/' '-')
    output_dir="$ARTIFACTS_DIR/evals/${slug}-B-L${level}"

    echo "[$RUN_INDEX/$TOTAL_RUNS] model=$model  mode=$MODE  level=$level"
    echo "  output -> $output_dir"

    EVAL_CMD=(
      env "TORCHINDUCTOR_CACHE_DIR=/data/long/hai/tmp/torchinductor"
      "$PYTHON_BIN"
      scripts/Exp2_Local_Model/evaluation_continuation_vllm.py
      "$DATASET"
      --model "$model"
      --mode "$MODE"
      --turn-2-prompt-level "$level"
      --output-dir "$output_dir"
      --gpus "${GPU_ARRAY[@]}"
      --tensor-parallel-size "$TP_SIZE"
      --gpu-memory-utilization "$GPU_MEM_UTIL"
    )
    if (( ${#EXTRA_ARGS[@]} > 0 )); then
      EVAL_CMD+=("${EXTRA_ARGS[@]}")
    fi

    if "${EVAL_CMD[@]}"; then
      SUCCESS_COUNT=$(( SUCCESS_COUNT + 1 ))
      SUCCESSFUL_OUTPUT_DIRS+=("$output_dir")
    else
      FAILED_RUNS+=("model=$model level=$level")
      echo "FAILED: model=$model level=$level" >&2
    fi
    echo
  done
done

echo "Eval phase completed: $SUCCESS_COUNT/$TOTAL_RUNS runs successful."
echo

if (( ${#SUCCESSFUL_OUTPUT_DIRS[@]} > 0 )); then
  for output_dir in "${SUCCESSFUL_OUTPUT_DIRS[@]}"; do
    echo "Scoring: $output_dir"
    if "$PYTHON_BIN" scripts/Exp2_Local_Model/score_continuation.py "$output_dir"; then
      SCORE_SUCCESS_COUNT=$(( SCORE_SUCCESS_COUNT + 1 ))
    else
      SCORE_FAILED_RUNS+=("$output_dir")
      echo "SCORE FAILED: $output_dir" >&2
    fi
  done
else
  echo "No successful eval runs to score."
fi

echo
echo "Score phase completed: $SCORE_SUCCESS_COUNT/${#SUCCESSFUL_OUTPUT_DIRS[@]} runs successful."

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
if (( ${#FAILED_RUNS[@]} > 0 )); then
  echo "Failed eval runs:"
  for failed in "${FAILED_RUNS[@]}"; do
    echo "  - $failed"
  done
fi

if (( ${#SCORE_FAILED_RUNS[@]} > 0 )); then
  echo "Failed score runs:"
  for failed in "${SCORE_FAILED_RUNS[@]}"; do
    echo "  - $failed"
  done
fi

if (( ${#FAILED_RUNS[@]} > 0 || ${#SCORE_FAILED_RUNS[@]} > 0 || PLOT_FAILED > 0 )); then
  exit 1
fi

echo "All eval, score, and plot steps completed successfully."
