#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./run.sh [dataset_path] [-- extra evaluate_continuation.py args...]

Examples:
  ./run.sh
  ./run.sh artifacts/emoji-bench-dataset-100
  ./run.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8

Notes:
  - Defaults to artifacts/emoji-bench-dataset-100
  - Runs only the B-L0 headline slice:
      9 models x prefill delivery x L0 ("Please continue.")
  - Any args after -- are forwarded to every evaluate_continuation.py call
  - Scores final-answer-only after the eval phase finishes
  - Generates B-variant final-answer plots in artifacts/plots/
  - Continues past failed cells and prints a final failure summary
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-artifacts/emoji-bench-dataset-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"

declare -a EXTRA_ARGS=()
if [[ $# -ge 2 ]]; then
  if [[ "$2" == "--" ]]; then
    EXTRA_ARGS=("${@:3}")
  else
    EXTRA_ARGS=("${@:2}")
  fi
fi

if (( ${#EXTRA_ARGS[@]} > 0 )); then
  for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--output-dir" ]]; then
      echo "run.sh does not support forwarding --output-dir because it breaks per-cell score routing." >&2
      exit 2
    fi
  done
fi

MODELS=(
  "claude-opus-4-7-reasoning-max"
  "claude-opus-4-6-reasoning-max"
  "claude-sonnet-4-6-reasoning-max"
  "gpt-5.4-reasoning-xhigh"
  "gpt-5.4-mini-reasoning-xhigh"
  "gemini-3.1-pro-preview-thinking-high"
  "gemini-3-flash-preview-thinking-high"
  "mistral-large-2512"
  "magistral-medium-2509"
)

MODE="prefill"
LEVEL="0"
TOTAL_RUNS=${#MODELS[@]}
RUN_INDEX=0
SUCCESS_COUNT=0
FAILED_RUNS=()
SUCCESSFUL_OUTPUT_DIRS=()
SCORE_SUCCESS_COUNT=0
SCORE_FAILED_RUNS=()

for model in "${MODELS[@]}"; do
  RUN_INDEX=$((RUN_INDEX + 1))
  echo "[$RUN_INDEX/$TOTAL_RUNS] model=$model mode=$MODE turn_2_level=$LEVEL"
  EVAL_CMD=(
    "$PYTHON_BIN"
    scripts/evaluate_continuation.py
    "$DATASET"
    --model "$model"
    --mode "$MODE"
    --turn-2-prompt-level "$LEVEL"
  )
  if (( ${#EXTRA_ARGS[@]} > 0 )); then
    EVAL_CMD+=("${EXTRA_ARGS[@]}")
  fi
  if "${EVAL_CMD[@]}"; then
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    SUCCESSFUL_OUTPUT_DIRS+=("artifacts/evals/${model}-B-L${LEVEL}")
  else
    FAILED_RUNS+=("model=$model mode=$MODE turn_2_level=$LEVEL")
    echo "FAILED: model=$model mode=$MODE turn_2_level=$LEVEL" >&2
  fi
done

echo
echo "Eval phase completed: $SUCCESS_COUNT/$TOTAL_RUNS runs successful."

for output_dir in "${SUCCESSFUL_OUTPUT_DIRS[@]}"; do
  echo "Scoring: $output_dir"
  if "$PYTHON_BIN" scripts/score_continuation.py "$output_dir"; then
    SCORE_SUCCESS_COUNT=$((SCORE_SUCCESS_COUNT + 1))
  else
    SCORE_FAILED_RUNS+=("$output_dir")
    echo "SCORE FAILED: $output_dir" >&2
  fi
done

echo
echo "Score phase completed: $SCORE_SUCCESS_COUNT/${#SUCCESSFUL_OUTPUT_DIRS[@]} runs successful."

PLOT_FAILED=0
echo
echo "Generating plots..."
if "$PYTHON_BIN" scripts/plot_b_final_answer.py; then
  echo "Plots written to artifacts/plots/"
else
  echo "PLOT FAILED" >&2
  PLOT_FAILED=1
fi

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
