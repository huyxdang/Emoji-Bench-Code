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
  - Runs the full 32-run matrix:
      8 models x 2 modes (prefill, single_turn) x 2 prompt levels (L0, L1)
  - Any args after -- are forwarded to every evaluate_continuation.py call
  - Runs judge+score after the eval phase finishes
  - Continues past failed cells and prints a final failure summary
  - Uses JUDGE_MODEL (default: gpt-5.4-mini-no-reasoning) and JUDGE_MAX_CONCURRENT (default: 8)
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-artifacts/emoji-bench-dataset-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-no-reasoning}"
JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-8}"

EXTRA_ARGS=()
if [[ $# -ge 2 ]]; then
  if [[ "$2" == "--" ]]; then
    EXTRA_ARGS=("${@:3}")
  else
    EXTRA_ARGS=("${@:2}")
  fi
fi

for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "$arg" == "--output-dir" ]]; then
    echo "run.sh does not support forwarding --output-dir because it breaks per-cell judge/score routing." >&2
    exit 2
  fi
done

MODELS=(
  "claude-opus-4-6-reasoning-high"
  "claude-sonnet-4-6-reasoning-high"
  "gpt-5.4-reasoning-xhigh"
  "gpt-5.4-mini-reasoning-xhigh"
  "gemini-3.1-pro-preview-thinking-high"
  "gemini-3-flash-preview-thinking-high"
  "mistral-large-2512"
  "magistral-medium-2509"
)

MODES=("prefill" "single_turn")
LEVELS=("0" "1")
TOTAL_RUNS=$(( ${#MODELS[@]} * ${#MODES[@]} * ${#LEVELS[@]} ))
RUN_INDEX=0
SUCCESS_COUNT=0
FAILED_RUNS=()
SUCCESSFUL_OUTPUT_DIRS=()
JUDGE_SUCCESS_COUNT=0
JUDGE_FAILED_RUNS=()
SCORE_SUCCESS_COUNT=0
SCORE_FAILED_RUNS=()

matrix_variant() {
  local mode="$1"
  if [[ "$mode" == "prefill" ]]; then
    echo "B"
  else
    echo "C"
  fi
}

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    for level in "${LEVELS[@]}"; do
      RUN_INDEX=$((RUN_INDEX + 1))
      echo "[$RUN_INDEX/$TOTAL_RUNS] model=$model mode=$mode turn_2_level=$level"
      if "$PYTHON_BIN" scripts/evaluate_continuation.py "$DATASET" \
        --model "$model" \
        --mode "$mode" \
        --turn-2-prompt-level "$level" \
        "${EXTRA_ARGS[@]}"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        SUCCESSFUL_OUTPUT_DIRS+=("artifacts/evals/${model}-$(matrix_variant "$mode")-L${level}")
      else
        FAILED_RUNS+=("model=$model mode=$mode turn_2_level=$level")
        echo "FAILED: model=$model mode=$mode turn_2_level=$level" >&2
      fi
    done
  done
done

echo
echo "Eval phase completed: $SUCCESS_COUNT/$TOTAL_RUNS runs successful."

for output_dir in "${SUCCESSFUL_OUTPUT_DIRS[@]}"; do
  echo "Judging: $output_dir"
  if "$PYTHON_BIN" scripts/judge_continuation.py "$output_dir" \
    --judge-model "$JUDGE_MODEL" \
    --max-concurrent "$JUDGE_MAX_CONCURRENT"; then
    JUDGE_SUCCESS_COUNT=$((JUDGE_SUCCESS_COUNT + 1))
  else
    JUDGE_FAILED_RUNS+=("$output_dir")
    echo "JUDGE FAILED: $output_dir" >&2
    continue
  fi

  echo "Scoring: $output_dir"
  if "$PYTHON_BIN" scripts/score_continuation.py "$output_dir"; then
    SCORE_SUCCESS_COUNT=$((SCORE_SUCCESS_COUNT + 1))
  else
    SCORE_FAILED_RUNS+=("$output_dir")
    echo "SCORE FAILED: $output_dir" >&2
  fi
done

echo
echo "Judge phase completed: $JUDGE_SUCCESS_COUNT/${#SUCCESSFUL_OUTPUT_DIRS[@]} runs successful."
echo "Score phase completed: $SCORE_SUCCESS_COUNT/${#SUCCESSFUL_OUTPUT_DIRS[@]} runs successful."

if (( ${#FAILED_RUNS[@]} > 0 )); then
  echo "Failed eval runs:"
  for failed in "${FAILED_RUNS[@]}"; do
    echo "  - $failed"
  done
fi

if (( ${#JUDGE_FAILED_RUNS[@]} > 0 )); then
  echo "Failed judge runs:"
  for failed in "${JUDGE_FAILED_RUNS[@]}"; do
    echo "  - $failed"
  done
fi

if (( ${#SCORE_FAILED_RUNS[@]} > 0 )); then
  echo "Failed score runs:"
  for failed in "${SCORE_FAILED_RUNS[@]}"; do
    echo "  - $failed"
  done
fi

if (( ${#FAILED_RUNS[@]} > 0 || ${#JUDGE_FAILED_RUNS[@]} > 0 || ${#SCORE_FAILED_RUNS[@]} > 0 )); then
  exit 1
fi

echo "All eval, judge, and score runs completed successfully."
