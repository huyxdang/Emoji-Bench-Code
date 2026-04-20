#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./run_magistral.sh [dataset_path] [-- extra evaluate_continuation.py args...]

Examples:
  ./run_magistral.sh
  ./run_magistral.sh artifacts/emoji-bench-dataset-100

Notes:
  - Defaults to artifacts/emoji-bench-dataset-100
  - Runs only magistral-medium-2509 on the B slice (L0 only)
  - Magistral is Mistral's reasoning model — reasoning is always-on (no effort dial),
    so "max reasoning" is inherent to the model selection. This script just maxes out
    the output-token budget: --max-output-tokens 40000 (Magistral Medium 1.2's ceiling).
  - Any args after -- are forwarded to every evaluate_continuation.py call
  - Runs judge+score after the eval phase finishes
  - Uses JUDGE_MODEL (default: gpt-5.4-mini-no-reasoning) and JUDGE_MAX_CONCURRENT (default: 8)
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-artifacts/emoji-bench-dataset-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODELS=("magistral-medium-2509")
MODE="prefill"
LEVELS=("0")
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-40000}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-mini-no-reasoning}"
JUDGE_MAX_CONCURRENT="${JUDGE_MAX_CONCURRENT:-8}"

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
      echo "run_magistral.sh does not support forwarding --output-dir because it breaks judge/score routing." >&2
      exit 2
    fi
    if [[ "$arg" == "--model" ]]; then
      echo "run_magistral.sh does not support forwarding --model because it is dedicated to: ${MODELS[*]}." >&2
      exit 2
    fi
  done
fi

TOTAL_RUNS=$(( ${#MODELS[@]} * ${#LEVELS[@]} ))
RUN_INDEX=0
SUCCESS_COUNT=0
SUCCESSFUL_OUTPUT_DIRS=()
FAILED_RUNS=()
JUDGE_SUCCESS_COUNT=0
JUDGE_FAILED_RUNS=()
SCORE_SUCCESS_COUNT=0
SCORE_FAILED_RUNS=()

for model in "${MODELS[@]}"; do
  for level in "${LEVELS[@]}"; do
    RUN_INDEX=$((RUN_INDEX + 1))
    echo "[$RUN_INDEX/$TOTAL_RUNS] model=$model mode=$MODE turn_2_level=$level max_output_tokens=$MAX_OUTPUT_TOKENS"
    EVAL_CMD=(
      "$PYTHON_BIN"
      scripts/evaluate_continuation.py
      "$DATASET"
      --model "$model"
      --mode "$MODE"
      --turn-2-prompt-level "$level"
      --max-output-tokens "$MAX_OUTPUT_TOKENS"
    )
    if (( ${#EXTRA_ARGS[@]} > 0 )); then
      EVAL_CMD+=("${EXTRA_ARGS[@]}")
    fi

    if "${EVAL_CMD[@]}"; then
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
      SUCCESSFUL_OUTPUT_DIRS+=("artifacts/evals/${model}-B-L${level}")
    else
      FAILED_RUNS+=("model=$model mode=$MODE turn_2_level=$level")
      echo "FAILED: model=$model mode=$MODE turn_2_level=$level" >&2
    fi
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

echo "All Magistral Medium 1.2 eval, judge, and score runs completed successfully."
