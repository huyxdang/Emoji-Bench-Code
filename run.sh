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
  - Continues past failed cells and prints a final failure summary
EOF
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

DATASET="${1:-artifacts/emoji-bench-dataset-100}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=()
if [[ $# -ge 2 ]]; then
  if [[ "$2" == "--" ]]; then
    EXTRA_ARGS=("${@:3}")
  else
    EXTRA_ARGS=("${@:2}")
  fi
fi

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
      else
        FAILED_RUNS+=("model=$model mode=$mode turn_2_level=$level")
        echo "FAILED: model=$model mode=$mode turn_2_level=$level" >&2
      fi
    done
  done
done

echo
echo "Completed $SUCCESS_COUNT/$TOTAL_RUNS runs successfully."

if (( ${#FAILED_RUNS[@]} > 0 )); then
  echo "Failed runs:"
  for failed in "${FAILED_RUNS[@]}"; do
    echo "  - $failed"
  done
  exit 1
fi

echo "All runs completed successfully."
