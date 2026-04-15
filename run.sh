#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage:
  ./run_32_model_matrix.sh [dataset_path] [-- extra evaluate_continuation.py args...]

Examples:
  ./run_32_model_matrix.sh
  ./run_32_model_matrix.sh artifacts/emoji-bench-dataset-100
  ./run_32_model_matrix.sh artifacts/emoji-bench-dataset-100 -- --max-concurrent 8

Notes:
  - Defaults to artifacts/emoji-bench-dataset-100
  - Runs the full 32-run matrix:
      8 models x 2 modes (prefill, single_turn) x 2 prompt levels (L0, L1)
  - Any args after -- are forwarded to every evaluate_continuation.py call
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

for model in "${MODELS[@]}"; do
  for mode in "${MODES[@]}"; do
    for level in "${LEVELS[@]}"; do
      RUN_INDEX=$((RUN_INDEX + 1))
      echo "[$RUN_INDEX/$TOTAL_RUNS] model=$model mode=$mode turn_2_level=$level"
      "$PYTHON_BIN" scripts/evaluate_continuation.py "$DATASET" \
        --model "$model" \
        --mode "$mode" \
        --turn-2-prompt-level "$level" \
        "${EXTRA_ARGS[@]}"
    done
  done
done
