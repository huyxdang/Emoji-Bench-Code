#!/bin/bash
# GRPO training for E-CONTINUE benchmark (emoji-bench)
#
# Usage:
#   bash scripts/Exp2_Local_Model/grpo_train/emoji_grpo.sh [model]
#
# Example:
#   bash scripts/Exp2_Local_Model/grpo_train/emoji_grpo.sh Qwen/Qwen3-8B
#   bash scripts/Exp2_Local_Model/grpo_train/emoji_grpo.sh meta-llama/Llama-3.1-8B-Instruct

set -e

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6

base_model="${1:-Qwen/Qwen3-8B}"
model="${base_model##*/}"
dataset_name="emoji_grpo"

# Navigate to repo root regardless of where the script is called from
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"


# FSDP config: match model family (same logic as SAI.sh)
case "${base_model}" in
  *[Ll]lama*)    fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_llama.json" ;;
  *[Qq]wen3*)    fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_qwen3.json" ;;
  *[Qq]wen*)     fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_qwen.json"  ;;
  *[Mm]istral*)  fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_mistral.json" ;;
  *[Gg]emma*)    fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_gemma.json" ;;
  *[Ff]alcon3*)  fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_llama.json" ;;
  *[Ff]alcon*)   fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_falcon.json" ;;
  *)             fsdp_config="scripts/Exp2_Local_Model/grpo_train/fsdp_config_qwen.json"  ;;
esac

# Count only the GPUs that are actually visible to this process.
# CUDA_VISIBLE_DEVICES remaps physical GPUs to logical indices 0..N-1,
# so nproc-per-node must equal N, not the total number on the machine.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -ra _visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
    gpu_count=${#_visible_gpus[@]}
else
    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
    gpu_count=${gpu_count:-1}
fi

grpo_dataset="scripts/Exp2_Local_Model/grpo_train/ft_dataset/${dataset_name}"
output_dir="scripts/Exp2_Local_Model/grpo_train/finetuned_models/${model}/${dataset_name}"

echo "Model:      ${base_model}"
echo "GPUs:       ${gpu_count}"
echo "FSDP cfg:   ${fsdp_config}"
echo "Output dir: ${output_dir}"
echo

echo "=== Step 1: Prepare GRPO dataset ==="
python scripts/Exp2_Local_Model/grpo_train/prepare_emoji_grpo_data.py \
    --input  "artifacts/emoji-bench-dataset-300-train-boxed-ver" \
    --output "${grpo_dataset}" \
    --test_split 0.05 \
    --seed 42

echo
echo "=== Step 2: GRPO training ==="
torchrun --nproc-per-node "${gpu_count}" --master_port 12349 \
    scripts/Exp2_Local_Model/grpo_train/emoji_grpo.py \
    --model_name                 "${base_model}" \
    --train_file_path            "${grpo_dataset}" \
    --output_dir                 "${output_dir}" \
    --per_device_train_batch_size  2 \
    --per_device_eval_batch_size   2 \
    --gradient_accumulation_steps  4 \
    --num_train_epochs             3 \
    --learning_rate                5e-7 \
    --max_completion_length        2048 \
    --num_generations              8 \
    --temperature                  0.8 \
    --top_p                        1.0 \
    --repetition_penalty           1.0 \
    --warmup_ratio                 0.1 \
    --lr_scheduler_type            cosine \
    --weight_decay                 0.05 \
    --bf16                         True \
    --logging_steps                5 \
    --save_strategy                "no" \
    --eval_strategy                "epoch" \
    --remove_unused_columns        False \
    --use_vllm                     True \
    --vllm_gpu_memory_utilization   0.8 \
    --vllm_mode                    colocate \
    --vllm_max_model_length        8192 \
    --fsdp full_shard --fsdp auto_wrap \
    --fsdp_config                  "${fsdp_config}" \
    --gradient_checkpointing       True \
    --push_to_hub                  False

echo
echo "Training complete. Model saved to ${output_dir}"
