#!/bin/bash
# =============================================================================
# InternVL-U LoRA SFT (text-only CE on understanding side).
#
# Required environment variables:
#   INTERNVLU_CKPT        Path or HF id of the InternVL-U base checkpoint.
#                         Must be loadable by InternVLUChatConfig.from_pretrained.
#   META_PATH             Path to the dataset meta JSON (same format as InternVL
#                         finetune meta files).
#   OUTPUT_DIR            Where to write checkpoints / logs.
#   GPUS                  Number of GPUs on this node (default: 8).
#   PER_DEVICE_BATCH_SIZE Microbatch size per GPU (default: 2).
#
# Optional:
#   BATCH_SIZE            Global effective batch size (default: 128). Used to
#                         derive gradient_accumulation_steps.
#   MASTER_PORT           torchrun master_port (default: 34229).
# =============================================================================

export TRITON_CACHE_DIR=/tmp/triton-cache-$USER
mkdir -p $TRITON_CACHE_DIR

set -x
export CUDA_VISIBLE_DEVICES=2,3,4,5
GPUS=${GPUS:-4}
BATCH_SIZE=${BATCH_SIZE:-128}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

: "${INTERNVLU_CKPT:=/homes/55/junlin/.cache/huggingface/hub/models--InternVL-U--InternVL-U/snapshots/f012d760e69712bb47f7d3d09a24280f346cee01/vlm}"
: "${META_PATH:=/scratch/network/ssd2/junlin/ssl_mllm/data/meta/spatial_ssrl_co_sft_meta.json}"
: "${OUTPUT_DIR:=/scratch/network/ssd2/junlin/models/internvl-u-4epoch}"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=${MASTER_PORT:-34229}
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

mkdir -p "$OUTPUT_DIR"

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune_u.py \
  --model_name_or_path "${INTERNVLU_CKPT}" \
  --conv_style "qwen2_5-chat-v3" \
  --use_fast_tokenizer False \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 6 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.0 \
  --freeze_llm True \
  --freeze_mlp False \
  --freeze_backbone True \
  --use_llm_lora 16 \
  --vision_select_layer -1 \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 4 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --eval_strategy "no" \
  --save_strategy "no" \
  --save_total_limit 1 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
