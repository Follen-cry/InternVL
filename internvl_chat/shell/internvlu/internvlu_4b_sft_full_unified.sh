#!/bin/bash
# =============================================================================
# InternVL-U full unified SFT (VLM + generation_decoder).
#
# Trains both the understanding side (text-CE, with LoRA on the LLM by default)
# and the generation side (flow-matching MSE) jointly. The VAE is frozen.
#
# Required environment variables:
#   INTERNVLU_CKPT        Path or HF id of the InternVL-U vlm/ subdir.
#                         (e.g. /path/to/snapshot/vlm). The pipeline siblings
#                         (generation_decoder/, vae/, scheduler/, processor/)
#                         are auto-detected as the parent of this path; pass
#                         SNAPSHOT_DIR explicitly to override.
#   META_PATH             Path to the dataset meta JSON. Each entry may carry
#                         an optional "task_type": "imgen" field; entries
#                         without one default to "understanding" so existing
#                         metas keep working.
#   OUTPUT_DIR            Where to write checkpoints / logs.
#   GPUS                  Number of GPUs on this node (default: 8).
#   PER_DEVICE_BATCH_SIZE Microbatch size per GPU (default: 1).
#
# Optional:
#   SNAPSHOT_DIR          Pipeline snapshot dir. Default: parent(INTERNVLU_CKPT).
#   BATCH_SIZE            Global effective batch size (default: 64).
#   MASTER_PORT           torchrun master_port (default: 34230).
#   IMGEN_RATIO           Fraction of imgen-only batches per epoch (default: 0.3).
#   GEN_LOSS_WEIGHT       Final gen-loss weight after warmup (default: 0.1).
#   GEN_LOSS_WARMUP       Linear warmup steps for gen-loss weight (default: 1000).
#   CFG_DROPOUT           Probability of replacing imgen prompt with <img_uncond> (0.1).
#   GEN_DECODER_LR        LR for the generation_decoder param group (default: 5e-5).
#   GEN_IMAGE_SIZE        Imgen target resolution before VAE encode (default: 1024).
#   INTERNVLU_PKG_PATH    Filesystem path to the `internvlu` Python package
#                         (default: /scratch/network/ssd2/junlin/ssl_mllm/InternVL-U).
# =============================================================================

set -x

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton-cache-$USER}
mkdir -p $TRITON_CACHE_DIR

GPUS=${GPUS:-8}
BATCH_SIZE=${BATCH_SIZE:-64}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

: "${INTERNVLU_CKPT:?Set INTERNVLU_CKPT to the InternVL-U vlm/ subdir path}"
: "${META_PATH:?Set META_PATH to the dataset meta JSON}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export INTERNVLU_PKG_PATH=${INTERNVLU_PKG_PATH:-/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U}
export MASTER_PORT=${MASTER_PORT:-34230}
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

mkdir -p "$OUTPUT_DIR"

SNAPSHOT_ARG=""
if [[ -n "${SNAPSHOT_DIR}" ]]; then
  SNAPSHOT_ARG="--snapshot_dir ${SNAPSHOT_DIR}"
fi

torchrun \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr=127.0.0.1 \
  --nproc_per_node=${GPUS} \
  --master_port=${MASTER_PORT} \
  internvl/train/internvl_chat_finetune_u_full.py \
  --model_name_or_path "${INTERNVLU_CKPT}" \
  ${SNAPSHOT_ARG} \
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
  --freeze_gen_decoder False \
  --freeze_vae True \
  --gen_decoder_lr ${GEN_DECODER_LR:-5e-5} \
  --gen_loss_weight ${GEN_LOSS_WEIGHT:-0.1} \
  --gen_loss_warmup_steps ${GEN_LOSS_WARMUP:-1000} \
  --imgen_ratio ${IMGEN_RATIO:-0.3} \
  --cfg_dropout ${CFG_DROPOUT:-0.1} \
  --gen_image_size ${GEN_IMAGE_SIZE:-1024} \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs 1 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --eval_strategy "no" \
  --save_strategy "steps" \
  --save_steps 500 \
  --save_total_limit 2 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
