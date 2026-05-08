"""Full unified-model SFT for InternVL-U.

Mirrors :mod:`internvl_chat_finetune_u` for arg parsing, dataset construction,
and tokenizer wiring, but trains both the VLM (understanding branch) **and**
the diffusion ``generation_decoder`` (imgen branch) jointly via
:class:`InternVLUUnifiedModel`.

The existing VLM-only script (``internvl_chat_finetune_u.py``) is unchanged.
This script lives next to it.

Usage: see ``shell/internvlu/internvlu_4b_sft_full_unified.sh``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import sys
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional

# --- internvlu inference package on sys.path so InternVLUUnifiedModel can import it ---
_INTERNVLU_PKG = os.environ.get(
    "INTERNVLU_PKG_PATH", "/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U"
)
if _INTERNVLU_PKG and os.path.isdir(_INTERNVLU_PKG) and _INTERNVLU_PKG not in sys.path:
    sys.path.insert(0, _INTERNVLU_PKG)

import numpy as np
import torch
import torch.distributed as dist
import transformers
from PIL import Image, ImageFile, PngImagePlugin
from torch.utils.data import ConcatDataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (
    enable_default_handler,
    enable_explicit_format,
    set_verbosity,
)

from internvl.dist_utils import init_dist
from internvl.model.internvlu import (
    InternVLUChatConfig,
    InternVLUChatModel,
    InternVLUUnifiedModel,
)
from internvl.model.internvlu.constants import SPECIAL_TOKEN_LIST
from internvl.patch import (
    concat_pad_data_collator,
    replace_internlm2_attention_class,
    replace_llama_attention_class,
    replace_llama_rmsnorm_with_fused_rmsnorm,
    replace_phi3_attention_class,
    replace_qwen2_attention_class,
    replace_train_dataloader,
    replace_train_sampler,
)
from internvl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    QUAD_END_TOKEN,
    QUAD_START_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
)
from internvl.train.dataset import TCSLoader
from internvl.train.dataset_unified import (
    ImgenLazyDataset,
    TaskTypeBatchSampler,
    UnifiedCollator,
    get_task_type,
)
from internvl.train.internvl_chat_finetune_u import (
    DataTrainingArguments,
    LazySupervisedDataset,
)

try:
    from petrel_client.client import Client  # noqa: F401
    has_tcs_loader = True
except ImportError:
    has_tcs_loader = False

IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
PngImagePlugin.MAX_TEXT_CHUNK = 1024 * (2 ** 20)
warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
@dataclass
class FullUnifiedModelArguments:
    """Arguments controlling model loading + per-component freeze / LoRA flags."""

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path or HF id of the InternVL-U snapshot dir."},
    )
    snapshot_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Pipeline snapshot dir containing generation_decoder/, vae/, "
                "scheduler/, processor/. Defaults to the parent of "
                "model_name_or_path if it points to a `vlm/` subdir, else "
                "model_name_or_path itself."
            )
        },
    )
    freeze_llm: bool = field(default=False)
    freeze_backbone: bool = field(default=True)
    freeze_mlp: bool = field(default=False)
    unfreeze_vit_layers: int = field(default=0)
    vision_select_layer: int = field(default=-1)
    use_backbone_lora: int = field(default=0)
    use_llm_lora: int = field(default=0)
    unfreeze_lm_head: bool = field(default=False)
    grad_checkpoint: bool = field(default=True)
    drop_path_rate: float = field(default=0.0)
    ps_version: str = field(default="v2")
    use_fast_tokenizer: bool = field(default=False)
    use_liger: bool = field(default=False)

    # New full-unified flags.
    freeze_gen_decoder: bool = field(
        default=False,
        metadata={"help": "Freeze the generation_decoder; only train the VLM side."},
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Freeze the VAE (default; matches published recipes)."},
    )
    gen_decoder_lr: float = field(
        default=5e-5,
        metadata={"help": "Learning rate for the generation_decoder param group."},
    )
    gen_loss_weight: float = field(
        default=0.1,
        metadata={"help": "Final weight applied to gen_loss after warmup."},
    )
    gen_loss_warmup_steps: int = field(
        default=1000,
        metadata={"help": "Linear ramp from 0 to gen_loss_weight over this many steps."},
    )
    imgen_ratio: float = field(
        default=0.3,
        metadata={"help": "Fraction of imgen-only batches per epoch."},
    )
    cfg_dropout: float = field(
        default=0.1,
        metadata={"help": "Probability of replacing imgen prompts with <img_uncond>."},
    )
    gen_image_size: int = field(
        default=1024,
        metadata={"help": "Square resolution that imgen target images are resized to."},
    )
    gen_max_seq_length: int = field(
        default=768,
        metadata={"help": "Tokenizer truncation length for imgen prompts."},
    )


# ---------------------------------------------------------------------------
# Trainer with per-group optimizer + grad-norm logging
# ---------------------------------------------------------------------------
class UnifiedTrainer(Trainer):
    """HF Trainer subclass that splits trainable params into named groups.

    The per-group structure makes it easy to apply different learning rates
    (e.g. ``gen_decoder_lr`` vs the global ``learning_rate``) and to log
    per-group grad norms to TensorBoard so regressions in any one branch
    surface immediately.
    """

    def __init__(
        self,
        *args: Any,
        gen_decoder_lr: float = 5e-5,
        log_per_group_grads: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._gen_decoder_lr = gen_decoder_lr
        self._log_per_group_grads = log_per_group_grads

    # -- per-component param groups ---------------------------------------
    def _build_param_groups(self) -> List[Dict[str, Any]]:
        model = self.model
        weight_decay = self.args.weight_decay
        base_lr = self.args.learning_rate

        def _group(params: List[Any], lr: float, name: str, decay: bool) -> Dict[str, Any]:
            return {
                "params": params,
                "lr": lr,
                "weight_decay": weight_decay if decay else 0.0,
                "name": name,
            }

        groups: List[Dict[str, Any]] = []

        bins: Dict[str, List[Any]] = {
            "vit": [],
            "mlp": [],
            "llm": [],
            "special_tokens": [],
            "gen_decoder": [],
            "vae": [],
            "other": [],
        }
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("vlm.vision_model"):
                bins["vit"].append(p)
            elif name.startswith("vlm.mlp1"):
                bins["mlp"].append(p)
            elif name.startswith("vlm.special_token_embedding"):
                bins["special_tokens"].append(p)
            elif name.startswith("vlm.language_model"):
                bins["llm"].append(p)
            elif name.startswith("generation_decoder"):
                bins["gen_decoder"].append(p)
            elif name.startswith("vae"):
                bins["vae"].append(p)
            else:
                bins["other"].append(p)

        if bins["vit"]:
            groups.append(_group(bins["vit"], base_lr, "vit", decay=True))
        if bins["mlp"]:
            groups.append(_group(bins["mlp"], base_lr, "mlp", decay=True))
        if bins["llm"]:
            groups.append(_group(bins["llm"], base_lr, "llm", decay=True))
        if bins["special_tokens"]:
            groups.append(_group(bins["special_tokens"], base_lr, "special_tokens", decay=False))
        if bins["gen_decoder"]:
            groups.append(_group(bins["gen_decoder"], self._gen_decoder_lr, "gen_decoder", decay=True))
        if bins["vae"]:
            groups.append(_group(bins["vae"], self._gen_decoder_lr, "vae", decay=True))
        if bins["other"]:
            groups.append(_group(bins["other"], base_lr, "other", decay=True))

        if not groups:
            raise RuntimeError(
                "UnifiedTrainer: no trainable parameters were assigned to any group; "
                "check the freeze_* flags."
            )
        for g in groups:
            if not g["params"]:
                raise RuntimeError(f"UnifiedTrainer: group {g['name']!r} has zero params.")
        return groups

    def create_optimizer(self):  # type: ignore[override]
        if self.optimizer is None:
            param_groups = self._build_param_groups()
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            optimizer_kwargs.pop("lr", None)
            optimizer_kwargs.pop("weight_decay", None)
            self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
            if self.args.local_rank in (-1, 0):
                logger.info("Optimizer param groups (lr, count):")
                for g in param_groups:
                    logger.info(f"  {g['name']:<14} lr={g['lr']:.3e}  n={len(g['params'])}")
        return self.optimizer

    # -- loss extraction --------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        # Pass the current global step so the unified model can apply the
        # gen_loss warmup linearly without needing a callback.
        inputs = dict(inputs)
        inputs["global_step"] = self.state.global_step

        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]

        if self._log_per_group_grads and self.state.global_step % max(1, self.args.logging_steps) == 0:
            extra = {
                "lm_loss": float(getattr(outputs, "lm_loss", outputs.get("lm_loss", 0.0)).detach()),
                "gen_loss": float(getattr(outputs, "gen_loss", outputs.get("gen_loss", 0.0)).detach()),
            }
            if hasattr(outputs, "gen_weight"):
                extra["gen_weight"] = float(outputs.gen_weight.detach())
            self.log(extra)
        return (loss, outputs) if return_outputs else loss

    # -- grad-norm logging ------------------------------------------------
    def training_step(self, model, inputs, num_items_in_batch=None):  # type: ignore[override]
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        if self._log_per_group_grads and self.state.global_step % max(1, self.args.logging_steps) == 0:
            metrics: Dict[str, float] = {}
            for group in self.optimizer.param_groups:
                name = group.get("name", "group")
                norms = [
                    p.grad.detach().data.norm(2)
                    for p in group["params"]
                    if p.grad is not None
                ]
                if norms:
                    metrics[f"grad_norm/{name}"] = float(torch.stack(norms).norm(2))
            if metrics:
                self.log(metrics)
        return loss


# ---------------------------------------------------------------------------
# Dataset / collator construction
# ---------------------------------------------------------------------------
def build_unified_datasets(
    *,
    data_args: DataTrainingArguments,
    tokenizer: Any,
    tcs_loader: Any,
    model: InternVLUUnifiedModel,
    gen_image_size: int,
):
    """Construct concatenated dataset + the per-pool indices for the batch sampler.

    Returns ``(concat_dataset, understanding_indices, imgen_indices)``.
    """
    ds_collections: Dict[str, Dict[str, Any]] = json.loads(open(data_args.meta_path).read())
    datasets = []
    u_indices: List[int] = []
    g_indices: List[int] = []

    cursor = 0
    data_rank = dist.get_rank() if dist.is_initialized() else 0
    data_world_size = dist.get_world_size() if dist.is_initialized() else 1

    for ds_idx, (ds_name, entry) in enumerate(ds_collections.items()):
        repeat_time = entry.get("repeat_time", 1)
        task_type = get_task_type(entry)

        if task_type == "imgen":
            ds = ImgenLazyDataset(
                meta_entry=entry,
                ds_name=ds_name,
                gen_image_size=gen_image_size,
                repeat_time=repeat_time,
                random_seed=ds_idx,
            )
            datasets.append(ds)
            g_indices.extend(range(cursor, cursor + len(ds)))
            cursor += len(ds)
            logger.info(f"Add imgen dataset: {ds_name} with length: {len(ds)}")
            continue

        max_num = entry.get("max_dynamic_patch", data_args.max_dynamic_patch)
        ds = LazySupervisedDataset(
            data_args.conv_style,
            entry,
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=entry.get("data_augment", False),
            pad2square=data_args.pad2square,
            group_by_length=False,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=data_args.min_num_frame,
            max_num_frame=data_args.max_num_frame,
            repeat_time=repeat_time,
            normalize_type=data_args.normalize_type,
            use_packed_ds=False,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=False,
            force_shuffle=False,
            random_seed=ds_idx,
        )
        datasets.append(ds)
        u_indices.extend(range(cursor, cursor + len(ds)))
        cursor += len(ds)
        logger.info(f"Add understanding dataset: {ds_name} with length: {len(ds)}")

    return ConcatDataset(datasets), u_indices, g_indices


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def _resolve_snapshot_dir(model_path: str, override: Optional[str]) -> str:
    """Pick the directory that contains generation_decoder/, vae/, scheduler/.

    If ``override`` is given, use it. Else if the basename of ``model_path``
    is ``vlm`` and its parent has a ``model_index.json``, use the parent.
    Else assume ``model_path`` itself is the snapshot dir.
    """
    if override:
        return override
    parent = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(os.path.abspath(model_path)) == "vlm" and os.path.isfile(
        os.path.join(parent, "model_index.json")
    ):
        return parent
    return model_path


def main():
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    launcher = os.environ.get("LAUNCHER", "slurm")
    init_dist(launcher=launcher, backend="nccl")
    parser = HfArgumentParser(
        (FullUnifiedModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.use_packed_ds = False  # Packed mode disabled in the unified path.

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, "
        f"n_gpu: {training_args.n_gpu}, distributed: {bool(training_args.local_rank != -1)}"
    )

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)

    set_seed(training_args.seed)

    # ---- tokenizer ----
    tokenizer_path = model_args.model_name_or_path
    logger.info(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader("~/petreloss.conf") if has_tcs_loader else None

    # ---- VLM (mirrors internvl_chat_finetune_u.py) ----
    logger.info("Loading InternVLUChatModel...")
    config = InternVLUChatConfig.from_pretrained(model_args.model_name_or_path)
    config.vision_config.drop_path_rate = model_args.drop_path_rate
    if config.llm_config.model_type == "internlm2":
        config.llm_config.attn_implementation = "flash_attention_2"
    else:
        config.llm_config._attn_implementation = "flash_attention_2"
    config.template = data_args.conv_style
    config.select_layer = model_args.vision_select_layer
    config.dynamic_image_size = data_args.dynamic_image_size
    config.use_thumbnail = data_args.use_thumbnail
    config.ps_version = model_args.ps_version
    config.min_dynamic_patch = data_args.min_dynamic_patch
    config.max_dynamic_patch = data_args.max_dynamic_patch
    config.anyres_image_size = False
    vlm = InternVLUChatModel.from_pretrained(
        model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config
    )
    vlm.img_context_token_id = img_context_token_id

    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKEN_LIST)})
    vlm.special_token_id_list = [
        tokenizer.convert_tokens_to_ids(t) for t in vlm.special_token_list
    ]
    vlm.im_start_token_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    vlm.im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    vlm.img_start_token_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
    vlm.img_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
    vlm.img_uncond_token_id = tokenizer.convert_tokens_to_ids("<img_uncond>")

    patch_size = vlm.config.vision_config.patch_size
    if vlm.config.vision_config.image_size != data_args.force_image_size:
        vlm.vision_model.resize_pos_embeddings(
            old_size=vlm.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size,
        )
        vlm.config.vision_config.image_size = data_args.force_image_size
    vlm.config.force_image_size = data_args.force_image_size
    vlm.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2
        * (data_args.down_sample_ratio ** 2)
    )

    if num_new_tokens > 0:
        vlm.language_model.resize_token_embeddings(len(tokenizer))
        out_emb = vlm.language_model.get_output_embeddings().weight.data
        avg = out_emb[:-num_new_tokens].mean(dim=0, keepdim=True)
        out_emb[-num_new_tokens:] = avg
        vlm.config.llm_config.vocab_size = len(tokenizer)
        vlm.language_model.config.vocab_size = len(tokenizer)

    vlm.language_model.config.use_cache = False
    vlm.vision_model.gradient_checkpointing = True
    vlm.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        vlm.language_model._set_gradient_checkpointing()

    # ---- compose unified model ----
    snapshot_dir = _resolve_snapshot_dir(model_args.model_name_or_path, model_args.snapshot_dir)
    logger.info(f"Loading pipeline subcomponents from {snapshot_dir}")
    model = InternVLUUnifiedModel.from_pretrained_components(
        snapshot_dir, vlm=vlm, torch_dtype=torch.bfloat16
    )
    model.configure_gen_loss_schedule(
        target_weight=model_args.gen_loss_weight,
        warmup_steps=model_args.gen_loss_warmup_steps,
    )

    # ---- freeze plumbing (mirrors the existing recipe; new gen-side blocks too) ----
    def _freeze(module):
        for p in module.parameters():
            p.requires_grad = False

    if model_args.freeze_backbone:
        _freeze(model.vlm.vision_model)
    if model_args.freeze_llm:
        model.vlm.language_model = model.vlm.language_model.eval()
        _freeze(model.vlm.language_model)
    if model_args.unfreeze_lm_head:
        model.vlm.language_model.lm_head.requires_grad = True
    if model_args.use_backbone_lora:
        model.vlm.wrap_backbone_lora(
            r=model_args.use_backbone_lora,
            lora_alpha=2 * model_args.use_backbone_lora,
        )
        model.vlm.config.use_backbone_lora = model_args.use_backbone_lora
    if model_args.use_llm_lora:
        model.vlm.wrap_llm_lora(
            r=model_args.use_llm_lora,
            lora_alpha=2 * model_args.use_llm_lora,
        )
        model.vlm.config.use_llm_lora = model_args.use_llm_lora
    if model_args.freeze_mlp:
        _freeze(model.vlm.mlp1)
    if model_args.unfreeze_vit_layers != 0:
        layers = model.vlm.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            v.requires_grad = True

    # gen-side freezing.
    if model_args.freeze_gen_decoder:
        _freeze(model.generation_decoder)
    if model_args.freeze_vae:
        _freeze(model.vae)

    # The U special-token embedding is always trained (preserves existing
    # invariant from internvl_chat_finetune_u.py — kept AFTER the freeze
    # branches).
    for p in model.special_token_embedding.parameters():
        p.requires_grad = True

    # The encoder_padding_token is part of the gen_decoder; if the decoder
    # is unfrozen this is already trainable. Mirror the special-token-pattern
    # by ensuring it stays trainable even if some sub-rule turned it off.
    if not model_args.freeze_gen_decoder and hasattr(
        model.generation_decoder, "encoder_padding_token"
    ):
        model.generation_decoder.encoder_padding_token.requires_grad = True

    if (not dist.is_initialized()) or dist.get_rank() == 0:
        logger.info("Trainable params:")
        for name, p in model.named_parameters():
            if p.requires_grad:
                logger.info(f"  {name}")

    # ---- datasets / collator ----
    train_dataset, u_indices, g_indices = build_unified_datasets(
        data_args=data_args,
        tokenizer=tokenizer,
        tcs_loader=tcs_loader,
        model=model,
        gen_image_size=model_args.gen_image_size,
    )

    base_collator = concat_pad_data_collator
    collator = UnifiedCollator(
        tokenizer=tokenizer,
        understanding_collator=base_collator,
        cfg_dropout=model_args.cfg_dropout,
        max_length=model_args.gen_max_seq_length,
        rng_seed=training_args.seed,
    )

    sampler = TaskTypeBatchSampler(
        understanding_indices=u_indices,
        imgen_indices=g_indices,
        batch_size=training_args.per_device_train_batch_size,
        imgen_ratio=model_args.imgen_ratio,
        rng_seed=training_args.seed,
        drop_last=True,
    )

    set_seed(training_args.seed)

    trainer = UnifiedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=None,
        tokenizer=tokenizer,
        data_collator=collator,
        gen_decoder_lr=model_args.gen_decoder_lr,
        log_per_group_grads=True,
    )

    # Replace the default data sampler with our task-type-aware one. HF's Trainer
    # builds the dataloader inside `trainer.get_train_dataloader`; the simplest
    # injection is to override that method here.
    def _get_dataloader_with_batch_sampler(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    trainer.get_train_dataloader = _get_dataloader_with_batch_sampler.__get__(
        trainer, type(trainer)
    )

    if training_args.do_train:
        ckpt = (
            training_args.resume_from_checkpoint
            if training_args.resume_from_checkpoint is not None
            else last_checkpoint
        )
        train_result = trainer.train(resume_from_checkpoint=ckpt)
        trainer.save_model()  # writes the pipeline directory layout via save_pretrained
        metrics = train_result.metrics
        try:
            metrics["train_samples"] = len(train_dataset)
        except Exception:
            metrics["train_samples"] = -1
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
