"""Dataset / sampler / collator for InternVL-U full unified SFT.

The new full-unified path runs *both* understanding (text-CE) and imgen
(flow-matching MSE) samples in the same training loop. To keep the existing
``LazySupervisedDataset`` untouched, we add:

* :class:`ImgenLazyDataset` — reads imgen-formatted JSONL rows
  (``{"task_type": "imgen", "caption": str, "target_image": path}``) and
  produces tensorised samples with a ``task_type="imgen"`` marker.
* :class:`TaskTypeBatchSampler` — yields homogeneous-task index batches at a
  configurable mix ratio so each forward sees one branch only.
* :func:`unified_collate_fn` — branches on ``task_type`` and produces a batch
  dict the :class:`InternVLUUnifiedModel.forward` understands.

The schema for understanding rows is unchanged from the existing
``LazySupervisedDataset`` (with an optional ``task_type="understanding"``
which is the default if absent). Old metas continue to load.
"""

from __future__ import annotations

import json
import math
import os
import random
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision.transforms.functional import InterpolationMode

from internvl.model.internvlu.constants import IMG_UNCOND_TOKEN


# ---------------------------------------------------------------------------
# meta-schema helpers
# ---------------------------------------------------------------------------
def get_task_type(meta_entry: Dict[str, Any]) -> str:
    """Return ``"imgen"`` if a meta-JSON entry is an imgen entry, else ``"understanding"``."""
    return str(meta_entry.get("task_type", "understanding")).lower()


# ---------------------------------------------------------------------------
# Imgen dataset
# ---------------------------------------------------------------------------
class ImgenLazyDataset(Dataset):
    """Tiny lazy JSONL reader for image-generation samples.

    Each JSONL row must contain a string ``caption`` and a path-like
    ``target_image`` (relative to the meta entry's ``root``). Optional
    fields: ``conditional_image`` for editing, ``id``, ``length``.

    Samples returned by :meth:`__getitem__` are *not* tokenised here — the
    collator does that to keep tokenizer references out of dataloader workers.
    """

    def __init__(
        self,
        meta_entry: Dict[str, Any],
        ds_name: str,
        *,
        gen_image_size: int = 1024,
        repeat_time: float = 1.0,
        random_seed: int = 0,
    ) -> None:
        super().__init__()
        self.ds_name = ds_name
        self.gen_image_size = gen_image_size
        self.root = meta_entry["root"]

        ann_path = meta_entry["annotation"]
        if not ann_path.endswith(".jsonl"):
            raise ValueError(f"ImgenLazyDataset expects a .jsonl annotation, got {ann_path}")
        with open(ann_path, "r") as f:
            raw = f.readlines()
        if 0 < repeat_time < 1:
            raw = raw[: int(len(raw) * repeat_time)]
        elif repeat_time > 1:
            assert isinstance(repeat_time, int)
            raw = raw * repeat_time
        self.raw_data: List[str] = raw
        self.rng = np.random.default_rng(seed=random_seed)

        # VAE input normalisation. Hard-coded from `processor_config.json`
        # `image_gen_processor_kwargs.image_mean/image_std = (0.5, 0.5, 0.5)`.
        self.transform = T.Compose(
            [
                T.Resize(
                    (gen_image_size, gen_image_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    def __len__(self) -> int:
        return len(self.raw_data)

    def _resolve_path(self, p: str) -> str:
        if p.startswith("/"):
            return p
        return os.path.join(self.root, p)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = json.loads(self.raw_data[i])
        target_path = self._resolve_path(row["target_image"])
        target = Image.open(target_path).convert("RGB")
        target_pixel_values = self.transform(target)
        return {
            "task_type": "imgen",
            "caption": row["caption"],
            "target_pixel_values": target_pixel_values,
            "ds_name": self.ds_name,
        }


# ---------------------------------------------------------------------------
# Mixed-task batch sampler
# ---------------------------------------------------------------------------
class TaskTypeBatchSampler(Sampler[List[int]]):
    """Yields homogeneous-task index batches at the requested mix ratio.

    Args:
        understanding_indices: Indices in the *concatenated* dataset that
            correspond to understanding samples.
        imgen_indices: Indices in the concatenated dataset for imgen samples.
        batch_size: Per-rank batch size.
        imgen_ratio: Target fraction of imgen-only batches per epoch in
            ``[0, 1]``. ``0`` reproduces the existing all-understanding path.
        rng_seed: Seed for shuffling and ratio coin-flips.
        drop_last: Whether to drop the trailing partial batch of either pool.
    """

    def __init__(
        self,
        *,
        understanding_indices: Sequence[int],
        imgen_indices: Sequence[int],
        batch_size: int,
        imgen_ratio: float = 0.3,
        rng_seed: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if not 0.0 <= imgen_ratio <= 1.0:
            raise ValueError(f"imgen_ratio must be in [0, 1], got {imgen_ratio}")
        self.u_pool = list(understanding_indices)
        self.g_pool = list(imgen_indices)
        self.batch_size = batch_size
        self.imgen_ratio = imgen_ratio
        self.rng_seed = rng_seed
        self.drop_last = drop_last
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Bump the epoch counter so each epoch reshuffles deterministically."""
        self._epoch = epoch

    def _epoch_rng(self) -> random.Random:
        return random.Random(self.rng_seed + self._epoch)

    def __iter__(self) -> Iterator[List[int]]:
        rng = self._epoch_rng()
        u = list(self.u_pool)
        g = list(self.g_pool)
        rng.shuffle(u)
        rng.shuffle(g)

        u_batches: List[List[int]] = [
            u[i : i + self.batch_size] for i in range(0, len(u), self.batch_size)
        ]
        g_batches: List[List[int]] = [
            g[i : i + self.batch_size] for i in range(0, len(g), self.batch_size)
        ]
        if self.drop_last:
            u_batches = [b for b in u_batches if len(b) == self.batch_size]
            g_batches = [b for b in g_batches if len(b) == self.batch_size]

        # Interleave according to imgen_ratio.
        total_u, total_g = len(u_batches), len(g_batches)
        if total_u == 0 and total_g == 0:
            return iter([])
        if self.imgen_ratio <= 0.0:
            ordered = u_batches
        elif self.imgen_ratio >= 1.0:
            ordered = g_batches
        else:
            ordered = []
            ui = gi = 0
            # Bernoulli-style scheduling that respects pool exhaustion.
            while ui < total_u or gi < total_g:
                pick_g = (
                    rng.random() < self.imgen_ratio
                    and gi < total_g
                    or ui >= total_u
                ) and gi < total_g
                if pick_g:
                    ordered.append(g_batches[gi])
                    gi += 1
                else:
                    ordered.append(u_batches[ui])
                    ui += 1
        return iter(ordered)

    def __len__(self) -> int:
        n_u = len(self.u_pool) // self.batch_size if self.drop_last else math.ceil(len(self.u_pool) / self.batch_size)
        n_g = len(self.g_pool) // self.batch_size if self.drop_last else math.ceil(len(self.g_pool) / self.batch_size)
        if self.imgen_ratio <= 0.0:
            return n_u
        if self.imgen_ratio >= 1.0:
            return n_g
        return n_u + n_g


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------
def _stack_pad(values: Sequence[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    """Right-pad a list of 1-D tensors to the max length and stack them."""
    if len(values) == 0:
        return torch.empty(0, dtype=torch.long)
    max_len = max(v.shape[0] for v in values)
    out = torch.full((len(values), max_len), pad_value, dtype=values[0].dtype)
    for i, v in enumerate(values):
        out[i, : v.shape[0]] = v
    return out


def _build_imgen_input_ids(
    captions: Sequence[str],
    *,
    tokenizer: Any,
    img_start_token: str,
    img_uncond_token: str,
    cfg_dropout: float,
    rng: random.Random,
    max_length: int,
    system_prompt: Optional[str] = None,
) -> Tuple[torch.LongTensor, torch.Tensor, torch.LongTensor]:
    """Tokenise imgen prompts and append a single ``<img_start>`` generation marker.

    The prompt format mirrors the inference processor's imgen template (see
    ``internvlu/processing_internvlu.py`` and the ``qwen2_5-chat-v3-imgen``
    conv template):

        <|im_start|>system
        <system_msg><|im_end|>
        <|im_start|>user
        <caption><|im_end|>
        <|im_start|>assistant
        <img>

    With probability ``cfg_dropout`` the caption is replaced by a single
    ``<img_uncond>`` token so the unconditional embedding produced by the
    VLM's special-token table is used instead.

    Args:
        captions: Per-sample text prompts.
        tokenizer: A HuggingFace tokenizer.
        img_start_token: The literal ``<img>`` token string.
        img_uncond_token: The literal ``<img_uncond>`` token string.
        cfg_dropout: Probability of replacing the caption with the uncond token.
        rng: Local RNG so dropout is reproducible.
        max_length: Truncation cap.
        system_prompt: Optional system message (defaults to a generic instruction).

    Returns:
        ``(input_ids, attention_mask, generation_flags)``. ``generation_flags``
        is a 1-D tensor of length ``B`` (one ``<img>`` per sample) marking the
        single image-to-generate.
    """
    sys_msg = system_prompt or (
        "You are a helpful assistant that turns a description into an image."
    )
    out_ids: List[torch.Tensor] = []
    out_mask: List[torch.Tensor] = []

    for cap in captions:
        if rng.random() < cfg_dropout:
            user_msg = img_uncond_token
        else:
            user_msg = cap
        prompt = (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{img_start_token}"
        )
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        ids = enc["input_ids"][0]
        # If the prompt was truncated, ensure it still ends with an <img>; if not, append it.
        ids_list = ids.tolist()
        img_id = tokenizer.convert_tokens_to_ids(img_start_token)
        if ids_list[-1] != img_id:
            ids = torch.cat([ids, torch.tensor([img_id], dtype=ids.dtype)])
        out_ids.append(ids)
        out_mask.append(torch.ones_like(ids))

    input_ids = _stack_pad(out_ids, pad_value=tokenizer.pad_token_id or 0)
    attention_mask = _stack_pad(out_mask, pad_value=0)
    # Exactly one <img> per sample, all are generation targets.
    generation_flags = torch.ones(len(captions), dtype=torch.long)
    return input_ids, attention_mask, generation_flags


class UnifiedCollator:
    """Collator that builds a batch dict the unified model can route on.

    Under the hood the collator delegates the **understanding** branch to the
    existing ``concat_pad_data_collator`` so behaviour for the old code path
    is bit-identical. The **imgen** branch is built here.

    The output dict either contains:

    * ``input_ids`` / ``attention_mask`` / ``labels`` / ``pixel_values`` /
      ``image_flags`` / ``position_ids`` (homogeneous understanding batch), or
    * ``gen_input_ids`` / ``gen_attention_mask`` / ``gen_target_pixel_values`` /
      ``gen_generation_flags`` (homogeneous imgen batch).

    The unified model's ``forward`` checks whether each branch is populated
    and returns a zero tensor for the unused side.
    """

    def __init__(
        self,
        *,
        tokenizer: Any,
        understanding_collator: Any,
        cfg_dropout: float = 0.1,
        max_length: int = 768,
        rng_seed: int = 0,
        img_start_token: str = "<img>",
        img_uncond_token: str = IMG_UNCOND_TOKEN,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.understanding_collator = understanding_collator
        self.cfg_dropout = float(cfg_dropout)
        self.max_length = int(max_length)
        self.img_start_token = img_start_token
        self.img_uncond_token = img_uncond_token
        self.system_prompt = system_prompt
        self._rng = random.Random(rng_seed)

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        if len(samples) == 0:
            return {}

        first_task = samples[0].get("task_type", "understanding")
        if not all(s.get("task_type", "understanding") == first_task for s in samples):
            raise RuntimeError(
                "UnifiedCollator received a mixed-task batch; the BatchSampler "
                "must produce homogeneous-task batches."
            )

        if first_task == "imgen":
            return self._collate_imgen(samples)
        return self._collate_understanding(samples)

    # -- understanding branch --
    def _collate_understanding(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Strip the optional task_type marker before delegating.
        cleaned = [
            {k: v for k, v in s.items() if k != "task_type"} for s in samples
        ]
        return self.understanding_collator(cleaned)

    # -- imgen branch --
    def _collate_imgen(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        captions = [s["caption"] for s in samples]
        target_pixel_values = torch.stack([s["target_pixel_values"] for s in samples])

        input_ids, attention_mask, gen_flags = _build_imgen_input_ids(
            captions,
            tokenizer=self.tokenizer,
            img_start_token=self.img_start_token,
            img_uncond_token=self.img_uncond_token,
            cfg_dropout=self.cfg_dropout,
            rng=self._rng,
            max_length=self.max_length,
            system_prompt=self.system_prompt,
        )

        return {
            "gen_input_ids": input_ids,
            "gen_attention_mask": attention_mask,
            "gen_pixel_values": None,
            "gen_image_flags": None,
            "gen_target_pixel_values": target_pixel_values,
            "gen_generation_flags": gen_flags,
        }
