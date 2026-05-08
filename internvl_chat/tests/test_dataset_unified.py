"""Unit tests for ImgenLazyDataset, TaskTypeBatchSampler, UnifiedCollator."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import torch

from internvl.train.dataset_unified import (
    ImgenLazyDataset,
    TaskTypeBatchSampler,
    UnifiedCollator,
    _build_imgen_input_ids,
    get_task_type,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_get_task_type_default_understanding():
    assert get_task_type({}) == "understanding"
    assert get_task_type({"task_type": "imgen"}) == "imgen"
    assert get_task_type({"task_type": "Understanding"}) == "understanding"


def test_imgen_dataset_loads_and_normalises():
    meta = json.loads((FIXTURES / "synth_meta_unified.json").read_text())
    ds = ImgenLazyDataset(meta["synth_imgen"], "synth_imgen", gen_image_size=32)
    assert len(ds) == 4
    s = ds[0]
    assert s["task_type"] == "imgen"
    assert isinstance(s["caption"], str) and len(s["caption"]) > 0
    target = s["target_pixel_values"]
    assert target.shape == (3, 32, 32)
    # mean/std normalisation: input pixels in [0,1] then (x-0.5)/0.5 in [-1, 1]
    assert target.min() >= -1.0 - 1e-6
    assert target.max() <= 1.0 + 1e-6


def test_task_type_batch_sampler_produces_homogeneous_batches():
    sampler = TaskTypeBatchSampler(
        understanding_indices=[0, 1, 2, 3],
        imgen_indices=[10, 11, 12, 13],
        batch_size=2,
        imgen_ratio=0.5,
        rng_seed=0,
    )
    batches = list(sampler)
    assert len(batches) == 4
    # Each batch is either fully-understanding or fully-imgen.
    for b in batches:
        ids = set(b)
        assert ids.issubset({0, 1, 2, 3}) or ids.issubset({10, 11, 12, 13})


def test_task_type_batch_sampler_zero_ratio_is_understanding_only():
    sampler = TaskTypeBatchSampler(
        understanding_indices=list(range(8)),
        imgen_indices=list(range(8, 16)),
        batch_size=2,
        imgen_ratio=0.0,
        rng_seed=0,
    )
    batches = list(sampler)
    for b in batches:
        for idx in b:
            assert idx < 8


def test_task_type_batch_sampler_full_ratio_is_imgen_only():
    sampler = TaskTypeBatchSampler(
        understanding_indices=list(range(8)),
        imgen_indices=list(range(8, 16)),
        batch_size=2,
        imgen_ratio=1.0,
        rng_seed=0,
    )
    batches = list(sampler)
    for b in batches:
        for idx in b:
            assert idx >= 8


def test_build_imgen_input_ids_appends_img_token():
    class _StubTokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            ids = torch.tensor([[hash(c) % 200 for c in text]])
            return {"input_ids": ids}

        def convert_tokens_to_ids(self, tok):
            return 999 if tok == "<img>" else 998

    rng = random.Random(0)
    ids, mask, flags = _build_imgen_input_ids(
        ["a red square", "a blue circle"],
        tokenizer=_StubTokenizer(),
        img_start_token="<img>",
        img_uncond_token="<img_uncond>",
        cfg_dropout=0.0,
        rng=rng,
        max_length=128,
    )
    assert ids.shape[0] == 2
    assert mask.shape == ids.shape
    # Last non-pad token must be the <img> id.
    for row, m in zip(ids, mask):
        last = row[m.bool()][-1].item()
        assert last == 999
    assert flags.tolist() == [1, 1]


def test_unified_collator_routes_imgen_branch():
    class _StubTokenizer:
        pad_token_id = 0

        def __call__(self, text, **kwargs):
            ids = torch.tensor([[hash(c) % 200 for c in text]])
            return {"input_ids": ids}

        def convert_tokens_to_ids(self, tok):
            return 999 if tok == "<img>" else 998

    def _stub_understanding_collator(samples):
        return {"input_ids": torch.zeros(len(samples), 4, dtype=torch.long)}

    coll = UnifiedCollator(
        tokenizer=_StubTokenizer(),
        understanding_collator=_stub_understanding_collator,
        cfg_dropout=0.0,
        max_length=64,
    )

    imgen_batch = [
        {
            "task_type": "imgen",
            "caption": f"caption {i}",
            "target_pixel_values": torch.zeros(3, 8, 8),
        }
        for i in range(2)
    ]
    out = coll(imgen_batch)
    assert "gen_input_ids" in out and "input_ids" not in out
    assert out["gen_target_pixel_values"].shape == (2, 3, 8, 8)
    assert out["gen_generation_flags"].tolist() == [1, 1]

    u_batch = [{"task_type": "understanding", "input_ids": torch.zeros(4, dtype=torch.long)}]
    out_u = coll(u_batch)
    assert "input_ids" in out_u and "gen_input_ids" not in out_u


def test_unified_collator_rejects_mixed_task_batch():
    coll = UnifiedCollator(
        tokenizer=None,
        understanding_collator=lambda s: {},
        cfg_dropout=0.0,
        max_length=64,
    )
    with pytest.raises(RuntimeError, match="homogeneous"):
        coll([
            {"task_type": "imgen", "caption": "x", "target_pixel_values": torch.zeros(3, 4, 4)},
            {"task_type": "understanding"},
        ])
