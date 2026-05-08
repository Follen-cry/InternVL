"""Unit tests for InternVLUUnifiedModel — built around tiny stand-ins.

These tests deliberately avoid loading the real 4B checkpoint: they use a
hand-rolled minimal VLM stand-in plus a single-layer transformer for the
generation_decoder. The point is to exercise the freeze / param-group logic,
the forward routing, the conditioning math, and the save/load roundtrip
without GPU + minutes of weight loading.

The end-to-end real-checkpoint check lives in
internvl_chat/tests/run_smoke.sh and is exercised by Phase 6 verification.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from internvl.model.internvlu.modeling_internvlu_unified import (
    InternVLUUnifiedConfig,
    InternVLUUnifiedModel,
)


# ---------------------------------------------------------------------------
# Stand-ins. We mirror just the attribute surface that the unified model pokes
# at — enough to drive the imgen forward end-to-end on CPU.
# ---------------------------------------------------------------------------
class _StubLM(nn.Module):
    """Tiny stand-in for the LLM: a single embedding + a 2-layer block."""

    def __init__(self, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.block1 = nn.Linear(hidden_size, hidden_size)
        self.block2 = nn.Linear(hidden_size, hidden_size)
        self.config = SimpleNamespace(vocab_size=vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=True,
        return_dict=True,
        padding_type=None,
        **_,
    ):
        h0 = inputs_embeds
        h1 = self.block1(h0)
        h2 = self.block2(h1)
        # Mimic HF: returned hidden_states is (embeds, layer1_out, ..., final).
        return SimpleNamespace(
            logits=h2,
            past_key_values=None,
            hidden_states=(h0, h1, h2),
            attentions=None,
            loss=None,
        )


class _StubVLM(nn.Module):
    """Tiny stand-in for InternVLUChatModel."""

    def __init__(self, *, vocab_size: int = 32, hidden_size: int = 16):
        super().__init__()
        self.language_model = _StubLM(vocab_size=vocab_size, hidden_size=hidden_size)
        self.vision_model = nn.Linear(8, hidden_size)
        self.mlp1 = nn.Linear(hidden_size, hidden_size)
        self.special_token_embedding = nn.Embedding(4, hidden_size)
        self.special_token_id_list = [0, 1, 2, 3]
        self.img_context_token_id = 5
        self.img_start_token_id = 6
        self.im_start_token_id = 7
        self.im_end_token_id = 8
        self.img_uncond_token_id = 9

    def replace_img_special_tokens(self, embeds, ids):
        for i, tok in enumerate(self.special_token_id_list):
            mask = ids == tok
            embeds[mask] = embeds[mask] * 0.0 + self.special_token_embedding.weight[i]
        return embeds

    def extract_feature(self, pixel_values, grid_thw=None):
        return torch.zeros(1, 0, self.special_token_embedding.embedding_dim)

    def forward(self, **kwargs):
        # The understanding-branch path delegates to language_model + computes
        # CE if labels are given. For the freeze-logic tests we only need it
        # to return *something* with .loss attribute.
        labels = kwargs.get("labels")
        ids = kwargs.get("input_ids")
        out = self.language_model(
            inputs_embeds=self.language_model.embed(ids),
            attention_mask=kwargs.get("attention_mask"),
            output_hidden_states=True,
            return_dict=True,
        )
        loss = None
        if labels is not None:
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        return SimpleNamespace(
            loss=loss,
            logits=out.logits,
            hidden_states=out.hidden_states,
            past_key_values=None,
        )

    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"_stub_vlm": True}, f)


class _StubDecoderInner(nn.Module):
    def __init__(self, hidden_size: int = 16, latent_channels: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(latent_channels, latent_channels, 3, padding=1)
        self.cond_proj = nn.Linear(hidden_size, latent_channels)

    def forward(
        self,
        noisy,
        encoder_hidden_states,
        encoder_attention_mask=None,
        encoder_image_token_mask=None,
        timestep=None,
        return_dict=False,
        **_,
    ):
        cond_summary = encoder_hidden_states.mean(dim=1)
        cond_summary = self.cond_proj(cond_summary)
        out = self.proj(noisy) + cond_summary[..., None, None]
        return (out,)


class _StubGenDecoder(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = SimpleNamespace(
            vlm_select_layer=[-1, -2],
            input_hidden_size=2 * hidden_size,
            padding_encoder_hidden_states=True,
            pad_to_fix_length=False,
            max_sequence_length=8,
            weighting_scheme="logit_normal",
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
            flow_shift=3.0,
        )
        self.encoder_padding_token = nn.Parameter(torch.zeros(2 * hidden_size))
        self.decoder = _StubDecoderInner(hidden_size=2 * hidden_size, latent_channels=4)
        self.decoder_projector = nn.Identity()

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def prepare_forward_input(self, encoder_hidden_states, encoder_image_token_mask=None, **_):
        max_len = max(8, max(x.shape[0] for x in encoder_hidden_states))
        B = len(encoder_hidden_states)
        H = encoder_hidden_states[0].shape[-1]
        out = self.encoder_padding_token[None, None].expand(B, max_len, H).clone()
        masks = torch.zeros(B, max_len, dtype=torch.bool, device=self.encoder_padding_token.device)
        tok_mask = (
            torch.zeros(B, max_len, dtype=torch.bool)
            if encoder_image_token_mask is not None
            else None
        )
        for i, x in enumerate(encoder_hidden_states):
            n = min(x.shape[0], max_len)
            out[i, :n] = x[:n]
            masks[i, :n] = True
            if encoder_image_token_mask is not None:
                tok_mask[i, :n] = encoder_image_token_mask[i][:n]
        return out, masks, tok_mask

    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"_stub_dec": True}, f)


class _StubVAE(nn.Module):
    def __init__(self, latent_channels: int = 4):
        super().__init__()
        self.proj = nn.Conv2d(3, latent_channels, kernel_size=8, stride=8)
        self.config = SimpleNamespace(
            latents_mean=[0.0] * latent_channels,
            latents_std=[1.0] * latent_channels,
            z_dim=latent_channels,
        )

    def encode(self, x):
        # x is (B, C, T, H, W); collapse T to keep the test shape-correct.
        x_2d = x.squeeze(2)
        z = self.proj(x_2d).unsqueeze(2)
        return SimpleNamespace(latent_dist=SimpleNamespace(sample=lambda: z))

    def save_pretrained(self, path, **kwargs):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "diffusion_pytorch_model.bin"))
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump({"_stub_vae": True}, f)


class _StubScheduler:
    def __init__(self):
        self.config = {"num_train_timesteps": 1000}

    def save_pretrained(self, path):
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "scheduler_config.json"), "w") as f:
            json.dump(self.config, f)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def _make_model() -> InternVLUUnifiedModel:
    cfg = InternVLUUnifiedConfig()
    return InternVLUUnifiedModel(
        config=cfg,
        vlm=_StubVLM(),
        generation_decoder=_StubGenDecoder(),
        vae=_StubVAE(),
        scheduler=_StubScheduler(),
        processor=None,
    )


def test_save_pretrained_emits_pipeline_layout(tmp_path: Path):
    model = _make_model()
    model.save_pretrained(tmp_path / "out")

    out = tmp_path / "out"
    assert (out / "model_index.json").is_file()
    manifest = json.loads((out / "model_index.json").read_text())
    assert manifest["_class_name"] == "InternVLUPipeline"
    for sub in ("vlm", "generation_decoder", "vae", "scheduler"):
        assert (out / sub).is_dir(), f"missing subdir: {sub}"


def test_save_pretrained_skips_processor_when_absent(tmp_path: Path):
    model = _make_model()
    model.save_pretrained(tmp_path / "out")
    manifest = json.loads((tmp_path / "out" / "model_index.json").read_text())
    assert "processor" not in manifest


def test_freeze_special_tokens_remain_trainable_after_freeze_all():
    model = _make_model()
    for p in model.parameters():
        p.requires_grad = False
    # Now apply the same "always-trainable" rule the trainer uses.
    for p in model.special_token_embedding.parameters():
        p.requires_grad = True
    assert model.special_token_embedding.weight.requires_grad
    assert not model.vlm.language_model.block1.weight.requires_grad


def test_compute_gen_loss_finite_with_grad():
    """Drive the imgen branch end-to-end through the stand-ins."""
    torch.manual_seed(0)
    model = _make_model()
    # Build a tiny imgen batch: one sample, one <img> token at the end.
    # Token plan: [<|im_start|>=7, dummy=11, <|im_start|>=7, prompt=12, <img>=6]
    input_ids = torch.tensor([[7, 11, 7, 12, 6]])
    attention_mask = torch.ones_like(input_ids)
    target = torch.zeros(1, 3, 8, 8)
    gen_flags = torch.tensor([1])

    out = model.forward(
        gen_input_ids=input_ids,
        gen_attention_mask=attention_mask,
        gen_target_pixel_values=target,
        gen_generation_flags=gen_flags,
        global_step=0,
    )
    assert torch.isfinite(out.gen_loss)
    assert torch.isfinite(out.loss)
    assert out.lm_loss.item() == 0.0  # understanding branch not invoked


def test_gen_loss_warmup_schedule():
    model = _make_model()
    model.configure_gen_loss_schedule(target_weight=0.2, warmup_steps=10)
    assert model._gen_loss_weight(0) == 0.0
    assert pytest.approx(model._gen_loss_weight(5), rel=1e-6) == 0.1
    assert pytest.approx(model._gen_loss_weight(10), rel=1e-6) == 0.2
    assert model._gen_loss_weight(50) == 0.2


def test_param_group_classification_for_freeze_logic():
    model = _make_model()
    # Mimic the "freeze ViT, train everything else" recipe.
    for name, p in model.named_parameters():
        p.requires_grad = not name.startswith("vlm.vision_model")
    bins = {"vit": 0, "mlp": 0, "llm": 0, "special_tokens": 0, "gen_decoder": 0, "vae": 0, "other": 0}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            bins["vit"] += 1 if name.startswith("vlm.vision_model") else 0
            continue
        if name.startswith("vlm.mlp1"):
            bins["mlp"] += 1
        elif name.startswith("vlm.special_token_embedding"):
            bins["special_tokens"] += 1
        elif name.startswith("vlm.language_model"):
            bins["llm"] += 1
        elif name.startswith("generation_decoder"):
            bins["gen_decoder"] += 1
        elif name.startswith("vae"):
            bins["vae"] += 1
        else:
            bins["other"] += 1
    # Each named bucket has at least one parameter — exactly the precondition
    # UnifiedTrainer asserts before optimizer creation.
    for k in ("mlp", "llm", "special_tokens", "gen_decoder", "vae"):
        assert bins[k] > 0, f"bucket {k} unexpectedly empty"


def test_save_then_reload_roundtrip(tmp_path: Path):
    """Round-trip just checks save_pretrained writes loadable subdirs."""
    model = _make_model()
    model.save_pretrained(tmp_path / "round")
    state_back = torch.load(
        tmp_path / "round" / "vlm" / "pytorch_model.bin", map_location="cpu"
    )
    assert any(k.startswith("language_model.") for k in state_back)
    state_dec = torch.load(
        tmp_path / "round" / "generation_decoder" / "pytorch_model.bin", map_location="cpu"
    )
    assert any(k.startswith("decoder.") for k in state_dec)
