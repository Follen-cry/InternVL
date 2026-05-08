"""InternVL-U full unified model used for end-to-end SFT.

The class composes the four trainable / loadable subcomponents that make up
the InternVL-U inference pipeline:

* ``vlm`` — the existing :class:`InternVLUChatModel` (vision tower + ViT->LLM
  projector + LLM + special-token embedding table).
* ``generation_decoder`` — :class:`InternVLUGenerationDecoder` from the
  ``internvlu`` package.
* ``vae`` — Qwen-Image autoencoder used to encode generation targets / decode
  generated latents.
* ``scheduler`` — DPM-Solver scheduler used at inference (training uses its own
  flow-matching schedule; the saved config is preserved for inference).

The model is **not** a drop-in replacement for :class:`InternVLUChatModel`. It
adds a single new entry point — :meth:`forward` — that routes by ``task_type``
to either the existing text-CE loss or a flow-matching MSE loss against
VLM-conditioned VAE latents.

The ``vlm`` subcomponent is expected to already be initialised by the existing
training script's loader path (so LoRA wrapping, special-token-embedding
resize, gradient-checkpointing, etc. all happen exactly as before). This
class never re-loads it.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from . import flow_matching_utils as fm_utils
from .modeling_internvlu_chat import InternVLUChatModel

logger = logging.get_logger(__name__)


def _import_internvlu_package() -> Any:
    """Return the imported ``internvlu`` package, adding common locations to ``sys.path``."""
    import importlib
    import sys

    candidates = [
        os.environ.get("INTERNVLU_PKG_PATH"),
        "/scratch/network/ssd2/junlin/ssl_mllm/InternVL-U",
    ]
    for cand in candidates:
        if cand and Path(cand).is_dir() and cand not in sys.path:
            sys.path.insert(0, cand)
    return importlib.import_module("internvlu")


@dataclass
class UnifiedForwardOutput:
    """Container returned by :meth:`InternVLUUnifiedModel.forward`.

    Attributes:
        loss: Combined scalar loss (``lm_loss + gen_weight * gen_loss``).
        lm_loss: Text-CE loss for the understanding branch (zero if no understanding sample).
        gen_loss: Flow-matching MSE for the imgen branch (zero if no imgen sample).
        gen_weight: The weight applied to ``gen_loss`` for this step (post-warmup).
    """

    loss: torch.Tensor
    lm_loss: torch.Tensor
    gen_loss: torch.Tensor
    gen_weight: torch.Tensor


class InternVLUUnifiedConfig(PretrainedConfig):
    """Lightweight config wrapper for the unified model directory layout.

    Stores the same field set as the pipeline's ``model_index.json`` so the
    output directory of :meth:`InternVLUUnifiedModel.save_pretrained` can be
    re-loaded by ``InternVLUPipeline.from_pretrained``. The actual subcomponent
    configs live next to this one inside their respective subdirs.
    """

    model_type = "internvlu_unified"

    def __init__(
        self,
        pipeline_class_name: str = "InternVLUPipeline",
        diffusers_version: Optional[str] = None,
        components: Optional[Dict[str, List[str]]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.pipeline_class_name = pipeline_class_name
        self.diffusers_version = diffusers_version
        # Default mirror of the released model_index.json.
        self.components: Dict[str, List[str]] = components or {
            "vlm": ["internvlu.vlm", "InternVLUChatModel"],
            "generation_decoder": ["internvlu.diffusion", "InternVLUGenerationDecoder"],
            "vae": ["diffusers", "AutoencoderKLQwenImage"],
            "scheduler": ["diffusers", "DPMSolverMultistepScheduler"],
            "processor": ["internvlu.processing_internvlu", "InternVLUProcessor"],
        }


class InternVLUUnifiedModel(PreTrainedModel):
    """Composite model owning the VLM, generation decoder, VAE, and scheduler.

    It is intentionally **not** a HuggingFace generation model — text inference
    still goes through the original :class:`InternVLUChatModel` and image
    inference still goes through :class:`InternVLUPipeline`. This class only
    exposes a training-time :meth:`forward` and the
    save/load plumbing required by the HF :class:`Trainer`.
    """

    config_class = InternVLUUnifiedConfig
    main_input_name = "input_ids"
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: InternVLUUnifiedConfig,
        vlm: InternVLUChatModel,
        generation_decoder: nn.Module,
        vae: nn.Module,
        scheduler: Any,
        processor: Any = None,
    ) -> None:
        super().__init__(config)
        self.vlm = vlm
        self.generation_decoder = generation_decoder
        self.vae = vae
        # ``scheduler`` and ``processor`` are not nn.Modules; keep references but skip
        # registration so DDP / DeepSpeed doesn't try to manage them.
        object.__setattr__(self, "scheduler", scheduler)
        object.__setattr__(self, "processor", processor)

        # Cache for the gen-loss weight schedule (set externally by the trainer).
        self._gen_loss_warmup_steps: int = 0
        self._gen_loss_target_weight: float = 0.0

    # ------------------------------------------------------------------
    # Convenience aliases so existing training-script code that pokes at
    # `model.<vlm-attr>` keeps working when handed a unified model.
    # ------------------------------------------------------------------
    @property
    def vision_model(self):
        return self.vlm.vision_model

    @property
    def language_model(self):
        return self.vlm.language_model

    @property
    def mlp1(self):
        return self.vlm.mlp1

    @property
    def special_token_embedding(self):
        return self.vlm.special_token_embedding

    @property
    def num_image_token(self) -> int:
        return self.vlm.num_image_token

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained_components(
        cls,
        snapshot_dir: Union[str, os.PathLike],
        *,
        vlm: InternVLUChatModel,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "InternVLUUnifiedModel":
        """Compose a unified model around a *prebuilt* VLM and the snapshot's other components.

        The VLM is supplied by the caller (the existing training script already
        builds it with all the right LoRA / freeze / token-embedding handling).
        The other three subdirs are loaded from ``snapshot_dir``.

        Args:
            snapshot_dir: Path to a directory containing ``generation_decoder/``,
                ``vae/``, ``scheduler/`` (and optionally ``processor/``).
            vlm: An already-initialised :class:`InternVLUChatModel`.
            torch_dtype: Optional dtype to cast ``generation_decoder`` and ``vae`` to.
        """
        snap = Path(snapshot_dir)
        if not snap.is_dir():
            raise FileNotFoundError(f"snapshot_dir does not exist: {snap}")

        internvlu = _import_internvlu_package()
        from diffusers import AutoencoderKLQwenImage, DPMSolverMultistepScheduler

        gen_decoder = internvlu.diffusion.InternVLUGenerationDecoder.from_pretrained(
            str(snap / "generation_decoder"),
            torch_dtype=torch_dtype,
        )
        vae = AutoencoderKLQwenImage.from_pretrained(str(snap / "vae"), torch_dtype=torch_dtype)
        scheduler = DPMSolverMultistepScheduler.from_pretrained(str(snap / "scheduler"))

        processor = None
        if (snap / "processor").is_dir():
            try:
                processor = internvlu.processing_internvlu.InternVLUProcessor.from_pretrained(
                    str(snap / "processor")
                )
            except Exception as exc:  # pragma: no cover — processor not strictly required
                logger.warning(f"Could not load processor from {snap}/processor: {exc}")

        idx = snap / "model_index.json"
        if idx.is_file():
            with idx.open() as f:
                manifest = json.load(f)
            components = {
                k: v
                for k, v in manifest.items()
                if not k.startswith("_") and isinstance(v, list)
            }
            cfg = InternVLUUnifiedConfig(
                pipeline_class_name=manifest.get("_class_name", "InternVLUPipeline"),
                diffusers_version=manifest.get("_diffusers_version"),
                components=components,
            )
        else:
            cfg = InternVLUUnifiedConfig()

        return cls(
            cfg,
            vlm=vlm,
            generation_decoder=gen_decoder,
            vae=vae,
            scheduler=scheduler,
            processor=processor,
        )

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------
    def save_pretrained(  # type: ignore[override]
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        save_function: Optional[Any] = None,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Write the pipeline directory layout to ``save_directory``.

        Produces:

        ::

            save_directory/
            ├── model_index.json
            ├── vlm/
            ├── generation_decoder/
            ├── vae/
            ├── scheduler/
            └── processor/      (only if a processor was registered)

        The result is loadable by ``InternVLUPipeline.from_pretrained`` provided
        the ``internvlu`` package is importable in the consumer environment.
        """
        if push_to_hub:
            raise NotImplementedError("push_to_hub is not implemented for InternVLUUnifiedModel")
        if not is_main_process:
            return

        out = Path(save_directory)
        out.mkdir(parents=True, exist_ok=True)

        # 1. VLM — uses the existing InternVLUChatModel.save_pretrained path so
        #    LoRA / special-token-embedding state are preserved exactly.
        self.vlm.save_pretrained(
            out / "vlm",
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
            variant=variant,
        )

        # 2. Generation decoder.
        self.generation_decoder.save_pretrained(
            out / "generation_decoder",
            safe_serialization=safe_serialization,
            max_shard_size=max_shard_size,
            variant=variant,
        )

        # 3. VAE — diffusers' save_pretrained signature is narrower; only forward
        #    the args it accepts.
        self.vae.save_pretrained(out / "vae", safe_serialization=safe_serialization, variant=variant)

        # 4. Scheduler (config only).
        self.scheduler.save_pretrained(out / "scheduler")

        # 5. Processor (optional).
        if self.processor is not None:
            self.processor.save_pretrained(out / "processor")

        # 6. Pipeline manifest.
        manifest: Dict[str, Any] = {"_class_name": self.config.pipeline_class_name}
        if self.config.diffusers_version is not None:
            manifest["_diffusers_version"] = self.config.diffusers_version
        for name, value in self.config.components.items():
            # Skip components that were not actually saved (e.g. processor).
            if name == "processor" and self.processor is None:
                continue
            manifest[name] = list(value)
        with (out / "model_index.json").open("w") as f:
            json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Conditioning & forward
    # ------------------------------------------------------------------
    def _vlm_forward_with_hidden_states(
        self,
        *,
        pixel_values: Optional[torch.Tensor],
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Optional[torch.Tensor]]:
        """Run the VLM body with grads, returning all hidden states.

        Mirrors :meth:`InternVLUChatModel.generate_hidden_states` but **without** the
        ``@torch.no_grad`` decorator so the resulting hidden states carry grads.

        Args:
            pixel_values: Optional ViT inputs. ``None`` for text-only / pure imgen samples.
            input_ids: ``(B, N)`` token ids.
            attention_mask: ``(B, N)`` attention mask.
            image_grid_thw: Optional anyres metadata (rarely used here).

        Returns:
            ``(hidden_states_tuple, logits)``. ``logits`` is None — we don't need
            them when only training the decoder branch.
        """
        vlm = self.vlm
        assert vlm.img_context_token_id is not None, "vlm.img_context_token_id must be set"

        input_embeds = vlm.language_model.get_input_embeddings()(input_ids).clone()
        input_embeds = vlm.replace_img_special_tokens(input_embeds, input_ids)

        if pixel_values is not None and pixel_values.numel() > 0:
            vit_embeds = vlm.extract_feature(pixel_values, image_grid_thw)
            B, N, C = input_embeds.shape
            flat_embeds = input_embeds.reshape(B * N, C)
            flat_ids = input_ids.reshape(B * N)
            selected = flat_ids == vlm.img_context_token_id
            if selected.any():
                flat_embeds[selected] = (
                    flat_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
                )
            input_embeds = flat_embeds.reshape(B, N, C)

        outputs = vlm.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            padding_type="pad",
        )
        return outputs.hidden_states, None

    def _build_gen_conditioning(
        self,
        *,
        hidden_states: Tuple[torch.Tensor, ...],
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        generation_flags: torch.LongTensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Replicate the inference-time conditioning contract.

        Quoted lines from ``internvlu/pipeline_internvlu.py::_prepare_diffusion_inputs``::

            vlm_hidden_states = [vlm_hidden_states[i].view(B, L, -1)
                                 for i in self.generation_decoder.config.vlm_select_layer]
            vlm_hidden_states = torch.cat(vlm_hidden_states, dim=-1)  # B, N, C * num_layers
            state_mask = self._prepare_hidden_state_mask(...)
            vlm_hidden_states = [vlm_hidden_states[s] for s in state_mask]
            vlm_image_token_mask = [selected[s] for s in state_mask]

        Args:
            hidden_states: Tuple of LM hidden states (one per layer plus embeddings).
            input_ids: ``(B, N)`` token ids.
            attention_mask: ``(B, N)`` attention mask.
            generation_flags: ``(K_total,)`` boolean flags marking which ``<img>``
                positions are *generation* targets (vs. real input images).

        Returns:
            ``(encoder_hidden_states_list, encoder_image_token_mask_list)``,
            both length ``K`` (one entry per generation target).
        """
        vlm = self.vlm
        select_layers: Sequence[int] = self.generation_decoder.config.vlm_select_layer

        B, L = input_ids.shape
        layer_states = [hidden_states[i].view(B, L, -1) for i in select_layers]
        cond = torch.cat(layer_states, dim=-1)  # (B, L, C_total)

        state_mask = self._prepare_hidden_state_mask_pad(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_flags=generation_flags,
        )
        # `state_mask` has shape (K, B, L); `cond[s]` selects the contiguous span
        # for each k along its row, returning a 2-D tensor of shape (span_len, C).
        encoder_hidden_states = [cond[s] for s in state_mask]
        selected_ctx = (input_ids == vlm.img_context_token_id)
        encoder_image_token_mask = [selected_ctx[s] for s in state_mask]
        return encoder_hidden_states, encoder_image_token_mask

    def _prepare_hidden_state_mask_pad(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        generation_flags: torch.LongTensor,
    ) -> torch.Tensor:
        """Padded variant of ``InternVLUPipeline._prepare_hidden_state_mask``.

        Builds a boolean mask of shape ``(K, B, N)`` marking, for each
        generation target ``k``, the tokens from the second ``<|im_start|>`` of
        its row up to the corresponding ``<img>`` token (inclusive), masked by
        ``attention_mask``.
        """
        vlm = self.vlm
        assert vlm.im_start_token_id is not None, (
            "InternVLUUnifiedModel: vlm.im_start_token_id must be set before training."
        )
        assert vlm.img_start_token_id is not None, (
            "InternVLUUnifiedModel: vlm.img_start_token_id must be set before training."
        )

        B, N = input_ids.shape
        device = input_ids.device

        img_start_positions = (input_ids == vlm.img_start_token_id).nonzero()  # [K_total, 2]
        gen_img_start_positions = img_start_positions[generation_flags.bool()]  # [K, 2]

        rows = torch.arange(B, device=device)[None]  # (1, B)
        cols = torch.arange(N, device=device)[None]  # (1, N)
        state_mask_row = rows == gen_img_start_positions[:, :1]  # (K, B)
        state_mask_col = cols <= gen_img_start_positions[:, 1:]  # (K, N)

        im_start_positions = (input_ids == vlm.im_start_token_id).nonzero()  # (K_im, 2)
        im_state_mask_row = rows == im_start_positions[:, :1]  # (K_im, B)
        im_start_second_idxs = (im_state_mask_row.cumsum(dim=0) == 2).nonzero(as_tuple=True)[0]
        im_start_second_positions = im_start_positions[im_start_second_idxs]  # (K_im2, 2)

        gen_pos_global = gen_img_start_positions[:, 0] * N + gen_img_start_positions[:, 1]
        bos_pos_global = gen_img_start_positions[:, 0] * N
        im2_pos_global = im_start_second_positions[:, 0] * N + im_start_second_positions[:, 1]

        in_range = (
            (im2_pos_global[None, :] <= gen_pos_global[:, None])
            & (im2_pos_global[None, :] >= bos_pos_global[:, None])
        )  # (K, K_im2)
        chosen_idx = in_range.int().argmax(dim=1)  # (K,)
        chosen_im2 = im_start_second_positions[chosen_idx]  # (K, 2)

        state_mask_col = state_mask_col & (chosen_im2[:, 1:] <= cols)
        state_mask = state_mask_row[..., None] & state_mask_col[:, None]  # (K, B, N)
        state_mask = state_mask & attention_mask.bool()[None]
        return state_mask

    def _encode_target_images(self, target_pixel_values: torch.Tensor) -> torch.Tensor:
        """VAE-encode imgen target images to normalised latents.

        Args:
            target_pixel_values: ``(B, 3, H, W)`` float tensor in the VAE's input
                normalisation (mean/std = (0.5, 0.5, 0.5)).

        Returns:
            ``(B, z_dim, H', W')`` normalised latent tensor, on the same device.
        """
        vae = self.vae
        x = target_pixel_values.to(dtype=next(vae.parameters()).dtype)
        x = x.unsqueeze(2)  # (B, C, 1, H, W) — Qwen-Image VAE wants a T axis
        with torch.no_grad():
            enc = vae.encode(x)
            if hasattr(enc, "latent_dist"):
                z = enc.latent_dist.sample()
            else:
                z = enc.latents
        latents_mean = (
            torch.tensor(vae.config.latents_mean, device=z.device, dtype=z.dtype)
            .view(1, -1, 1, 1, 1)
        )
        latents_std = (
            torch.tensor(vae.config.latents_std, device=z.device, dtype=z.dtype)
            .view(1, -1, 1, 1, 1)
        )
        z = (z - latents_mean) / latents_std
        return z.squeeze(2)

    def configure_gen_loss_schedule(self, *, target_weight: float, warmup_steps: int) -> None:
        """Set the gen-loss warmup schedule. Called once by the trainer at startup."""
        self._gen_loss_target_weight = float(target_weight)
        self._gen_loss_warmup_steps = int(warmup_steps)

    def _gen_loss_weight(self, global_step: int) -> float:
        """Linear ramp from 0 to ``target_weight`` over ``warmup_steps``."""
        target = self._gen_loss_target_weight
        warmup = self._gen_loss_warmup_steps
        if warmup <= 0:
            return target
        return target * min(1.0, max(0.0, global_step / warmup))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(  # type: ignore[override]
        self,
        *,
        task_type: Optional[Union[str, Sequence[str]]] = None,
        # ---- understanding-branch inputs (mirror InternVLUChatModel.forward) ----
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        # ---- imgen-branch extra inputs ----
        gen_input_ids: Optional[torch.LongTensor] = None,
        gen_attention_mask: Optional[torch.Tensor] = None,
        gen_pixel_values: Optional[torch.FloatTensor] = None,
        gen_image_flags: Optional[torch.LongTensor] = None,
        gen_target_pixel_values: Optional[torch.FloatTensor] = None,
        gen_generation_flags: Optional[torch.LongTensor] = None,
        gen_image_grid_thw: Optional[torch.LongTensor] = None,
        # ---- bookkeeping ----
        global_step: Optional[int] = None,
        return_dict: bool = True,
        **unused: Any,
    ) -> Union[UnifiedForwardOutput, Dict[str, torch.Tensor]]:
        """Compute combined LM + flow-matching loss for a mixed-task batch.

        The collator partitions the batch into a homogeneous understanding tensor
        block (``input_ids`` / ``pixel_values`` / ``labels``) and an optional
        homogeneous imgen tensor block (``gen_*`` fields). Either block may be
        empty for a given step; the loss container always has both keys with
        zero tensors for the missing branch so DeepSpeed / the HF Trainer can
        compute scalar gradients without special-casing.
        """
        device = next(self.vlm.parameters()).device
        zero = torch.zeros((), device=device, dtype=torch.float32)
        lm_loss = zero
        gen_loss = zero

        # ---------------- understanding branch ----------------
        if input_ids is not None and input_ids.numel() > 0:
            u_out = self.vlm(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags,
                labels=labels,
                output_hidden_states=False,
                return_dict=True,
                image_grid_thw=image_grid_thw,
            )
            if u_out.loss is not None:
                lm_loss = u_out.loss

        # ---------------- imgen branch ----------------
        if (
            gen_input_ids is not None
            and gen_input_ids.numel() > 0
            and gen_target_pixel_values is not None
            and gen_target_pixel_values.numel() > 0
        ):
            gen_loss = self._compute_gen_loss(
                input_ids=gen_input_ids,
                attention_mask=gen_attention_mask,
                pixel_values=gen_pixel_values,
                image_flags=gen_image_flags,
                target_pixel_values=gen_target_pixel_values,
                generation_flags=gen_generation_flags,
                image_grid_thw=gen_image_grid_thw,
            )

        weight = self._gen_loss_weight(global_step or 0)
        weight_t = torch.tensor(weight, device=device, dtype=lm_loss.dtype if lm_loss.requires_grad else torch.float32)
        total = lm_loss + weight_t * gen_loss

        out = UnifiedForwardOutput(loss=total, lm_loss=lm_loss, gen_loss=gen_loss, gen_weight=weight_t)
        if return_dict:
            return out
        return {"loss": total, "lm_loss": lm_loss, "gen_loss": gen_loss}

    def _compute_gen_loss(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.FloatTensor],
        image_flags: Optional[torch.LongTensor],
        target_pixel_values: torch.FloatTensor,
        generation_flags: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor],
    ) -> torch.Tensor:
        """Flow-matching MSE loss for one homogeneous imgen sub-batch."""
        decoder_cfg = self.generation_decoder.config

        hidden_states, _ = self._vlm_forward_with_hidden_states(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
        )

        encoder_hidden_states, encoder_image_token_mask = self._build_gen_conditioning(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_flags=generation_flags,
        )

        clean = self._encode_target_images(target_pixel_values)
        B = clean.shape[0]
        if B != len(encoder_hidden_states):
            raise RuntimeError(
                f"InternVLUUnifiedModel: encoder_hidden_states length ({len(encoder_hidden_states)})"
                f" does not match target batch size ({B})."
            )

        sigma = fm_utils.compute_density_for_timestep_sampling(
            weighting_scheme=decoder_cfg.weighting_scheme,
            batch_size=B,
            logit_mean=decoder_cfg.logit_mean,
            logit_std=decoder_cfg.logit_std,
            mode_scale=decoder_cfg.mode_scale,
            device=clean.device,
        )
        sigma_shifted = fm_utils.shift_sigma(sigma, decoder_cfg.flow_shift)
        noisy, target = fm_utils.build_noisy_latents(clean, sigma_shifted)

        timestep = fm_utils.sigma_to_timestep(sigma_shifted)

        proj_hidden, attn_masks, image_token_mask = self.generation_decoder.prepare_forward_input(
            encoder_hidden_states,
            encoder_image_token_mask=encoder_image_token_mask,
        )

        decoder_dtype = next(self.generation_decoder.decoder.parameters()).dtype
        pred = self.generation_decoder.decoder(
            noisy.to(decoder_dtype),
            encoder_hidden_states=proj_hidden.to(decoder_dtype),
            encoder_attention_mask=attn_masks,
            encoder_image_token_mask=image_token_mask,
            timestep=timestep.to(decoder_dtype),
            return_dict=False,
        )[0].float()

        loss = torch.nn.functional.mse_loss(pred, target.float(), reduction="none")
        weighting = fm_utils.compute_loss_weighting(decoder_cfg.weighting_scheme, sigma_shifted)
        loss = (loss * weighting.to(loss.dtype)).mean()
        return loss
