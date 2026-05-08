"""Flow-matching helpers for InternVL-U full unified SFT.

These mirror the standard SD3 / FluxPipeline flow-matching training recipe
(`weighting_scheme="logit_normal"`) so the new training-time forward stays
in-distribution with how the released `InternVLUGenerationDecoder` was trained.

References:
    * `internvlu.diffusion.configuration_internvlu_generation_decoder.InternVLUGenerationDecoderConfig`
      (`weighting_scheme`, `logit_mean`, `logit_std`, `mode_scale`, `flow_shift`)
    * diffusers `examples/dreambooth/train_dreambooth_sd3.py`,
      `compute_density_for_timestep_sampling`, `compute_loss_weighting_for_sd3`
"""

from __future__ import annotations

from typing import Optional

import torch


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Sample `sigma in (0, 1)` per batch element.

    Args:
        weighting_scheme: One of ``logit_normal``, ``mode``, ``cosmap``, ``uniform`` / ``null``.
            ``logit_normal`` matches the released InternVL-U decoder config.
        batch_size: Number of sigmas to draw.
        logit_mean: Mean of the underlying normal (logit_normal scheme only).
        logit_std: Std of the underlying normal.
        mode_scale: Scale used in the ``mode`` scheme.
        device: Optional device for the returned tensor.

    Returns:
        Float tensor of shape ``(batch_size,)`` with values in ``(0, 1)``.
    """
    if weighting_scheme == "logit_normal":
        u = torch.normal(mean=logit_mean, std=logit_std, size=(batch_size,), device=device)
        sigma = torch.sigmoid(u)
    elif weighting_scheme == "mode":
        u = torch.rand(batch_size, device=device)
        sigma = 1.0 - u - mode_scale * (torch.cos(torch.pi * u / 2) ** 2 - 1.0 + u)
    elif weighting_scheme == "cosmap":
        u = torch.rand(batch_size, device=device)
        sigma = 1.0 - 1.0 / (torch.tan(torch.pi / 2.0 * u) + 1.0)
    else:
        sigma = torch.rand(batch_size, device=device)
    return sigma.clamp(min=1e-5, max=1.0 - 1e-5)


def shift_sigma(sigma: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply the SD3/Flux flow-matching ``shift`` reparameterisation.

    Args:
        sigma: Tensor of unshifted sigmas in ``(0, 1)``.
        shift: Shift parameter (``>= 1`` pushes mass toward the noisy end).

    Returns:
        Shifted sigma tensor of the same shape.
    """
    if shift is None or shift == 1.0:
        return sigma
    return (shift * sigma) / (1.0 + (shift - 1.0) * sigma)


def compute_loss_weighting(
    weighting_scheme: str, sigma: torch.Tensor
) -> torch.Tensor:
    """Compute per-sample MSE weighting for flow-matching losses.

    Args:
        weighting_scheme: One of ``sigma_sqrt`` or anything else (returns ones).
        sigma: Tensor of sigmas in ``(0, 1)``.

    Returns:
        Tensor of weights broadcastable to ``(batch, 1, 1, 1)``.
    """
    if weighting_scheme == "sigma_sqrt":
        return (sigma ** -2.0).reshape(-1, 1, 1, 1)
    return torch.ones_like(sigma).reshape(-1, 1, 1, 1)


def build_noisy_latents(
    clean_latents: torch.Tensor,
    sigma_shifted: torch.Tensor,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample noise and build noisy latents under flow matching.

    `noisy = (1 - sigma) * clean + sigma * noise`. The flow-prediction
    target (``noise - clean``) is returned so the caller can compute MSE
    against the model output without reconstructing ``noise`` again.

    Args:
        clean_latents: Latent tensor of shape ``(B, C, H, W)`` (already mean/std-normalised).
        sigma_shifted: Shifted sigmas of shape ``(B,)`` in ``(0, 1)``.
        generator: Optional CUDA generator for noise sampling.

    Returns:
        ``(noisy_latents, target)`` where ``target = noise - clean``.
    """
    noise = torch.randn(
        clean_latents.shape,
        generator=generator,
        device=clean_latents.device,
        dtype=clean_latents.dtype,
    )
    sigma_b = sigma_shifted.view(-1, 1, 1, 1).to(clean_latents.dtype)
    noisy = (1.0 - sigma_b) * clean_latents + sigma_b * noise
    target = noise - clean_latents
    return noisy, target


def sigma_to_timestep(sigma_shifted: torch.Tensor, num_train_timesteps: int = 1000) -> torch.Tensor:
    """Map a shifted-sigma value to the integer timestep the decoder expects.

    The released `InternVLUGenerationDecoder` was trained with
    ``DPMSolverMultistepScheduler`` whose default `num_train_timesteps=1000`,
    and `prediction_type="flow_prediction"`. The decoder's
    `time_text_embed` consumes a continuous-valued timestep, so we just
    multiply by the schedule length and keep float dtype.
    """
    return (sigma_shifted * num_train_timesteps).to(sigma_shifted.dtype)
