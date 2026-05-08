"""Unit tests for the flow-matching helpers used by the unified-SFT path."""

import pytest
import torch

from internvl.model.internvlu import flow_matching_utils as fm


def test_logit_normal_returns_in_range():
    sigma = fm.compute_density_for_timestep_sampling(
        "logit_normal", batch_size=128, logit_mean=0.0, logit_std=1.0
    )
    assert sigma.shape == (128,)
    assert (sigma > 0).all() and (sigma < 1).all()


def test_shift_sigma_identity_when_one():
    s = torch.linspace(0.01, 0.99, 16)
    out = fm.shift_sigma(s, 1.0)
    assert torch.allclose(s, out)


def test_shift_sigma_3_pushes_mass_up():
    """For shift > 1 every sigma must increase, and 0/1 are fixed points."""
    s = torch.tensor([0.1, 0.5, 0.9])
    out = fm.shift_sigma(s, 3.0)
    assert (out > s).all()


def test_build_noisy_latents_shapes_and_target():
    clean = torch.randn(4, 16, 8, 8)
    sigma = torch.full((4,), 0.5)
    noisy, target = fm.build_noisy_latents(clean, sigma)
    assert noisy.shape == clean.shape
    assert target.shape == clean.shape
    # noisy = (1-sigma)*clean + sigma*noise => sigma=0.5 => noise = 2*noisy - clean
    noise = 2.0 * noisy - clean
    assert torch.allclose(target, noise - clean, atol=1e-5)


def test_sigma_to_timestep_scales_to_1000():
    s = torch.tensor([0.0, 0.5, 1.0])
    t = fm.sigma_to_timestep(s)
    assert torch.allclose(t, torch.tensor([0.0, 500.0, 1000.0]))
