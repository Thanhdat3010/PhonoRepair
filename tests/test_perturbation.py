"""Unit tests for perturbation utilities (E6 and E10)."""

import numpy as np
import pytest
import torch

from phonorepair.evaluation.perturbation import add_acoustic_noise, perturb_latent_frame
from phonorepair.artifacts.schema import PhonoRepairArtifact


def test_add_acoustic_noise_snr():
    """Verify that adding noise scales properly with SNR."""
    # Synthetic sine wave
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    noisy_20 = add_acoustic_noise(signal, snr_db=20.0, seed=42)
    noisy_5 = add_acoustic_noise(signal, snr_db=5.0, seed=42)

    # 5dB noise should have higher deviation from signal than 20dB noise
    diff_20 = np.mean((noisy_20 - signal) ** 2)
    diff_5 = np.mean((noisy_5 - signal) ** 2)
    assert diff_5 > diff_20


def test_perturb_latent_frame_norm():
    """Verify that latent perturbation norm matches kappa * nu."""
    r = 16
    u = torch.zeros(r)
    artifact = PhonoRepairArtifact(
        model_id="test",
        feature_dim=32,
        repair_dim=r,
        num_classes=2,
        mu=torch.zeros(32),
        Q=torch.eye(32, r),
        anchors=torch.zeros(2, r),
        bases=[torch.zeros(r, 0), torch.zeros(r, 0)],
        radii=torch.zeros(2),
        patch_labels=torch.tensor([0, 1]),
        patch_dimensions=torch.tensor([0, 0]),
        eta=2.0,
        nu=1.5,
        q_cal=2.0,
        q_off=2.0,
        w=1.5,
        correction_budget=3.5,
        is_active=True,
    )

    kappa = 0.5
    u_pert = perturb_latent_frame(u, true_class=0, artifact=artifact, kappa=kappa, direction_mode="random")
    displacement = torch.norm(u_pert - u, p=2).item()
    expected_displacement = kappa * artifact.nu
    assert abs(displacement - expected_displacement) < 1e-5
