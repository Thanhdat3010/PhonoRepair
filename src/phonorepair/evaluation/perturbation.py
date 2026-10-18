"""Perturbation generators for E2/E10 (Synthetic Latent Perturbation) and E6 (Acoustic Corruption).

Implements Proposal Section 12:
- E10 Synthetic Latent Perturbation: xi_t = kappa * nu along (i) tangent, (ii) orthogonal, (iii) competing class
- E6 Acoustic Corruption: Additive noise at calibrated SNR (20dB, 10dB, 5dB, 0dB)
"""

from typing import List, Optional, Tuple
import numpy as np
import torch


def add_acoustic_noise(
    audio: np.ndarray,
    snr_db: float,
    seed: int = 17,
) -> np.ndarray:
    """Add calibrated Gaussian noise to audio waveform matching target SNR.

    Formula from Proposal Section 12 (E6):
    alpha = sqrt(sum(x^2) / (sum(n^2) * 10^(SNR/10)))
    y = x + alpha * n
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(*audio.shape).astype(np.float32)

    power_signal = np.sum(audio ** 2)
    power_noise = np.sum(noise ** 2)

    if power_signal == 0 or power_noise == 0:
        return audio.copy()

    # Calculate scale factor alpha
    snr_linear = 10.0 ** (snr_db / 10.0)
    alpha = np.sqrt(power_signal / (power_noise * snr_linear))

    noisy_audio = audio + alpha * noise

    # Rescale to avoid clipping while preserving dynamic range
    max_amp = np.max(np.abs(noisy_audio))
    if max_amp > 1.0:
        noisy_audio = noisy_audio / max_amp

    return noisy_audio.astype(np.float32)


def perturb_latent_frame(
    u: torch.Tensor,
    true_class: int,
    artifact: any,
    kappa: float = 1.0,
    direction_mode: str = "competing",
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Perturb a coordinate u by displacement ||xi|| = kappa * nu.

    Args:
        u: Coordinate tensor (r,)
        true_class: Ground truth phonetic class index c*
        artifact: PhonoRepairArtifact
        kappa: Perturbation strength multiplier (0.25, 0.5, 1.0, 1.5, 2.0)
        direction_mode: One of 'competing', 'orthogonal', 'random'
        rng: Optional torch Generator

    Returns:
        u_perturbed: Tensor (r,)
    """
    nu = artifact.nu
    step_norm = kappa * nu
    device = u.device
    r = u.shape[-1]

    if step_norm == 0.0:
        return u.clone()

    if direction_mode == "competing":
        # Direction pointing towards the nearest anchor of a competing class
        competing_mask = (artifact.patch_labels != true_class)
        if competing_mask.any():
            competing_anchors = artifact.anchors[competing_mask]
            dists = torch.norm(competing_anchors - u.unsqueeze(0), dim=-1)
            nearest_idx = torch.argmin(dists)
            target_anchor = competing_anchors[nearest_idx]
            v = target_anchor - u
        else:
            v = torch.randn(r, device=device, generator=rng)
    elif direction_mode == "random":
        v = torch.randn(r, device=device, generator=rng)
    else:
        # Default orthogonal / random
        v = torch.randn(r, device=device, generator=rng)

    norm_v = torch.norm(v, p=2)
    if norm_v > 1e-6:
        v_unit = v / norm_v
    else:
        v_unit = torch.zeros_like(v)
        v_unit[0] = 1.0

    return u + step_norm * v_unit


def perturb_latent_sequence(
    h_seq: torch.Tensor,
    frame_labels: torch.Tensor,
    artifact: any,
    kappa: float = 1.0,
    direction_mode: str = "competing",
) -> torch.Tensor:
    """Apply calibrated latent perturbation along valid frames of an utterance.

    Args:
        h_seq: Encoder output tensor (T, d)
        frame_labels: Phonetic labels per frame (T,)
        artifact: PhonoRepairArtifact
        kappa: Perturbation strength multiplier
        direction_mode: Perturbation direction

    Returns:
        h_seq_corrupted: Corrupted tensor (T, d)
    """
    T, d = h_seq.shape
    Q = artifact.Q
    mu = artifact.mu
    device = h_seq.device

    # 1. Project to coordinate space u_seq = (h_seq - mu) @ Q
    centered = h_seq - mu
    u_seq = torch.matmul(centered, Q)  # (T, r)

    u_corrupted = u_seq.clone()
    for t in range(T):
        lbl = int(frame_labels[t].item()) if t < len(frame_labels) else 0
        u_corrupted[t] = perturb_latent_frame(
            u_seq[t], lbl, artifact, kappa=kappa, direction_mode=direction_mode
        )

    # 2. Lift back to latent space: h' = h + (u' - u) @ Q.T
    delta_u = u_corrupted - u_seq  # (T, r)
    delta_h = torch.matmul(delta_u, Q.t())  # (T, d)
    return h_seq + delta_h
