"""Conformal finite-sample calibration on clean speaker blocks.

Implements Proposal Section 7.7:
- Calibration unit: Speaker block of up to 10 episodes
- Block score A_i^block = max_{k <= 10, t} A(u_{i, k, t})
- Finite-sample quantile k = ceil((n_cal + 1)(1 - beta))
- Final threshold q_off = max(q_cal, eta_used)
- Usable gate width w = max(0, eta_used + nu - q_off)
"""

import math
from typing import Dict, List, Optional, Tuple
import torch

from ..core.geometry import read_coordinates, compute_all_patch_distances
from ..core.gate import aggregate_class_distances, compute_gate_width
from .schema import PhonoRepairArtifact


class ConformalCalibrator:
    """Computes distribution-free conformal bounds on clean speaker blocks."""

    def __init__(self, beta: float = 0.01):
        self.beta = beta

    def compute_block_score(
        self,
        episodes_features: List[torch.Tensor],
        artifact: PhonoRepairArtifact,
    ) -> float:
        """Compute maximum frame score A(u) across all episodes in a speaker block.

        A_i^block = max_{k <= 10, t} A(u_{i, k, t})
        """
        max_score = 0.0

        for ep_feat in episodes_features:
            ep_feat = ep_feat.to(artifact.Q.device).float()
            # shape (T, d)
            u_seq = read_coordinates(ep_feat, artifact.Q, artifact.mu)
            patch_dists = compute_all_patch_distances(
                u_seq, artifact.anchors, artifact.bases, artifact.radii
            )
            _, d_1, _, _ = aggregate_class_distances(
                patch_dists, artifact.patch_labels, artifact.num_classes
            )
            ep_max = float(torch.max(d_1).item())
            if ep_max > max_score:
                max_score = ep_max

        return max_score

    def calibrate(
        self,
        speaker_blocks: List[List[torch.Tensor]],
        artifact: PhonoRepairArtifact,
    ) -> Tuple[PhonoRepairArtifact, Dict[str, float]]:
        """Calibrate the artifact using n_cal speaker blocks.

        Args:
            speaker_blocks: List of n_cal blocks, each containing a list of episode feature tensors
            artifact: Initial uncalibrated artifact

        Returns:
            calibrated_artifact: Artifact with updated q_cal, q_off, and w
            stats: Summary metrics dictionary
        """
        n_cal = len(speaker_blocks)
        assert n_cal > 0, "No calibration speaker blocks provided"

        block_scores = []
        for block in speaker_blocks:
            score = self.compute_block_score(block, artifact)
            block_scores.append(score)

        block_scores.sort()

        # Finite-sample conformal rank: k = ceil((n_cal + 1)(1 - beta))
        k = math.ceil((n_cal + 1) * (1.0 - self.beta))
        if k <= n_cal:
            # 1-indexed k maps to index k-1
            q_cal = float(block_scores[k - 1])
            is_formal_conformal = True
        else:
            # Pilot fallback when n_cal < ceil(1/beta - 1):
            # Use empirical maximum across available speaker blocks
            q_cal = float(block_scores[-1])
            is_formal_conformal = False

        q_off = max(q_cal, artifact.eta)
        w = compute_gate_width(artifact.eta, artifact.nu, q_cal)

        # Update artifact
        artifact.q_cal = q_cal
        artifact.q_off = q_off
        artifact.w = w

        stats = {
            "n_cal_speakers": float(n_cal),
            "beta": float(self.beta),
            "rank_k": float(k),
            "q_cal": q_cal,
            "q_off": q_off,
            "eta": artifact.eta,
            "nu": artifact.nu,
            "w": w,
            "is_gate_open": 1.0 if w > 0.0 else 0.0,
            "is_formal_conformal": 1.0 if is_formal_conformal else 0.0,
        }
        return artifact, stats
