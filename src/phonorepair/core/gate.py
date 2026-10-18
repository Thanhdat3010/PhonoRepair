"""Gate logic and feasibility scoring for PhonoRepair.

Implements the exact gate conditions from Proposal Section 7.7 & 8.2:
- Score width w = max(0, eta + nu - max(q_cal, eta))
- Frame score A(u) = d_{(1)}(u)
- Identity fallback when A(u) <= q_off = max(q_cal, eta)
- Feasible classes: { c : d_c(u) <= eta + nu }
- Singleton routing condition: exactly 1 feasible class
"""

from typing import Dict, List, Optional, Tuple
import torch


def compute_gate_width(eta: float, nu: float, q_cal: float) -> float:
    """Compute the usable distance-score gate width w.

    w = max(0, eta + nu - max(q_cal, eta))
    If w == 0, the gate is mathematically empty and will reject 100% of inputs.
    """
    lower_threshold = max(q_cal, eta)
    upper_threshold = eta + nu
    return max(0.0, upper_threshold - lower_threshold)


def aggregate_class_distances(
    patch_distances: torch.Tensor,
    patch_labels: torch.Tensor,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Aggregate M patch distances into class-level minimum distances.

    Args:
        patch_distances: (..., M) distance to each patch
        patch_labels: (M,) class label for each patch in 0 .. num_classes-1
        num_classes: Total number of phonetic classes C

    Returns:
        class_distances: (..., C) minimum distance to each class
        d_1: (...,) distance to nearest class
        d_2: (...,) distance to second nearest class
        nearest_class: (...,) class index of d_1
    """
    prefix_shape = patch_distances.shape[:-1]
    flat_distances = patch_distances.reshape(-1, patch_distances.shape[-1])
    N = flat_distances.shape[0]

    class_dists = torch.full((N, num_classes), float("inf"), dtype=patch_distances.dtype, device=patch_distances.device)

    for c in range(num_classes):
        mask = (patch_labels == c)
        if mask.any():
            class_dists[:, c] = torch.min(flat_distances[:, mask], dim=-1).values

    # Sort class distances to get d_(1) and d_(2)
    sorted_dists, sorted_indices = torch.sort(class_dists, dim=-1)
    d_1 = sorted_dists[:, 0]
    d_2 = sorted_dists[:, 1]
    nearest_class = sorted_indices[:, 0]

    out_shape_class = (*prefix_shape, num_classes)
    return (
        class_dists.reshape(out_shape_class),
        d_1.reshape(prefix_shape),
        d_2.reshape(prefix_shape),
        nearest_class.reshape(prefix_shape),
    )


def evaluate_gate(
    class_distances: torch.Tensor,
    q_off: float,
    eta: float,
    nu: float,
    margin_gamma: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
    """Evaluate whether frames are accepted for geometric repair.

    Args:
        class_distances: (N, C) minimum distance to each class
        q_off: Statistical open threshold max(q_cal, eta)
        eta: Envelope radius eta_used
        nu: Threat/noise budget nu
        margin_gamma: Singleton margin requirement d_(2) - d_(1) >= margin_gamma

    Returns:
        is_accepted: (N,) boolean tensor, True if frame is accepted for repair
        target_class: (N,) long tensor, selected class ID (meaningful only when is_accepted is True)
        rejection_reasons: List of N strings explaining the rejection reason
    """
    N, C = class_distances.shape
    is_accepted = torch.zeros(N, dtype=torch.bool, device=class_distances.device)
    target_class = torch.zeros(N, dtype=torch.long, device=class_distances.device)
    rejection_reasons: List[str] = []

    upper_bound = eta + nu

    for i in range(N):
        dists = class_distances[i]
        d_1 = torch.min(dists).item()

        # Check 1: Inside clean/unrepairable envelope (Fallback to identity)
        if d_1 <= q_off:
            rejection_reasons.append("below_q_off_identity")
            continue

        # Check 2: Feasible classes
        feasible_mask = dists <= upper_bound
        num_feasible = int(torch.sum(feasible_mask).item())

        if num_feasible == 0:
            rejection_reasons.append("feasible_empty")
        elif num_feasible > 1:
            if margin_gamma > 0.0:
                sorted_d, sorted_idx = torch.sort(dists)
                if (sorted_d[1] - sorted_d[0]).item() >= margin_gamma:
                    # Singleton margin separation satisfied
                    best_c = int(sorted_idx[0].item())
                    is_accepted[i] = True
                    target_class[i] = best_c
                    rejection_reasons.append("accepted")
                else:
                    rejection_reasons.append("feasible_ambiguous_multiple")
            else:
                rejection_reasons.append("feasible_ambiguous_multiple")
        else:
            # Singleton accepted!
            best_c = int(torch.nonzero(feasible_mask, as_tuple=True)[0][0].item())
            is_accepted[i] = True
            target_class[i] = best_c
            rejection_reasons.append("accepted")

    return is_accepted, target_class, rejection_reasons

