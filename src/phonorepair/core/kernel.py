"""Unified REPAIR_FRAME kernel for PhonoRepair.

Implements the exact execution contract from Proposal Section 8.2:
~~~text
REPAIR_FRAME(h, artifact):
    if outside declared scope/cap or artifact inactive: return h
    u = Q.T @ (h - mean)
    calculate distances of all patches; min by class
    A = min class distance
    if A <= q_off: return h
    feasible = {p: d_p <= eta_used + nu}
    if feasible not singleton: return h
    candidate = exact nearest-patch projection of the unique class
    if norm(candidate - u) > declared correction budget: return h
    return h + Q @ (candidate - u)
~~~
"""

from typing import Dict, List, Optional, Tuple, Any
import torch

from .geometry import (
    read_coordinates,
    lift_coordinates,
    project_to_patch,
    compute_patch_distance,
    compute_all_patch_distances,
)
from .gate import aggregate_class_distances, evaluate_gate


def repair_frame(
    h: torch.Tensor,
    artifact: Any,
    margin_gamma: Optional[float] = None,
    alpha_blend: Optional[float] = None,
) -> Tuple[torch.Tensor, str, Optional[int]]:
    """Apply PhonoRepair to a single frame latent vector h.

    Args:
        h: Latent tensor of shape (d,)
        artifact: Loaded PhonoRepair artifact object containing:
            - Q: Orthonormal basis (d, r)
            - mu: Mean vector (d,)
            - anchors: Patch anchors (M, r)
            - bases: List of patch bases (r, s_j)
            - radii: Patch radii (M,)
            - patch_labels: Tensor (M,) mapping patch index -> class index
            - num_classes: Total phonetic classes C
            - q_off: Gate open threshold max(q_cal, eta)
            - eta: Envelope radius
            - nu: Noise budget
            - correction_budget: Maximum allowed displacement norm (default eta + nu)
            - is_active: bool
        margin_gamma: Optional singleton margin threshold (overrides artifact)
        alpha_blend: Optional projection shrinkage factor in (0, 1] (overrides artifact)

    Returns:
        h_out: Repaired or original latent tensor of shape (d,)
        reason: Execution status / rejection reason string
        target_class: Selected phonetic class ID if repaired, else None
    """
    if not getattr(artifact, "is_active", True):
        return h, "artifact_inactive", None

    Q = artifact.Q
    mu = artifact.mu
    anchors = artifact.anchors
    bases = artifact.bases
    radii = artifact.radii
    patch_labels = artifact.patch_labels
    num_classes = artifact.num_classes
    q_off = artifact.q_off
    eta = artifact.eta
    nu = artifact.nu
    correction_budget = getattr(artifact, "correction_budget", eta + nu)
    gamma = margin_gamma if margin_gamma is not None else getattr(artifact, "margin_gamma", 0.0)
    alpha = alpha_blend if alpha_blend is not None else getattr(artifact, "alpha_blend", 1.0)

    # 1. Project to repair space u = Q^T (h - mu)
    u = read_coordinates(h, Q, mu)

    # 2. Compute distances to all M patches
    patch_dists = compute_all_patch_distances(u, anchors, bases, radii)

    # 3. Aggregate class distances and evaluate gate
    class_dists, d_1, d_2, nearest_class = aggregate_class_distances(
        patch_dists.unsqueeze(0), patch_labels, num_classes
    )

    is_accepted, target_classes, reasons = evaluate_gate(class_dists, q_off, eta, nu, margin_gamma=gamma)

    if not is_accepted[0].item():
        return h, reasons[0], None

    c_star = int(target_classes[0].item())

    # 4. Find the nearest patch within the chosen singleton class c_star
    class_patch_indices = torch.nonzero(patch_labels == c_star, as_tuple=True)[0]
    best_dist = float("inf")
    best_proj = None

    for patch_idx in class_patch_indices:
        j = int(patch_idx.item())
        a_j = anchors[j]
        B_j = bases[j]
        rho_j = float(radii[j].item())

        dist_j = compute_patch_distance(u, a_j, B_j, rho_j).item()
        if dist_j < best_dist:
            best_dist = dist_j
            proj_j, _ = project_to_patch(u, a_j, B_j, rho_j)
            best_proj = proj_j

    if best_proj is None:
        return h, "no_valid_patch_in_class", None

    # 5. Check correction budget: ||candidate - u|| <= correction_budget
    displacement = torch.norm(best_proj - u, p=2).item()
    if displacement > correction_budget:
        return h, "displacement_exceeds_budget", None

    # 6. Apply alpha shrinkage and lift repaired coordinate: h_hat = h + alpha * Q (candidate - u)
    target_coord = u + alpha * (best_proj - u) if alpha < 1.0 else best_proj
    h_hat = lift_coordinates(target_coord, u, h, Q)
    return h_hat, "accepted", c_star


def repair_sequence(
    h_seq: torch.Tensor,
    artifact: Any,
    margin_gamma: Optional[float] = None,
    alpha_blend: Optional[float] = None,
) -> Tuple[torch.Tensor, List[str], List[Optional[int]]]:
    """Apply PhonoRepair vectorially over an acoustic sequence with fast reject path.

    Args:
        h_seq: Latent sequence tensor of shape (T, d)
        artifact: PhonoRepair artifact
        margin_gamma: Optional singleton margin threshold
        alpha_blend: Optional projection shrinkage factor in (0, 1]

    Returns:
        h_seq_repaired: Tensor of shape (T, d)
        reasons: List of T rejection/acceptance strings
        classes: List of T selected phonetic classes
    """
    T, d = h_seq.shape
    if not getattr(artifact, "is_active", True) or T == 0:
        return h_seq, ["artifact_inactive"] * T, [None] * T

    Q = artifact.Q
    mu = artifact.mu
    anchors = artifact.anchors
    bases = artifact.bases
    radii = artifact.radii
    patch_labels = artifact.patch_labels
    num_classes = artifact.num_classes
    q_off = artifact.q_off
    eta = artifact.eta
    nu = artifact.nu
    correction_budget = getattr(artifact, "correction_budget", eta + nu)
    gamma = margin_gamma if margin_gamma is not None else getattr(artifact, "margin_gamma", 0.0)
    alpha = alpha_blend if alpha_blend is not None else getattr(artifact, "alpha_blend", 1.0)

    # 1. Project sequence u_seq = (h_seq - mu) @ Q  -> shape (T, r)
    u_seq = read_coordinates(h_seq, Q, mu)

    # 2. Vectorized patch distances for all T frames across M patches -> shape (T, M)
    patch_dists = compute_all_patch_distances(u_seq, anchors, bases, radii)

    # 3. Class aggregation -> shape (T, C)
    class_dists, _, _, _ = aggregate_class_distances(patch_dists, patch_labels, num_classes)

    # 4. Evaluate gate across all frames
    is_accepted, target_classes, reasons = evaluate_gate(class_dists, q_off, eta, nu, margin_gamma=gamma)

    classes: List[Optional[int]] = [None] * T
    h_out = h_seq.clone()

    # Fast path: If no frames accepted (standard for clean speech under conformal envelope)
    if not is_accepted.any():
        return h_out, reasons, classes

    # Slow path: only for accepted frames (dysarthric or corrupted frames)
    accepted_indices = torch.nonzero(is_accepted, as_tuple=True)[0]
    for idx_t in accepted_indices:
        t = int(idx_t.item())
        c_star = int(target_classes[t].item())
        u_t = u_seq[t]
        h_t = h_seq[t]

        class_patch_indices = torch.nonzero(patch_labels == c_star, as_tuple=True)[0]
        best_dist = float("inf")
        best_proj = None

        for patch_idx in class_patch_indices:
            j = int(patch_idx.item())
            a_j = anchors[j]
            B_j = bases[j]
            rho_j = float(radii[j].item())

            dist_j = compute_patch_distance(u_t, a_j, B_j, rho_j).item()
            if dist_j < best_dist:
                best_dist = dist_j
                proj_j, _ = project_to_patch(u_t, a_j, B_j, rho_j)
                best_proj = proj_j

        if best_proj is None:
            reasons[t] = "no_valid_patch_in_class"
            continue

        displacement = torch.norm(best_proj - u_t, p=2).item()
        if displacement > correction_budget:
            reasons[t] = "displacement_exceeds_budget"
            continue

        target_coord = u_t + alpha * (best_proj - u_t) if alpha < 1.0 else best_proj
        h_hat = lift_coordinates(target_coord, u_t, h_t, Q)
        h_out[t] = h_hat
        classes[t] = c_star

    return h_out, reasons, classes
