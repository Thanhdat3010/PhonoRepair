"""Geometric operations for PhonoRepair.

Implements affine coordinate transforms, patch projections, and distance calculations:
- Read coordinates: u = Q^T (h - mu)
- Lift coordinates: h_hat = h + Q (u_hat - u) [Preserves orthogonal complement, Theorem A.2]
- Patch projection: pi_j(u) = a_j + B_j clip_rho_j(B_j^T (u - a_j))
- Patch distance: d_j(u)^2 = ||u - a_j||^2 - ||v_j||^2 + (max(0, ||v_j|| - rho_j))^2
"""

from typing import List, Tuple
import torch
import torch.nn.functional as F


def read_coordinates(h: torch.Tensor, Q: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Project high-dimensional latent h into reduced repair space u.

    Args:
        h: Latent tensor of shape (..., d)
        Q: Orthonormal basis matrix of shape (d, r) where Q^T Q = I_r
        mu: Mean vector of shape (d,)

    Returns:
        u: Reduced coordinates of shape (..., r)
    """
    centered = h - mu
    return torch.matmul(centered, Q)


def lift_coordinates(u_hat: torch.Tensor, u: torch.Tensor, h: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Lift repaired coordinate u_hat back into the original feature space.

    Guarantees Theorem A.2:
    - (I - Q Q^T) (h_hat - mu) = (I - Q Q^T) (h - mu)  [Orthogonal complement strictly preserved]
    - ||h_hat - h||^2 = ||u_hat - u||^2                [Iso-magnitude displacement]

    Args:
        u_hat: Repaired coordinate of shape (..., r)
        u: Original coordinate of shape (..., r)
        h: Original feature tensor of shape (..., d)
        Q: Orthonormal basis matrix of shape (d, r)

    Returns:
        h_hat: Repaired feature tensor of shape (..., d)
    """
    diff_u = u_hat - u
    delta_h = torch.matmul(diff_u, Q.t())
    return h + delta_h


def project_to_patch(
    u: torch.Tensor, anchor: torch.Tensor, basis: torch.Tensor, rho: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project coordinate u onto a single local affine patch S_j = { a_j + B_j w : ||w|| <= rho_j }.

    Args:
        u: Point coordinate of shape (r,) or (..., r)
        anchor: Patch center a_j of shape (r,)
        basis: Patch orthonormal basis B_j of shape (r, s_j) where B_j^T B_j = I_s
        rho: Patch radius rho_j >= 0

    Returns:
        proj: Projected coordinate pi_j(u) on the patch of shape (..., r)
        v: Projection coefficients B_j^T (u - a_j) of shape (..., s_j)
    """
    diff = u - anchor
    if basis.numel() == 0 or basis.shape[1] == 0:
        # 0-dimensional patch (Anchor only, s=0)
        proj = anchor.expand_as(u)
        v = torch.empty((*u.shape[:-1], 0), dtype=u.dtype, device=u.device)
        return proj, v

    # v = B_j^T (u - a_j)
    v = torch.matmul(diff, basis)
    v_norm = torch.norm(v, p=2, dim=-1, keepdim=True)

    # clip_rho(v) = v * min(1, rho / ||v||)
    scale = torch.clamp(rho / torch.clamp(v_norm, min=1e-12), max=1.0)
    clipped_v = v * scale

    # pi_j(u) = a_j + B_j clipped_v
    proj = anchor + torch.matmul(clipped_v, basis.t())
    return proj, v


def compute_patch_distance(
    u: torch.Tensor, anchor: torch.Tensor, basis: torch.Tensor, rho: float
) -> torch.Tensor:
    """Compute exact Euclidean distance from point u to patch S_j.

    Using the orthogonal decomposition from Section 8.1 / Appendix A.3:
    d_j(u)^2 = ||u - a_j||^2 - ||v_j||^2 + max(0, ||v_j|| - rho_j)^2

    Args:
        u: Point of shape (..., r)
        anchor: a_j of shape (r,)
        basis: B_j of shape (r, s_j)
        rho: rho_j >= 0

    Returns:
        dist: Exact Euclidean distance d_j(u) of shape (...)
    """
    diff = u - anchor
    sq_dist_diff = torch.sum(diff * diff, dim=-1)

    if basis.numel() == 0 or basis.shape[1] == 0:
        # Distance to point anchor
        return torch.sqrt(torch.clamp(sq_dist_diff, min=0.0))

    v = torch.matmul(diff, basis)
    sq_norm_v = torch.sum(v * v, dim=-1)
    norm_v = torch.sqrt(torch.clamp(sq_norm_v, min=0.0))

    excess = torch.clamp(norm_v - rho, min=0.0)
    sq_excess = excess * excess

    sq_d = sq_dist_diff - sq_norm_v + sq_excess
    return torch.sqrt(torch.clamp(sq_d, min=0.0))


def compute_all_patch_distances(
    u: torch.Tensor,
    anchors: torch.Tensor,
    bases: List[torch.Tensor],
    radii: torch.Tensor,
) -> torch.Tensor:
    """Compute distances from u to all M patches in the dictionary.

    Args:
        u: Query point of shape (r,) or (N, r)
        anchors: (M, r) tensor of all patch anchors
        bases: List of M tensors, each (r, s_j)
        radii: (M,) tensor of patch radii

    Returns:
        distances: (..., M) tensor of Euclidean distances
    """
    is_single = u.ndim == 1
    if is_single:
        u = u.unsqueeze(0)

    N = u.shape[0]
    M = anchors.shape[0]
    distances = torch.empty((N, M), dtype=u.dtype, device=u.device)

    for j in range(M):
        a_j = anchors[j]
        B_j = bases[j]
        rho_j = float(radii[j].item())
        distances[:, j] = compute_patch_distance(u, a_j, B_j, rho_j)

    return distances.squeeze(0) if is_single else distances
