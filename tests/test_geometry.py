"""Unit tests for core geometric operations (E0 correctness)."""

import pytest
import torch

from phonorepair.core.geometry import (
    read_coordinates,
    lift_coordinates,
    project_to_patch,
    compute_patch_distance,
)


def test_read_and_lift_orthogonality():
    """Verify Theorem A.2: Orthogonal components are preserved and displacement norm matches."""
    torch.manual_seed(42)
    d, r = 16, 4

    # Generate random orthonormal basis Q
    A = torch.randn(d, r)
    Q, _ = torch.linalg.qr(A)
    mu = torch.randn(d)

    h = torch.randn(d)
    u = read_coordinates(h, Q, mu)

    # Apply arbitrary modification in reduced repair space
    delta_u = torch.randn(r) * 0.5
    u_hat = u + delta_u

    h_hat = lift_coordinates(u_hat, u, h, Q)

    # 1. Coordinate check: Q^T (h_hat - mu) == u_hat
    u_reconstructed = read_coordinates(h_hat, Q, mu)
    assert torch.allclose(u_reconstructed, u_hat, atol=1e-5)

    # 2. Orthogonal complement check: (I - Q Q^T)(h_hat - mu) == (I - Q Q^T)(h - mu)
    proj_matrix = torch.matmul(Q, Q.t())
    ortho_h = (h - mu) - torch.matmul(proj_matrix, (h - mu))
    ortho_h_hat = (h_hat - mu) - torch.matmul(proj_matrix, (h_hat - mu))
    assert torch.allclose(ortho_h, ortho_h_hat, atol=1e-5)

    # 3. Norm check: ||h_hat - h||^2 == ||u_hat - u||^2
    disp_h = torch.norm(h_hat - h, p=2)
    disp_u = torch.norm(u_hat - u, p=2)
    assert torch.allclose(disp_h, disp_u, atol=1e-5)


def test_project_to_patch_inside_radius():
    """Verify that a point already inside the patch projection remains unchanged."""
    r, s = 8, 2
    anchor = torch.zeros(r)

    B, _ = torch.linalg.qr(torch.randn(r, s))
    rho = 2.0

    # Point directly on the subspace inside rho
    w = torch.tensor([0.5, -0.3])
    u = anchor + torch.matmul(w, B.t())

    proj, v = project_to_patch(u, anchor, B, rho)
    dist = compute_patch_distance(u, anchor, B, rho)

    assert torch.allclose(proj, u, atol=1e-5)
    assert dist.item() < 1e-6


def test_project_to_patch_outside_radius_clipped():
    """Verify that points exceeding rho are clipped to the boundary ||v|| == rho."""
    r, s = 8, 2
    anchor = torch.zeros(r)

    B, _ = torch.linalg.qr(torch.randn(r, s))
    rho = 1.0

    # Point along subspace exceeding rho
    w = torch.tensor([3.0, 4.0])  # norm = 5.0
    u = anchor + torch.matmul(w, B.t())

    proj, v = project_to_patch(u, anchor, B, rho)
    dist = compute_patch_distance(u, anchor, B, rho)

    # Distance to patch should be exactly ||w|| - rho = 5.0 - 1.0 = 4.0
    assert abs(dist.item() - 4.0) < 1e-5
    # Projected point should have distance rho = 1.0 from anchor
    assert abs(torch.norm(proj - anchor, p=2).item() - 1.0) < 1e-5


def test_patch_dimension_zero():
    """Verify that s=0 reduces correctly to point anchor distance."""
    r = 6
    anchor = torch.randn(r)
    basis = torch.empty((r, 0))
    rho = 0.0

    u = torch.randn(r)
    proj, _ = project_to_patch(u, anchor, basis, rho)
    dist = compute_patch_distance(u, anchor, basis, rho)

    expected_dist = torch.norm(u - anchor, p=2).item()
    assert torch.allclose(proj, anchor, atol=1e-5)
    assert abs(dist.item() - expected_dist) < 1e-5
