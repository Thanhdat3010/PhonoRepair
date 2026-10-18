"""Generates synthetic geometry arrays for E0 unit testing.

Produces data/processed/toy/e0_geometry_cases.npz containing:
- Orthogonal projection edge cases
- Points inside, on, and outside patch radii
- Boundary tie-breaking test points
"""

import os
import numpy as np


def generate_toy_cases(output_path: str = "data/processed/toy/e0_geometry_cases.npz") -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    rng = np.random.default_rng(20261018)

    r = 16
    s = 2

    # Patch 1: at origin, radius = 2.0
    a1 = np.zeros(r, dtype=np.float32)
    B1, _ = np.linalg.qr(rng.standard_normal((r, s)))
    B1 = B1.astype(np.float32)
    rho1 = 2.0

    # Point 1: exactly inside patch (dist = 0)
    w_inside = np.array([0.5, -0.5], dtype=np.float32)
    p_inside = a1 + (B1 @ w_inside)

    # Point 2: along subspace but outside radius (dist = ||w|| - rho)
    w_outside = np.array([3.0, 4.0], dtype=np.float32)  # norm = 5.0
    p_outside = a1 + (B1 @ w_outside)

    # Point 3: strictly orthogonal to subspace
    ortho_vec = rng.standard_normal(r).astype(np.float32)
    ortho_vec = ortho_vec - B1 @ (B1.T @ ortho_vec)
    ortho_vec = ortho_vec / np.linalg.norm(ortho_vec) * 3.5
    p_ortho = a1 + ortho_vec

    np.savez_compressed(
        output_path,
        anchor=a1,
        basis=B1,
        rho=rho1,
        p_inside=p_inside,
        p_outside=p_outside,
        p_ortho=p_ortho,
    )
    print(f"[+] Generated toy geometry test cases at: {output_path}")


if __name__ == "__main__":
    generate_toy_cases()
