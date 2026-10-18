"""Offline artifact builder using clean source speech data only."""

from typing import Dict, List, Optional, Tuple
import torch

from ..core.geometry import project_to_patch, compute_patch_distance
from .schema import PhonoRepairArtifact


class ArtifactBuilder:
    """Builds clean geometric dictionary artifacts according to Policy v7.1."""

    def __init__(
        self,
        repair_dim: int = 64,
        total_patches: int = 64,
        patch_dim: int = 2,
        neighborhood_size: int = 128,
        device: str = "cpu",
    ):
        self.repair_dim = repair_dim
        self.total_patches = total_patches
        self.patch_dim = patch_dim
        self.neighborhood_size = neighborhood_size
        self.device = device

    def fit_basis(self, features: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fit global PCA basis Q and mean mu on clean features.

        Args:
            features: Tensor of shape (N, d)
            rank: Dimensionality r <= d

        Returns:
            mu: Mean vector (d,)
            Q: Orthonormal basis (d, r)
        """
        features = features.to(self.device).float()
        N, d = features.shape
        mu = torch.mean(features, dim=0)
        centered = features - mu

        # Compute SVD: centered = U S V^T
        # Covariance eigenvectors correspond to columns of V (rows of Vh)
        actual_rank = min(rank, d, N)
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        Q = vh[:actual_rank, :].t()  # (d, actual_rank)

        # Orthonormalize with QR to ensure Q^T Q == I
        Q, _ = torch.linalg.qr(Q)
        return mu, Q

    @staticmethod
    def farthest_first_anchors(
        points: torch.Tensor,
        num_anchors: int,
    ) -> List[int]:
        """Greedy farthest-first traversal (Gonzalez 1985) on Euclidean points.

        Args:
            points: Tensor of shape (N, r)
            num_anchors: K anchors to select

        Returns:
            indices: List of K indices selected
        """
        N = points.shape[0]
        if N <= num_anchors:
            return list(range(N))

        # Anchor 1: First sample by convention
        selected = [0]
        # Keep track of minimum distance from each point to already selected anchors
        dists = torch.norm(points - points[0], dim=-1)

        for _ in range(1, num_anchors):
            # Pick point with maximum distance to the nearest existing anchor
            farthest_idx = int(torch.argmax(dists).item())
            selected.append(farthest_idx)

            # Update minimum distances
            new_dists = torch.norm(points - points[farthest_idx], dim=-1)
            dists = torch.minimum(dists, new_dists)

        return selected

    def fit_single_patch(
        self,
        u_samples: torch.Tensor,
        anchor: torch.Tensor,
        s_dim: int,
    ) -> Tuple[torch.Tensor, float]:
        """Fit a single local affine patch S_j around anchor a_j using local SVD."""
        diff = u_samples - anchor
        dist_to_anchor = torch.norm(diff, dim=-1)

        # Take closest neighborhood_size points to anchor
        k_neighbors = min(self.neighborhood_size, u_samples.shape[0])
        _, neighbor_indices = torch.topk(dist_to_anchor, k=k_neighbors, largest=False)
        neighbors = diff[neighbor_indices]  # (k, r)

        if s_dim == 0 or neighbors.shape[0] <= 1:
            basis = torch.empty((u_samples.shape[-1], 0), dtype=u_samples.dtype, device=u_samples.device)
            radius = 0.0
            return basis, radius

        # Local SVD on neighbors
        actual_s = min(s_dim, neighbors.shape[-1], neighbors.shape[0])
        _, _, vh = torch.linalg.svd(neighbors, full_matrices=False)
        B_j = vh[:actual_s, :].t()  # (r, actual_s)
        B_j, _ = torch.linalg.qr(B_j)

        # Radius = max norm of projected coordinates of selected neighbors
        v = torch.matmul(neighbors, B_j)
        v_norms = torch.norm(v, dim=-1)
        radius = float(torch.max(v_norms).item()) if v_norms.numel() > 0 else 0.0

        return B_j, radius

    def build_from_class_features(
        self,
        model_id: str,
        features_by_class: Dict[int, torch.Tensor],
        policy_config: Optional[dict] = None,
    ) -> PhonoRepairArtifact:
        """Construct a complete PhonoRepairArtifact from clean class-labeled features."""
        classes = sorted(list(features_by_class.keys()))
        num_classes = len(classes)
        assert num_classes > 0, "No classes provided"

        # 1. Pool all features to fit global basis Q and mean mu
        all_features = torch.cat([features_by_class[c] for c in classes], dim=0)
        feature_dim = all_features.shape[-1]
        repair_dim = min(self.repair_dim, feature_dim)

        mu, Q = self.fit_basis(all_features, rank=repair_dim)

        # 2. Project class features into repair coordinate space u
        u_by_class = {}
        for c in classes:
            centered = features_by_class[c].to(self.device).float() - mu
            u_by_class[c] = torch.matmul(centered, Q)

        # 3. Allocate patches per class
        # Equal distribution across classes
        patches_per_class = max(1, self.total_patches // num_classes)

        all_anchors = []
        all_bases = []
        all_radii = []
        all_labels = []
        all_dims = []

        for c in classes:
            u_c = u_by_class[c]
            num_c_patches = min(patches_per_class, u_c.shape[0])
            anchor_indices = self.farthest_first_anchors(u_c, num_c_patches)

            for idx in anchor_indices:
                a_j = u_c[idx]
                B_j, rho_j = self.fit_single_patch(u_c, a_j, self.patch_dim)

                all_anchors.append(a_j)
                all_bases.append(B_j)
                all_radii.append(rho_j)
                all_labels.append(c)
                all_dims.append(B_j.shape[1])

        anchors_tensor = torch.stack(all_anchors, dim=0)
        radii_tensor = torch.tensor(all_radii, dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.long, device=self.device)
        dims_tensor = torch.tensor(all_dims, dtype=torch.long, device=self.device)

        # 4. Compute working envelope eta_work (robust 95% quantile on clean samples)
        # 4. Working envelope eta: Quantile of dictionary anomaly score A(u) = min_p d(u, S_p)
        # Matches the exact inference score definition in Proposal Section 8.2
        residuals = []
        num_all_patches = len(all_labels)
        for c in classes:
            u_c = u_by_class[c]
            for u_point in u_c[:min(100, u_c.shape[0])]:
                min_d = min(
                    compute_patch_distance(u_point, all_anchors[p], all_bases[p], all_radii[p]).item()
                    for p in range(num_all_patches)
                )
                residuals.append(min_d)

        if residuals:
            res_tensor = torch.tensor(residuals, dtype=torch.float32)
            eta_work = float(torch.quantile(res_tensor, 0.95).item())
        else:
            eta_work = 1.0

        # 5. Estimate threat budget nu_local proportional to acoustic dispersion scale
        gaps = []
        for c in classes:
            u_c = u_by_class[c]
            other_patch_indices = [i for i, lbl in enumerate(all_labels) if lbl != c]
            if not other_patch_indices:
                continue

            for u_point in u_c[:min(100, u_c.shape[0])]:
                d_c = min(
                    compute_patch_distance(u_point, all_anchors[p], all_bases[p], all_radii[p]).item()
                    for p in [i for i, lbl in enumerate(all_labels) if lbl == c]
                )
                d_other = min(
                    compute_patch_distance(u_point, all_anchors[p], all_bases[p], all_radii[p]).item()
                    for p in other_patch_indices
                )
                gap = d_other - d_c
                if gap > 0:
                    gaps.append(gap)

        # Scale-proportional threat budget (acoustically calibrated)
        scale_budget = 0.25 * eta_work
        if gaps:
            gaps_tensor = torch.tensor(gaps, dtype=torch.float32)
            q_gap = float(torch.median(gaps_tensor).item())
            nu_local = max(0.25 * q_gap, scale_budget)
        else:
            nu_local = scale_budget

        # Initial provisional thresholds before calibration
        q_cal_provisional = eta_work
        q_off_provisional = max(q_cal_provisional, eta_work)
        w_provisional = max(0.0, eta_work + nu_local - q_off_provisional)

        return PhonoRepairArtifact(
            model_id=model_id,
            feature_dim=feature_dim,
            repair_dim=repair_dim,
            num_classes=num_classes,
            mu=mu,
            Q=Q,
            anchors=anchors_tensor,
            bases=all_bases,
            radii=radii_tensor,
            patch_labels=labels_tensor,
            patch_dimensions=dims_tensor,
            eta=eta_work,
            nu=nu_local,
            q_cal=q_cal_provisional,
            q_off=q_off_provisional,
            w=w_provisional,
            correction_budget=eta_work + nu_local,
            is_active=True,
            policy_version="phonorepair-v7.1",
        )
