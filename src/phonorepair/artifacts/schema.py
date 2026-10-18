"""Artifact container, validation, and serialization for PhonoRepair."""

from dataclasses import dataclass, field
import hashlib
import io
import os
from typing import Any, Dict, List, Optional
import torch


@dataclass
class PhonoRepairArtifact:
    """Immutable artifact container storing geometric dictionaries and calibration gates."""

    # Model metadata
    model_id: str
    feature_dim: int
    repair_dim: int
    num_classes: int

    # Affine coordinate system
    mu: torch.Tensor                   # Mean vector of shape (d,)
    Q: torch.Tensor                    # Orthonormal basis matrix of shape (d, r)

    # Dictionary affine patches
    anchors: torch.Tensor              # Patch centers a_j of shape (M, r)
    bases: List[torch.Tensor]          # List of M orthonormal bases, each (r, s_j)
    radii: torch.Tensor                # Patch radii rho_j of shape (M,)
    patch_labels: torch.Tensor         # Patch phonetic class label (M,) in 0..C-1
    patch_dimensions: torch.Tensor     # Patch subspace dimension s_j (M,)

    # Gate thresholds
    eta: float                         # Working envelope max-residual
    nu: float                          # Noise/threat budget
    q_cal: float                       # Empirical calibration threshold
    q_off: float                       # Final threshold max(q_cal, eta)
    w: float                           # Usable gate width
    correction_budget: float           # Max displacement norm constraint

    # Status & provenance
    is_active: bool = True
    policy_version: str = "phonorepair-v7.1"
    provenance: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

    def validate(self) -> None:
        """Validate dimensional consistency and contract rules."""
        d, r = self.feature_dim, self.repair_dim
        assert self.mu.shape == (d,), f"mu shape mismatch: expected ({d},), got {self.mu.shape}"
        assert self.Q.shape == (d, r), f"Q shape mismatch: expected ({d}, {r}), got {self.Q.shape}"

        M = self.anchors.shape[0]
        assert len(self.bases) == M, f"bases count mismatch: expected {M}, got {len(self.bases)}"
        assert self.radii.shape == (M,), f"radii shape mismatch: expected ({M},), got {self.radii.shape}"
        assert self.patch_labels.shape == (M,), f"patch_labels shape mismatch"

        # Check Q orthogonality: Q^T Q == I_r
        identity = torch.eye(r, dtype=self.Q.dtype, device=self.Q.device)
        qt_q = torch.matmul(self.Q.t(), self.Q)
        max_ortho_err = torch.max(torch.abs(qt_q - identity)).item()
        assert max_ortho_err < 1e-4, f"Q is not orthonormal: max error {max_ortho_err}"

    def to(self, device: Any) -> "PhonoRepairArtifact":
        """Move all artifact tensors to the specified device."""
        self.mu = self.mu.to(device)
        self.Q = self.Q.to(device)
        self.anchors = self.anchors.to(device)
        self.bases = [b.to(device) for b in self.bases]
        self.radii = self.radii.to(device)
        self.patch_labels = self.patch_labels.to(device)
        self.patch_dimensions = self.patch_dimensions.to(device)
        return self

    def compute_checksum(self) -> str:
        """Compute SHA256 of the artifact parameters."""
        buffer = io.BytesIO()
        torch.save(
            {
                "mu": self.mu.cpu(),
                "Q": self.Q.cpu(),
                "anchors": self.anchors.cpu(),
                "radii": self.radii.cpu(),
                "patch_labels": self.patch_labels.cpu(),
                "eta": self.eta,
                "nu": self.nu,
                "q_cal": self.q_cal,
                "q_off": self.q_off,
            },
            buffer,
        )
        return hashlib.sha256(buffer.getvalue()).hexdigest()


def save_artifact(artifact: PhonoRepairArtifact, output_path: str) -> str:
    """Save artifact to a .pt file with calculated checksum."""
    artifact.validate()
    artifact.checksum = artifact.compute_checksum()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {
        "model_id": artifact.model_id,
        "feature_dim": artifact.feature_dim,
        "repair_dim": artifact.repair_dim,
        "num_classes": artifact.num_classes,
        "mu": artifact.mu.cpu(),
        "Q": artifact.Q.cpu(),
        "anchors": artifact.anchors.cpu(),
        "bases": [b.cpu() for b in artifact.bases],
        "radii": artifact.radii.cpu(),
        "patch_labels": artifact.patch_labels.cpu(),
        "patch_dimensions": artifact.patch_dimensions.cpu(),
        "eta": artifact.eta,
        "nu": artifact.nu,
        "q_cal": artifact.q_cal,
        "q_off": artifact.q_off,
        "w": artifact.w,
        "correction_budget": artifact.correction_budget,
        "is_active": artifact.is_active,
        "policy_version": artifact.policy_version,
        "provenance": artifact.provenance,
        "checksum": artifact.checksum,
    }
    torch.save(payload, output_path)
    return artifact.checksum


def load_artifact(artifact_path: str, device: str = "cpu") -> PhonoRepairArtifact:
    """Load and validate an artifact from a .pt file."""
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(f"Artifact file not found: {artifact_path}")

    payload = torch.load(artifact_path, map_location=device, weights_only=False)

    artifact = PhonoRepairArtifact(
        model_id=payload["model_id"],
        feature_dim=payload["feature_dim"],
        repair_dim=payload["repair_dim"],
        num_classes=payload["num_classes"],
        mu=payload["mu"].to(device),
        Q=payload["Q"].to(device),
        anchors=payload["anchors"].to(device),
        bases=[b.to(device) for b in payload["bases"]],
        radii=payload["radii"].to(device),
        patch_labels=payload["patch_labels"].to(device),
        patch_dimensions=payload["patch_dimensions"].to(device),
        eta=float(payload["eta"]),
        nu=float(payload["nu"]),
        q_cal=float(payload["q_cal"]),
        q_off=float(payload["q_off"]),
        w=float(payload["w"]),
        correction_budget=float(payload["correction_budget"]),
        is_active=bool(payload.get("is_active", True)),
        policy_version=payload.get("policy_version", "phonorepair-v7.1"),
        provenance=payload.get("provenance", {}),
        checksum=payload.get("checksum"),
    )
    artifact.validate()
    return artifact
