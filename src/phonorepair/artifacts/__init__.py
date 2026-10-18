"""Artifact construction, calibration, and serialization for PhonoRepair.

Provides offline fit utilities using clean data only:
- schema: PhonoRepairArtifact container and serialization
- builder: Basis fitting, farthest-first clustering, local SVD patches
- calibrator: Conformal finite-sample calibration on speaker blocks
"""

from .schema import PhonoRepairArtifact, load_artifact, save_artifact
from .builder import ArtifactBuilder
from .calibrator import ConformalCalibrator

__all__ = [
    "PhonoRepairArtifact",
    "load_artifact",
    "save_artifact",
    "ArtifactBuilder",
    "ConformalCalibrator",
]
