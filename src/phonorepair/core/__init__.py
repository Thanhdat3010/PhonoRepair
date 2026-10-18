"""Core mathematical engine for PhonoRepair.

Contains pure PyTorch/NumPy algorithms for:
- Geometry: coordinate reading, lifting, projection, patch distances
- Gate: class score aggregation, envelope checking, singleton routing, width w
- Kernel: unified REPAIR_FRAME pipeline matching Proposal Section 8.2
"""

from .geometry import (
    read_coordinates,
    lift_coordinates,
    project_to_patch,
    compute_patch_distance,
    compute_all_patch_distances,
)
from .gate import (
    compute_gate_width,
    aggregate_class_distances,
    evaluate_gate,
)
from .kernel import repair_frame, repair_sequence

__all__ = [
    "read_coordinates",
    "lift_coordinates",
    "project_to_patch",
    "compute_patch_distance",
    "compute_all_patch_distances",
    "compute_gate_width",
    "aggregate_class_distances",
    "evaluate_gate",
    "repair_frame",
    "repair_sequence",
]
