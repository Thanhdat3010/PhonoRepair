"""Utility functions for hashing, locking, and audio I/O."""

from .hashing import compute_speaker_hash, assign_split
from .locking import compute_file_sha256, generate_bundle_manifest, verify_bundle_integrity
from .audio import load_and_resample_audio

__all__ = [
    "compute_speaker_hash",
    "assign_split",
    "compute_file_sha256",
    "generate_bundle_manifest",
    "verify_bundle_integrity",
    "load_and_resample_audio",
]
