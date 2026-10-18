"""Locking and Anti-Drift utilities for PhonoRepair.

Ensures complete scientific reproducibility and immutability:
1. Data manifest SHA256 verification
2. Python bundle code manifest verification
3. WORM (Write-Once-Read-Many) receipt generation
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def compute_file_sha256(filepath: str) -> str:
    """Compute the SHA256 checksum of a file in binary chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def generate_bundle_manifest(
    source_dir: str,
    extensions: Optional[List[str]] = None,
    exclude_filenames: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Scan source_dir and compute SHA256 for all relevant files (default .py, .yaml)."""
    if extensions is None:
        extensions = [".py", ".yaml"]
    if exclude_filenames is None:
        exclude_filenames = ["bundle_manifest.json", "manifest.json"]

    manifest = {}
    base_path = Path(source_dir).resolve()

    for root, _, files in sorted(os.walk(base_path)):
        for file in sorted(files):
            if file in exclude_filenames:
                continue
            file_path = Path(root) / file
            if file_path.suffix in extensions:
                rel_path = file_path.relative_to(base_path).as_posix()
                manifest[rel_path] = compute_file_sha256(str(file_path))

    return manifest


def save_bundle_manifest(manifest: Dict[str, str], output_file: str) -> None:
    """Save the bundle manifest as a formatted JSON file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def verify_bundle_integrity(source_dir: str, bundle_manifest_path: str) -> Tuple[bool, List[str]]:
    """Verify that current files in source_dir match the bundle manifest exactly."""
    if not os.path.exists(bundle_manifest_path):
        return False, [f"Bundle manifest file not found: {bundle_manifest_path}"]

    with open(bundle_manifest_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    manifest_filename = os.path.basename(bundle_manifest_path)
    current = generate_bundle_manifest(source_dir, exclude_filenames=["bundle_manifest.json", "manifest.json", manifest_filename])
    errors = []

    for file_rel, expected_hash in expected.items():
        if file_rel not in current:
            errors.append(f"Missing file: {file_rel}")
        elif current[file_rel] != expected_hash:
            errors.append(f"Hash mismatch for {file_rel}: expected {expected_hash}, got {current[file_rel]}")

    for file_rel in current:
        if file_rel not in expected:
            errors.append(f"Untracked new file in bundle: {file_rel}")

    is_valid = len(errors) == 0
    return is_valid, errors


def write_worm_receipt(receipt_data: dict, receipt_path: str, allow_overwrite: bool = False) -> None:
    """Write an execution receipt with Write-Once-Read-Many protection."""
    if os.path.exists(receipt_path) and not allow_overwrite:
        raise FileExistsError(
            f"WORM Violation: Receipt already exists at {receipt_path}. "
            "Overwriting verified runs is strictly forbidden to prevent baseline drift."
        )

    os.makedirs(os.path.dirname(os.path.abspath(receipt_path)), exist_ok=True)
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt_data, f, indent=2, sort_keys=True)
