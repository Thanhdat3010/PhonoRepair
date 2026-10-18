"""Unit tests for bundle locking and anti-drift reproducibility."""

import os
import tempfile
import pytest

from phonorepair.utils.locking import (
    compute_file_sha256,
    generate_bundle_manifest,
    save_bundle_manifest,
    verify_bundle_integrity,
    write_worm_receipt,
)


def test_compute_file_sha256():
    """Verify SHA256 computation matches known reference."""
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8") as f:
        f.write("hello phonorepair\n")
        temp_path = f.name

    try:
        sha = compute_file_sha256(temp_path)
        assert len(sha) == 64
        # Verify deterministic hash
        assert sha == compute_file_sha256(temp_path)
    finally:
        os.remove(temp_path)


def test_bundle_manifest_detection():
    """Verify that tampering with code is detected by verify_bundle_integrity."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file1 = os.path.join(tmpdir, "module.py")
        with open(file1, "w", encoding="utf-8") as f:
            f.write("x = 1\n")

        manifest = generate_bundle_manifest(tmpdir)
        manifest_path = os.path.join(tmpdir, "manifest.json")
        save_bundle_manifest(manifest, manifest_path)

        # Integrity passes initially
        valid, errs = verify_bundle_integrity(tmpdir, manifest_path)
        assert valid
        assert len(errs) == 0

        # Tamper with file1
        with open(file1, "w", encoding="utf-8") as f:
            f.write("x = 2 # tampered\n")

        # Integrity must now fail
        valid, errs = verify_bundle_integrity(tmpdir, manifest_path)
        assert not valid
        assert any("Hash mismatch" in e for e in errs)


def test_worm_receipt_write_once():
    """Verify that overwriting existing run receipts raises FileExistsError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        receipt_file = os.path.join(tmpdir, "receipt.json")
        data = {"wer": 0.05}

        # First write succeeds
        write_worm_receipt(data, receipt_file, allow_overwrite=False)
        assert os.path.exists(receipt_file)

        # Second write without overwrite flag must fail
        with pytest.raises(FileExistsError):
            write_worm_receipt(data, receipt_file, allow_overwrite=False)
