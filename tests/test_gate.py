"""Unit tests for gate logic, width w, and singleton routing."""

import math
import pytest
import torch

from phonorepair.core.gate import compute_gate_width, evaluate_gate


def test_gate_width_formula():
    """Verify algebraic gate width: w = max(0, eta + nu - max(q_cal, eta))."""
    # Case 1: Standard open gate
    # eta = 2.0, nu = 1.0, q_cal = 1.5 -> q_off = 2.0 -> w = 2.0 + 1.0 - 2.0 = 1.0
    assert compute_gate_width(eta=2.0, nu=1.0, q_cal=1.5) == 1.0

    # Case 2: Dominant q_cal
    # eta = 2.0, nu = 1.0, q_cal = 2.5 -> q_off = 2.5 -> w = 3.0 - 2.5 = 0.5
    assert compute_gate_width(eta=2.0, nu=1.0, q_cal=2.5) == 0.5

    # Case 3: Empty gate (q_cal exceeds upper bound)
    # eta = 2.0, nu = 1.0, q_cal = 3.5 -> q_off = 3.5 -> w = max(0, 3.0 - 3.5) = 0.0
    assert compute_gate_width(eta=2.0, nu=1.0, q_cal=3.5) == 0.0


def test_gate_identity_fallback_below_q_off():
    """Verify that any input within q_off is rejected (identity fallback)."""
    # 2 classes, distances: class 0 is 1.2, class 1 is 3.0
    class_dists = torch.tensor([[1.2, 3.0]])
    q_off = 1.5  # d_1 (1.2) <= q_off (1.5)
    eta = 1.5
    nu = 1.0

    is_accepted, _, reasons = evaluate_gate(class_dists, q_off, eta, nu)
    assert not is_accepted[0].item()
    assert reasons[0] == "below_q_off_identity"


def test_gate_singleton_accepted():
    """Verify that an input with exactly one feasible class outside q_off is accepted."""
    # class 0: 2.2, class 1: 4.5
    class_dists = torch.tensor([[2.2, 4.5]])
    q_off = 2.0
    eta = 2.0
    nu = 1.0  # upper bound = 3.0. Class 0 (2.2) <= 3.0, Class 1 (4.5) > 3.0

    is_accepted, target_class, reasons = evaluate_gate(class_dists, q_off, eta, nu)
    assert is_accepted[0].item()
    assert target_class[0].item() == 0
    assert reasons[0] == "accepted"


def test_gate_ambiguous_multiple_rejected():
    """Verify that an input with two competing feasible classes is rejected."""
    # class 0: 2.2, class 1: 2.4 -> both <= upper bound (3.0)
    class_dists = torch.tensor([[2.2, 2.4]])
    q_off = 2.0
    eta = 2.0
    nu = 1.0

    is_accepted, _, reasons = evaluate_gate(class_dists, q_off, eta, nu)
    assert not is_accepted[0].item()
    assert reasons[0] == "feasible_ambiguous_multiple"


def test_conformal_calibrator_pilot_fallback():
    """Verify ConformalCalibrator with pilot speaker blocks."""
    from phonorepair.artifacts.calibrator import ConformalCalibrator
    from phonorepair.artifacts.schema import PhonoRepairArtifact

    artifact = PhonoRepairArtifact(
        model_id="test",
        feature_dim=16,
        repair_dim=8,
        num_classes=2,
        mu=torch.zeros(16),
        Q=torch.eye(16, 8),
        anchors=torch.zeros(2, 8),
        bases=[torch.zeros(8, 0), torch.zeros(8, 0)],
        radii=torch.zeros(2),
        patch_labels=torch.tensor([0, 1]),
        patch_dimensions=torch.tensor([0, 0]),
        eta=2.0,
        nu=1.0,
        q_cal=2.0,
        q_off=2.0,
        w=1.0,
        correction_budget=3.0,
        is_active=True,
    )

    # 3 speaker blocks, each with 1 episode
    calibrator = ConformalCalibrator(beta=0.01)
    spk_blocks = [
        [torch.zeros(5, 16)],
        [torch.zeros(5, 16) + 0.5],
        [torch.zeros(5, 16) + 1.0],
    ]
    cal_art, stats = calibrator.calibrate(spk_blocks, artifact)
    assert stats["is_gate_open"] == 1.0
    assert stats["w"] > 0.0
    assert not math.isinf(stats["q_cal"])

