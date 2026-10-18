"""Unit tests for Whisper binding and E0 identity contract."""

import pytest
import torch
import torch.nn as nn

from phonorepair.binding.whisper import WhisperBinding


class DummyWhisperEncoder(nn.Module):
    """Dummy Whisper encoder module mimicking openai/whisper encoder output."""

    def __init__(self, d_model: int = 512):
        super().__init__()
        self.linear = nn.Linear(80, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (B, 80, T) mel input -> output is (B, T, d_model)
        x_trans = x.transpose(1, 2)
        return self.linear(x_trans)


class DummyWhisperModel(nn.Module):
    def __init__(self, d_model: int = 512):
        super().__init__()
        self.encoder = DummyWhisperEncoder(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def test_whisper_identity_forward_hook():
    """Verify that attaching WhisperBinding with identity callback produces bit-exact output."""
    torch.manual_seed(42)
    model = DummyWhisperModel(d_model=512)
    model.eval()

    mel_input = torch.randn(2, 80, 50)

    # 1. Native baseline output
    with torch.no_grad():
        native_out = model(mel_input)

    # 2. Attach binding with identity callback
    binding = WhisperBinding()

    def identity_callback(features: torch.Tensor) -> torch.Tensor:
        return features  # No modification

    binding.attach_hook(model, identity_callback)

    with torch.no_grad():
        hooked_out = model(mel_input)

    # Must be bit-exact (delta == 0)
    delta = torch.max(torch.abs(hooked_out - native_out)).item()
    assert delta == 0.0, f"Identity violation: delta = {delta}"

    # 3. Detach hook and verify still identical
    binding.detach_hook()
    with torch.no_grad():
        detached_out = model(mel_input)

    delta_detached = torch.max(torch.abs(detached_out - native_out)).item()
    assert delta_detached == 0.0
