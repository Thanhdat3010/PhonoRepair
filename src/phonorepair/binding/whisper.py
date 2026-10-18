"""OpenAI Whisper binding implementation for PhonoRepair (W1)."""

from typing import Any, Callable, Optional
import torch
import torch.nn as nn

from .base import AbstractBinding, MemoryPacket


class WhisperBinding(AbstractBinding):
    """Binding for OpenAI Whisper base.en (Seq2Seq Transformer)."""

    def __init__(self, contract_config: Optional[dict] = None):
        super().__init__(contract_config or {})
        self.hook_handle = None
        self.active_callback: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def extract(self, native_output: Any) -> MemoryPacket:
        """Extract memory packet from Whisper encoder output tensor (B, T, D)."""
        if isinstance(native_output, torch.Tensor):
            features = native_output
        elif hasattr(native_output, "last_hidden_state"):
            features = native_output.last_hidden_state
        else:
            raise TypeError(f"Unexpected Whisper encoder output type: {type(native_output)}")

        return MemoryPacket(
            features=features,
            valid_lengths=None,
            timebase_subsampling=2,
            stream_id="whisper_w1",
            memory_version=1,
        )

    def replace(self, native_output: Any, repaired_features: torch.Tensor) -> Any:
        """Replace features in Whisper encoder output."""
        if isinstance(native_output, torch.Tensor):
            return repaired_features
        elif hasattr(native_output, "last_hidden_state"):
            native_output.last_hidden_state = repaired_features
            return native_output
        return repaired_features

    def attach_hook(
        self,
        model: nn.Module,
        repair_callback: Callable[[torch.Tensor], torch.Tensor],
    ) -> Any:
        """Attach forward hook to whisper encoder."""
        self.detach_hook()
        self.active_callback = repair_callback

        encoder = getattr(model, "encoder", None)
        if encoder is None and hasattr(model, "model"):
            encoder = getattr(model.model, "encoder", None)

        if encoder is None:
            raise AttributeError("Could not locate encoder module on the provided Whisper model.")

        def forward_hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            if self.active_callback is None:
                return output

            # output is (B, T, D)
            packet = self.extract(output)
            repaired_features = self.active_callback(packet.features)
            return self.replace(output, repaired_features)

        self.hook_handle = encoder.register_forward_hook(forward_hook)
        return self.hook_handle

    def detach_hook(self) -> None:
        """Remove the forward hook cleanly."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None
        self.active_callback = None
