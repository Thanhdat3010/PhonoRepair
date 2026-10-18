"""NeMo Conformer CTC binding implementation for PhonoRepair (C1)."""

from typing import Any, Callable, Optional, Tuple
import torch
import torch.nn as nn

from .base import AbstractBinding, MemoryPacket


class ConformerBinding(AbstractBinding):
    """Binding for NeMo Conformer CTC Large."""

    def __init__(self, contract_config: Optional[dict] = None):
        super().__init__(contract_config or {})
        self.hook_handle = None
        self.active_callback: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def extract(self, native_output: Any) -> MemoryPacket:
        """Extract memory packet from NeMo encoder output.

        NeMo encoder returns tuple (encoded, encoded_len).
        """
        if isinstance(native_output, tuple):
            features, lengths = native_output[0], native_output[1]
        elif isinstance(native_output, torch.Tensor):
            features, lengths = native_output, None
        else:
            raise TypeError(f"Unexpected Conformer encoder output type: {type(native_output)}")

        return MemoryPacket(
            features=features,
            valid_lengths=lengths,
            timebase_subsampling=4,
            stream_id="conformer_c1",
            memory_version=1,
        )

    def replace(self, native_output: Any, repaired_features: torch.Tensor) -> Any:
        """Replace features in Conformer encoder output."""
        if isinstance(native_output, tuple):
            return (repaired_features, native_output[1])
        return repaired_features

    def attach_hook(
        self,
        model: nn.Module,
        repair_callback: Callable[[torch.Tensor], torch.Tensor],
    ) -> Any:
        """Attach forward hook to Conformer encoder."""
        self.detach_hook()
        self.active_callback = repair_callback

        encoder = getattr(model, "encoder", None)
        if encoder is None:
            raise AttributeError("Could not locate encoder module on the provided Conformer model.")

        def forward_hook(module: nn.Module, inputs: Any, output: Any) -> Any:
            if self.active_callback is None:
                return output
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
