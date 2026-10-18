"""Abstract base binding interface for ASR backbones."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Optional
import torch


@dataclass
class MemoryPacket:
    """Represents an acoustic memory tensor extracted from a frozen encoder."""
    features: torch.Tensor             # Shape (B, T, d) or (T, d)
    valid_lengths: Optional[torch.Tensor] = None # Lengths or mask
    timebase_subsampling: int = 2      # Ratio of frontend frames to encoder frames
    stream_id: str = "default"
    memory_version: int = 1


class AbstractBinding(ABC):
    """Abstract interface for intercepting encoder outputs and injecting repairs."""

    def __init__(self, contract_config: dict):
        self.contract = contract_config

    @abstractmethod
    def extract(self, native_output: Any) -> MemoryPacket:
        """Extract the acoustic features into a standardized MemoryPacket."""
        pass

    @abstractmethod
    def replace(self, native_output: Any, repaired_features: torch.Tensor) -> Any:
        """Return native output with features replaced by repaired_features."""
        pass

    @abstractmethod
    def attach_hook(self, model: torch.nn.Module, repair_callback: Callable[[torch.Tensor], torch.Tensor]) -> Any:
        """Attach PyTorch forward hook into encoder to run repair_callback."""
        pass

    @abstractmethod
    def detach_hook(self) -> None:
        """Remove previously attached forward hook."""
        pass
