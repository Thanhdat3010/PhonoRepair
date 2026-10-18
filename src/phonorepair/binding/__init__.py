"""Backbone integration bindings for PhonoRepair.

Provides declarative tensor extraction and replacement hooks without embedding
repair logic or brand-specific thresholds into model definitions:
- base: MemoryPacket and AbstractBinding interface
- whisper: OpenAI Whisper base.en hook
- conformer: NeMo Conformer CTC hook
- zipformer: Icefall Zipformer Transducer hook
"""

from .base import MemoryPacket, AbstractBinding
from .whisper import WhisperBinding
from .conformer import ConformerBinding
from .zipformer import ZipformerBinding

__all__ = [
    "MemoryPacket",
    "AbstractBinding",
    "WhisperBinding",
    "ConformerBinding",
    "ZipformerBinding",
]
