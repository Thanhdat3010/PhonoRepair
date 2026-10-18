"""Unified data schema and records for PhonoRepair datasets."""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


@dataclass
class DatasetRecord:
    """Standardized record for speech recognition evaluation."""

    utterance_id: str
    speaker_id: str
    audio_path: str
    text: str
    duration_sec: float
    corpus: str                        # 'uaspeech', 'torgo', 'librispeech'
    task_type: str                     # 'word', 'sentence'
    channel: str = "default"           # 'M5', 'headMic', etc.
    intelligibility: Optional[str] = None # 'Very Low', 'Low', 'Mid', 'High', 'Control'
    metadata: Dict[str, Any] = field(default_factory=dict)
    cache_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a serializable dictionary for JSONL manifests."""
        d = asdict(self)
        if self.cache_path is None:
            d.pop("cache_path", None)
        return d
