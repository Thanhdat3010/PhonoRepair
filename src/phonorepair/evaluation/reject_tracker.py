"""Diagnostic tracking of gate decisions and harm vs correction accounting."""

from collections import Counter
from typing import Dict, List, Optional


class RejectTracker:
    """Tracks frame-level gate acceptance and utterance-level correction/harm."""

    def __init__(self):
        self.frame_reasons = Counter()
        self.total_frames = 0
        self.accepted_frames = 0

        # Utterance level metrics
        self.total_utterances = 0
        self.corrected_utterances = 0   # Base had errors, Repair was 100% exact
        self.harmed_utterances = 0      # Base was 100% exact, Repair introduced errors
        self.both_correct = 0
        self.both_error = 0
        self.unchanged_hypothesis = 0

    def record_frames(self, reasons: List[str]) -> None:
        """Record frame-level decisions."""
        for r in reasons:
            self.total_frames += 1
            self.frame_reasons[r] += 1
            if r == "accepted":
                self.accepted_frames += 1

    def record_utterance(
        self,
        base_wer: float,
        repair_wer: float,
        hypothesis_base: str,
        hypothesis_repair: str,
    ) -> None:
        """Record utterance-level outcome comparison."""
        self.total_utterances += 1

        if hypothesis_base == hypothesis_repair:
            self.unchanged_hypothesis += 1

        base_exact = (base_wer == 0.0)
        repair_exact = (repair_wer == 0.0)

        if not base_exact and repair_exact:
            self.corrected_utterances += 1
        elif base_exact and not repair_exact:
            self.harmed_utterances += 1
        elif base_exact and repair_exact:
            self.both_correct += 1
        else:
            self.both_error += 1

    @property
    def acceptance_rate(self) -> float:
        return (self.accepted_frames / self.total_frames) if self.total_frames > 0 else 0.0

    def get_summary(self) -> Dict[str, float]:
        """Return a structured summary dict for Table D."""
        return {
            "total_frames": float(self.total_frames),
            "accepted_frames": float(self.accepted_frames),
            "acceptance_rate_pct": float(self.acceptance_rate * 100.0),
            "total_utterances": float(self.total_utterances),
            "corrected_utterances": float(self.corrected_utterances),
            "harmed_utterances": float(self.harmed_utterances),
            "both_correct": float(self.both_correct),
            "both_error": float(self.both_error),
            "unchanged_hypothesis_pct": float((self.unchanged_hypothesis / max(1, self.total_utterances)) * 100.0),
            "reasons_breakdown": dict(self.frame_reasons),
        }
