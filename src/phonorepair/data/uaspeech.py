"""UASpeech dataset loader and manifest builder.

Implements Proposal Section 10.2 & 12 (E4):
- Primary word-level dysarthric recognition.
- Default channel: M5 (Proposal line 828).
- Subgroup intelligibility stratification (Very Low, Low, Mid, High, Control).
- Intended prompt matching and normalizer v1 compatibility.
"""

import os
import re
from typing import Dict, List, Optional
import soundfile as sf

from .schema import DatasetRecord
from ..evaluation.normalizer import normalize_text_v1


# UASpeech official intelligibility levels (Kim et al., 2008)
UASPEECH_INTELLIGIBILITY = {
    # Dysarthric Speakers
    "M04": "Very Low", "M12": "Very Low", "F03": "Very Low", "M01": "Very Low",
    "M07": "Low", "M16": "Low", "M05": "Low",
    "M11": "Mid", "F02": "Mid", "M13": "Mid",
    "M08": "High", "M09": "High", "M10": "High", "M14": "High", "M15": "High",
    "F04": "High", "F05": "High",
    # Healthy Controls
    "CM01": "Control", "CM02": "Control", "CM03": "Control", "CM04": "Control",
    "CF01": "Control", "CF02": "Control", "CF03": "Control", "CF04": "Control", "CF05": "Control",
}

# Standard UASpeech common word inventory (CW1 - CW19)
COMMON_WORDS = {
    "CW1": "COMMAND", "CW2": "ENTER", "CW3": "ESCAPE", "CW4": "FILE",
    "CW5": "HELP", "CW6": "LINE", "CW7": "MENU", "CW8": "MOVE",
    "CW9": "NEXT", "CW10": "NO", "CW11": "PAGE", "CW12": "PRINT",
    "CW13": "QUIT", "CW14": "REPEAT", "CW15": "RETURN", "CW16": "SAVE",
    "CW17": "STOP", "CW18": "VIEW", "CW19": "YES",
}

# Digits (D1 - D10)
DIGIT_WORDS = {
    "D1": "ONE", "D2": "TWO", "D3": "THREE", "D4": "FOUR", "D5": "FIVE",
    "D6": "SIX", "D7": "SEVEN", "D8": "EIGHT", "D9": "NINE", "D10": "ZERO",
    "D11": "OH",
}

# Letters (L1 - L26)
LETTER_WORDS = {f"L{i+1}": chr(ord('A') + i) for i in range(26)}

# Computer commands (C1 - C19)
COMMAND_WORDS = {
    "C1": "BACKSPACE", "C2": "DELETE", "C3": "DOWN", "C4": "END",
    "C5": "HOME", "C6": "INSERT", "C7": "LEFT", "C8": "PAGE DOWN",
    "C9": "PAGE UP", "C10": "PAUSE", "C11": "RIGHT", "C12": "SPACE",
    "C13": "TAB", "C14": "UP", "C15": "ALT", "C16": "CONTROL",
    "C17": "SHIFT", "C18": "ESCAPE", "C19": "ENTER",
}


class UASpeechLoader:
    """Parses and standardizes the UASpeech word-level speech dataset."""

    def __init__(
        self,
        root_dir: str,
        preferred_channel: str = "M5",
        custom_word_table: Optional[Dict[str, str]] = None,
    ):
        self.root_dir = root_dir
        self.preferred_channel = preferred_channel
        self.word_table = custom_word_table or {}

        # Merge standard tables
        self.word_table.update(COMMON_WORDS)
        self.word_table.update(DIGIT_WORDS)
        self.word_table.update(LETTER_WORDS)
        self.word_table.update(COMMAND_WORDS)

    def load_word_table_from_file(self, filepath: str) -> None:
        """Load tab/comma/space separated word table (WordID -> WordText)."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.word_table[parts[0].upper()] = parts[1].upper()

    def parse_filename(self, filename: str) -> Optional[Dict[str, str]]:
        """Parse standard UASpeech filename: <speaker>_<block>_<word_id>_<channel>.wav"""
        base = os.path.splitext(filename)[0]
        parts = base.split("_")
        if len(parts) < 4:
            return None

        speaker_id = parts[0].upper()
        block_id = parts[1].upper()
        word_id = parts[2].upper()
        channel = parts[3].upper()

        return {
            "speaker_id": speaker_id,
            "block_id": block_id,
            "word_id": word_id,
            "channel": channel,
            "utterance_id": base,
        }

    def scan_dataset(
        self,
        speakers: Optional[List[str]] = None,
        include_controls: bool = False,
    ) -> List[DatasetRecord]:
        """Scan directory and return list of valid DatasetRecord instances.

        Applies Proposal Section 12 E4 exclusion criteria:
        - Exclude missing audio or zero-length audio files.
        - Exclude missing word reference mappings.
        - Match preferred acoustic channel (M5).
        """
        records: List[DatasetRecord] = []
        if not os.path.exists(self.root_dir):
            return records

        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if not file.lower().endswith(".wav"):
                    continue

                parsed = self.parse_filename(file)
                if not parsed:
                    continue

                spk = parsed["speaker_id"]
                ch = parsed["channel"]
                wid = parsed["word_id"]

                # Channel filter
                if self.preferred_channel and ch != self.preferred_channel.upper():
                    continue

                # Speaker filter
                is_control = spk.startswith("C")
                if not include_controls and is_control:
                    continue

                if speakers and spk not in speakers:
                    continue

                # Text prompt resolution
                raw_text = self.word_table.get(wid)
                if not raw_text:
                    continue

                norm_text = normalize_text_v1(raw_text)
                if not norm_text:
                    continue

                audio_path = os.path.normpath(os.path.join(root, file)).replace("\\", "/")

                # Audio validation
                try:
                    info = sf.info(audio_path)
                    duration = info.duration
                    if duration < 0.1 or duration > 30.0:
                        continue
                except Exception:
                    continue

                intelligibility = UASPEECH_INTELLIGIBILITY.get(
                    spk, "Control" if is_control else "Unknown"
                )

                rec = DatasetRecord(
                    utterance_id=parsed["utterance_id"],
                    speaker_id=spk,
                    audio_path=audio_path,
                    text=norm_text,
                    duration_sec=round(duration, 2),
                    corpus="uaspeech",
                    task_type="word",
                    channel=ch,
                    intelligibility=intelligibility,
                    metadata={
                        "block_id": parsed["block_id"],
                        "word_id": wid,
                    },
                )
                records.append(rec)

        records.sort(key=lambda r: (r.speaker_id, r.utterance_id))
        return records
