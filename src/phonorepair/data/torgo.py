"""TORGO dataset loader and manifest builder.

Implements Proposal Section 10.2 & 12 (E5):
- Primary sentence-level dysarthric recognition (>= 2 words).
- Secondary single-word subset.
- Primary acoustic channel: wav_headMic (Proposal line 848).
- Strict exclusion filtering: exclude 'xxx', bracketed noise tags, non-word tasks, missing transcripts.
"""

import os
import re
from typing import Dict, List, Optional, Tuple
import soundfile as sf

from .schema import DatasetRecord
from ..evaluation.normalizer import normalize_text_v1


# TORGO official speaker demographics & dysarthria etiologies
TORGO_SPEAKERS = {
    # Dysarthric
    "F01": {"gender": "F", "type": "dysarthric", "etiology": "CP"},
    "F03": {"gender": "F", "type": "dysarthric", "etiology": "CP"},
    "F04": {"gender": "F", "type": "dysarthric", "etiology": "CP"},
    "M01": {"gender": "M", "type": "dysarthric", "etiology": "ALS"},
    "M02": {"gender": "M", "type": "dysarthric", "etiology": "ALS"},
    "M03": {"gender": "M", "type": "dysarthric", "etiology": "CP"},
    "M04": {"gender": "M", "type": "dysarthric", "etiology": "ALS"},
    "M05": {"gender": "M", "type": "dysarthric", "etiology": "ALS"},
    # Controls
    "FC01": {"gender": "F", "type": "control", "etiology": "None"},
    "FC02": {"gender": "F", "type": "control", "etiology": "None"},
    "MC01": {"gender": "M", "type": "control", "etiology": "None"},
    "MC02": {"gender": "M", "type": "control", "etiology": "None"},
    "MC03": {"gender": "M", "type": "control", "etiology": "None"},
    "MC04": {"gender": "M", "type": "control", "etiology": "None"},
}


class TORGOLoader:
    """Parses and standardizes the TORGO sentence- and word-level speech dataset."""

    def __init__(
        self,
        root_dir: str,
        preferred_channel: str = "wav_headMic",
    ):
        self.root_dir = root_dir
        self.preferred_channel = preferred_channel

    def clean_prompt(self, raw_text: str) -> Optional[str]:
        """Apply Proposal Section 12 E5 exclusion rules to prompt text.

        - Discard if contains 'xxx' (unintelligible prompt).
        - Remove [bracketed] descriptions.
        - Normalize and discard if empty.
        """
        if not raw_text or not raw_text.strip():
            return None

        # Rule 1: Exclude prompts containing 'xxx'
        if re.search(r"\bxxx\b", raw_text, re.IGNORECASE):
            return None

        # Rule 2: Remove bracketed annotations [like this]
        text = re.sub(r"\[.*?\]", "", raw_text)
        text = re.sub(r"\(.*?\)", "", text)

        # Rule 3: Normalize
        norm = normalize_text_v1(text)
        if not norm or not norm.strip():
            return None

        return norm

    def scan_dataset(
        self,
        speakers: Optional[List[str]] = None,
        task_filter: Optional[str] = None,  # 'sentence', 'word', or None (both)
        include_controls: bool = False,
    ) -> List[DatasetRecord]:
        """Scan directory and return list of valid DatasetRecord instances.

        Returns:
            records: Sorted list of DatasetRecord objects.
        """
        records: List[DatasetRecord] = []
        if not os.path.exists(self.root_dir):
            return records

        # Scan speaker directories
        spk_dirs = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]

        for spk in spk_dirs:
            spk_upper = spk.upper()
            is_control = spk_upper.startswith("FC") or spk_upper.startswith("MC")
            if not include_controls and is_control:
                continue

            if speakers and spk_upper not in speakers:
                continue

            spk_path = os.path.join(self.root_dir, spk)
            session_dirs = [s for s in os.listdir(spk_path) if os.path.isdir(os.path.join(spk_path, s))]

            for session in session_dirs:
                sess_path = os.path.join(spk_path, session)
                audio_dir = os.path.join(sess_path, self.preferred_channel)
                prompt_dir = os.path.join(sess_path, "prompts")

                if not os.path.exists(audio_dir) or not os.path.exists(prompt_dir):
                    continue

                audio_files = [f for f in os.listdir(audio_dir) if f.lower().endswith(".wav")]

                for a_file in audio_files:
                    stem = os.path.splitext(a_file)[0]
                    p_file = os.path.join(prompt_dir, f"{stem}.txt")

                    if not os.path.exists(p_file):
                        continue

                    try:
                        with open(p_file, "r", encoding="utf-8", errors="ignore") as pf:
                            raw_prompt = pf.read()
                    except Exception:
                        continue

                    clean_text = self.clean_prompt(raw_prompt)
                    if not clean_text:
                        continue

                    # Sentence vs word split
                    word_count = len(clean_text.split())
                    task_type = "sentence" if word_count >= 2 else "word"

                    if task_filter and task_type != task_filter:
                        continue

                    audio_path = os.path.join(audio_dir, a_file)
                    try:
                        info = sf.info(audio_path)
                        duration = info.duration
                        if duration < 0.2 or duration > 30.0:
                            continue
                    except Exception:
                        continue

                    spk_meta = TORGO_SPEAKERS.get(spk_upper, {})
                    utt_id = f"TORGO_{spk_upper}_{session}_{stem}"

                    rec = DatasetRecord(
                        utterance_id=utt_id,
                        speaker_id=spk_upper,
                        audio_path=audio_path,
                        text=clean_text,
                        duration_sec=round(duration, 2),
                        corpus="torgo",
                        task_type=task_type,
                        channel=self.preferred_channel,
                        intelligibility=spk_meta.get("type", "dysarthric"),
                        metadata={
                            "session": session,
                            "word_count": word_count,
                            "etiology": spk_meta.get("etiology", "Unknown"),
                            "gender": spk_meta.get("gender", "Unknown"),
                        },
                    )
                    records.append(rec)

        records.sort(key=lambda r: (r.speaker_id, r.utterance_id))
        return records
