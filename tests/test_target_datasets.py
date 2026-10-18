"""Unit tests for UASpeech and TORGO data loaders and schemas."""

import os
import pytest
import soundfile as sf
import numpy as np

from phonorepair.data import DatasetRecord, UASpeechLoader, TORGOLoader


def test_dataset_record_serialization():
    """Verify DatasetRecord dataclass and dictionary conversion."""
    rec = DatasetRecord(
        utterance_id="test_01",
        speaker_id="M01",
        audio_path="/path/to/test.wav",
        text="HELLO WORLD",
        duration_sec=1.5,
        corpus="uaspeech",
        task_type="word",
        channel="M5",
        intelligibility="Very Low",
    )
    d = rec.to_dict()
    assert d["utterance_id"] == "test_01"
    assert d["intelligibility"] == "Very Low"
    assert "cache_path" not in d


def test_uaspeech_filename_parsing():
    """Verify standard UASpeech filename parsing."""
    loader = UASpeechLoader(root_dir="/nonexistent")
    parsed = loader.parse_filename("M04_B1_CW1_M5.wav")
    assert parsed is not None
    assert parsed["speaker_id"] == "M04"
    assert parsed["block_id"] == "B1"
    assert parsed["word_id"] == "CW1"
    assert parsed["channel"] == "M5"

    # Non-conforming filename
    assert loader.parse_filename("random_audio.wav") is None


def test_uaspeech_word_table_lookup():
    """Verify standard common words and digits resolution."""
    loader = UASpeechLoader(root_dir="/nonexistent")
    assert loader.word_table["CW1"] == "COMMAND"
    assert loader.word_table["D1"] == "ONE"
    assert loader.word_table["L1"] == "A"
    assert loader.word_table["C1"] == "BACKSPACE"


def test_torgo_prompt_cleaning():
    """Verify Proposal Section 12 E5 exclusion rules for TORGO prompts."""
    loader = TORGOLoader(root_dir="/nonexistent")

    # 1. Reject 'xxx' unintelligible
    assert loader.clean_prompt("xxx [snort]") is None
    assert loader.clean_prompt("the dog xxx jumped") is None

    # 2. Strip bracketed noise annotations
    clean = loader.clean_prompt("the dog [cough] jumped over the fence")
    assert clean == "the dog jumped over the fence"

    # 3. Strip parentheses
    clean_paren = loader.clean_prompt("SHE HAD YOUR (laughter) DARK SUIT")
    assert clean_paren == "she had your dark suit"

    # 4. Empty or whitespace
    assert loader.clean_prompt("   ") is None
    assert loader.clean_prompt("") is None


def test_uaspeech_and_torgo_mock_scanning():
    """Test full scanning and parsing on a synthetic directory structure."""
    import tempfile
    from pathlib import Path

    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    dummy_wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        tmp_path = Path(td)
        # 1. Setup mock UASpeech
        ua_dir = tmp_path / "uaspeech"
        ua_dir.mkdir()
        ua_wav = ua_dir / "M04_B1_CW1_M5.wav"
        sf.write(str(ua_wav), dummy_wav, sr, subtype="PCM_16")

        ua_loader = UASpeechLoader(root_dir=str(ua_dir), preferred_channel="M5")
        ua_records = ua_loader.scan_dataset()
        assert len(ua_records) == 1
        assert ua_records[0].speaker_id == "M04"
        assert ua_records[0].intelligibility == "Very Low"
        assert ua_records[0].text == "command"

        # 2. Setup mock TORGO
        torgo_dir = tmp_path / "torgo"
        spk_sess = torgo_dir / "M01" / "Session1"
        audio_dir = spk_sess / "wav_headMic"
        prompt_dir = spk_sess / "prompts"
        audio_dir.mkdir(parents=True)
        prompt_dir.mkdir(parents=True)

        # Sentence audio + prompt
        sf.write(str(audio_dir / "0001.wav"), dummy_wav, sr, subtype="PCM_16")
        (prompt_dir / "0001.txt").write_text("THIS IS A FULL SENTENCE", encoding="utf-8")

        # Rejected 'xxx' audio + prompt
        sf.write(str(audio_dir / "0002.wav"), dummy_wav, sr, subtype="PCM_16")
        (prompt_dir / "0002.txt").write_text("xxx [unintelligible]", encoding="utf-8")

        torgo_loader = TORGOLoader(root_dir=str(torgo_dir), preferred_channel="wav_headMic")
        torgo_records = torgo_loader.scan_dataset()
        assert len(torgo_records) == 1
        assert torgo_records[0].speaker_id == "M01"
        assert torgo_records[0].task_type == "sentence"
        assert torgo_records[0].text == "this is a full sentence"
