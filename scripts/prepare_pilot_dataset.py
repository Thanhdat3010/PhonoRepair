"""Download, preprocess, align, and cache LibriSpeech pilot dataset for PhonoRepair.

This script:
1. Downloads clean speech audio from Hugging Face `openslr/librispeech_asr`
2. Converts transcripts to 39 ARPABET phonetic tokens + SIL using G2P
3. Aligns frame timestamps with Whisper encoder subsampling (50 fps, 20ms per frame)
4. Passes audio through Whisper base.en encoder to extract and cache features (T, 512)
5. Generates frame-level class labels in {0, ..., 39} for dictionary fitting
6. Updates data/manifests/ with exact file paths and hashes
"""

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List, Tuple
import numpy as np
import soundfile as sf
import torch
from datasets import load_dataset
from g2p_en import G2p
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# 39 ARPABET phonetic router inventory + SIL (Proposal Appendix B.5)
ROUTER_INVENTORY = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH", "SIL"
]
PHONE_TO_ID = {phone: i for i, phone in enumerate(ROUTER_INVENTORY)}
ID_TO_PHONE = {i: phone for i, phone in enumerate(ROUTER_INVENTORY)}
SIL_ID = PHONE_TO_ID["SIL"]


def clean_phoneme(p: str) -> str:
    """Strip stress digits from phonemes (e.g. 'AH0' -> 'AH')."""
    p_clean = re.sub(r"\d+", "", p).upper().strip()
    if p_clean in PHONE_TO_ID:
        return p_clean
    if p_clean in [" ", "", "'"]:
        return "SIL"
    return "AH"  # Fallback neutral vowel for unmapped symbols


def text_to_phonemes(text: str, g2p: G2p) -> List[int]:
    """Convert raw text to list of phonetic class IDs in {0..39}."""
    raw_phones = g2p(text)
    phone_ids = []
    for p in raw_phones:
        cleaned = clean_phoneme(p)
        phone_ids.append(PHONE_TO_ID[cleaned])
    return phone_ids if phone_ids else [SIL_ID]


def main():
    parser = argparse.ArgumentParser(description="Prepare LibriSpeech Pilot Data for PhonoRepair")
    parser.add_argument("--num-train100-utterances", type=int, default=60, help="Number of train.clean.100 utterances")
    parser.add_argument("--num-train360-utterances", type=int, default=30, help="Number of train.clean.360 utterances")
    parser.add_argument("--num-test-utterances", type=int, default=15, help="Number of test.clean utterances")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save audio & alignments")
    parser.add_argument("--cache-dir", default="cache/features", help="Directory to cache encoder latents")
    parser.add_argument("--manifest-dir", default="data/manifests", help="Directory to save manifests")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.manifest_dir, exist_ok=True)
    audio_dir = os.path.join(args.output_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    print("=" * 60)
    print("  PHONOREPAIR: LIBRISPEECH PILOT PREPARATION")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # 1. Initialize Whisper and G2P
    print("[1/5] Loading Whisper base.en model & G2P engine...")
    g2p = G2p()
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").to(args.device)
    model.eval()

    splits_to_download = [
        ("train.100", args.num_train100_utterances, "train100"),
        ("train.360", args.num_train360_utterances, "train360"),
        ("test", args.num_test_utterances, "test_clean"),
    ]

    collected_records = {k: [] for _, _, k in splits_to_download}

    # 2. Download and process audio streams
    for hf_split, max_count, split_key in splits_to_download:
        print(f"\n[2/5] Streaming {max_count} utterances from LibriSpeech split '{hf_split}'...")
        ds = load_dataset("openslr/librispeech_asr", "clean", split=hf_split, streaming=True)

        count = 0
        spk_counts = {}
        max_per_speaker = 5
        pbar = tqdm(total=max_count, desc=f"Processing {split_key}")
        for sample in ds:
            if count >= max_count:
                break

            utt_id = sample["id"]
            spk_id = str(sample["speaker_id"])
            if spk_counts.get(spk_id, 0) >= max_per_speaker:
                continue

            text = sample["text"]
            audio_data = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            # Ensure 16kHz
            if sr != 16000:
                continue

            # Skip audio that is too short (< 1s) or too long (> 20s)
            duration = len(audio_data) / 16000.0
            if duration < 1.0 or duration > 20.0:
                continue

            # Save audio file
            spk_audio_dir = os.path.join(audio_dir, split_key, spk_id)
            os.makedirs(spk_audio_dir, exist_ok=True)
            wav_path = os.path.join(spk_audio_dir, f"{utt_id}.wav")
            sf.write(wav_path, audio_data, 16000, subtype="PCM_16")

            # G2P Phonetic classes
            phone_ids = text_to_phonemes(text, g2p)

            # 3. Extract Whisper encoder latents: H in (T, 512)
            inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(args.device)

            with torch.no_grad():
                encoder_outputs = model.model.encoder(input_features)
                # Shape is (1, T_frames, 512) -> Whisper encoder generates 1500 frames for 30s mel
                # Calculate actual valid audio frames: Whisper downsamples mel by 2x (100 fps -> 50 fps)
                valid_frames = min(int(duration * 50), encoder_outputs.last_hidden_state.shape[1])
                latent_tensor = encoder_outputs.last_hidden_state[0, :valid_frames, :].cpu()  # (valid_frames, 512)

            # Assign phoneme labels across valid frames uniformly/proportionally
            # (interior frame mapping as specified in Proposal E1)
            frame_labels = np.zeros(valid_frames, dtype=np.int64)
            n_phones = len(phone_ids)
            frames_per_phone = max(1, valid_frames / n_phones)
            for f in range(valid_frames):
                p_idx = min(int(f / frames_per_phone), n_phones - 1)
                frame_labels[f] = phone_ids[p_idx]

            # Cache features & frame labels
            cache_file = os.path.join(args.cache_dir, f"{utt_id}.pt")
            torch.save(
                {
                    "utt_id": utt_id,
                    "speaker_id": spk_id,
                    "features": latent_tensor,              # (T, 512)
                    "frame_labels": torch.from_numpy(frame_labels), # (T,) in 0..39
                    "text": text,
                    "duration_sec": duration,
                },
                cache_file,
            )

            record = {
                "utterance_id": utt_id,
                "speaker_id": spk_id,
                "audio_path": wav_path,
                "cache_path": cache_file,
                "text": text,
                "duration_sec": duration,
                "num_frames": valid_frames,
            }
            collected_records[split_key].append(record)
            spk_counts[spk_id] = spk_counts.get(spk_id, 0) + 1
            count += 1
            pbar.update(1)

        pbar.close()

    # 4. Partition into Proposal Manifests
    print("\n[4/5] Partitioning records into Proposal Seed-17 Manifests...")
    train100_records = collected_records["train100"]
    unique_train100_spks = sorted(list(set(r["speaker_id"] for r in train100_records)))

    # Deterministic split seed 17
    from phonorepair.utils.hashing import split_train_clean_100, split_train_clean_360
    basis_spks, dict_spks = split_train_clean_100(unique_train100_spks, seed=17)

    basis_records = [r for r in train100_records if r["speaker_id"] in basis_spks]
    dict_records = [r for r in train100_records if r["speaker_id"] in dict_spks]

    train360_records = collected_records["train360"]
    unique_360_spks = sorted(list(set(r["speaker_id"] for r in train360_records)))
    n_sel = max(1, len(unique_360_spks) // 2)
    n_cal = max(1, len(unique_360_spks) - n_sel)
    spk_360_splits = split_train_clean_360(unique_360_spks, seed=17, n_selection=n_sel, n_calibration=n_cal)

    selection_records = [r for r in train360_records if r["speaker_id"] in spk_360_splits["clean_selection"]]
    calibration_records = [r for r in train360_records if r["speaker_id"] in spk_360_splits["calibration"]]
    test_records = collected_records["test_clean"]

    manifest_map = {
        "librispeech_basis_fit.jsonl": basis_records,
        "librispeech_dict_fit.jsonl": dict_records,
        "librispeech_selection.jsonl": selection_records,
        "librispeech_calibration.jsonl": calibration_records,
        "test_clean.jsonl": test_records,
    }

    checksums = {}
    for fname, recs in manifest_map.items():
        m_path = os.path.join(args.manifest_dir, fname)
        with open(m_path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")

        with open(m_path, "rb") as f:
            checksums[fname] = hashlib.sha256(f.read()).hexdigest()
        print(f"    - {fname}: {len(recs)} utterances ({checksums[fname][:12]}...)")

    # 5. Lock manifest checksums
    checksum_file = os.path.join(args.manifest_dir, "checksums.sha256")
    with open(checksum_file, "w", encoding="utf-8") as f:
        for fname, sha in checksums.items():
            f.write(f"{sha}  {fname}\n")
    print(f"[5/5] Manifests locked at: {checksum_file}")

    print("\n" + "=" * 60)
    print("  [+] DATA PREPARATION COMPLETE! ALL LATENTS & AUDIO PREPROCESSED.")
    print("=" * 60)


if __name__ == "__main__":
    main()
