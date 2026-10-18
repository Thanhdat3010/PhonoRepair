"""Download, extract, and build locked JSONL evaluation manifests from Hugging Face datasets.

Datasets:
1. LibriSpeech test-clean: 'openslr/librispeech_asr' (clean baseline preservation E0/E3)
2. UASpeech: 'extraordinarylab/ua-speech' (dysarthric word evaluation E4)
3. TORGO: 'extraordinarylab/torgo' (dysarthric sentence/word evaluation E5, headMic only)

Stores audio in data/raw/audio/<corpus>/ and writes manifests to data/manifests/<corpus>_eval.jsonl.
Computes and locks SHA256 in data/manifests/checksums.sha256.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional
import numpy as np
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from phonorepair.data.schema import DatasetRecord
from phonorepair.evaluation.normalizer import normalize_text_v1
from phonorepair.utils.locking import compute_file_sha256


def build_librispeech_clean(
    output_dir: str = "data/raw/audio/librispeech",
    manifest_path: str = "data/manifests/librispeech_test_clean_full.jsonl",
    max_samples: Optional[int] = None,
) -> int:
    """Build locked manifest for LibriSpeech test-clean."""
    print(f"\n[*] Processing LibriSpeech test-clean -> {manifest_path}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    ds = load_dataset("openslr/librispeech_asr", split="test.clean", streaming=True)
    records: List[DatasetRecord] = []

    count = 0
    for item in tqdm(ds, desc="LibriSpeech clean", total=max_samples or 2620):
        if max_samples and count >= max_samples:
            break

        audio_data = item["audio"]
        arr = audio_data["array"]
        sr = audio_data["sampling_rate"]
        raw_text = item.get("text", "")
        norm_text = normalize_text_v1(raw_text)
        if not norm_text:
            continue

        spk_id = str(item.get("speaker_id", "clean_spk"))
        utt_id = str(item.get("id", f"ls_{spk_id}_{count:05d}"))

        # Save audio
        wav_name = f"{utt_id}.wav"
        wav_path = os.path.join(output_dir, wav_name).replace("\\", "/")
        if not os.path.exists(wav_path):
            sf.write(wav_path, arr.astype(np.float32), sr, subtype="PCM_16")

        dur = round(len(arr) / sr, 2)
        rec = DatasetRecord(
            utterance_id=utt_id,
            speaker_id=spk_id,
            audio_path=wav_path,
            text=norm_text,
            duration_sec=dur,
            corpus="librispeech",
            task_type="sentence",
            channel="clean",
            intelligibility="Control",
            metadata={"chapter_id": item.get("chapter_id")},
        )
        records.append(rec)
        count += 1

    records.sort(key=lambda r: (r.speaker_id, r.utterance_id))
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    sha256 = compute_file_sha256(manifest_path)
    print(f"[+] Saved {len(records)} LibriSpeech records to {manifest_path} (SHA256: {sha256})")
    return len(records)


def build_uaspeech(
    output_dir: str = "data/raw/audio/uaspeech",
    manifest_path: str = "data/manifests/uaspeech_eval_full.jsonl",
    max_samples: Optional[int] = None,
) -> int:
    """Build locked manifest for UASpeech test split."""
    print(f"\n[*] Processing UASpeech test split -> {manifest_path}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    ds = load_dataset("extraordinarylab/ua-speech", split="test", streaming=True)
    records: List[DatasetRecord] = []

    count = 0
    for item in tqdm(ds, desc="UASpeech test", total=max_samples or 18740):
        if max_samples and count >= max_samples:
            break

        audio_data = item["audio"]
        arr = audio_data["array"]
        sr = audio_data["sampling_rate"]
        raw_text = item.get("text", "")
        norm_text = normalize_text_v1(raw_text)
        if not norm_text:
            continue

        spk = item.get("speaker", "unknown").upper()
        sev = item.get("severity", "unknown").capitalize()
        utt_id = f"UASPEECH_{spk}_{count:06d}"

        wav_name = f"{utt_id}.wav"
        wav_path = os.path.join(output_dir, wav_name).replace("\\", "/")
        if not os.path.exists(wav_path):
            sf.write(wav_path, arr.astype(np.float32), sr, subtype="PCM_16")

        dur = round(len(arr) / sr, 2)
        rec = DatasetRecord(
            utterance_id=utt_id,
            speaker_id=spk,
            audio_path=wav_path,
            text=norm_text,
            duration_sec=dur,
            corpus="uaspeech",
            task_type="word",
            channel="M5",
            intelligibility=sev,
            metadata={"original_severity": item.get("severity")},
        )
        records.append(rec)
        count += 1

    records.sort(key=lambda r: (r.speaker_id, r.utterance_id))
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    sha256 = compute_file_sha256(manifest_path)
    print(f"[+] Saved {len(records)} UASpeech records to {manifest_path} (SHA256: {sha256})")
    return len(records)


def build_torgo(
    output_dir: str = "data/raw/audio/torgo",
    manifest_path: str = "data/manifests/torgo_eval_full.jsonl",
    max_samples: Optional[int] = None,
) -> int:
    """Build locked manifest for TORGO dataset (filtering for headMic and valid prompt)."""
    print(f"\n[*] Processing TORGO dataset (headMic channel) -> {manifest_path}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    ds = load_dataset("extraordinarylab/torgo", split="test", streaming=True)
    records: List[DatasetRecord] = []

    count = 0
    for item in tqdm(ds, desc="TORGO test", total=max_samples or 16552):
        if max_samples and count >= max_samples:
            break

        # Filter strictly for preferred headMic channel (Proposal line 848)
        mic = item.get("microphone", "")
        if mic != "headMic":
            continue

        raw_text = item.get("text", "")
        # Filter 'xxx' unintelligible tags
        if not raw_text or re.search(r"\bxxx\b", raw_text, re.IGNORECASE):
            continue

        norm_text = normalize_text_v1(raw_text)
        if not norm_text:
            continue

        audio_data = item["audio"]
        arr = audio_data["array"]
        sr = audio_data["sampling_rate"]

        spk = item.get("speaker", "unknown").upper()
        status = item.get("speech_status", "unknown")
        intelligibility = "Control" if status == "healthy" or spk.startswith("C") else "dysarthric"

        # Determine task type: sentence (>= 2 words) or single word
        word_count = len(norm_text.split())
        task_type = "sentence" if word_count >= 2 else "word"
        utt_id = f"TORGO_{spk}_{count:06d}"

        wav_name = f"{utt_id}.wav"
        wav_path = os.path.join(output_dir, wav_name).replace("\\", "/")
        if not os.path.exists(wav_path):
            sf.write(wav_path, arr.astype(np.float32), sr, subtype="PCM_16")

        dur = round(len(arr) / sr, 2)
        rec = DatasetRecord(
            utterance_id=utt_id,
            speaker_id=spk,
            audio_path=wav_path,
            text=norm_text,
            duration_sec=dur,
            corpus="torgo",
            task_type=task_type,
            channel="headMic",
            intelligibility=intelligibility,
            metadata={"status": status, "word_count": word_count},
        )
        records.append(rec)
        count += 1

    records.sort(key=lambda r: (r.speaker_id, r.utterance_id))
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    sha256 = compute_file_sha256(manifest_path)
    print(f"[+] Saved {len(records)} TORGO records to {manifest_path} (SHA256: {sha256})")
    return len(records)


def lock_checksums(manifest_paths: List[str], checksum_file: str = "data/manifests/checksums.sha256") -> None:
    """Lock SHA256 checksums of all manifests."""
    print(f"\n[*] Locking checksums into {checksum_file}...")
    existing = {}
    if os.path.exists(checksum_file):
        with open(checksum_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    existing[parts[1]] = parts[0]

    for p in manifest_paths:
        if os.path.exists(p):
            sha = compute_file_sha256(p)
            rel = os.path.relpath(p, os.path.dirname(checksum_file)).replace("\\", "/")
            existing[rel] = sha

    with open(checksum_file, "w", encoding="utf-8") as f:
        for rel in sorted(existing.keys()):
            f.write(f"{existing[rel]}  {rel}\n")
    print(f"[+] Updated and locked {len(existing)} manifest checksums in {checksum_file}")


def main():
    parser = argparse.ArgumentParser(description="Build and lock evaluation manifests from Hugging Face")
    parser.add_argument("--corpora", nargs="+", choices=["librispeech", "uaspeech", "torgo", "all"], default=["all"])
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples per corpus for pilot/test")
    args = parser.parse_args()

    targets = ["librispeech", "uaspeech", "torgo"] if "all" in args.corpora else args.corpora
    manifests = []

    if "librispeech" in targets:
        m = "data/manifests/librispeech_test_clean_full.jsonl"
        build_librispeech_clean(manifest_path=m, max_samples=args.max_samples)
        manifests.append(m)

    if "uaspeech" in targets:
        m = "data/manifests/uaspeech_eval_full.jsonl"
        build_uaspeech(manifest_path=m, max_samples=args.max_samples)
        manifests.append(m)

    if "torgo" in targets:
        m = "data/manifests/torgo_eval_full.jsonl"
        build_torgo(manifest_path=m, max_samples=args.max_samples)
        manifests.append(m)

    lock_checksums(manifests)
    print("\n[+] ALL EVALUATION MANIFESTS BUILT AND CRYPTOGRAPHICALLY LOCKED!")


if __name__ == "__main__":
    main()
