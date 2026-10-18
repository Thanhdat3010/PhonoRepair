"""CLI utility to scan, validate, and build JSONL manifests for UASpeech and TORGO.

Usage:
  python scripts/prepare_target_datasets.py --corpus uaspeech --data-dir /path/to/uaspeech --output data/manifests/uaspeech_eval.jsonl
  python scripts/prepare_target_datasets.py --corpus torgo --data-dir /path/to/torgo --output data/manifests/torgo_eval.jsonl
  python scripts/prepare_target_datasets.py --generate-pilot-mocks
"""

import argparse
import hashlib
import json
import os
import numpy as np
import soundfile as sf

from phonorepair.data import UASpeechLoader, TORGOLoader
from phonorepair.utils.locking import compute_file_sha256


def generate_pilot_mocks(base_dir: str = "data/raw/mocks") -> None:
    """Generate small representative mock directories for UASpeech and TORGO."""
    print(f"[*] Generating mock dataset structures in: {base_dir}")

    # 1. Mock UASpeech
    ua_dir = os.path.join(base_dir, "uaspeech")
    os.makedirs(ua_dir, exist_ok=True)
    sr = 16000

    # 3 speakers: 1 Very Low (M04), 1 Mid (M11), 1 High (M08)
    ua_samples = [
        ("M04", "B1", "CW1", "M5"),   # COMMAND
        ("M04", "B1", "D1", "M5"),    # ONE
        ("M11", "B2", "CW2", "M5"),   # ENTER
        ("M11", "B2", "D2", "M5"),    # TWO
        ("M08", "B3", "CW3", "M5"),   # ESCAPE
        ("M08", "B3", "D3", "M5"),    # THREE
    ]

    for spk, blk, wid, ch in ua_samples:
        fname = f"{spk}_{blk}_{wid}_{ch}.wav"
        fpath = os.path.join(ua_dir, fname)
        # 1-sec audio tone
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
        sf.write(fpath, audio, sr, subtype="PCM_16")

    print(f"    - UASpeech mock: {len(ua_samples)} files in {ua_dir}")

    # 2. Mock TORGO
    torgo_dir = os.path.join(base_dir, "torgo")
    # 2 speakers: M01 (ALS), F01 (CP)
    torgo_samples = [
        ("M01", "Session1", "0001", "SHE HAD YOUR DARK SUIT IN GREASY WASH WATER ALL YEAR"),  # Sentence
        ("M01", "Session1", "0002", "UP"),                                                  # Single word
        ("M01", "Session1", "0003", "xxx [unintelligible]"),                                 # To be excluded!
        ("F01", "Session1", "0001", "DON'T ASK ME TO CARRY AN OILY RAG LIKE THAT"),         # Sentence
        ("F01", "Session1", "0002", "FEED"),                                                # Single word
    ]

    for spk, sess, stem, prompt in torgo_samples:
        sess_dir = os.path.join(torgo_dir, spk, sess)
        audio_dir = os.path.join(sess_dir, "wav_headMic")
        prompt_dir = os.path.join(sess_dir, "prompts")
        os.makedirs(audio_dir, exist_ok=True)
        os.makedirs(prompt_dir, exist_ok=True)

        # Audio
        t = np.linspace(0, 2.0, sr, endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 250 * t)).astype(np.float32)
        sf.write(os.path.join(audio_dir, f"{stem}.wav"), audio, sr, subtype="PCM_16")

        # Prompt
        with open(os.path.join(prompt_dir, f"{stem}.txt"), "w", encoding="utf-8") as pf:
            pf.write(prompt)

    print(f"    - TORGO mock: {len(torgo_samples)} files in {torgo_dir}")


def main():
    parser = argparse.ArgumentParser(description="Target Dysarthric Dataset Manifest Builder")
    parser.add_argument("--corpus", choices=["uaspeech", "torgo"], help="Corpus name")
    parser.add_argument("--data-dir", help="Path to raw corpus directory")
    parser.add_argument("--output", help="Output JSONL manifest path")
    parser.add_argument("--task-filter", choices=["sentence", "word"], default=None, help="TORGO task filter")
    parser.add_argument("--generate-pilot-mocks", action="store_true", help="Generate mock directories for testing")
    parser.add_argument("--checksum-file", default="data/manifests/checksums.sha256", help="Checksum file to append")
    args = parser.parse_args()

    if args.generate_pilot_mocks:
        generate_pilot_mocks()
        return

    if not args.corpus or not args.data_dir or not args.output:
        parser.error("--corpus, --data-dir, and --output are required unless --generate-pilot-mocks is specified.")

    print(f"[*] Scanning {args.corpus.upper()} in: {args.data_dir}")

    if args.corpus == "uaspeech":
        loader = UASpeechLoader(root_dir=args.data_dir)
        records = loader.scan_dataset()
    else:
        loader = TORGOLoader(root_dir=args.data_dir)
        records = loader.scan_dataset(task_filter=args.task_filter)

    print(f"[+] Found {len(records)} valid utterances matching criteria.")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r.to_dict()) + "\n")

    # Compute checksum
    sha256 = compute_file_sha256(args.output)
    print(f"[+] Manifest saved to: {args.output}")
    print(f"    SHA256: {sha256}")

    # Append to checksum file
    if args.checksum_file and os.path.exists(os.path.dirname(args.checksum_file)):
        with open(args.checksum_file, "a", encoding="utf-8") as cf:
            rel_path = os.path.relpath(args.output, os.path.dirname(args.checksum_file))
            cf.write(f"{sha256}  {rel_path}\n")
        print(f"[+] Locked checksum in: {args.checksum_file}")


if __name__ == "__main__":
    main()
