"""Fetch and inspect TORGO dataset from Hugging Face hub (abnerh/TORGO-database).

This script allows downloading and formatting the public TORGO dataset directly
onto the RTX 6000 server without needing manual institutional portal access.
"""

import os
import sys
from datasets import load_dataset
import soundfile as sf
import numpy as np

def main():
    print("[*] Inspecting and loading 'abnerh/TORGO-database' from Hugging Face...")
    ds = load_dataset("abnerh/TORGO-database", split="train", streaming=True)
    first_sample = next(iter(ds))
    print("[+] Successfully connected to 'abnerh/TORGO-database'!")
    print("    Sample keys:", list(first_sample.keys()))
    for k, v in first_sample.items():
        if k != "audio":
            print(f"    {k}: {v}")
        else:
            print(f"    audio: sampling_rate={v.get('sampling_rate')}, len={len(v.get('array', []))}")

if __name__ == "__main__":
    main()
