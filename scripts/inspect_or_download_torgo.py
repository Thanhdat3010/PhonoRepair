"""Fetch and thoroughly inspect real data samples from TORGO and UASpeech on Hugging Face.

Verifies:
1. Audio format, sampling rate, duration, clarity.
2. Metadata (speaker, gender, etiology, intelligibility status).
3. Text transcripts and canonical normalization via normalize_text_v1.
4. Saves physical audio samples to data/raw/samples/ for verification.
"""

import os
import json
import numpy as np
import soundfile as sf
from datasets import load_dataset
from phonorepair.evaluation.normalizer import normalize_text_v1

def inspect_torgo():
    print("\n" + "="*70)
    print("   INSPECTING TORGO SAMPLES ('abnerh/TORGO-database')")
    print("="*70)
    out_dir = "data/raw/samples/torgo"
    os.makedirs(out_dir, exist_ok=True)
    
    ds = load_dataset("abnerh/TORGO-database", split="train", streaming=True)
    samples = []
    
    for i, item in enumerate(ds):
        if i >= 6:
            break
        
        audio = item.get("audio", {})
        arr = audio.get("array")
        sr = audio.get("sampling_rate", 16000)
        dur = len(arr) / sr if arr is not None else 0.0
        
        raw_text = item.get("text", item.get("transcription", ""))
        norm_text = normalize_text_v1(raw_text)
        
        info = {
            "index": i + 1,
            "speaker_id": item.get("speaker_id", item.get("speaker", "unknown")),
            "status": item.get("status", item.get("dysarthria", "unknown")),
            "gender": item.get("gender", "unknown"),
            "raw_text": raw_text,
            "norm_text": norm_text,
            "sampling_rate": sr,
            "duration_sec": round(dur, 2),
            "num_samples": len(arr) if arr is not None else 0
        }
        samples.append(info)
        
        # Save first 2 wavs
        if i < 2 and arr is not None:
            wav_path = os.path.join(out_dir, f"sample_{i+1}_{info['speaker_id']}.wav")
            sf.write(wav_path, arr.astype(np.float32), sr)
            info["saved_wav"] = wav_path
            
        print(f"\n[Sample {i+1}]")
        for k, v in info.items():
            print(f"  {k}: {v}")
            
    return samples

def inspect_uaspeech():
    print("\n" + "="*70)
    print("   INSPECTING UASPEECH SAMPLES ('extraordinarylab/ua-speech')")
    print("="*70)
    out_dir = "data/raw/samples/uaspeech"
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        ds = load_dataset("extraordinarylab/ua-speech", split="train", streaming=True)
    except Exception as e:
        print(f"[!] Could not load 'extraordinarylab/ua-speech': {e}")
        print("[*] Trying alternative 'ngdiana/uaspeech'...")
        try:
            ds = load_dataset("ngdiana/uaspeech", split="train", streaming=True)
        except Exception as e2:
            print(f"[!] Alternative also failed: {e2}")
            return []
            
    samples = []
    for i, item in enumerate(ds):
        if i >= 6:
            break
        
        audio = item.get("audio", {})
        arr = audio.get("array")
        sr = audio.get("sampling_rate", 16000)
        dur = len(arr) / sr if arr is not None else 0.0
        
        raw_text = item.get("text", item.get("transcription", item.get("word", "")))
        norm_text = normalize_text_v1(raw_text)
        
        info = {
            "index": i + 1,
            "speaker_id": item.get("speaker_id", item.get("speaker", "unknown")),
            "severity": item.get("severity", item.get("intelligibility", "unknown")),
            "raw_text": raw_text,
            "norm_text": norm_text,
            "sampling_rate": sr,
            "duration_sec": round(dur, 2),
            "num_samples": len(arr) if arr is not None else 0
        }
        samples.append(info)
        
        if i < 2 and arr is not None:
            wav_path = os.path.join(out_dir, f"sample_{i+1}_{info['speaker_id']}.wav")
            sf.write(wav_path, arr.astype(np.float32), sr)
            info["saved_wav"] = wav_path
            
        print(f"\n[Sample {i+1}]")
        for k, v in info.items():
            print(f"  {k}: {v}")
            
    return samples

def main():
    torgo_samples = inspect_torgo()
    uaspeech_samples = inspect_uaspeech()
    
    summary = {
        "torgo_samples_inspected": len(torgo_samples),
        "uaspeech_samples_inspected": len(uaspeech_samples),
        "torgo_ok": len(torgo_samples) > 0,
        "uaspeech_ok": len(uaspeech_samples) > 0,
    }
    os.makedirs("outputs/reports", exist_ok=True)
    with open("outputs/reports/dataset_inspection_report.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "torgo": torgo_samples, "uaspeech": uaspeech_samples}, f, indent=2)
    print("\n[+] Inspection report written to: outputs/reports/dataset_inspection_report.json")

if __name__ == "__main__":
    main()
