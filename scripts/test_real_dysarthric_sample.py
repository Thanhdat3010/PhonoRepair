"""Test real dysarthric speech samples on local Whisper base model with immediate unbuffered output."""

import sys
import torch
from datasets import load_dataset
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from phonorepair.evaluation.normalizer import normalize_text_v1
from phonorepair.evaluation.metrics import compute_wer_counts, compute_corpus_wer

def test_samples():
    print("[*] Loading Whisper base model locally...", flush=True)
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en")
    model.eval()

    # 1. Test UASpeech Dysarthric Samples
    print("\n" + "="*65, flush=True)
    print("   EVALUATING REAL UASPEECH SAMPLES ('extraordinarylab/ua-speech')", flush=True)
    print("="*65, flush=True)
    ua_ds = load_dataset("extraordinarylab/ua-speech", split="test", streaming=True)
    for i, item in enumerate(ua_ds):
        if i >= 3:
            break
        spk = item["speaker"]
        sev = item["severity"]
        ref_text = item["text"]
        ref_norm = normalize_text_v1(ref_text)
        audio_arr = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]

        inputs = processor(audio_arr, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out_ids = model.generate(inputs.input_features, temperature=0.0, num_beams=1)
            hyp_raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            hyp_norm = normalize_text_v1(hyp_raw)

        counts = compute_wer_counts(ref_norm.split(), hyp_norm.split())
        wer = compute_corpus_wer([counts])
        print(f"\nUASpeech Sample {i+1} [Speaker: {spk} | Severity: {sev}]", flush=True)
        print(f"  Ref Text:       '{ref_text}' -> '{ref_norm}'", flush=True)
        print(f"  Whisper Hyp:    '{hyp_raw}' -> '{hyp_norm}'", flush=True)
        print(f"  WER:            {wer*100:.1f}% (Sub={counts.substitutions}, Del={counts.deletions}, Ins={counts.insertions})", flush=True)

    # 2. Test TORGO Samples
    print("\n" + "="*65, flush=True)
    print("   EVALUATING REAL TORGO SAMPLES ('extraordinarylab/torgo')", flush=True)
    print("="*65, flush=True)
    torgo_ds = load_dataset("extraordinarylab/torgo", split="test", streaming=True)
    for i, item in enumerate(torgo_ds):
        if i >= 3:
            break
        spk = item.get("speaker", "unknown")
        status = item.get("speech_status", "unknown")
        mic = item.get("microphone", "unknown")
        ref_text = item.get("text", "")
        ref_norm = normalize_text_v1(ref_text)
        audio_arr = item["audio"]["array"]
        sr = item["audio"]["sampling_rate"]

        inputs = processor(audio_arr, sampling_rate=sr, return_tensors="pt")
        with torch.no_grad():
            out_ids = model.generate(inputs.input_features, temperature=0.0, num_beams=1)
            hyp_raw = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
            hyp_norm = normalize_text_v1(hyp_raw)

        counts = compute_wer_counts(ref_norm.split(), hyp_norm.split())
        wer = compute_corpus_wer([counts])
        print(f"\nTORGO Sample {i+1} [Speaker: {spk} | Status: {status} | Mic: {mic}]", flush=True)
        print(f"  Ref Text:       '{ref_text}' -> '{ref_norm}'", flush=True)
        print(f"  Whisper Hyp:    '{hyp_raw}' -> '{hyp_norm}'", flush=True)
        print(f"  WER:            {wer*100:.1f}% (Sub={counts.substitutions}, Del={counts.deletions}, Ins={counts.insertions})", flush=True)

if __name__ == "__main__":
    test_samples()
