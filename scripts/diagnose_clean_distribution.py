import json
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from phonorepair.artifacts.schema import load_artifact
from phonorepair.core.geometry import read_coordinates, compute_all_patch_distances
from phonorepair.core.gate import aggregate_class_distances


def diagnose_clean():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    artifact = load_artifact("outputs/artifacts/whisper_base_artifact.pt")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en").to(device)

    records = []
    with open("data/manifests/librispeech_test_clean_full.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    Q = artifact.Q.to(device)
    mu = artifact.mu.to(device)
    anchors = artifact.anchors.to(device)
    bases = [b.to(device) for b in artifact.bases]
    radii = artifact.radii.to(device)
    patch_labels = artifact.patch_labels.to(device)
    num_classes = artifact.num_classes

    d1_all = []
    gap_all = []

    for r in records:
        audio, sr = sf.read(r["audio_path"])
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features.to(device)
        with torch.no_grad():
            h_seq = model.model.encoder(feat).last_hidden_state
        B, T, d = h_seq.shape
        h_flat = h_seq.reshape(-1, d)
        u_flat = read_coordinates(h_flat, Q, mu)
        pd = compute_all_patch_distances(u_flat, anchors, bases, radii)
        _, d1, d2, _ = aggregate_class_distances(pd, patch_labels, num_classes)
        d1_all.append(d1.cpu())
        gap_all.append((d2 - d1).cpu())

    d1 = torch.cat(d1_all)
    gap = torch.cat(gap_all)

    print("=" * 60)
    print("   CLEAN SPEECH (LIBRISPEECH) DISTANCE DISTRIBUTION")
    print("=" * 60)
    for q in [0.50, 0.75, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]:
        val = float(d1.quantile(q).item())
        print(f"Quantile {q*100:5.1f}%: {val:.4f}")
    print(f"Max distance:   {float(d1.max().item()):.4f}")
    print(f"Min distance:   {float(d1.min().item()):.4f}")
    print(f"Mean distance:  {float(d1.mean().item()):.4f}")
    print(f"Median margin (d2-d1): {float(gap.quantile(0.5).item()):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    diagnose_clean()
