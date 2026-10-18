import json
import os
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from phonorepair.artifacts.schema import load_artifact
from phonorepair.core.geometry import read_coordinates, compute_all_patch_distances
from phonorepair.core.gate import aggregate_class_distances, evaluate_gate


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute device: {device}")

    artifact = load_artifact("outputs/artifacts/whisper_base_artifact.pt")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en").to(device)

    print(f"[*] Artifact: eta={artifact.eta:.4f}, nu={artifact.nu:.4f}, q_off={artifact.q_off:.4f}, w={artifact.w:.4f}")
    upper_bound = artifact.eta + artifact.nu
    print(f"    Open window: [{artifact.q_off:.4f}, {upper_bound:.4f}]")

    # Sample 50 dysarthric utterances from UASpeech
    records = []
    with open("data/manifests/uaspeech_eval_full.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            if "Very low" in line or "Low" in line:
                records.append(json.loads(line))
            if len(records) >= 50:
                break

    print(f"[*] Analyzing {len(records)} dysarthric utterances...")

    all_d1 = []
    all_d2 = []
    all_reasons = []

    Q = artifact.Q.to(device)
    mu = artifact.mu.to(device)
    anchors = artifact.anchors.to(device)
    bases = [b.to(device) for b in artifact.bases]
    radii = artifact.radii.to(device)
    patch_labels = artifact.patch_labels.to(device)
    num_classes = artifact.num_classes

    for idx, r in enumerate(records):
        audio, sr = sf.read(r["audio_path"])
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)

        with torch.no_grad():
            encoder_outputs = model.model.encoder(input_features)
            h_seq = encoder_outputs.last_hidden_state[0]  # (T, d)

        u_seq = read_coordinates(h_seq, Q, mu)
        patch_dists = compute_all_patch_distances(u_seq, anchors, bases, radii)
        class_dists, d1, d2, _ = aggregate_class_distances(patch_dists, patch_labels, num_classes)

        _, _, reasons = evaluate_gate(class_dists, artifact.q_off, artifact.eta, artifact.nu)

        all_d1.extend(d1.cpu().tolist())
        all_d2.extend(d2.cpu().tolist())
        all_reasons.extend(reasons)

    d1_tensor = torch.tensor(all_d1)
    d2_tensor = torch.tensor(all_d2)
    gap_tensor = d2_tensor - d1_tensor

    print("\n" + "=" * 65)
    print("   DYSARTHRIC LATENT DISTANCE DISTRIBUTION")
    print("=" * 65)
    print(f"Total evaluated frames: {len(d1_tensor)}")
    print(f"d_1 min:    {d1_tensor.min():.4f}")
    print(f"d_1 10%:    {d1_tensor.quantile(0.10):.4f}")
    print(f"d_1 25%:    {d1_tensor.quantile(0.25):.4f}")
    print(f"d_1 Median: {d1_tensor.median():.4f}")
    print(f"d_1 Mean:   {d1_tensor.mean():.4f}")
    print(f"d_1 75%:    {d1_tensor.quantile(0.75):.4f}")
    print(f"d_1 90%:    {d1_tensor.quantile(0.90):.4f}")
    print(f"d_1 max:    {d1_tensor.max():.4f}")

    print("\n[+] Class separation gap (d_2 - d_1):")
    print(f"Gap Median: {gap_tensor.median():.4f}")
    print(f"Gap Mean:   {gap_tensor.mean():.4f}")

    print("\n[+] Reason Breakdown:")
    from collections import Counter
    counts = Counter(all_reasons)
    for r, c in counts.items():
        print(f"  {r}: {c} ({c / len(all_reasons) * 100:.2f}%)")

    # Simulation of different threshold pairs:
    print("\n" + "=" * 65)
    print("   THRESHOLD SIMULATION GRID (Searching for optimal w)")
    print("=" * 65)
    print(f"{'q_off':<10} | {'upper':<10} | {'w':<8} | {'Accepted %':<12} | {'Below q_off %':<15} | {'Ambiguous %':<12}")
    print("-" * 75)

    for q_cand in [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.635]:
        for up_cand in [20.0, 22.0, 24.0, 26.0, 28.0, 30.0]:
            if up_cand <= q_cand:
                continue
            w_cand = up_cand - q_cand
            # Count accepted
            acc = 0
            below = 0
            ambig = 0
            for d, gap in zip(all_d1, (d2_tensor - d1_tensor).tolist()):
                if d <= q_cand:
                    below += 1
                elif d <= up_cand:
                    # check if d_2 <= up_cand
                    # if d_2 <= up_cand, then 2 classes are feasible!
                    d2_val = d + gap
                    if d2_val <= up_cand:
                        ambig += 1
                    else:
                        acc += 1
            N_tot = len(all_d1)
            print(f"{q_cand:<10.2f} | {up_cand:<10.2f} | {w_cand:<8.2f} | {acc/N_tot*100:<12.2f}% | {below/N_tot*100:<15.2f}% | {ambig/N_tot*100:<12.2f}%")


if __name__ == "__main__":
    main()
