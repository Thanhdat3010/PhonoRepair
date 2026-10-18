import json
import os
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from phonorepair.artifacts.schema import load_artifact
from phonorepair.evaluation.normalizer import normalize_text_v1
from phonorepair.evaluation.metrics import compute_wer_counts, compute_corpus_wer
from phonorepair.core.geometry import read_coordinates, lift_coordinates, project_to_patch, compute_patch_distance, compute_all_patch_distances
from phonorepair.core.gate import aggregate_class_distances


def test_tuning_on_gpu():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute device: {device}")

    artifact = load_artifact("outputs/artifacts/whisper_base_artifact.pt")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en").to(device)

    # 1. Load 50 real UASpeech samples (Very Low / Low intelligibility)
    records = []
    with open("data/manifests/uaspeech_eval_full.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("intelligibility") in ["Very low", "Low"]:
                records.append(r)
            if len(records) >= 50:
                break

    print(f"[*] Loaded {len(records)} dysarthric utterances from UASpeech (Very low / Low).")

    # Pre-extract encoder features and references
    eval_items = []
    for r in records:
        audio, sr = sf.read(r["audio_path"])
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        ref_norm = normalize_text_v1(r["text"])
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features.to(device)
        eval_items.append((feat, ref_norm))

    # Evaluate Native Baseline
    base_counts = []
    base_hyps = []
    with torch.no_grad():
        for feat, ref in eval_items:
            out_ids = model.generate(feat, temperature=0.0, num_beams=1)
            hyp = normalize_text_v1(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
            base_hyps.append(hyp)
            base_counts.append(compute_wer_counts(ref.split(), hyp.split()))

    base_wer = compute_corpus_wer(base_counts) * 100.0
    print(f"\n[+] Native Baseline WER: {base_wer:.2f}%")

    # Test grid of repair parameters:
    # q_off candidates, upper_bound candidates, margin gamma candidates, alpha blending candidates
    Q = artifact.Q.to(device)
    mu = artifact.mu.to(device)
    anchors = artifact.anchors.to(device)
    bases = [b.to(device) for b in artifact.bases]
    radii = artifact.radii.to(device)
    patch_labels = artifact.patch_labels.to(device)
    num_classes = artifact.num_classes

    from phonorepair.binding.whisper import WhisperBinding
    binding = WhisperBinding()

    # Define repair hook function generator
    def run_repair_experiment(q_off_val, upper_val, margin_val, alpha_blend):
        rep_counts = []
        total_accepted = 0
        total_frames = 0

        def repair_callback(h_seq: torch.Tensor) -> torch.Tensor:
            nonlocal total_accepted, total_frames
            # h_seq is shape (B, T, d)
            B, T, d = h_seq.shape
            total_frames += B * T
            h_flat = h_seq.reshape(-1, d)
            u_flat = read_coordinates(h_flat, Q, mu)

            patch_dists = compute_all_patch_distances(u_flat, anchors, bases, radii)
            class_dists, d1, d2, _ = aggregate_class_distances(patch_dists, patch_labels, num_classes)

            # Upgraded routing condition:
            # 1. d1 > q_off_val (needs correction)
            # 2. d1 <= upper_val (within reach)
            # 3. Margin check: (d2 - d1) >= margin_val
            needs_corr = (d1 > q_off_val) & (d1 <= upper_val)
            gap = d2 - d1
            unambiguous = gap >= margin_val

            accept_mask = needs_corr & unambiguous
            num_acc = int(accept_mask.sum().item())
            total_accepted += num_acc

            if num_acc == 0:
                return h_seq

            # Project accepted frames
            accepted_indices = torch.nonzero(accept_mask, as_tuple=True)[0]
            # Find best class for each
            best_classes = torch.argmin(class_dists[accepted_indices], dim=-1)

            h_out_flat = h_flat.clone()
            for idx_in_acc, flat_idx in enumerate(accepted_indices):
                c_star = int(best_classes[idx_in_acc].item())
                u_t = u_flat[flat_idx]
                h_t = h_flat[flat_idx]

                class_patch_indices = torch.nonzero(patch_labels == c_star, as_tuple=True)[0]
                best_d = float("inf")
                best_p = None
                for p_idx in class_patch_indices:
                    j = int(p_idx.item())
                    a_j = anchors[j]
                    B_j = bases[j]
                    rho_j = float(radii[j].item())
                    dist_j = compute_patch_distance(u_t, a_j, B_j, rho_j).item()
                    if dist_j < best_d:
                        best_d = dist_j
                        proj_j, _ = project_to_patch(u_t, a_j, B_j, rho_j)
                        best_p = proj_j

                if best_p is not None:
                    # Apply alpha blending
                    target_p = u_t + alpha_blend * (best_p - u_t)
                    h_hat = lift_coordinates(target_p, u_t, h_t, Q)
                    h_out_flat[flat_idx] = h_hat

            return h_out_flat.reshape(B, T, d)

        # Attach hook using official WhisperBinding
        binding.attach_hook(model, repair_callback)

        with torch.no_grad():
            for feat, ref in eval_items:
                out_ids = model.generate(feat, temperature=0.0, num_beams=1)
                hyp = normalize_text_v1(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
                rep_counts.append(compute_wer_counts(ref.split(), hyp.split()))

        binding.detach_hook()
        rep_wer = compute_corpus_wer(rep_counts) * 100.0
        werr = ((base_wer - rep_wer) / base_wer * 100.0) if base_wer > 0 else 0.0
        acc_pct = (total_accepted / max(1, total_frames)) * 100.0
        return rep_wer, werr, acc_pct

    # Run Grid Search
    print("\n" + "=" * 70)
    print("   RUNNING PARAMETER TUNING GRID ON DYSARTHRIC SAMPLES")
    print("=" * 70)
    print(f"{'q_off':<8} | {'upper':<8} | {'margin':<8} | {'alpha':<6} | {'Rep WER %':<10} | {'WERR %':<10} | {'Accepted %':<12}")
    print("-" * 75)

    best_werr = -999.0
    best_config = None

    for q_off in [12.0, 14.0, 16.0]:
        for upper in [20.0, 24.0]:
            for margin in [0.2, 0.4, 0.6]:
                for alpha in [0.3, 0.5, 0.8]:
                    rep_wer, werr, acc_pct = run_repair_experiment(q_off, upper, margin, alpha)
                    print(f"{q_off:<8.1f} | {upper:<8.1f} | {margin:<8.2f} | {alpha:<6.2f} | {rep_wer:<10.2f} | {werr:<+10.2f} | {acc_pct:<12.2f}%")
                    if werr > best_werr:
                        best_werr = werr
                        best_config = (q_off, upper, margin, alpha, rep_wer, werr, acc_pct)

    print("\n" + "=" * 70)
    print(f"[+] BEST CONFIGURATION FOUND:")
    print(f"    q_off: {best_config[0]}, upper: {best_config[1]}, margin: {best_config[2]}, alpha: {best_config[3]}")
    print(f"    Baseline WER: {base_wer:.2f}% -> Repaired WER: {best_config[4]:.2f}% (WERR: {best_config[5]:+.2f}%)")
    print(f"    Accepted Frames: {best_config[6]:.2f}%")
    print("=" * 70)


if __name__ == "__main__":
    test_tuning_on_gpu()
