import json
import torch
import soundfile as sf
import numpy as np
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from phonorepair.artifacts.schema import load_artifact
from phonorepair.evaluation.normalizer import normalize_text_v1
from phonorepair.evaluation.metrics import compute_wer_counts, compute_corpus_wer
from phonorepair.core.geometry import read_coordinates, lift_coordinates, project_to_patch, compute_patch_distance, compute_all_patch_distances
from phonorepair.core.gate import aggregate_class_distances
from phonorepair.binding.whisper import WhisperBinding


def run_clean_check():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute device: {device}")

    artifact = load_artifact("outputs/artifacts/whisper_base_artifact.pt")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en").to(device)

    # Load 100 clean LibriSpeech test-clean samples
    records = []
    with open("data/manifests/librispeech_test_clean_full.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
            if len(records) >= 100:
                break

    print(f"[*] Loaded {len(records)} clean utterances from LibriSpeech test-clean.")

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
    with torch.no_grad():
        for feat, ref in eval_items:
            out_ids = model.generate(feat, temperature=0.0, num_beams=1)
            hyp = normalize_text_v1(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
            base_counts.append(compute_wer_counts(ref.split(), hyp.split()))

    base_wer = compute_corpus_wer(base_counts) * 100.0
    print(f"[+] Clean Native Baseline WER: {base_wer:.2f}%\n")

    Q = artifact.Q.to(device)
    mu = artifact.mu.to(device)
    anchors = artifact.anchors.to(device)
    bases = [b.to(device) for b in artifact.bases]
    radii = artifact.radii.to(device)
    patch_labels = artifact.patch_labels.to(device)
    num_classes = artifact.num_classes

    binding = WhisperBinding()

    configs = [
        ("Config A (12.0, 20.0, 0.2, 0.3)", 12.0, 20.0, 0.20, 0.30),
        ("Config B (14.0, 20.0, 0.2, 0.3)", 14.0, 20.0, 0.20, 0.30),
        ("Config C (14.0, 24.0, 0.2, 0.8)", 14.0, 24.0, 0.20, 0.80),
        ("Config D (14.0, 20.0, 0.4, 0.3)", 14.0, 20.0, 0.40, 0.30),
    ]

    print(f"{'Config Name':<35} | {'Clean WER %':<12} | {'Delta WER %':<12} | {'Accepted %':<12}")
    print("-" * 75)

    for name, q_off_val, upper_val, margin_val, alpha_blend in configs:
        rep_counts = []
        total_accepted = 0
        total_frames = 0

        def repair_callback(h_seq: torch.Tensor) -> torch.Tensor:
            nonlocal total_accepted, total_frames
            B, T, d = h_seq.shape
            total_frames += B * T
            h_flat = h_seq.reshape(-1, d)
            u_flat = read_coordinates(h_flat, Q, mu)

            patch_dists = compute_all_patch_distances(u_flat, anchors, bases, radii)
            class_dists, d1, d2, _ = aggregate_class_distances(patch_dists, patch_labels, num_classes)

            needs_corr = (d1 > q_off_val) & (d1 <= upper_val)
            gap = d2 - d1
            unambiguous = gap >= margin_val

            accept_mask = needs_corr & unambiguous
            num_acc = int(accept_mask.sum().item())
            total_accepted += num_acc

            if num_acc == 0:
                return h_seq

            accepted_indices = torch.nonzero(accept_mask, as_tuple=True)[0]
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
                    target_p = u_t + alpha_blend * (best_p - u_t)
                    h_hat = lift_coordinates(target_p, u_t, h_t, Q)
                    h_out_flat[flat_idx] = h_hat

            return h_out_flat.reshape(B, T, d)

        binding.attach_hook(model, repair_callback)

        with torch.no_grad():
            for feat, ref in eval_items:
                out_ids = model.generate(feat, temperature=0.0, num_beams=1)
                hyp = normalize_text_v1(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
                rep_counts.append(compute_wer_counts(ref.split(), hyp.split()))

        binding.detach_hook()
        rep_wer = compute_corpus_wer(rep_counts) * 100.0
        delta_wer = rep_wer - base_wer
        acc_pct = (total_accepted / max(1, total_frames)) * 100.0
        print(f"{name:<35} | {rep_wer:<12.2f} | {delta_wer:<+12.2f} | {acc_pct:<12.2f}%")


if __name__ == "__main__":
    run_clean_check()
