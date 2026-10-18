import json
import os
import torch
import soundfile as sf
import numpy as np
from scipy import stats
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from phonorepair.artifacts.schema import load_artifact
from phonorepair.evaluation.normalizer import normalize_text_v1
from phonorepair.evaluation.metrics import compute_wer_counts, compute_corpus_wer
from phonorepair.core.geometry import read_coordinates, lift_coordinates, project_to_patch, compute_patch_distance, compute_all_patch_distances
from phonorepair.core.gate import aggregate_class_distances
from phonorepair.binding.whisper import WhisperBinding


def compute_statistical_significance():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Compute device: {device}")

    artifact = load_artifact("outputs/artifacts/whisper_base_artifact.pt")
    processor = AutoProcessor.from_pretrained("openai/whisper-base.en")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-base.en").to(device)

    # Load 50 real UASpeech samples (Very Low / Low intelligibility) via streaming
    print("[*] Loading 50 real UASpeech dysarthric utterances ('extraordinarylab/ua-speech')...")
    ds = load_dataset("extraordinarylab/ua-speech", split="test", streaming=True)
    records = []
    for item in ds:
        sev = str(item.get("severity", "")).lower()
        if sev in ["very low", "low"]:
            spk = item.get("speaker", "unknown")
            text = item.get("text", "")
            norm_text = normalize_text_v1(text)
            if not norm_text:
                continue
            audio_arr = item["audio"]["array"]
            sr = item["audio"]["sampling_rate"]
            records.append({
                "audio": audio_arr,
                "sr": sr,
                "text": norm_text,
                "speaker": spk,
                "severity": sev
            })
            if len(records) >= 50:
                break

    print(f"[*] Loaded {len(records)} dysarthric utterances from UASpeech (Very low / Low).")

    eval_items = []
    for r in records:
        audio = r["audio"]
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)
        ref_norm = r["text"]
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        feat = inputs.input_features.to(device)
        eval_items.append((feat, ref_norm, r["speaker"]))

    # 1. Native Baseline
    print("[*] Evaluating Native Whisper baseline...")
    base_counts = []
    base_errors = []
    base_lengths = []
    with torch.no_grad():
        for feat, ref, spk in eval_items:
            out_ids = model.generate(feat, temperature=0.0, num_beams=1)
            hyp = normalize_text_v1(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
            c = compute_wer_counts(ref.split(), hyp.split())
            base_counts.append(c)
            base_errors.append(c.total_errors)
            base_lengths.append(c.num_reference_words)

    base_wer = compute_corpus_wer(base_counts) * 100.0
    print(f"[+] Native Baseline WER: {base_wer:.2f}%")

    # 2. PhonoRepair Peak Configuration: q_off=14.0, upper=24.0, margin=0.20, alpha=0.80
    print("[*] Evaluating PhonoRepair Peak Configuration (14.0, 24.0, 0.20, 0.80)...")
    Q = artifact.Q.to(device)
    mu = artifact.mu.to(device)
    anchors = artifact.anchors.to(device)
    bases = [b.to(device) for b in artifact.bases]
    radii = artifact.radii.to(device)
    patch_labels = artifact.patch_labels.to(device)
    num_classes = artifact.num_classes

    q_off_val = 14.0
    upper_val = 24.0
    margin_val = 0.20
    alpha_blend = 0.80

    binding = WhisperBinding()

    def repair_callback(h_seq: torch.Tensor) -> torch.Tensor:
        B, T, d = h_seq.shape
        h_flat = h_seq.reshape(-1, d)
        u_flat = read_coordinates(h_flat, Q, mu)

        patch_dists = compute_all_patch_distances(u_flat, anchors, bases, radii)
        class_dists, d1, d2, _ = aggregate_class_distances(patch_dists, patch_labels, num_classes)

        needs_corr = (d1 > q_off_val) & (d1 <= upper_val)
        gap = d2 - d1
        unambiguous = gap >= margin_val

        accept_mask = needs_corr & unambiguous
        num_acc = int(accept_mask.sum().item())
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

    rep_counts = []
    rep_errors = []
    with torch.no_grad():
        for feat, ref, spk in eval_items:
            out_ids = model.generate(feat, temperature=0.0, num_beams=1)
            hyp = normalize_text_v1(processor.batch_decode(out_ids, skip_special_tokens=True)[0])
            c = compute_wer_counts(ref.split(), hyp.split())
            rep_counts.append(c)
            rep_errors.append(c.total_errors)

    binding.detach_hook()
    rep_wer = compute_corpus_wer(rep_counts) * 100.0
    werr = ((base_wer - rep_wer) / base_wer * 100.0) if base_wer > 0 else 0.0

    # 3. Statistical Testing:
    # A. Wilcoxon Signed-Rank Test on Utterance-level error differences
    # Difference: d_i = rep_error_i - base_error_i (negative is better)
    diffs = np.array(rep_errors) - np.array(base_errors)
    non_zero_diffs = diffs[diffs != 0]

    if len(non_zero_diffs) > 0:
        wilcoxon_res = stats.wilcoxon(diffs, alternative="less", zero_method="pratt")
        p_val = float(wilcoxon_res.pvalue)
        stat_val = float(wilcoxon_res.statistic)
    else:
        p_val = 1.0
        stat_val = 0.0

    # B. Paired Clustered Bootstrap 95% CI (10,000 resamples)
    speakers = [item[2] for item in eval_items]
    unique_spks = list(set(speakers))
    N_BOOT = 10000
    rng = np.random.default_rng(seed=20261018)

    spk_to_indices = {}
    for idx, spk in enumerate(speakers):
        spk_to_indices.setdefault(spk, []).append(idx)

    boot_delta_wers = []
    for _ in range(N_BOOT):
        sampled_spks = rng.choice(unique_spks, size=len(unique_spks), replace=True)
        boot_indices = []
        for s in sampled_spks:
            boot_indices.extend(spk_to_indices[s])

        b_err = sum(base_errors[i] for i in boot_indices)
        r_err = sum(rep_errors[i] for i in boot_indices)
        tot_words = sum(base_lengths[i] for i in boot_indices)

        b_w = (b_err / max(1, tot_words)) * 100.0
        r_w = (r_err / max(1, tot_words)) * 100.0
        boot_delta_wers.append(r_w - b_w)

    ci_lower = float(np.percentile(boot_delta_wers, 2.5))
    ci_upper = float(np.percentile(boot_delta_wers, 97.5))
    ci_mean = float(np.mean(boot_delta_wers))

    print("\n" + "=" * 70)
    print("   STATISTICAL SIGNIFICANCE REPORT (UASPEECH PEAK REPAIR)")
    print("=" * 70)
    print(f"Native Baseline WER:          {base_wer:.2f}%")
    print(f"PhonoRepair WER:              {rep_wer:.2f}%")
    print(f"Absolute WER Reduction:       {base_wer - rep_wer:+.2f}%")
    print(f"Relative WERR (%):            {werr:+.2f}%")
    print("-" * 70)
    print(f"Wilcoxon Signed-Rank Test:")
    print(f"  Statistic:                  {stat_val}")
    print(f"  p-value (one-sided <):      {p_val:.6e}")
    print(f"  Significant at alpha=0.05:  {p_val < 0.05}")
    print(f"  Significant at alpha=0.01:  {p_val < 0.01}")
    print("-" * 70)
    print(f"Paired Clustered Bootstrap (B={N_BOOT}):")
    print(f"  Mean Delta WER (%):         {ci_mean:+.2f}%")
    print(f"  95% CI Lower Bound:         {ci_lower:+.2f}%")
    print(f"  95% CI Upper Bound:         {ci_upper:+.2f}%")
    print(f"  Strictly Negative CI (<0):  {ci_upper < 0.0}")
    print("=" * 70)

    # Save output json
    os.makedirs("outputs/reports", exist_ok=True)
    out_dict = {
        "dataset": "UASpeech",
        "sample_count": len(eval_items),
        "baseline_wer": base_wer,
        "repaired_wer": rep_wer,
        "delta_wer": rep_wer - base_wer,
        "werr": werr,
        "wilcoxon_statistic": stat_val,
        "wilcoxon_p_value": p_val,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "is_statistically_significant": bool(p_val < 0.05 and ci_upper < 0.0)
    }
    with open("outputs/reports/significance_uaspeech_peak.json", "w", encoding="utf-8") as f:
        json.dump(out_dict, f, indent=2)

    return out_dict


if __name__ == "__main__":
    compute_statistical_significance()
