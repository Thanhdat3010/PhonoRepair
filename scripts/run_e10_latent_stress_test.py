"""E10 Synthetic Latent Perturbation & Repair Stress Test.

Implements Proposal Section 12 (E2/E10 Synthetic mechanism pilot):
- Applies phonetic latent displacement xi_t = kappa * nu to clean speech representations.
- Compares Baseline Whisper (unrepaired corrupted latent) vs PhonoRepair (geometrically projected latent).
- Evaluates Frame Acceptance Rate, Corrected Utterances, Harmed Utterances, and WER improvement.
"""

import json
import os
import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from phonorepair.binding.whisper import WhisperBinding
from phonorepair.core.kernel import repair_sequence
from phonorepair.artifacts.schema import load_artifact
from phonorepair.evaluation.normalizer import normalize_text_v1
from phonorepair.evaluation.metrics import compute_wer_counts
from phonorepair.evaluation.perturbation import perturb_latent_sequence


def run_e10_experiment(
    manifest_path: str = "data/manifests/test_clean.jsonl",
    artifact_path: str = "outputs/artifacts/whisper_base_artifact.pt",
    output_report: str = "outputs/reports/e10_latent_stress_report.json",
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running E10 Latent Perturbation Experiment on: {device}")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").to(device)
    model.eval()

    artifact = load_artifact(artifact_path).to(device)
    binding = WhisperBinding()

    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    print(f"[*] Loaded {len(samples)} test utterances.")
    print(f"[*] Artifact: eta={artifact.eta:.4f}, nu={artifact.nu:.4f}, q_off={artifact.q_off:.4f}, w={artifact.w:.4f}")

    # Perturbation strengths kappa:
    # kappa=0.0: clean baseline
    # kappa=1.0: within threat budget
    # kappa=2.0: moderate perturbation
    # kappa=3.0: strong perturbation
    kappa_levels = [0.0, 1.0, 2.0, 3.0]
    results = []

    for kappa in kappa_levels:
        print(f"\n==================================================")
        print(f"  E10 PERTURBATION STRENGTH: kappa = {kappa} (displacement = {kappa * artifact.nu:.2f})")
        print(f"==================================================")

        baseline_corrupt_hyps = []
        repaired_hyps = []
        refs = []
        rejection_stats = {}

        for s_idx, sample in enumerate(samples):
            wav_path = sample["audio_path"]
            cache_path = sample["cache_path"]
            ref_text = normalize_text_v1(sample["text"])
            refs.append(ref_text)

            # Load true frame labels from cache
            cache_data = torch.load(cache_path, weights_only=False)
            frame_labels = torch.tensor(cache_data["frame_labels"], device=device)

            audio_data, sr = sf.read(wav_path)
            inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            # 1. Baseline with corrupted latent
            # We apply perturbation hook that corrupts latent without repair
            def make_corrupt_hook(k):
                def hook(module, args, output):
                    feat = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
                    if k > 0:
                        corrupted = perturb_latent_sequence(feat[0], frame_labels, artifact, kappa=k, direction_mode="random")
                        feat = corrupted.unsqueeze(0)
                    if hasattr(output, "last_hidden_state"):
                        output.last_hidden_state = feat
                        return output
                    return feat
                return hook

            # 2. Repair hook that first applies perturbation, then applies PhonoRepair
            def make_repair_hook(k):
                def hook(module, args, output):
                    feat = output.last_hidden_state if hasattr(output, "last_hidden_state") else output
                    if k > 0:
                        corrupted = perturb_latent_sequence(feat[0], frame_labels, artifact, kappa=k, direction_mode="random")
                    else:
                        corrupted = feat[0]
                    # Apply PhonoRepair on the corrupted latent
                    repaired, reasons, _ = repair_sequence(corrupted, artifact)
                    for r in reasons:
                        rejection_stats[r] = rejection_stats.get(r, 0) + 1
                    repaired_batch = repaired.unsqueeze(0)
                    if hasattr(output, "last_hidden_state"):
                        output.last_hidden_state = repaired_batch
                        return output
                    return repaired_batch
                return hook

            # Run baseline corrupted Whisper
            handle_corrupt = model.model.encoder.register_forward_hook(make_corrupt_hook(kappa))
            with torch.no_grad():
                out_corrupt = model.generate(input_features)
                text_corrupt = normalize_text_v1(processor.batch_decode(out_corrupt, skip_special_tokens=True)[0])
                baseline_corrupt_hyps.append(text_corrupt)
            handle_corrupt.remove()

            # Run PhonoRepair on corrupted Whisper
            handle_repair = model.model.encoder.register_forward_hook(make_repair_hook(kappa))
            with torch.no_grad():
                out_rep = model.generate(input_features)
                text_rep = normalize_text_v1(processor.batch_decode(out_rep, skip_special_tokens=True)[0])
                repaired_hyps.append(text_rep)
            handle_repair.remove()

        # Compute WER
        ref_tokens = (" ".join(refs)).split()
        base_tokens = (" ".join(baseline_corrupt_hyps)).split()
        rep_tokens = (" ".join(repaired_hyps)).split()

        counts_base = compute_wer_counts(ref_tokens, base_tokens)
        counts_rep = compute_wer_counts(ref_tokens, rep_tokens)

        wer_base = counts_base.wer * 100.0
        wer_rep = counts_rep.wer * 100.0
        delta_wer = wer_rep - wer_base
        werr = ((wer_base - wer_rep) / max(0.01, wer_base)) * 100.0 if wer_base > 0 else 0.0

        total_f = sum(rejection_stats.values())
        acc_f = rejection_stats.get("accepted", 0)
        acc_rate = (acc_f / max(1, total_f)) * 100.0

        print(f"  Unrepaired Corrupted WER: {wer_base:.2f}%")
        print(f"  PhonoRepair Repaired WER: {wer_rep:.2f}%")
        print(f"  WER Reduction:            {delta_wer:+.2f}% (Rel. WERR: {werr:+.2f}%)")
        print(f"  Acceptance Rate:          {acc_rate:.2f}% ({acc_f}/{total_f} frames)")
        print(f"  Gate Breakdown:           {rejection_stats}")

        results.append({
            "kappa": kappa,
            "displacement": round(kappa * artifact.nu, 2),
            "unrepaired_wer": round(wer_base, 2),
            "repaired_wer": round(wer_rep, 2),
            "delta_wer": round(delta_wer, 2),
            "relative_werr": round(werr, 2),
            "acceptance_rate": round(acc_rate, 2),
            "accepted_frames": acc_f,
            "total_frames": total_f,
            "gate_reasons": rejection_stats,
        })

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] E10 Stress Test Report saved to: {output_report}")
    return results


if __name__ == "__main__":
    run_e10_experiment()
