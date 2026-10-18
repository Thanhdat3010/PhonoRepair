"""Run E6 Acoustic Corruption & E10 Latent Perturbation Experiments on Whisper base.en.

Generates empirical evidence for Table 1 (Acoustic Corruption) & Table 2 (Mechanism Diagnostic).
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
from phonorepair.evaluation.perturbation import add_acoustic_noise


def run_experiment(
    manifest_path: str = "data/manifests/test_clean.jsonl",
    artifact_path: str = "outputs/artifacts/whisper_base_artifact.pt",
    output_report: str = "outputs/reports/synthetic_corruption_report.json",
):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[*] Running Corruption Experiment on device: {device}")

    # 1. Load model & artifact
    processor = WhisperProcessor.from_pretrained("openai/whisper-base.en")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base.en").to(device)
    model.eval()

    artifact = load_artifact(artifact_path).to(device)
    binding = WhisperBinding()

    # 2. Load test-clean manifest
    with open(manifest_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    print(f"[*] Loaded {len(samples)} test utterances.")

    # Conditions to test:
    # 1. Clean (SNR = inf)
    # 2. SNR 20 dB (mild noise)
    # 3. SNR 10 dB (moderate noise)
    # 4. SNR 5 dB (strong noise)
    conditions = [
        {"name": "Clean (No Corruption)", "snr": None},
        {"name": "Acoustic Noise (20 dB)", "snr": 20.0},
        {"name": "Acoustic Noise (10 dB)", "snr": 10.0},
        {"name": "Acoustic Noise (5 dB)", "snr": 5.0},
    ]

    all_results = []

    for cond in conditions:
        cond_name = cond["name"]
        snr_db = cond["snr"]
        print(f"\n==================================================")
        print(f"  EVALUATING CONDITION: {cond_name}")
        print(f"==================================================")

        baseline_hyps = []
        repaired_hyps = []
        refs = []
        rejection_stats = {}

        def make_repair_cb(art):
            def callback(features: torch.Tensor) -> torch.Tensor:
                B, T, D = features.shape
                out = features.clone()
                for b in range(B):
                    h_rep, reasons, _ = repair_sequence(features[b], art)
                    out[b] = h_rep
                    for r in reasons:
                        rejection_stats[r] = rejection_stats.get(r, 0) + 1
                return out
            return callback

        repair_cb = make_repair_cb(artifact)

        for s_idx, sample in enumerate(samples):
            wav_path = sample["audio_path"]
            ref_text = normalize_text_v1(sample["text"])
            refs.append(ref_text)

            audio_data, sr = sf.read(wav_path)
            if snr_db is not None:
                audio_data = add_acoustic_noise(audio_data, snr_db, seed=17 + s_idx)

            inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to(device)

            # Baseline Whisper
            with torch.no_grad():
                out_base = model.generate(input_features)
                base_text = normalize_text_v1(processor.batch_decode(out_base, skip_special_tokens=True)[0])
                baseline_hyps.append(base_text)

            # PhonoRepair Hooked Whisper
            binding.attach_hook(model, repair_cb)
            with torch.no_grad():
                out_rep = model.generate(input_features)
                rep_text = normalize_text_v1(processor.batch_decode(out_rep, skip_special_tokens=True)[0])
                repaired_hyps.append(rep_text)
            binding.detach_hook()

        # Metrics computation
        ref_tokens = (" ".join(refs)).split()
        base_tokens = (" ".join(baseline_hyps)).split()
        rep_tokens = (" ".join(repaired_hyps)).split()

        counts_base = compute_wer_counts(ref_tokens, base_tokens)
        counts_rep = compute_wer_counts(ref_tokens, rep_tokens)

        wer_base = counts_base.wer * 100.0
        wer_rep = counts_rep.wer * 100.0
        delta_wer = wer_rep - wer_base

        if wer_base > 0:
            werr = ((wer_base - wer_rep) / wer_base) * 100.0
        else:
            werr = 0.0

        total_frames = sum(rejection_stats.values())
        accepted_frames = rejection_stats.get("accepted", 0)
        acceptance_rate = (accepted_frames / max(1, total_frames)) * 100.0

        # Utterance level changes
        improved_utts = 0
        degraded_utts = 0
        identical_utts = 0

        for r, b, rep in zip(refs, baseline_hyps, repaired_hyps):
            r_t = r.split()
            err_b = compute_wer_counts(r_t, b.split()).total_errors
            err_rep = compute_wer_counts(r_t, rep.split()).total_errors
            if err_rep < err_b:
                improved_utts += 1
            elif err_rep > err_b:
                degraded_utts += 1
            else:
                identical_utts += 1

        print(f"  Baseline WER:      {wer_base:.2f}%")
        print(f"  PhonoRepair WER:   {wer_rep:.2f}%")
        print(f"  Delta WER:         {delta_wer:+.2f}%")
        print(f"  Rel. WERR:         {werr:+.2f}%")
        print(f"  Acceptance Rate:   {acceptance_rate:.2f}% ({accepted_frames}/{total_frames} frames)")
        print(f"  Gate Breakdown:    {rejection_stats}")
        print(f"  Utterance Changes: Improved={improved_utts}, Degraded={degraded_utts}, Identical={identical_utts}")

        cond_result = {
            "condition": cond_name,
            "snr_db": snr_db,
            "baseline_wer": round(wer_base, 2),
            "phonorepair_wer": round(wer_rep, 2),
            "delta_wer": round(delta_wer, 2),
            "relative_werr": round(werr, 2),
            "acceptance_rate_percent": round(acceptance_rate, 2),
            "accepted_frames": accepted_frames,
            "total_frames": total_frames,
            "improved_utterances": improved_utts,
            "degraded_utterances": degraded_utts,
            "identical_utterances": identical_utts,
            "gate_reasons": rejection_stats,
        }
        all_results.append(cond_result)

    # Save summary report
    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[+] Full experiment report saved to: {output_report}")
    return all_results


if __name__ == "__main__":
    run_experiment()
