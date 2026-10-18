"""Unit tests for EvaluationRunner and paired benchmark execution."""

import os
import json
import tempfile
import numpy as np
import pytest
import soundfile as sf
import torch

from phonorepair.evaluation.runner import EvaluationRunner


def test_evaluation_runner_provenance():
    """Verify that provenance collection returns valid cryptographic hashes."""
    runner = EvaluationRunner(
        model_id="whisper",
        artifact_path="outputs/artifacts/whisper_base_artifact.pt",
        device="cpu",
    )
    with tempfile.NamedTemporaryFile(suffix=".jsonl", mode="w", delete=False) as f:
        f.write('{"utterance_id": "u1", "audio_path": "a.wav", "text": "hello"}\n')
        tmp_manifest = f.name

    try:
        prov = runner._collect_provenance(tmp_manifest)
        assert "git_commit" in prov
        assert "code_bundle_sha256" in prov
        assert len(prov["code_bundle_sha256"]) == 64
        assert "platform" in prov
        assert "torch_version" in prov
    finally:
        if os.path.exists(tmp_manifest):
            os.remove(tmp_manifest)


def test_evaluation_runner_mock_manifest():
    """Test paired evaluation loop on a minimal synthetic audio manifest."""
    sr = 16000
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    dummy_wav = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    with tempfile.TemporaryDirectory() as td:
        wav_path = os.path.join(td, "test_utt.wav")
        sf.write(wav_path, dummy_wav, sr, subtype="PCM_16")

        manifest_path = os.path.join(td, "eval_mock.jsonl")
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({
                "utterance_id": "test_001",
                "speaker_id": "spk01",
                "audio_path": wav_path,
                "text": "hello world",
                "intelligibility": "Mid",
                "task_type": "sentence",
            }) + "\n")

        receipt_path = os.path.join(td, "receipt.json")

        runner = EvaluationRunner(
            model_id="whisper",
            artifact_path="outputs/artifacts/whisper_base_artifact.pt",
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )

        summary = runner.run_manifest(
            manifest_path=manifest_path,
            output_receipt_path=receipt_path,
            max_samples=1,
        )

        assert os.path.exists(receipt_path)
        assert summary["overall_metrics"]["num_utterances"] == 1
        assert "Mid" in summary["subgroups"]
        assert "gate_diagnostics" in summary
