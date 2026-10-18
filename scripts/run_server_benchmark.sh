#!/usr/bin/env bash
# ==============================================================================
# PHONOREPAIR: 1-CLICK SERVER BENCHMARK ORCHESTRATOR
# Target Environment: RTX 6000 Ada / A100 / Colab GPU
# Conforms to Proposal v7.1 and Universal Anti-Drift Protocol (v1.0)
# ==============================================================================
set -euo pipefail

echo "================================================================="
echo "   PHONOREPAIR: FULL BENCHMARK ORCHESTRATION PIPELINE"
echo "   Submission Target: Interspeech 2027 (4 Pages + References)"
echo "================================================================="

# ------------------------------------------------------------------------------
# STEP 0: PRE-FLIGHT ENVIRONMENT & INTEGRITY AUDIT
# ------------------------------------------------------------------------------
echo -e "\n[*] [STEP 0/5] Auditing environment and code integrity..."
python -c "import torch; print(f'    PyTorch: {torch.__version__} | CUDA: {torch.version.cuda} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

# Verify code bundle SHA256 integrity
echo "    Verifying code bundle against bundle_manifest.json..."
python -m phonorepair.cli verify-lock --code-dir src/phonorepair --manifest outputs/locks/bundle_manifest.json

# Run complete unit test suite (22/22 tests)
echo "    Running pre-flight unit tests..."
pytest tests/ -q

echo "[+] Step 0 PASSED: Environment and code integrity verified."

# ------------------------------------------------------------------------------
# STEP 1: CLEAN MANIFESTS PREPARATION (SEED 17)
# ------------------------------------------------------------------------------
echo -e "\n[*] [STEP 1/5] Partitioning clean source manifests (LibriSpeech)..."
python -m phonorepair.cli prepare-source \
    --config configs/shared_policy.yaml \
    --output-dir data/manifests \
    --max-speakers-pilot 100 \
    --split-seed 17 \
    --lock-checksums

# ------------------------------------------------------------------------------
# STEP 2: BUILD CLEAN ARTIFACT & CONFORMAL CALIBRATION
# ------------------------------------------------------------------------------
echo -e "\n[*] [STEP 2/5] Building clean dictionary artifact and calibrating gate..."
python -m phonorepair.cli build-artifact \
    --model whisper \
    --policy configs/shared_policy.yaml \
    --basis-manifest data/manifests/librispeech_basis_fit.jsonl \
    --dict-manifest data/manifests/librispeech_dict_fit.jsonl \
    --output outputs/artifacts/whisper_base_artifact.pt

python -m phonorepair.cli pilot-feasibility \
    --artifact outputs/artifacts/whisper_base_artifact.pt \
    --selection-manifest data/manifests/librispeech_selection.jsonl \
    --calibration-manifest data/manifests/librispeech_calibration.jsonl \
    --report outputs/reports/w_feasibility_report.json

# ------------------------------------------------------------------------------
# STEP 3: E0 CLEAN SPEECH BASELINE VERIFICATION
# ------------------------------------------------------------------------------
echo -e "\n[*] [STEP 3/5] Verifying clean baseline protection (Delta WER <= 0.00%)..."
python -m phonorepair.cli evaluate \
    --suite clean \
    --manifest data/manifests/test_clean.jsonl \
    --artifact outputs/artifacts/whisper_base_artifact.pt \
    --model whisper \
    --export-table-latex

# ------------------------------------------------------------------------------
# STEP 4: TARGET DYSARTHRIC EVALUATION (UASPEECH & TORGO)
# ------------------------------------------------------------------------------
echo -e "\n[*] [STEP 4/5] Checking for target evaluation datasets..."

# UASpeech Evaluation
if [ -f "data/manifests/uaspeech.jsonl" ]; then
    echo "    Running blind benchmark on UASpeech (Word-level, M5 channel)..."
    python -m phonorepair.cli evaluate \
        --suite disease \
        --manifest data/manifests/uaspeech.jsonl \
        --artifact outputs/artifacts/whisper_base_artifact.pt \
        --model whisper \
        --receipt outputs/receipts/worm_receipt_uaspeech.json \
        --export-table-latex
else
    echo "    [i] Info: data/manifests/uaspeech.jsonl not found. (To scan: python scripts/prepare_target_datasets.py --corpus uaspeech --data-dir <path>)"
fi

# TORGO Evaluation
if [ -f "data/manifests/torgo.jsonl" ]; then
    echo "    Running blind benchmark on TORGO (Sentence-level, headMic)..."
    python -m phonorepair.cli evaluate \
        --suite disease \
        --manifest data/manifests/torgo.jsonl \
        --artifact outputs/artifacts/whisper_base_artifact.pt \
        --model whisper \
        --receipt outputs/receipts/worm_receipt_torgo.json \
        --export-table-latex
else
    echo "    [i] Info: data/manifests/torgo.jsonl not found. (To scan: python scripts/prepare_target_datasets.py --corpus torgo --data-dir <path>)"
fi

# ------------------------------------------------------------------------------
# STEP 5: FINAL RECEIPT & LOG AUDIT
# ------------------------------------------------------------------------------
echo -e "\n[*] [STEP 5/5] Finalizing run receipts and audit trail..."
ls -lh outputs/receipts/

echo -e "\n================================================================="
echo "   [+] BENCHMARK EXECUTION COMPLETED SUCCESSFULLY!"
echo "   WORM receipts archived at: outputs/receipts/"
echo "================================================================="
