#!/usr/bin/env bash
# PhonoRepair Remote Execution Runner (Colab / RTX 6000)
set -euo pipefail

echo "=== PHONOREPAIR: REMOTE PILOT SUITE ==="

# 1. Prepare data splits
echo "[1/4] Preparing deterministic data manifests (Seed 17)..."
python -m phonorepair.cli prepare-source --config configs/shared_policy.yaml --output-dir data/manifests --max-speakers-pilot 50 --lock-checksums

# 2. Build clean artifact
echo "[2/4] Building clean dictionary artifact..."
python -m phonorepair.cli build-artifact --model whisper --policy configs/shared_policy.yaml --output outputs/artifacts/whisper_base_artifact.pt

# 3. Pilot gate feasibility
echo "[3/4] Measuring gate score width w and calibration..."
python -m phonorepair.cli pilot-feasibility --artifact outputs/artifacts/whisper_base_artifact.pt --report outputs/reports/w_feasibility_report.json

# 4. Evaluate on clean speech
echo "[4/4] Evaluating clean speech baseline protection..."
python -m phonorepair.cli evaluate --locked-run configs/locked_run.json --artifact outputs/artifacts/whisper_base_artifact.pt --suite clean --export-table-latex

echo "[+] ALL PILOT STEPS COMPLETED!"
