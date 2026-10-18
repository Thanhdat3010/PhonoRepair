#!/usr/bin/env bash
# ==============================================================================
# Script: scripts/verify_server_env.sh
# Purpose: Pre-flight smoke test on RTX 6000 server for PhonoRepair
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "================================================================="
echo "   PHONOREPAIR: REMOTE GPU SERVER PRE-FLIGHT VERIFICATION"
echo "================================================================="

# Source environment
if [ -f "environment.sh" ]; then
    echo "[*] Sourcing environment.sh..."
    source environment.sh
fi

echo "[*] Project Root: $PWD"
echo "[*] Python Executable: $(command -v python)"
python -V

# GPU & PyTorch verification
echo -e "\n[*] GPU and PyTorch Status:"
python -c "
import torch
print(f'    PyTorch Version: {torch.__version__}')
print(f'    CUDA Available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'    Device Name:     {torch.cuda.get_device_name(0)}')
    print(f'    Device Count:    {torch.cuda.device_count()}')
    vram_free = torch.cuda.mem_get_info()[0] / (1024**3)
    vram_total = torch.cuda.mem_get_info()[1] / (1024**3)
    print(f'    VRAM:            {vram_free:.2f} GB free / {vram_total:.2f} GB total')
"

# Verify bundle lock
echo -e "\n[*] Verifying code bundle integrity lock..."
python -m phonorepair.cli verify-lock --code-dir src/phonorepair --manifest outputs/locks/bundle_manifest.json

# Run unit tests
echo -e "\n[*] Executing unit tests (22 tests)..."
pytest tests/ -v

# Smoke paired evaluation on clean manifest
echo -e "\n[*] Executing smoke paired evaluation on test_clean.jsonl (first 3 samples)..."
python -m phonorepair.cli evaluate \
    --manifest data/manifests/test_clean.jsonl \
    --artifact outputs/artifacts/whisper_base_artifact.pt \
    --max-samples 3 \
    --device cuda:0

echo -e "\n================================================================="
echo "[+] ALL PRE-FLIGHT CHECKS PASSED SUCCESSFULLY ON GPU SERVER!"
echo "================================================================="
