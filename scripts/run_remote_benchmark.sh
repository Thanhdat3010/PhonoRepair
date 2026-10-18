#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "==========================================================================="
echo "   PHONOREPAIR RTX 6000 BENCHMARK EXECUTION SCRIPT"
echo "==========================================================================="
echo "Current directory: $(pwd)"
echo "Host: $(hostname)"
echo "Date: $(date -u)"

# 1. Source environment
if [ -f "environment.sh" ]; then
    echo "[*] Sourcing environment.sh..."
    source environment.sh
fi
export PYTHONUNBUFFERED=1

# 2. Check Python and PyTorch
echo "[*] Python executable: $(which python)"
python -c "import sys, torch; print(f'Python {sys.version.split()[0]} | PyTorch {torch.__version__} | CUDA available: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 3. Create logs directory
mkdir -p logs outputs/reports outputs/receipts docs/paper

# 4. Run master benchmark
echo "[*] Starting Master Interspeech 2027 Benchmark..."
python scripts/run_full_paper_benchmarks.py "$@"

echo "==========================================================================="
echo "   BENCHMARK PIPELINE EXECUTION COMPLETED SUCCESSFULLY!"
echo "==========================================================================="
