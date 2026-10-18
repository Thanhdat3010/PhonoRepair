#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

mkdir -p logs outputs/reports outputs/receipts docs/paper

# Kill existing benchmark session if any
tmux kill-session -t phonorepair_bench 2>/dev/null || true

# Start fresh tmux session in background
tmux new-session -d -s phonorepair_bench "bash scripts/run_remote_benchmark.sh > logs/master_benchmark.log 2>&1"

echo "[+] Started tmux session 'phonorepair_bench' in background."
echo "[+] Logging to logs/master_benchmark.log"
tmux ls
