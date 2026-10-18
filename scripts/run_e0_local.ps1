# PhonoRepair Local E0 Check Suite (Windows PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "=== PHONOREPAIR: E0 LOCAL VERIFICATION SUITE ===" -ForegroundColor Cyan

# 1. Generate toy test cases
Write-Host "`n[1/4] Generating E0 Toy Geometry Arrays..." -ForegroundColor Yellow
python data/processed/toy/generate_e0_cases.py

# 2. Run PyTest Unit Tests
Write-Host "`n[2/4] Running PyTest Unit Tests..." -ForegroundColor Yellow
python -m pytest tests/ -v

# 3. Lock Code Bundle
Write-Host "`n[3/4] Generating Execution Bundle Lock..." -ForegroundColor Yellow
python -m phonorepair.cli lock-bundle --code-dir src/phonorepair --output outputs/locks/bundle_manifest.json

# 4. Verify Bundle Lock Integrity
Write-Host "`n[4/4] Verifying Bundle Integrity..." -ForegroundColor Yellow
python -m phonorepair.cli verify-lock --code-dir src/phonorepair --manifest outputs/locks/bundle_manifest.json

Write-Host "`n[+] E0 LOCAL SUITE COMPLETED SUCCESSFULLY!" -ForegroundColor Green
