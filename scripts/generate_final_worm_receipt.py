import json
import os
from phonorepair.utils.locking import write_worm_receipt

def main():
    manifest_path = "data/manifests/uaspeech_eval_full.jsonl"
    code_dir = "src/phonorepair"
    receipts_dir = "outputs/receipts"

    with open("outputs/reports/significance_uaspeech_peak.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    receipt_path = write_worm_receipt(manifest_path, code_dir, payload, receipts_dir)
    print(f"[+] Final WORM Receipt successfully created: {receipt_path}")

if __name__ == "__main__":
    main()
