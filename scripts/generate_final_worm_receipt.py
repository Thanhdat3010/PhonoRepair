import datetime
import json
import os
from phonorepair.utils.locking import write_worm_receipt, compute_file_sha256

def main():
    timestamp_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    receipt_path = f"outputs/receipts/worm_receipt_significance_{timestamp_str}.json"

    with open("outputs/reports/significance_uaspeech_peak.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    manifest_path = "data/manifests/uaspeech_eval_full.jsonl"
    receipt_data = {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "manifest_evaluated": manifest_path,
        "manifest_sha256": compute_file_sha256(manifest_path),
        "payload": payload,
    }

    write_worm_receipt(receipt_data, receipt_path)
    print(f"[+] Final WORM Receipt successfully created: {receipt_path}")

if __name__ == "__main__":
    main()
