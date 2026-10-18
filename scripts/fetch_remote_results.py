import base64
import os
import subprocess
import tarfile

def main():
    cmd = [
        r"C:\Program Files\Tailscale\tailscale.exe",
        "ssh",
        "ura@100.90.53.12",
        "cd /workspace/ura/thanhdat/projects/PhonoRepair && base64 -w 0 results.tar.gz"
    ]
    print("[*] Fetching results.tar.gz from RTX 6000 server...")
    res = subprocess.check_output(cmd)
    b64_str = res.decode("utf-8").strip()
    tar_bytes = base64.b64decode(b64_str)
    
    with open("results.tar.gz", "wb") as f:
        f.write(tar_bytes)
    print(f"[+] Downloaded results.tar.gz ({len(tar_bytes)} bytes)")
    
    with tarfile.open("results.tar.gz", "r:gz") as tar:
        tar.extractall(".")
    
    print("[+] Successfully extracted all remote benchmark deliverables locally:")
    for path in ["docs/paper", "outputs/reports", "outputs/receipts"]:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    print(f"  - {os.path.join(root, f)}")

if __name__ == "__main__":
    main()
