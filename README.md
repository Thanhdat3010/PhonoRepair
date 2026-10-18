# PhonoRepair: Clean-Only Geometric Latent Repair across Frozen Speech Recognition Backbones

Framework nghiên cứu can thiệp hình học affine cục bộ tại đầu ra encoder của các mô hình ASR đóng băng (**Whisper**, **Conformer**, **Zipformer**) cho tiếng nói bệnh (**Dysarthric Speech**), hoàn toàn không học từ dữ liệu bệnh và bảo vệ nguyên vẹn tiếng nói chuẩn.

---

## 1. Cấu trúc Dự án

```text
PhonoRepair/
├── configs/
│   ├── shared_policy.yaml          # Tham số Policy v7.1 chuẩn hóa
│   ├── locked_run.json             # Protocol và seed cố định
│   └── bindings/                   # Cấu hình bindings: Whisper, Conformer, Zipformer
├── src/phonorepair/
│   ├── core/                       # Toán hình học, SVD, patch projection, gate width w, kernel
│   ├── binding/                    # Hook trích xuất/thay thế tensor encoder output
│   ├── artifacts/                  # Farthest-first clustering, builder, conformal calibrator
│   ├── evaluation/                 # Normalizer v1, WER (S/D/I), bootstrap CI, Table 1 export
│   └── utils/                      # Hashing seed 17, bundle locking, audio I/O
├── tests/                          # Suite unit tests E0 toán học & identity contract
├── scripts/                        # Script chạy local Windows và runner Colab/RTX 6000
├── pyproject.toml
└── requirements.txt
```

---

## 2. Cài đặt Môi trường

```bash
# Kích hoạt môi trường conda đã tạo
conda activate phonorepair

# Cài đặt package dạng editable
pip install -e .
```

---

## 3. Các Lệnh CLI Chính

```bash
# 1. Chuẩn bị splits và manifests dữ liệu sạch (Seed 17)
python -m phonorepair.cli prepare-source --config configs/shared_policy.yaml --output-dir data/manifests --max-speakers-pilot 50 --lock-checksums

# 2. Khóa bundle mã nguồn chống trôi dạt kết quả (Anti-Drift Lock)
python -m phonorepair.cli lock-bundle --code-dir src/phonorepair --output outputs/locks/bundle_manifest.json

# 3. Kiểm định hợp đồng bảo vệ baseline E0 trên Whisper
python -m phonorepair.cli verify-identity --model whisper --contract configs/bindings/whisper_base.yaml

# 4. Xây dựng artifact dictionary sạch
python -m phonorepair.cli build-artifact --model whisper --output outputs/artifacts/whisper_base_artifact.pt

# 5. Đo tính khả thi của Gate (Đo bề rộng w và q_cal trên speaker blocks)
python -m phonorepair.cli pilot-feasibility --artifact outputs/artifacts/whisper_base_artifact.pt --report outputs/reports/w_feasibility_report.json

# 6. Đánh giá benchmark và xuất bảng kết quả Table 1 (Interspeech)
python -m phonorepair.cli evaluate --locked-run configs/locked_run.json --artifact outputs/artifacts/whisper_base_artifact.pt --suite clean --export-table-latex
```

---

## 4. Bộ Kiểm thử Độc lập

```bash
# Chạy toàn bộ unit tests E0
pytest tests/ -v
```
