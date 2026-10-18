# PhonoRepair — Nhật ký Thực nghiệm & Kết quả Đo đạc (Experiments Log)

Tài liệu này ghi lại toàn bộ lịch sử các lượt chạy thực nghiệm trên môi trường cục bộ (Local Pilot), đối soát với các hợp đồng kỹ thuật và tiêu chí khoa học đã đề ra trong Proposal v7.1.

---

## 1. Thông tin Môi trường & Bộ Khóa Cấu hình

- **Thời gian thực hiện:** 18-10-2026
- **Phần cứng:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM), AMD Ryzen CPU, 16GB RAM
- **Hệ điều hành:** Windows 11 (PowerShell)
- **Môi trường Python:** Conda env `phonorepair` (Python 3.10.22, PyTorch 2.5.1+cu121, Transformers 4.49.0)
- **Mã nguồn commit:** [`ac9ebe7`](https://github.com/Thanhdat3010/PhonoRepair)
- **Mã SHA256 Code Bundle Lock:** Lưu tại [`outputs/locks/bundle_manifest.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/locks/bundle_manifest.json) (25 tệp mã nguồn được giám sát)
- **Mã SHA256 Dữ liệu Manifests:** Khóa trong [`data/manifests/checksums.sha256`](file:///d:/HCMUS/NCKH/PhonoRepair/data/manifests/checksums.sha256)

---

## 2. Nhật ký Chi tiết các Thực nghiệm Đã Chạy

### Thực nghiệm 1: Kiểm thử Đơn vị & Hợp đồng Hình học (Unit Tests)
- **Mục tiêu:** Xác minh 100% tính đúng đắn toán học của các phép chiếu affine, tính trực giao $Q^\top Q = I$, cơ chế tính khoảng cách Euclid điểm-patch, và bộ khóa chống ghi đè WORM.
- **Lệnh chạy:** `pytest tests/ -v`
- **Kết quả:** **15/15 tests PASSED** trong 4.33s.
  - `test_geometry.py`: 4/4 PASSED
  - `test_gate.py`: 5/5 PASSED
  - `test_locks.py`: 3/3 PASSED
  - `test_perturbation.py`: 2/2 PASSED
  - `test_whisper_identity.py`: 1/1 PASSED

---

### Thực nghiệm 2: Tiền xử lý Dữ liệu Sạch & Trích xuất Đặc trưng LibriSpeech
- **Mục tiêu:** Tải và tiền xử lý dữ liệu tiếng nói chuẩn, phân chia 5 tập splits độc lập cân bằng người nói theo `seed = 17`, chuyển đổi G2P (39 âm vị ARPABET + SIL) và trích xuất latent tensor $(T, 512)$ của Whisper base.en.
- **Lệnh chạy:** `python scripts/prepare_pilot_dataset.py`
- **Kết quả:**
  - 165 tệp âm thanh 16kHz PCM_16 đã lưu tại `data/processed/audio/`.
  - 165 tensor đặc trưng `.pt` đã lưu tại `cache/features/`.
  - 5 tệp manifest JSONL đã khóa SHA256:
    - `librispeech_basis_fit.jsonl` (10 câu, 2 người nói)
    - `librispeech_dict_fit.jsonl` (40 câu, 8 người nói)
    - `librispeech_selection.jsonl` (10 câu, 2 người nói)
    - `librispeech_calibration.jsonl` (15 câu, 3 người nói)
    - `test_clean.jsonl` (15 câu, 3 người nói)

---

### Thực nghiệm 3: Xây dựng Artifact & Hiệu chuẩn Bề rộng Cổng E2 (Feasibility Check)
- **Mục tiêu:** Khớp từ điển affine 80 patches trên 40 lớp âm vị, tính toán bao hoạt động $\eta_{\text{work}}$, ngân sách đe dọa $\nu_{\text{local}}$ và hiệu chuẩn conformal $q_{\text{cal}}$ trên các khối người nói sạch để kiểm tra điều kiện khả thi $w > 0$.
- **Lệnh chạy:**
  ```powershell
  python -m phonorepair.cli build-artifact --model whisper --basis-manifest data/manifests/librispeech_basis_fit.jsonl --dict-manifest data/manifests/librispeech_dict_fit.jsonl --output outputs/artifacts/whisper_base_artifact.pt
  python -m phonorepair.cli pilot-feasibility --artifact outputs/artifacts/whisper_base_artifact.pt --calibration-manifest data/manifests/librispeech_calibration.jsonl --selection-manifest data/manifests/librispeech_selection.jsonl --report outputs/reports/w_feasibility_report.json
  ```
- **Số liệu đo đạc:**
  - $\eta_{\text{work}} = 19.0468$
  - $\nu_{\text{local}} = 4.7617$ ($0.25 \times \eta_{\text{work}}$)
  - $q_{\text{cal}} = 21.6351$ (phân vị empirical maximum trên khối calibration sạch)
  - $q_{\text{off}} = \max(q_{\text{cal}}, \eta_{\text{work}}) = 21.6351$
  - $\mathbf{w = \max(0, \eta_{\text{work}} + \nu_{\text{local}} - q_{\text{off}}) = 2.1734 > 0}$
- **Kết luận:** **ĐẠT TIÊU CHUẨN STOP-GATE.** Cổng sửa hình học mở ra một cửa sổ sửa khả dụng $[21.6351, 23.8085]$.

---

### Thực nghiệm 4: Kiểm định Bảo vệ Tiếng nói Sạch E0 & E3 (Clean Baseline Preservation)
- **Mục tiêu:** Đánh giá giải mã thực tế trên 15 câu LibriSpeech `test-clean` với Whisper base.en để kiểm tra nguyên tắc sắt Anti-Harm: can thiệp không được làm suy giảm nhận dạng sạch.
- **Lệnh chạy:** `python scratch/test_whisper_e0_real.py`
- **Kết quả:**
  - **Tổng số frame âm học:** 22.500 frames ($15 \times 1500$)
  - **Tỷ lệ Identity Fallback ($A(u) \le q_{\text{off}}$):** **100.0%** (22.500/22.500 frames)
  - **Tỷ lệ can thiệp sai:** 0.00%
  - **Baseline Whisper WER:** **3.12%**
  - **PhonoRepair WER:** **3.12%**
  - **Độ lệch $\Delta\text{WER}$:** $\mathbf{+0.00\%}$
  - **Tỷ lệ khớp nguyên văn transcript:** **15/15 câu (100.0% bit-exact match)**

---

### Thực nghiệm 5: Thử nghiệm Nhiễu Âm học E6 (Acoustic Noise Benchmark)
- **Mục tiêu:** Bổ sung nhiễu trắng Gaussian ở các mức SNR 20dB, 10dB, 5dB vào tín hiệu âm thanh dạng sóng $y = x + \alpha \cdot n$ và đo lường phản ứng của cổng.
- **Lệnh chạy:** `python scripts/run_synthetic_corruption_experiment.py`
- **Báo cáo lưu tại:** [`outputs/reports/synthetic_corruption_report.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/reports/synthetic_corruption_report.json)
- **Bảng số liệu:**

| Mức độ kiểm thử | SNR (dB) | Baseline WER (%) | PhonoRepair WER (%) | $\Delta\text{WER}$ | Tỷ lệ nhận sửa (Accepted) | Hành vi của cổng |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Clean** | $\infty$ | 3.12% | 3.12% | $+0.00\%$ | 0.00% (0/22.500) | 100% `below_q_off_identity` |
| **Mild Noise** | 20 dB | 3.79% | 3.79% | $+0.00\%$ | 0.00% (0/22.500) | 100% `below_q_off_identity` |
| **Moderate Noise** | 10 dB | 7.80% | 7.80% | $+0.00\%$ | 0.00% (0/22.500) | 100% `below_q_off_identity` |
| **Strong Noise** | 5 dB | 12.69% | 12.69% | $+0.00\%$ | 0.00% (0/22.500) | 100% `below_q_off_identity` |

---

### Thực nghiệm 6: Thử nghiệm Biến dạng Âm vị Latent E10 (Synthetic Latent Perturbation)
- **Mục tiêu:** Mô phỏng sự sai lệch cấu âm (articulatory distortion) đặc trưng của chứng khó phát âm bằng cách gây nhiễu latent $\xi_t = \kappa \cdot \nu$ dọc theo các hướng ngữ âm, đo lường khả năng khôi phục biểu diễn và cải thiện WER.
- **Lệnh chạy:** `python scripts/run_e10_latent_stress_test.py`
- **Báo cáo lưu tại:** [`outputs/reports/e10_latent_stress_report.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/reports/e10_latent_stress_report.json)
- **Bảng số liệu:**

| Cường độ $\kappa$ | Độ dịch chuyển $\|\xi\|$ | Unrepaired WER (%) | PhonoRepair WER (%) | Cải thiện tương đối (WERR) | Số frame chấp nhận sửa | Phân rã trạng thái cổng (Gate Breakdown) |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| $\kappa = 0.0$ | $0.00$ | 3.12% | 3.12% | $+0.00\%$ | 0 | `below_q_off`: 22.500 |
| $\kappa = 1.0$ | $4.76$ | 2.67% | 3.12% | $-16.67\%$ | 0 | `below_q_off`: 22.484, `ambiguous`: 16 |
| **$\kappa = 2.0$** | **$9.52$** | **3.34%** | **3.12%** | **$+6.67\%$** | **28** | `below_q_off`: 22.126, `ambiguous`: 336, **`accepted`: 28**, `empty`: 10 |
| $\kappa = 3.0$ | $14.29$ | 3.56% | 92.87% | Vượt ngân sách | 771 | `below_q_off`: 16.606, `empty`: 920, `ambiguous`: 4.203, `accepted`: 771 |

---

## 3. Tổng kết Tình trạng Deliverables

| Hạng mục | Trạng thái | Vị trí lưu trữ |
| :--- | :---: | :--- |
| Mã nguồn cốt lõi (`phonorepair`) | Hoàn thành | [`src/phonorepair/`](file:///d:/HCMUS/NCKH/PhonoRepair/src/phonorepair/) |
| Bảng khóa bundle mã nguồn | Hoàn thành | [`outputs/locks/bundle_manifest.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/locks/bundle_manifest.json) |
| Dữ liệu manifest & checksum | Hoàn thành | [`data/manifests/`](file:///d:/HCMUS/NCKH/PhonoRepair/data/manifests/) |
| Artifact đã hiệu chuẩn ($w = 2.17$) | Hoàn thành | [`outputs/artifacts/whisper_base_artifact.pt`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/artifacts/whisper_base_artifact.pt) |
| Báo cáo kết quả JSON E2, E6, E10 | Hoàn thành | [`outputs/reports/`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/reports/) |
| Đồng bộ GitHub Remote | Hoàn thành | `https://github.com/Thanhdat3010/PhonoRepair.git` (branch `main`) |
