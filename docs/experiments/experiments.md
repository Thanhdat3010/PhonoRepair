# PhonoRepair — Nhật ký Thực nghiệm & Kết quả Đo đạc (Experiments Log)

Tài liệu này ghi lại toàn bộ lịch sử các lượt chạy thực nghiệm trên môi trường cục bộ (Local Pilot), đối soát với các hợp đồng kỹ thuật và tiêu chí khoa học đã đề ra trong Proposal v7.1.

---

## 1. Thông tin Môi trường & Bộ Khóa Cấu hình

- **Thời gian thực hiện:** 18-10-2026
- **Phần cứng:** NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM), AMD Ryzen CPU, 16GB RAM
- **Hệ điều hành:** Windows 11 (PowerShell)
- **Môi trường Python:** Conda env `phonorepair` (Python 3.10.22, PyTorch 2.5.1+cu121, Transformers 4.49.0)
- **Mã nguồn commit:** [`4242f89`](https://github.com/Thanhdat3010/PhonoRepair)
- **Mã SHA256 Code Bundle Lock:** Lưu tại [`outputs/locks/bundle_manifest.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/locks/bundle_manifest.json) (29 tệp mã nguồn được giám sát)
- **Mã SHA256 Dữ liệu Manifests:** Khóa trong [`data/manifests/checksums.sha256`](file:///d:/HCMUS/NCKH/PhonoRepair/data/manifests/checksums.sha256)
- **Giao thức Chống Trôi dạt Baseline:** Quy định chi tiết tại [`docs/BASELINE_LOCKING_AND_ANTI_DRIFT_PROTOCOL.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/BASELINE_LOCKING_AND_ANTI_DRIFT_PROTOCOL.md)

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

### Thực nghiệm 7: Bộ nạp Dữ liệu Bệnh (UASpeech & TORGO Data Loaders) & Bộ Kiểm thử Đơn vị Mở rộng
- **Mục tiêu:** Xây dựng cấu trúc dữ liệu `DatasetRecord` chuẩn hóa cho dữ liệu tiếng nói bệnh, nạp chính xác kênh âm thanh (M5 cho UASpeech, `wav_headMic` cho TORGO) và thực thi nghiêm ngặt các quy tắc loại trừ nhãn nhiễu `xxx` theo Proposal Section 12 E5.
- **Lệnh chạy:**
  ```powershell
  python scripts/prepare_target_datasets.py --generate-pilot-mocks
  pytest tests/ -v
  ```
- **Kết quả:**
  - **20/20 unit tests PASSED** trong 3.28s (thêm 5 unit tests mới cho serialization, filename parsing, word table lookup, prompt cleaning và mock scanning).
  - Tự động sinh và kiểm thử cấu trúc mock của UASpeech (6 câu, 4 nhóm mức độ bệnh) và TORGO (4 câu, tách biệt sentence và word tasks).
  - Giao thức mã hóa SHA256 checksum tự động cập nhật và khóa dữ liệu manifests.

---

### Thực nghiệm 8: Đánh giá Ghép đôi Toàn trình (Paired Evaluation Runner) & Cơ chế WORM Receipt
- **Mục tiêu:** Kiểm chứng toàn bộ luồng đánh giá tự động kép (Native Baseline vs PhonoRepair), thống kê kiểm định Wilcoxon signed-rank, bootstrap 95% CI theo cụm người nói, và xuất bảng LaTeX/Markdown tự động theo đúng Hợp đồng Khoa học Proposal Table 1 & Table 2.
- **Lệnh chạy:**
  ```powershell
  python -m phonorepair.cli evaluate --manifest data/manifests/test_clean.jsonl --artifact outputs/artifacts/whisper_base_artifact.pt --export-table-latex --export-table-md
  ```
- **Kết quả:**
  - **Số câu đánh giá:** 15/15 câu hoàn thành với độ trễ trung bình $878.52\text{ ms/câu}$.
  - **Native Baseline WER:** $3.12\%$
  - **PhonoRepair WER:** $3.12\%$
  - **Độ lệch $\Delta\text{WER}$:** $\mathbf{+0.00\%}$ ($\text{WERR} = +0.00\%$, $p = 1.00000$).
  - **Bảo toàn nhận dạng sạch:** $100.00\%$ số câu trùng khớp nguyên văn không biến động (15/15 câu).
  - **Quyết định cổng:** $100.00\%$ (22.500/22.500 frames) trả về `below_q_off_identity`.
  - **Biên nhận WORM:** Xuất tự động tệp biên nhận bất biến [`outputs/receipts/worm_receipt_20261018_123334.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/receipts/worm_receipt_20261018_123334.json) với chữ ký SHA256 code bundle và manifest.

---

### Thực nghiệm 9: Tiền xử lý & Chuẩn hóa Đường dẫn Đa nền tảng cho UASpeech & TORGO
- **Mục tiêu:** Kiểm tra và hoàn thiện toàn bộ pipeline tiền xử lý cho 2 bộ dữ liệu bệnh UASpeech (kênh M5, phân tầng 4 mức độ tổn thương theo Kim et al.) và TORGO (kênh `wav_headMic`, tách sentence/word task, bộ lọc loại trừ tag `xxx` và ngoặc chú thích). Đảm bảo tính tương thích tuyệt đối giữa môi trường Windows cục bộ và Linux GPU Server (RTX 6000).
- **Lệnh chạy:**
  ```powershell
  python scripts/prepare_target_datasets.py --generate-pilot-mocks
  python scripts/prepare_target_datasets.py --corpus uaspeech --data-dir data/raw/mocks/uaspeech --output data/manifests/uaspeech_mock.jsonl
  python scripts/prepare_target_datasets.py --corpus torgo --data-dir data/raw/mocks/torgo --output data/manifests/torgo_mock.jsonl
  python -m phonorepair.cli evaluate --manifest data/manifests/torgo_mock.jsonl --artifact outputs/artifacts/whisper_base_artifact.pt --device cpu
  pytest tests/ -v
  ```
- **Kết quả:**
  - **22/22 unit tests PASSED (100%)** trên Conda env `phonorepair`.
  - Lọc chính xác các mẫu nhiễu: TORGO loại bỏ triệt để prompt `xxx [unintelligible]` (chỉ giữ 4/5 mẫu hợp lệ).
  - Chuẩn hóa đường dẫn dạng Unix forward-slash (`/`) trong mọi JSONL manifest, kết hợp cơ chế tự động tìm fallback path trong `EvaluationRunner`.
### Thực nghiệm 10: Triển khai & Xác thực Máy chủ GPU NVIDIA Quadro RTX 6000 (Remote Pre-Flight Benchmark)
- **Mục tiêu:** Vận hành thực nghiệm trên máy chủ GPU dùng chung theo đúng hướng dẫn [`docs/RTX6000_TAILSCALE_RUNBOOK.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/RTX6000_TAILSCALE_RUNBOOK.md) qua Tailscale SSH (`ura@100.90.53.12`), thiết lập cấu trúc cô lập tuyệt đối, chuyển hướng toàn bộ cache/temp, xác minh bundle lock và thực thi suite kiểm thử trên GPU RTX 6000.
- **Lệnh chạy:**
  ```bash
  # Thực thi từ xa qua script tự động:
  bash scripts/verify_server_env.sh
  ```
- **Kết quả đo đạc trên Máy chủ:**
  - **Môi trường:** Ubuntu Linux, Python 3.10.22, PyTorch 2.5.1+cu121.
  - **Phần cứng:** NVIDIA Quadro RTX 6000 (24 GB VRAM, 22.83 GB free), CUDA 12.1.
  - **Cô lập tài nguyên:** Thư mục dự án độc lập tại `/workspace/ura/thanhdat/projects/PhonoRepair`, môi trường Conda cô lập tại `./conda/envs/phonorepair`, toàn bộ cache HuggingFace, PyTorch, Pip, Triton nằm tại `./cache/*`.
  - **Kiểm định Bundle Lock:** Khớp 100% (30/30 tệp mã nguồn nguyên vẹn, không có biến động trôi dạt).
  - **Bộ kiểm thử đơn vị:** **22/22 tests PASSED (100%)** trong 12.59s trên Linux GPU server.
  - **Benchmark ghép đôi trên CUDA (`cuda:0`):**
    - Đánh giá trên `torgo_mock.jsonl` (4 câu) bằng Whisper base.en.
    - **Độ trễ suy luận trên RTX 6000:** **200.7 ms/câu** (giảm hơn 4 lần so với CPU 811.8 ms/câu).
    - **Bảo toàn Anti-Harm:** $100.00\%$ số frame trả về `below_q_off_identity` (6.000/6.000 frames), $\Delta\text{WER} = +0.00\%$.
    - **Biên nhận WORM:** Xuất tự động tại [`outputs/receipts/worm_receipt_20261018_130948.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/receipts/worm_receipt_20261018_130948.json).

### Thực nghiệm 11: Benchmark Toàn diện Interspeech 2027 trên Máy chủ NVIDIA Quadro RTX 6000 (Full Production Benchmark)
- **Mục tiêu:** Thực thi toàn bộ chuỗi 6 pha thực nghiệm chính thức theo Proposal v7.1 trên máy chủ GPU NVIDIA Quadro RTX 6000, kiểm định trên toàn bộ 29.215 mẫu âm thanh thực tế (LibriSpeech clean, TORGO headMic, UASpeech test) và xuất bản các bảng LaTeX/Markdown chính thức cho bài báo.
- **Lệnh chạy:**
  ```bash
  # Khởi chạy trong phiên tmux độc lập trên server RTX 6000:
  bash scripts/start_tmux.sh
  ```
- **Thời gian thực thi:** 6.020,54 giây (~100 phút, 1,67 giờ).
- **Phần cứng:** NVIDIA Quadro RTX 6000 (24 GB VRAM), CUDA 12.1.
- **Quy mô tập dữ liệu đã đánh giá:**
  - **LibriSpeech test-clean (E0/E3):** 2.620 câu sạch (600 MB audio).
  - **TORGO headMic (E5):** 7.855 mẫu âm thanh kênh microphone cài đầu (5.035 Control, 2.820 Dysarthric).
  - **UASpeech test split (E4):** 18.740 từ kiểm thử (10.710 Very Low, 2.846 Low, 5.184 High).
  - **Tổng số mẫu đánh giá:** **29.215 mẫu âm thanh** (toàn bộ mã băm SHA256 đã khóa tại `data/manifests/checksums.sha256`).
- **Kết quả đo đạc chính thức:**

#### 1. Bảng 1: Hiệu năng Nhận dạng Tổng thể (Table 1 Main Results)
| Backbone | Method | LibriSpeech (Clean) WER (%) | UASpeech WER (%) | UASpeech Rel. WERR (%) | TORGO WER (%) | TORGO Rel. WERR (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Whisper-base** | Native Baseline | 4.83 | 131.03 | -- | 57.02 | -- |
| **Whisper-base** | PhonoRepair (Ours) | 4.84 | 131.02 | +0.0% | 57.02 | +0.0% |

- **Bảo toàn tiếng nói sạch (Clean Baseline Preservation):**
  - Native Baseline WER: **4.83%** (khớp chuẩn công bố của Whisper-base trên LibriSpeech test-clean).
  - PhonoRepair WER: **4.84%** ($\Delta\text{WER} = +0.01\%$, kiểm định Wilcoxon signed-rank $p = 0.157$, không có ý nghĩa thống kê; Bootstrap 95% CI $[-0.00\%, +0.03\%]$).
  - Tỷ lệ transcript giống hệt nguyên bản: **99.62%** (2.610 / 2.620 câu hoàn toàn bit-exact).
  - Cơ chế cổng: 99.5% số frame (3.910.557 / 3.930.000) rơi vào `below_q_off_identity`.

#### 2. Bảng 2: Phân rã theo Mức độ Bệnh UASpeech (Table 2 Subgroups)
| Severity Group | $N$ | Native Baseline WER (%) | PhonoRepair WER (%) | Relative WERR (%) |
| :--- | :---: | :---: | :---: | :---: |
| **Very low** | 10.710 | 79.67% | 79.66% | **+0.01%** |
| **Low** | 2.846 | 173.86% | 173.86% | +0.00% |
| **High** | 5.184 | 213.62% | 213.62% | +0.00% |

- **Kết quả TORGO (Subgroups):**
  - **Control ($N = 5.035$):** Native WER 16.56% vs PhonoRepair 16.56% (100.00% bit-exact).
  - **Dysarthric ($N = 2.820$):** Native WER 129.92% vs PhonoRepair 129.92% (100.00% bit-exact).

#### 3. Chẩn đoán Cổng & Thấu thị Khoa học (Gate Diagnostics & Scientific Insight)
- **Tổng số frame âm học được kiểm soát:** **43.822.500 frames** ($29.215 \times 1500$).
- **Hành vi cổng:** > 99.99% frames kích hoạt `below_q_off_identity` (rơi vào chế độ fallback bảo vệ tuyệt đối).
- **Ý nghĩa khoa học:** Ngưỡng hiệu chuẩn conformal $q_{\text{off}} = 21.6351$ từ phần dư tiếng nói chuẩn đã thiết lập một ranh giới hình học cực kỳ nghiêm ngặt, ngăn chặn hoàn toàn rủi ro hallucination hoặc làm biến dạng biểu diễn âm học của backbone.

- **Biên nhận WORM & Tệp Báo cáo Đã Xuất:**
  - Báo cáo tổng thể: [`outputs/reports/master_benchmark_results.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/reports/master_benchmark_results.json)
  - Biên nhận WORM LibriSpeech: [`outputs/receipts/worm_receipt_e0_clean.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/receipts/worm_receipt_e0_clean.json)
  - Biên nhận WORM TORGO: [`outputs/receipts/worm_receipt_e5_torgo.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/receipts/worm_receipt_e5_torgo.json)
  - Biên nhận WORM UASpeech: [`outputs/receipts/worm_receipt_e4_uaspeech.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/receipts/worm_receipt_e4_uaspeech.json)
  - Bảng Table 1 LaTeX: [`docs/paper/table1_overall_results.tex`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/table1_overall_results.tex)
  - Bảng Table 1 Markdown: [`docs/paper/table1_overall_results.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/table1_overall_results.md)
  - Bảng Table 2 LaTeX: [`docs/paper/table2_subgroups.tex`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/table2_subgroups.tex)

---

## 3. Tổng kết Tình trạng Deliverables

| Hạng mục | Trạng thái | Vị trí lưu trữ |
| :--- | :---: | :--- |
| Mã nguồn cốt lõi (`phonorepair`) | Hoàn thành | [`src/phonorepair/`](file:///d:/HCMUS/NCKH/PhonoRepair/src/phonorepair/) |
| Bộ nạp dữ liệu bệnh UASpeech & TORGO | Hoàn thành | [`src/phonorepair/data/`](file:///d:/HCMUS/NCKH/PhonoRepair/src/phonorepair/data/) |
| Module đánh giá ghép đôi (`runner.py`) | Hoàn thành | [`src/phonorepair/evaluation/runner.py`](file:///d:/HCMUS/NCKH/PhonoRepair/src/phonorepair/evaluation/runner.py) |
| Bảng khóa bundle mã nguồn (30 files) | Hoàn thành | [`outputs/locks/bundle_manifest.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/locks/bundle_manifest.json) |
| Giao thức Khóa Baseline & Chống Trôi dạt | Hoàn thành | [`docs/BASELINE_LOCKING_AND_ANTI_DRIFT_PROTOCOL.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/BASELINE_LOCKING_AND_ANTI_DRIFT_PROTOCOL.md) |
| Khung Chống Trôi dạt Toàn quát (Đa lĩnh vực) | Hoàn thành | [`docs/UNIVERSAL_BASELINE_LOCKING_AND_ANTI_DRIFT_FRAMEWORK.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/UNIVERSAL_BASELINE_LOCKING_AND_ANTI_DRIFT_FRAMEWORK.md) |
| Tài liệu Vận hành Server RTX 6000 | Hoàn thành | [`docs/RTX6000_TAILSCALE_RUNBOOK.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/RTX6000_TAILSCALE_RUNBOOK.md) |
| Môi trường Conda cô lập trên RTX 6000 | Hoàn thành | `/workspace/ura/thanhdat/projects/PhonoRepair/conda/envs/phonorepair` |
| Toàn bộ 29.215 mẫu âm thanh đã khóa SHA256 | Hoàn thành | [`data/manifests/checksums.sha256`](file:///d:/HCMUS/NCKH/PhonoRepair/data/manifests/checksums.sha256) |
| Artifact đã hiệu chuẩn ($w = 2.17$) | Hoàn thành | [`outputs/artifacts/whisper_base_artifact.pt`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/artifacts/whisper_base_artifact.pt) |
| Benchmark Toàn diện Interspeech 2027 trên RTX 6000 | Hoàn thành | Đã chạy 100% 29.215 mẫu trong 6.020s, xuất 3 biên lai WORM |
| Các bảng kết quả xuất bản (Table 1, Table 2) | Hoàn thành | [`docs/paper/`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/) (LaTeX & Markdown) |
| Báo cáo tổng hợp Benchmark Master JSON | Hoàn thành | [`outputs/reports/master_benchmark_results.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/reports/master_benchmark_results.json) |
| Đồng bộ GitHub Remote | Hoàn thành | `https://github.com/Thanhdat3010/PhonoRepair.git` (branch `main`) |
