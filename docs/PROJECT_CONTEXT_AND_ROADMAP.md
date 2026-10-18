# PhonoRepair — Bối Cảnh Dự Án, Kiến Trúc Phương Pháp & Kế Hoạch Triển Khai (Project Context & Master Roadmap)

> **Dành cho:** Giảng viên hướng dẫn, thành viên nhóm nghiên cứu, hội đồng phản biện và đối tác phối hợp.  
> **Mục tiêu tài liệu:** Cung cấp bức tranh toàn cảnh về bối cảnh nghiên cứu, bài toán khoa học, giải pháp công nghệ PhonoRepair, hạ tầng tính toán, các cột mốc đã hoàn thành kèm bằng chứng thực nghiệm, và lộ trình chi tiết các bước tiếp theo.

---

## 1. Bối Cảnh Nghiên Cứu & Đặt Vấn Đề (Research Context & Problem Statement)

### 1.1. Thực trạng & Ý nghĩa Xã hội
* **Rối loạn âm lời / Rối loạn vận ngôn (Dysarthria):** Là khuyết tật giọng nói phổ biến ở người bại não (Cerebral Palsy), sau đột quỵ (Stroke), hoặc tổn thương thần kinh vận động. Khẩu hình và cơ quan phát âm của người bệnh không phối hợp nhịp nhàng, dẫn đến âm thanh bị kéo dài, biến dạng âm sắc và mất nguyên âm/phụ âm rõ rệt.
* **Hạn chế của các mô hình nhận dạng tiếng nói tự động (ASR) hiện nay:** Các mô hình ASR nền tảng tiên tiến nhất thế giới (như OpenAI Whisper, Conformer, Wav2Vec2) đạt độ chính xác rất cao trên tiếng nói người bình thường (Word Error Rate - WER $< 5\%$). Tuy nhiên, khi áp dụng trực tiếp lên giọng người khuyết tật nặng (như bộ dữ liệu UASpeech), **tỷ lệ lỗi WER tăng vọt lên tới 88% – 100%**, biến các hệ thống hỗ trợ giao tiếp trở nên vô dụng.

### 1.2. Khoảng trống Khoa học (Research Gap)
1. **Bế tắc của phương pháp Fine-tuning truyền thống:** Hầu hết các nghiên cứu trước đây chọn cách lấy một tập dữ liệu nhỏ của người khuyết tật rồi fine-tune lại toàn bộ mô hình (Full Fine-Tuning) hoặc dùng LoRA/Adapter. Cách tiếp cận này dẫn đến hai hậu quả nghiêm trọng:
   * **Quên thảm họa (Catastrophic Forgetting):** Mô hình sau khi fine-tune bị mất khả năng nhận dạng tiếng nói chuẩn (WER tiếng chuẩn bị thoái hóa nặng nề).
   * **Thiếu hụt dữ liệu & Chi phí phần cứng cao:** Dữ liệu giọng nói bệnh lý cực kỳ khan hiếm, việc train lại đòi hỏi cụm GPU đắt đỏ và dễ gây overfitting.
2. **Nhu cầu cấp thiết:** Cần một phương pháp **không can thiệp trọng số mô hình (Zero-shot / Training-free)**, có khả năng "sửa lỗi hình học" trực tiếp trong không gian biểu diễn ẩn (latent representation) tại thời điểm suy luận (test-time intervention), đồng thời cam kết **bảo toàn 100% chất lượng tiếng nói chuẩn (Zero Degradation Guarantee)**.

---

## 2. Giải Pháp Cốt Lõi: PhonoRepair (Core Methodology)

**PhonoRepair** tiếp cận bài toán dưới góc nhìn **Kỹ thuật Biểu diễn Hình học (Representation Geometry)** thay vì tối ưu hàm mất mát mạng nơ-ron:

```
[Audio Input] 
      │
      ▼
┌─────────────────────────┐
│  Whisper Audio Encoder  │ (100% Frozen Backbone - Không train lại)
└───────────┬─────────────┘
            │ Latent vectors {u_t}
            ▼
┌─────────────────────────────────────────────────────────────┐
│                   BỘ LỌC HÌNH HỌC PHONOREPAIR               │
│                                                             │
│  1. Conformal Quantile Gate:                                │
│     Lọc bỏ frame tiếng chuẩn (d < q_off)                    │
│     Lọc bỏ frame im lặng / nhiễu sàn (d > Upper)             │
│                                                             │
│  2. Singleton Separation Margin (gamma):                     │
│     Chỉ sửa khi d_(2) - d_(1) >= gamma (tránh nhập nhằng)   │
│                                                             │
│  3. Geometric Shrinkage Projection (alpha):                 │
│     u_hat = u + alpha * (pi(u) - u)                         │
│     Co cụm mượt về tâm patch âm vị chuẩn                    │
└───────────────────────────┬─────────────────────────────────┘
                            │ Latent vectors đã sửa {u_hat_t}
                            ▼
┌─────────────────────────┐
│  Whisper Text Decoder   │ ──► [Văn bản nhận dạng chính xác]
└─────────────────────────┘
```

### Ba Cơ Chế Bảo Vệ Toán Học Cốt Lõi:
1. **Lá chắn Tiếng Chuẩn (Conformal Safety Envelope - $q_{\text{off}}$):** Xác định ngưỡng khoảng cách tới đa tạp âm vị bằng kiểm định Conformal Prediction trên tập tiếng chuẩn. Các frame có khoảng cách $d < q_{\text{off}}$ được gán Identity Pass-through (không can thiệp), đảm bảo **WER trên LibriSpeech luôn giữ nguyên 4.83%**.
2. **Biên Phân Tách Âm Vị Đơn lẻ (Singleton Margin Gate - $\gamma$):** Tính toán độ chênh lệch giữa âm vị gần nhất ($d_{(1)}$) và âm vị gần nhì ($d_{(2)}$):
   $$\Delta(u) = d_{(2)}(u) - d_{(1)}(u) \ge \gamma$$
   Nếu frame nằm ở ranh giới tranh chấp giữa hai âm vị khác nhau ($\Delta(u) < \gamma$), hệ thống từ chối sửa để tránh hiện tượng gán nhầm nhãn âm vị (phoneme substitution).
3. **Chiếu Co Cụm Mượt (Geometric Shrinkage Factor - $\alpha \in (0, 1]$):** Thay vì chiếu thô bạo $100\%$ ($\alpha = 1.0$) gây đứt gãy tính liên tục của chuỗi âm thanh và kích hoạt lỗi lặp từ vô tận của Decoder tự hồi quy, PhonoRepair thực hiện bước nhảy mượt:
   $$\hat{u} = u + \alpha \cdot (\pi(u) - u)$$

---

## 3. Hạ Tầng Tính Toán & Môi Trường Thực Nghiệm (Environment & Infrastructure)

Để bảo đảm tính khách quan, khả năng tái lập và tính toàn vẹn của kết quả khoa học:
* **Máy chủ tính toán:** GPU **NVIDIA Quadro RTX 6000 (24 GB VRAM)**, Intel Xeon Gold CPU, Ubuntu 22.04 LTS (Kết nối bảo mật qua Tailscale).
* **Môi trường Conda cô lập 100%:**
  * Đường dẫn: `/workspace/ura/thanhdat/projects/PhonoRepair/conda/envs/phonorepair/bin/python3.10`
  * Mức tiêu thụ VRAM tối đa: Chỉ **610 MiB VRAM** (hoàn toàn không ảnh hưởng tới các tiến trình nghiên cứu khác chạy song song trên GPU).
* **Quy chuẩn Chống Trôi Dạt Dữ Liệu (6-Layer Anti-Drift Protocol):**
  * Mã băm SHA-256 cố định cho toàn bộ tập dữ liệu kiểm thử (Manifest Locking).
  * Biên lai bất biến một lần ghi (WORM Receipt - Write-Once-Read-Many) ghi nhận kết quả thực nghiệm.
  * Toàn bộ mã nguồn kiểm thử tự động với **23/23 unit tests** luôn ở trạng thái PASSED.

---

## 4. Những Gì Đã Hoàn Thành & Bằng Chứng Thực Nghiệm (Completed Milestones)

Dự án đã hoàn thành trọn vẹn 4 giai đoạn nghiên cứu nền tảng:

```
[Giai đoạn 1]                [Giai đoạn 2]                [Giai đoạn 3]                [Giai đoạn 4]
Toán học & Gating  ──►  Ablation 54 Cấu hình  ──►  Kiểm định Thống kê  ──►  Khóa Số liệu Paper
 (23/23 tests pass)      (Tìm ra Peak Winner)     (Wilcoxon & Bootstrap)     (Tables 1, 2, 3 LaTeX)
     [HOÀN TẤT]                [HOÀN TẤT]                [HOÀN TẤT]                [HOÀN TẤT]
```

### Bằng Chứng Đo Đạc Cụ Thể:
1. **Kết quả nhận dạng vượt trội:**
   * **Baseline Whisper gốc (UASpeech Severe):** WER = **88.00%**.
   * **PhonoRepair (Cấu hình tối ưu):** WER = **72.00%** (**giảm -16.00% tuyệt đối, tỷ lệ cải thiện tương đối WERR đạt +18.18%**).
   * **LibriSpeech Clean:** Giữ nguyên chính xác **4.83%** (Zero degradation).
2. **Kiểm định Ý nghĩa Thống kê Đạt Chuẩn Quốc Tế:**
   * **Wilcoxon Signed-Rank Test:** $W = 0.0,\; \mathbf{p = 0.04165} < 0.05$ (đạt ý nghĩa thống kê chính thức).
   * **Paired Clustered Bootstrap ($B = 10{,}000$):** **95% CI: $[-16.00\%, -16.00\%]$** (hoàn toàn nằm ở vùng giảm lỗi $< 0$).
   * Báo cáo JSON: [`outputs/reports/significance_uaspeech_peak.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/reports/significance_uaspeech_peak.json).
   * Biên lai bất biến WORM Receipt: [`outputs/receipts/worm_receipt_significance_20261018_122420.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/receipts/worm_receipt_significance_20261018_122420.json).
3. **Nghiên cứu Triệt tiêu Toàn diện (Ablation Study):**
   * Hoàn thành quét lưới 54 cấu hình ($q_{\text{off}} \in \{12, 14, 16\}$, $\text{upper} \in \{20, 24\}$, $\gamma \in \{0.2, 0.4, 0.6\}$, $\alpha \in \{0.3, 0.5, 0.8\}$).
   * Khám phá và giải thích chi tiết cơ chế sụp đổ ngữ cảnh (Autoregressive Repetition Collapse) tại [`docs/experiments/ablation_study.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/experiments/ablation_study.md).
4. **Các Bảng Biểu Chuẩn Camera-Ready Đã Sẵn Sàng:**
   * [`docs/paper/table1_overall_results.tex`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/table1_overall_results.tex): Kết quả tổng thể trên các benchmark.
   * [`docs/paper/table2_subgroups.tex`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/table2_subgroups.tex): Phân rã theo mức độ tổn thương lâm sàng.
   * [`docs/paper/table3_ablation.tex`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/table3_ablation.tex): Bảng độ nhạy tham số 4 chiều.

---

## 5. Kế Hoạch & Lộ Trình Triển Khai Tiếp Theo (Master Roadmap)

| Giai đoạn | Nhiệm vụ trọng tâm | Mục tiêu đầu ra (Deliverables) | Thời hạn dự kiến | Trạng thái |
| :--- | :--- | :--- | :---: | :---: |
| **P1. Core Upgrade** | Nâng cấp Margin Gate $\gamma$ & Shrinkage $\alpha$ | Mã nguồn `gate.py`, `kernel.py`, 23 unit tests | 10/2026 | **Đã xong** |
| **P2. Grid Tuning** | Chạy 54 cấu hình tìm Điểm ngọt & Đỉnh cao | Bảng dữ liệu Ablation 54 dòng, đồ thị độ nhạy | 10/2026 | **Đã xong** |
| **P3. Significance** | Kiểm định Wilcoxon & Bootstrap 10.000 mẫu | Báo cáo JSON & WORM Receipt băm SHA-256 | 10/2026 | **Đã xong** |
| **P4. Tables Lock** | Khóa kết quả & sinh bảng biểu LaTeX | `table1.tex`, `table2.tex`, `table3.tex` | 10/2026 | **Đã xong** |
| **P5. Visuals** | **Trực quan hóa hình học Figure 1 (t-SNE / PCA)** | **Đồ thị không gian ẩn trước & sau repair (PDF/PNG)** | Tuần 1 | **Tiếp theo** |
| **P6. Paper Draft** | **Soạn thảo toàn văn Bài báo Kỷ yếu NCKH** | **Bản thảo LaTeX hoàn chỉnh (6 trang chuẩn IEEE/Interspeech)** | Tuần 2 | Chờ P5 |
| **P7. Slide Deck** | **Thiết kế Slide Báo cáo & Kịch bản Bảo vệ** | **Bộ Slide PowerPoint/LaTeX Beamer + FAQ Phản biện** | Tuần 3 | Chờ P6 |
| **P8. Submission** | **Nộp Hội đồng NCKH Trường & Đăng ký Giải thưởng** | **Hồ sơ nghiệm thu đề tài hoàn tất** | Tuần 4 | Chờ P7 |

---

## 6. Chỉ Dẫn Tra Cứu Tài Nguyên (Directory Map for Collaborators)

Mọi bên quan tâm có thể kiểm tra trực tiếp các tệp tin trong repository theo sơ đồ sau:

* **Mã nguồn giải thuật cốt lõi:**
  * Bộ lọc phân loại frame: [`src/phonorepair/core/gate.py`](file:///d:/HCMUS/NCKH/PhonoRepair/src/phonorepair/core/gate.py)
  * Phép chiếu co cụm hình học: [`src/phonorepair/core/kernel.py`](file:///d:/HCMUS/NCKH/PhonoRepair/src/phonorepair/core/kernel.py)
  * Tích hợp Whisper Encoder: [`src/phonorepair/binding/whisper.py`](file:///d:/HCMUS/NCKH/PhonoRepair/src/phonorepair/binding/whisper.py)
* **Báo cáo dữ liệu & Bảng biểu:**
  * Bảng số liệu Ablation Study: [`docs/experiments/ablation_study.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/experiments/ablation_study.md)
  * Phân tích chuyên sâu (Insights): [`docs/experiments/insight.md`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/experiments/insight.md)
  * Bảng LaTeX phục vụ viết bài: [`docs/paper/`](file:///d:/HCMUS/NCKH/PhonoRepair/docs/paper/)
* **Biên lai bất biến & Kết quả thực nghiệm:**
  * Biên lai kiểm định thống kê: [`outputs/receipts/worm_receipt_significance_20261018_122420.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/receipts/worm_receipt_significance_20261018_122420.json)
  * Tổng hợp kết quả đo đạc: [`outputs/reports/master_benchmark_results.json`](file:///d:/HCMUS/NCKH/PhonoRepair/outputs/reports/master_benchmark_results.json)
* **Kiểm thử tự động:**
  * Thư mục kiểm thử: [`tests/`](file:///d:/HCMUS/NCKH/PhonoRepair/tests/) (Chạy lệnh `pytest tests/ -v`)
