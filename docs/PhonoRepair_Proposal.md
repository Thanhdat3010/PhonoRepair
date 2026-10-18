# PHONOREPAIR
## Proposal triển khai và thực nghiệm: sửa latent chỉ từ dữ liệu sạch cho ASR đóng băng

**Phiên bản:** 7.1 — 17-10-2026; chỉnh protocol sau review, giữ tên file hiện tại.  
**Tên tiếng Anh:** *PhonoRepair: Clean-Only Geometric Latent Repair across Frozen Speech Recognition Backbones*.  
**Định dạng:** tài liệu độc lập; đặc tả triển khai, protocol thực nghiệm và chứng minh trong Appendix A.  
**Trạng thái:** proposal nghiên cứu. Chưa có implementation, artifact đã fit hoặc kết quả benchmark/hardware được báo trong tài liệu này. Mọi số mục tiêu dưới đây là tiêu chí thiết kế đề xuất, không phải số đo.  
**Backbone mục tiêu:** Whisper, Zipformer và Conformer; FastConformer streaming là cấu hình mở rộng thuộc họ Conformer.  
**Ràng buộc:** không dùng audio bệnh, nhãn bệnh, pseudo-label bệnh hoặc dữ liệu mô phỏng bệnh để fit, chọn tham số, calibration hay cập nhật online.

## Mục lục

1. [Tóm tắt và mục tiêu](#s1)
2. [Motivation](#s2)
3. [Related work, novelty và phạm vi claim](#s3)
4. [Câu hỏi nghiên cứu và tiêu chí thành công](#s4)
5. [Bài toán, ký hiệu và giả định](#s5)
6. [Kiến trúc tổng thể và tích hợp backbone](#s6)
7. [Xây artifact chỉ bằng dữ liệu sạch](#s7)
8. [Inference và bảo toàn baseline](#s8)
9. [Yêu cầu phần mềm, tài nguyên và realtime](#s9)
10. [Dữ liệu, checkpoint và quyền truy cập](#s10)
11. [Protocol đánh giá và thống kê chung](#s11)
12. [Thực nghiệm E0–E10](#s12)
13. [Baselines, ablations và các bảng kết quả](#s13)
14. [Quy tắc xử lý kết quả xấu và tránh leakage](#s14)
15. [Lộ trình triển khai và deliverables](#s15)
16. [Claim được phép và giới hạn](#s16)
17. [Appendix A: chứng minh toán học](#app-a)
18. [Appendix B: cấu hình, pseudocode và checklist tái lập](#app-b)
19. [References](#refs)

<a id="s1"></a>
## 1. Tóm tắt và mục tiêu

PhonoRepair là một framework tạo và vận hành adapter hình học tại đầu ra acoustic encoder của một ASR đóng băng. Adapter dùng dictionary affine có nhãn được xây từ tiếng nói sạch; inference không học lại encoder, decoder, dictionary hoặc gate. Một core sửa chung phục vụ các backbone có giao diện tích hợp hợp lệ. Mỗi checkpoint nhận artifact riêng, được tạo theo cùng policy.

Mục tiêu chính là giữ hoặc cải thiện đầu ra nhận dạng. Không cần sinh waveform, tạo âm thanh tự nhiên cho người nghe, hay phục hồi đúng từng tọa độ latent sạch. Tuy nhiên, label của dictionary đúng chưa đủ bảo đảm transcript đúng: phải đánh giá decoder nguyên bản và, nếu tuyên bố certificate đầu ra, xác lập điều kiện quyết định của chính decoder đó.

Luồng online là:

$$
\text{audio}\rightarrow F_b\rightarrow H
\rightarrow R_{\theta_b}(H)\rightarrow G_b\rightarrow \widehat Y,
$$

với encoder $$F_b$$, decoder $$G_b$$ đóng băng; $$R$$ có cùng thuật toán ở mọi backbone, còn $$\theta_b$$ là artifact của checkpoint.

Bốn mục tiêu bắt buộc:

1. **Không học từ bệnh:** toàn bộ onboarding chỉ dùng clean source data.
2. **Dùng chung core:** không phân nhánh theo tên Whisper/Zipformer/Conformer để đổi thuật toán hoặc threshold.
3. **Bảo vệ baseline sạch:** identity đúng ở mức tensor/state; calibration hạn chế can thiệp sạch. Bảo đảm không sửa mọi clean input chỉ được tuyên bố khi có coverage toàn support hợp lệ.
4. **Hữu ích và có ngân sách:** không coi artifact từ chối mọi input là thành công; phải đo WER, correction/harm, repair overhead và memory. Realtime SLA, số phút repeat onboarding và certificate transcript cho mọi decoder là mục tiêu mở rộng.

Phạm vi chính giữ ba backbone W1/Z1/C1. “General” nghĩa là cùng repair core và cùng policy source-only tạo artifact; basis, dictionary và tham số số có thể khác theo checkpoint do policy chọn. Binding chỉ xử lý giao diện, timebase và state. Thành công tích hợp không tự chứng minh hiệu quả recognition xuyên backbone; ba mức kết luận được định nghĩa tại §4.3.

Tài liệu phân biệt ba mức chứng cứ:

| Mức | Nội dung | Claim được phép |
| --- | --- | --- |
| Chứng minh có điều kiện | Mệnh đề trong Appendix A | Kết luận đúng nếu các giả định được xác lập |
| Certificate của artifact | Cận support, số học và decoder thực sự hợp lệ trên miền khai báo | Certificate trên chính miền và checkpoint đó |
| Chẩn đoán thực nghiệm | Residual, margin, gradient, CI trên dữ liệu hữu hạn | Bằng chứng thực nghiệm; không đổi tên thành uniform certificate |

<a id="s2"></a>
## 2. Motivation

### 2.1. Vấn đề nghiên cứu

ASR pretrained có thể suy giảm chất lượng nhận dạng trên tiếng nói dysarthric; mức suy giảm phụ thuộc model, người nói và protocol. Các nghiên cứu pathological ASR với pretraining/adaptation [R01], [R37] tạo động lực khảo sát khả năng cải thiện khi không được thu dữ liệu bệnh để huấn luyện. Chưa mặc định rằng mọi lỗi đều do lệch phát âm/nhịp so miền huấn luyện, hoặc mọi backbone đều suy giảm như nhau; causal explanation này cần bằng chứng riêng.

Trong triển khai, yêu cầu là một người nói mới có thể dùng hệ thống ngay, không phải ghi dataset cá nhân để fine-tune. Hệ thống cần hạn chế làm hỏng tiếng nói chuẩn, duy trì ngân sách inference và cung cấp lý do rõ khi không đủ khả năng sửa.

### 2.2. Vì sao sửa latent

Backbone đã chứa cấu trúc âm học và ngôn ngữ học hữu ích. Can thiệp vào encoder output có thể tận dụng decoder sẵn có mà không phải tái tạo waveform. Một correction có thể xa clean latent hơn nhưng vẫn đưa model tới quyết định đúng; reconstruction error vì thế không phải objective tác vụ cuối cùng.

Ngược lại, projection có thể chọn nhầm lớp hoặc xóa biến thiên sạch cần thiết. Vì vậy, thiết kế phải nối bốn vấn đề: coverage của dictionary, routing, tác động lên decoder và bảo toàn đường identity.

### 2.3. Đánh đổi có chủ đích

Có thể bỏ qua chi tiết người nói/chất giọng khi tác vụ chỉ là nội dung. Không được mặc định thanh điệu, tương phản âm vị hoặc thứ tự thời gian là nuisance. Việc loại bỏ thông tin qua representation không khả nghịch có thể gây lỗi dù làm các miền nhìn giống nhau [R02].

Mục tiêu được giới hạn ở những biến dạng còn giữ đủ thông tin trong biểu diễn được truy cập. Nếu hai nội dung khác nhau tạo cùng toàn bộ quan sát khả dụng, mọi thuật toán đều gặp giới hạn bất khả phân biệt; Appendix A.1 chứng minh điều này.

Giả thuyết cần kiểm chứng là tồn tại đủ input mà geometry sạch cho correction hữu ích đối với native decoder. Chưa biết tỷ lệ này trên dysarthria. Mô hình nhiễu latent bị chặn không đại diện toàn bộ thay thế âm vị, mất thông tin hoặc thay đổi correspondence theo thời gian. Repair từng frame không thực hiện time warping hay đổi số frame; nó vẫn có thể ảnh hưởng blank/token/repetition thông qua features đã ngữ cảnh hóa. Hiệu ứng temporal phải được đo bằng S/D/I và transcript, không được tuyên bố đã giải quyết từ routing theorem.

<a id="s3"></a>
## 3. Related work, novelty và phạm vi claim

| Nhóm | Nguồn | Khoảng cách nghiên cứu PhonoRepair cần kiểm tra |
| --- | --- | --- |
| Pathological ASR với SSL | Violeta et al. [R01] | Clean-only onboarding, khóa trước disease-test |
| Hướng ngữ âm từ healthy references | Muller et al., preprint 2026 [R03] | Assessment khác mục tiêu can thiệp latent cho recognition |
| Local affine approximation | Localized K-Flats [R04] | Bản thân affine patches/PCA không là novelty |
| Farthest-first clustering | Gonzalez [R05] | Cận phủ neo không phải cận optimum affine dictionary |
| Test-time adaptation | SUTA [R06]; DSUTA [R07] | Có cập nhật từ input đích; là setting khác với artifact bất động |
| Invariance và generalization | Johansson et al. [R02]; Ben-David et al. [R08] | Cần giả định thông tin/support; không có bảo đảm mọi domain |
| Calibration | Angelopoulos & Bates [R09] | Áp dụng thứ hạng vào intervention theo episode/cluster |
| Backbones và decoder | Whisper [R10]; Zipformer [R11]; Conformer [R12]; CTC [R13] | Một ranh giới acoustic memory, giữ decoder gốc |
| Dysarthric ASR với target supervision/augmentation | Leung et al. [R37] | Bối cảnh hiệu quả; khác quyền dùng dữ liệu bệnh và protocol |
| Time stretching cho dysarthric recognition | Prananta et al. [R36] | Cần đối chứng preprocessing cố định; không mặc định làm chậm luôn có lợi |
| Enhancement dùng recognition perceptual loss | Plantinga et al. [R38] | Học enhancement theo acoustic model; khác artifact projection bất động |
| Quantized speech representations | wav2vec 2.0 [R39] | Codebook phục vụ representation learning, không tự là latent repair sau encoder đóng băng |
| Nearest-neighbor feature replacement | kNN-VC [R40] | Cơ chế retrieval liên quan; tác vụ voice conversion và resynthesis khác recognition repair |

Các đóng góp **dự kiến**, cần được chứng minh bằng implementation và thực nghiệm:

- **C1:** một core projection/routing chung với binding khai báo, không có repair module theo tên backbone.
- **C2:** quy trình clean-only tạo artifact và báo tách empirical fit, thiếu phủ support, sai số tối ưu.
- **C3:** routing cục bộ có điều kiện, cho phép sửa vùng đủ phân biệt dù global separation thất bại.
- **C4:** bảo toàn identity/state và gate sạch hữu hạn mẫu; tăng thành zero intervention trên support khi có clean envelope hợp lệ.
- **C5:** protocol mù trên ba backbone, đánh giá usefulness, harms, shared-core portability và overhead; hợp đồng decoder xác định giới hạn claim. Certificate transcript đa decoder và realtime SLA không là điều kiện bắt buộc của contribution chính.

Không tuyên bố “đầu tiên”, novelty đã được xác nhận exhaustive, hoặc các lemma projection/margin/entropy là định lý nghiên cứu mới. Nguồn [R03] là preprint; không mặc định đã peer-reviewed.

Các nguồn enhancement/codebook/retrieval là đối chiếu cơ chế, không được coi là baseline cùng setting nếu chưa kiểm tra dữ liệu và decoder. Khi trình bày paper, bảng literature phải ghi split, microphone, lexicon, quyền target supervision/update và vị trí can thiệp; số liệu khác protocol chỉ dùng làm bối cảnh, không trừ trực tiếp để tuyên bố “cách SOTA” một lượng WER.

<a id="s4"></a>
## 4. Câu hỏi nghiên cứu và tiêu chí thành công

### 4.1. Câu hỏi

| ID | Câu hỏi | Thực nghiệm chính |
| --- | --- | --- |
| RQ1 | Dictionary trong budget có residual và vùng routing hữu ích không? | E1, E2 |
| RQ2 | Clean input có giữ được baseline không? | E0, E3, E8 |
| RQ3 | Không học bệnh có cải thiện recognition trên bệnh chưa dùng thiết kế không? | E4, E5 |
| RQ4 | Có tác dụng trên biến dạng âm học không dùng để fit không? | E6 |
| RQ5 | Backbone mới có thể onboarding bằng cùng core/policy không? | E7 |
| RQ6 | Chi phí toàn hệ thống có đạt deployment budget không? | E9 |
| RQ7 | Geometry/protection thực sự cần thiết hay chỉ thêm overhead? | E10 |

### 4.2. Tiêu chí đăng ký ban đầu

Các margin dưới đây là mục tiêu engineering đề xuất. Chốt trước disease/corruption test; không đổi vì kết quả đích xấu.

| Mục tiêu | Điều kiện đánh giá |
| --- | --- |
| Tích hợp | 3/3 backbone chính chạy được cùng repair core; không thêm brand branch |
| Identity | Không có mismatch transcript/state trong E0 theo contract deterministic đã khóa |
| Clean accuracy | Cận trên một phía Bonferroni $$1-0.05/3\approx98.33\%$$ của $$\Delta\mathrm{WER}$$ không vượt $$+0.1$$ điểm phần trăm trên LibriSpeech test-clean; báo thêm cận 95% mô tả, quyết định dùng cận hiệu chỉnh |
| Hiệu quả bệnh đầy đủ | Cả 6 cặp 3 backbone × 2 corpus có relative WER reduction ≥5%, và CI đã hiệu chỉnh có $$\Delta\mathrm{WER}<0$$ |
| Kết quả bệnh một phần | Báo toàn bộ 6 cặp; chỉ claim các cặp có point WERR ≥5% và CI hiệu chỉnh dưới 0, đồng thời backbone đạt clean non-inferiority; không gọi “mọi backbone” |
| Usefulness | Tỷ lệ sửa >0 và có corrected cases; identity toàn bộ bị đánh dấu thất bại usefulness |
| Realtime — mở rộng | Chỉ checkpoint/mode eligible: acoustic chunk latency p99 ≤250 ms, RTF ≤0.8, miss rate ≤0.1% trên hardware khai báo; không quyết định thành công recognition chính |
| Repair overhead | p99 ≤5 ms/chunk; artifact ≤16 MiB; working memory phụ ≤64 MiB |
| Repeat onboarding — mở rộng | 0 sửa core, 0 threshold viết tay là yêu cầu shared policy; ≤15 phút thao tác kỹ sư là mục tiêu phụ, không gồm compute |
| Clean calibration | Đánh giá với đơn vị độc lập hợp lý; không đếm frame như episode độc lập |

Mục tiêu 5% relative reduction không phải dự đoán. CI quá rộng là **inconclusive**, không tự thành tốt hoặc xấu. Không dùng “không significant” làm bằng chứng non-inferiority.

Ngưỡng 5% áp lên point estimate; CI dưới 0 xác nhận superiority theo protocol, không chứng minh WERR tối thiểu 5%. Phải báo cả effect size và uncertainty. Success toàn panel còn cần E0, clean protection và usefulness đạt; không dùng một cell đẹp để thay tiêu chí sau test.

### 4.3. Ba mức kết luận general được khóa trước target

| Mức | Bằng chứng cần có | Phạm vi được nói |
| --- | --- | --- |
| Tích hợp chung | E0/E7 đạt trên W1/Z1/C1; cùng core/policy, không brand-specific repair | Khả năng dùng chung thuật toán và interface; chưa kết luận cải thiện WER |
| Hiệu quả trên nhiều backbone trong panel | Ít nhất 2/3 backbone có ít nhất một cặp đạt tiêu chí hiệu chỉnh ở §4.2 và clean protection; báo đủ sáu cặp và mọi harms | Bằng chứng hiệu quả trên các backbone/corpus được nêu tên; không suy ra mọi corpus hoặc backbone mới |
| Hiệu quả nhất quán toàn panel | Cả sáu cặp đạt, cả ba backbone đạt clean protection và các gate kỹ thuật | Hiệu quả nhất quán trên đúng panel đã đăng ký; chưa là universal guarantee |

Chỉ một backbone đạt là model-specific result, chưa xác nhận usefulness xuyên backbone. Một cặp còn lại có harm/inconclusive phải xuất hiện cạnh kết quả tích cực; mức “nhiều backbone” không có nghĩa toàn panel an toàn. Giữ cố định denominator ba backbone/sáu cặp; unsupported hoặc dừng feasibility không bị loại âm thầm để tính lại success.

<a id="s5"></a>
## 5. Bài toán, ký hiệu và giả định

### 5.1. Mục tiêu tác vụ

Với audio $$a$$, encoder output $$H=F_b(a)$$, decoder gốc $$G_b$$:

$$
Y_{\rm base}=G_b(H),\qquad
Y_{\rm repair}=G_b(R_{\theta_b}(H)).
$$

Mục tiêu đánh giá là transcript tham chiếu $$Y^\star$$, không phải tự động coi $$Y_{\rm base}$$ là ground truth. Bảo toàn baseline chỉ là một yêu cầu bảo vệ sạch.

### 5.2. Ký hiệu

| Ký hiệu | Ý nghĩa |
| --- | --- |
| $$b,d_b$$ | Checkpoint và chiều encoder output |
| $$T,B$$ | Frame của episode, frame của chunk |
| $$r_b,Q_b,\mu_b$$ | Chiều sửa, basis trực chuẩn, mean |
| $$u=Q_b^\top(h-\mu_b)$$ | Tọa độ routing/sửa |
| $$C,M,s_j$$ | Số lớp, tổng patch, chiều patch |
| $$a_j,B_j,\rho_j,\ell_j$$ | Neo, basis patch, radius, label |
| $$K_p,X_p$$ | Clean support và mẫu sạch lớp $$p$$ trong không gian $$u$$ |
| $$r_{{\rm cov},p},\widehat E_p,E_p$$ | Bán kính net, fit max-residual, support max-residual |
| $$\eta,\nu,\tau=\eta+\nu$$ | Coverage envelope, noise budget, routing threshold |
| $$q_{\rm cal},q_{\rm off}$$ | Ngưỡng thống kê và ngưỡng mở repair |
| $$\beta,T_{\rm cap},K_{\rm ep}$$ | Risk level, episode cap, số episode trong calibration block |
| $$\varepsilon_{\rm num}$$ | Sai số số học có cận nếu được chứng nhận |
| $$L_i,m_i$$ | Độ nhạy và margin quyết định decoder |

### 5.3. Miền biến dạng có điều kiện

Trong tọa độ sửa:

$$
u=x+\xi,\qquad x\in K_p,\qquad \|\xi\|\le\nu.
$$

Đây là giả định về latent, không phải ước lượng mức bệnh runtime. Tên loại bệnh và phân phối nhiễu không xuất hiện trong routing theorem.

Không mặc định tồn tại clean counterpart có correspondence frame đúng cho mọi utterance bệnh. Với biến dạng độ dài/nhịp, không ép padding để suy ra noise budget; chỉ dùng certificate khi correspondence, coverage và actual noise assumptions thực sự hợp lệ. Các trường hợp còn lại được đánh giá empirical bằng native decoder.

Phần latent ngoài $$Q_b$$ được giữ nguyên; nếu decoder nhạy với phần đó, cận transcript phải bao gồm nó. Không được suy từ gradient nhỏ tại vài clean points rằng nhiễu ở phần bỏ qua được phép lớn tùy ý.

### 5.4. Ba giới hạn bắt buộc

- Nhãn/alignment đúng là giả định của lý thuyết, phải audit trên dữ liệu thật.
- Frame cùng recording/speaker không mặc nhiên i.i.d.
- Clean support compact không cho sẵn net radius, minimum mass hoặc coverage hữu ích.

<a id="s6"></a>
## 6. Kiến trúc tổng thể và tích hợp backbone

### 6.1. Ranh giới chung

Can thiệp ở acoustic encoder output, trước decoder sử dụng memory. Không chọn một điểm “trước head CTC” rồi áp nguyên sang Whisper. Whisper đọc encoder memory qua text decoder; Icefall/NeMo cung cấp các cấu trúc encoder/head khác nhau [R14], [R15], [R16].

~~~mermaid
flowchart LR
    A["Audio / frontend gốc"] --> E["Encoder đóng băng"]
    E --> B["Binding: features + lengths + timebase"]
    B --> R["Repair core chung"]
    C["Clean source"] --> O["Onboarding offline"]
    O --> P["Artifact của checkpoint"]
    P --> R
    R --> D["Decoder gốc + cache hợp lệ"]
    D --> Y["Transcript"]
~~~

### 6.2. Contract tối thiểu

Một model package phải khai báo:

~~~yaml
model_contract:
  checkpoint_id: resolved_before_test
  checkpoint_revision: resolved_before_test
  checkpoint_sha256: computed_from_files
  inference_entrypoint: native_callable_or_exported_graph
  encoder_invocation: semantic_binding
  features_output_path: semantic_binding
  layout: native_axes
  valid_lengths_or_mask: semantic_binding
  frame_time_mapping: derived_from_frontend_and_subsampling
  state_outputs: declared_paths
  memory_consumers: declared_edges
  context: native_context_contract
  reset: native_reset_contract
  decoder_config_hash: computed_before_test
~~~

Không suy ra acoustic semantics từ shape đơn thuần. Auto-discovery có thể đề xuất binding, nhưng ambiguity phải báo lỗi. Cho phép binding theo runtime/call signature; cấm cài thuật toán sửa trong binding hoặc chọn ngưỡng theo tên model.

### 6.3. Wrapper/hook và export

Prototype dùng wrapper hoặc per-module output hook; chỉ sửa features, giữ lengths/masks/state output nguyên cấu trúc. Khi gate đóng, trả nguyên tensor/output tương ứng, không reconstruct bằng phép nhân basis.

Hook phải được kiểm tra trên đúng invocation. PyTorch gọi module qua cơ chế hook; gọi trực tiếp forward có thể bỏ hook [R17]. Streaming/export có thể dùng entrypoint khác; Icefall có export streaming riêng [R18]. Production cần wrapper/graph insertion được kiểm định trên runtime đó, không mặc định Python hook sống nguyên qua export.

### 6.4. Ba lớp phần mềm

| Lớp | Trách nhiệm | Điều không được chứa |
| --- | --- | --- |
| Tensor binding | Extract/replace features, axes, lengths, state contract | Disease heuristics, brand-specific thresholds |
| Repair core | Read coordinates, exact search, gate, projection, lift | Tokenizer hoặc vòng decoder riêng |
| Artifact factory | Fit/selection/calibration/audit sạch | Target disease/corruption feedback |

### 6.5. Cache ownership

Encoder state không nhận correction feedback. Memory sửa được tạo trước khi decoder cache phụ thuộc nó được ghi. Mỗi memory version chỉ gắn với cache sinh từ đúng version. Với overlapped windows, sửa lại memory không được tái dùng cache cũ mà không có quy tắc invalidation hợp lệ.

PhonoRepair không mutate in-place features alias với encoder state. Nếu binding không bảo đảm điều này, artifact không được triển khai.

### 6.6. Phép đọc và ghi chung

$$
Q_b^\top Q_b=I,\qquad
u=Q_b^\top(h-\mu_b),\qquad
\widehat h=h+Q_b(\widehat u-u).
$$

Appendix A.2 chứng minh đúng tọa độ sửa, giữ thành phần trực giao và kiểm soát biên độ. Với rank đầy đủ, dùng identity basis fast path; không cần lưu dense identity matrix.

<a id="s7"></a>
## 7. Xây artifact chỉ bằng dữ liệu sạch

### 7.1. Chia dữ liệu và trình tự khóa

Trình tự: freeze checkpoint/contract → source split → basis fit → dictionary fit → clean selection → khóa candidate/score → independent calibration → clean test → blind target test.

Basis và mean được học trên basis-fit. Dictionary dùng dictionary-fit. Việc tách này giúp chỉ rõ representation đã cố định trước mẫu dùng cho một coverage theorem; mẫu dictionary vẫn có thể tạo dictionary thích nghi trên biến cố phủ đồng thời.

Dataset thật không tự đáp ứng giả định net/i.i.d.; chỉ dùng theorem xác suất khi các đối tượng đó thực sự được xác lập.

**Vai trò synthetic pilot:** toy finite supports dùng kiểm thử theorem/kernel; perturbations trên clean source dùng diagnostic cơ chế; nếu kết quả được dùng sửa thiết kế thì phải ghi là synthetic development. Không gọi chúng là mô phỏng bệnh hoặc dùng để ước lượng hiệu quả dysarthria. Phiên bản này giữ objective candidate selection ở §7.6; synthetic development chỉ giúp quyết định dừng hoặc đăng ký policy phiên bản mới, không thêm tiêu chí xếp hạng ngầm.

Synthetic development chỉ dùng speakers của clean-selection. Synthetic evaluation chỉ dùng speakers giữ kín theo §10.1, với seed/recipe khóa trước khi mở. Không đưa final calibration, clean test, disease/corruption test vào vòng thử sửa policy. Recipe được định nghĩa tại E2; final calibration vẫn diễn ra một lần sau khi khóa candidate. Nếu cần phát triển phiên bản mới sau calibration failure, phải thay calibration data bằng units chưa dùng và khai báo lịch sử development.

### 7.2. Nhãn sạch và ánh xạ thời gian

Nguồn chính: transcript LibriSpeech và forced alignment offline bằng một aligner cố định [R19], [R20], [R21]. Không chạy alignment với transcript thật của input bệnh tại inference.

MFA paper gốc được dẫn ở [R34]. Pin acoustic model, pronunciation dictionary, tool version và các normalization/adaptation options; bước speaker normalization nội bộ aligner chỉ dùng source-clean, không truy cập bệnh. Unknown/OOV được log, không tự dùng target transcript để bổ sung lexicon.

Inventory router v1: 39 base ARPABET phones sau bỏ stress digit, cộng SIL; các nhãn aligner khác phải có explicit mapping hoặc là UNSUPPORTED. SIL là lớp dictionary, không dùng oracle silence mask runtime. Không bỏ hard transition frames khỏi WER.

Gán label cho encoder frame theo tâm interval thời gian khai báo bởi binding. Khi interval vượt phone boundary, ghi flag transition; vẫn giữ frame trong evaluation. Clean-fit có thể tạo một ablation context-state inventory riêng, nhưng không đổi inventory theo kết quả bệnh.

TIMIT cung cấp annotation phonetic/word đã kiểm định thủ công [R22]. Dùng để audit aligner và auxiliary phone evaluation. Bộ mapping TIMIT→router phải được version/hash; report các class nhiều-một và tỷ lệ không map được. TIMIT evaluation không tự biến alignment LibriSpeech thành ground truth.

Appendix B.5 định nghĩa mapping tối thiểu và cách xử lý SIL/Q. Các frame padding giữ nguyên. Nếu native decoder đọc cả padding memory, chúng vẫn phải có mặt trong cận sai số chuỗi A.12–A.14; không loại khỏi certificate chỉ vì valid-length mask không cho sửa.

### 7.3. Chọn basis theo cùng policy

Candidate ranks:

$$
r_b\in\operatorname{unique}\{\min(d_b,32),\min(d_b,64),
\min(d_b,128),\min(d_b,256),d_b\}.
$$

Basis candidates: (i) PCA của basis-fit; (ii) supervised Fisher span từ scatter giữa/trong phone classes rồi QR để có basis Euclid trực chuẩn; (iii) full-rank identity. Regularization Fisher cố định theo source variance, ví dụ $$10^{-3}\operatorname{tr}(S_W)/d_b$$; xử lý zero variance bằng floor $$10^{-8}$$ cùng policy.

Các basis chỉ là cách tạo candidate. Clean decoder validation quyết định feasibility. Không claim Fisher/PCA tìm được “task-sufficient quotient” hoặc invariant directions dưới mọi biến dạng.

### 7.4. Dictionary fit

Grid khởi đầu:

$$
M\in\{64,128,256\},\qquad s_{\max}\in\{0,2,4\}.
$$

- Mỗi lớp nhận có ít nhất một patch; ngân sách còn lại phân bổ đều, phần dư theo thứ tự label đã khóa.
- Nếu thiếu lớp trong fit, không tự tạo anchor; report unsupported scope hoặc ineligible.
- Với mỗi lớp, lấy tối đa 5,000 fit frames bằng sampler speaker-balanced, seed cố định.
- Chọn anchor đầu tiên là sample có ID nhỏ nhất; các anchor sau farthest-first. Ties theo ID.
- Láng giềng mỗi anchor: tối đa 128 mẫu gần nhất cùng lớp; SVD trên $$u_i-a_j$$, không dùng centering phụ làm anchor trôi.
- Basis patch dùng số chiều khả thi $$\min(s_{\max},r_b,\operatorname{rank})$$.
- Radius là max norm tọa độ các láng giềng đã chọn. Radius không được gọi là coverage toàn support.
- Audit lại max-residual sau mọi thay đổi. Optional minimax refinement chỉ nhận candidate không tăng objective trong cùng họ; chưa có global-optimum guarantee.

Farthest-first có cận 2 cho bán kính anchor [R05]; Appendix A.5 cho proof và giới hạn. Không dùng explained variance ratio thay cho max-residual.

### 7.5. Coverage, budgets và trạng thái certificate

Có hai loại $$\eta$$:

1. **Support envelope:** $$\eta_{\rm cert}=\max_p(r_{{\rm cov},p}+\widehat E_p)$$, chỉ khi $$r_{{\rm cov},p}$$ hợp lệ.
2. **Working envelope:** $$\eta_{\rm work}$$ là max-residual quan sát trên dictionary-fit và geometry audit của clean-selection. Đây là một tham số source-only; chưa phải support bound.

Đặt $$\eta_{\rm used}$$ theo loại có sẵn và ghi rõ provenance.

Global dictionary separation lower bound:

$$
\underline\Gamma=
\max\{0,\min_{\ell_j\ne\ell_k}
(\|a_j-a_k\|-\rho_j-\rho_k)\}.
$$

Nếu envelope hợp lệ và $$\underline\Gamma>2\eta_{\rm cert}$$:

$$
\nu_{\rm global}=\tfrac12(\underline\Gamma/2-\eta_{\rm cert}).
$$

Global certificate bảo đảm mọi input trong budget được routing duy nhất. Nếu nó thất bại, local routing vẫn được nghiên cứu.

Policy local source-only đề xuất: lấy positive class-gap $$g(u)$$ trên clean-selection mà nearest class khớp annotation; đặt $$\nu_{\rm local}=\tfrac14 Q_{0.10}(g)$$, quantile bằng nearest-rank. Nếu không có positive gap, $$\nu_{\rm local}=0$$. Đây là **declared threat budget**, không phải estimate actual disease noise hay certificate population. Appendix A.8 vẫn đúng nếu coverage và noise assumptions thực sự thỏa.

Trong policy này, $$g(u)=d_{(2)}(u)-d_{(1)}(u)$$ theo khoảng cách lớp; ties không thuộc positive-gap subset. Báo cả số frame bị loại khỏi subset này để tránh che overlap.

### 7.6. Candidate selection

Trước final calibration, đánh giá candidates trên clean-selection với provisional threshold lấy từ fit residual quantile 0.99. Provisional gate chỉ phục vụ selection, không mang claim conformal.

Policy cố định:

1. Loại candidates sai binding/identity hoặc vượt resource budget.
2. Loại candidates có clean-selection WER tăng hơn 0.1 điểm phần trăm so với native baseline.
3. Ưu tiên candidate có certified radius nếu có; trong cùng status ưu tiên $$\nu/(\eta_{\rm used}+10^{-8})$$, rồi lower max-residual, lower measured overhead, candidate ID.
4. Khóa một candidate. Final calibration có thể làm artifact không còn usefulness; báo thất bại, không quay lại chọn candidate bằng calibration.

Tỷ số $$\nu/(\eta_{\rm used}+10^{-8})$$ chỉ là tiêu chí hình học đã đăng ký, không là surrogate WER đã được xác nhận. Báo geometry/gate diagnostics cho toàn grid trên source; không dùng synthetic hoặc target winner để thay candidate được policy chọn. Provisional threshold chỉ dùng development và không dự đoán được ngưỡng final; không dùng final calibration để lập leaderboard candidates.

Giữ danh sách candidate grid để điều chỉnh multiplicity nếu thực hiện probabilistic certification sau selection. Không coi bộ chạy multistart tốt nhất là lower bound optimum.

### 7.7. Calibration và risk profile

Default pilot: $$\beta=0.01$$. Calibration unit là **speaker block** gồm $$K_{\rm ep}=10$$ episodes, mỗi episode tối đa $$T_{\rm cap}$$ tương đương 30 s tại checkpoint. Block score:

$$
A_i^{\rm block}=\max_{k\le10,t}A(u_{i,k,t}).
$$

Một người nói mới với block cùng protocol chỉ có bound nếu các block exchangeable. Dùng $$n_{\rm cal}=100$$ speakers cho finite threshold; không thay $$n_{\rm cal}$$ bằng số frame.

Mục tiêu nâng cấp $$\beta=0.001$$ cần ít nhất 999 calibration units tương ứng. Nếu dùng speaker-block guarantee thì cần 999 speakers/units hợp lệ; 999 utterances của vài speakers không thay thế được. Không tự hạ $$\beta$$ sau khi thấy target metrics.

Ngưỡng:

$$
k=\lceil(n_{\rm cal}+1)(1-\beta)\rceil,\quad
q_{\rm cal}=A_{(k)}^{\rm block}
$$

nếu $$k\le n_{\rm cal}$$, ngược lại $$+\infty$$.

Default protection:

$$
q_{\rm off}=\max(q_{\rm cal},\eta_{\rm used}).
$$

Conformal bound vẫn giữ vì gate nhận là tập con score $$>q_{\rm cal}$$. Zero intervention trên toàn clean support chỉ giữ khi $$\eta_{\rm used}$$ là support envelope hợp lệ. $$\eta_{\rm work}$$ không tạo claim đó.

Với default $$n_{\rm cal}=100,\ \beta=0.01$$, $$k=100$$: q_cal là maximum của 100 block scores. Không mô tả nó như percentile frame thông thường. Số frame/block phụ thuộc độ dài thực và timebase checkpoint.

**Kiểm tra vùng nhận sửa bắt buộc.** Trước guards số học/deadline, điều kiện là:

$$
\max(q_{\rm cal},\eta_{\rm used})<d_{(1)}(u)
\le\eta_{\rm used}+\nu<d_{(2)}(u).
$$

Đặt:

$$
w=\max\{0,\eta_{\rm used}+\nu-\max(q_{\rm cal},\eta_{\rm used})\}.
$$

Nếu $$w=0$$ thì artifact không thể nhận sửa; đánh dấu failure usefulness trước target accuracy. Nếu $$w>0$$ thì chỉ chứng minh một khoảng score còn mở, không phải acceptance rate dương hoặc WER improvement. Singleton/numerical/deadline guards có thể loại hết input còn lại. Báo w, q_cal, q_off, η, ν và counts theo từng nhánh reject.

Không tự thay η max bằng quantile để “mở gate”: khi q_cal chi phối, hạ η giữ ngưỡng dưới nhưng hạ ngưỡng trên, có thể làm w nhỏ hơn. Quantile envelope phải là biến thể với coverage failure và claim routing xác suất riêng. Phiên bản này giữ working/support envelope và gate cũ.

### 7.8. Artifact cần lưu

Checkpoint/code/decoder/binding hashes; data/split/aligner/inventory hashes; mean/basis; anchors/bases/radii/labels; all thresholds and units; metric; frame timing; episode/block policy; numerical status; resource measurements; certificate assumptions; training permissions; candidate selection log; unsupported cases.

<a id="s8"></a>
## 8. Inference và bảo toàn baseline

### 8.1. Khoảng cách và projection

Với $$v_j=B_j^\top(u-a_j)$$:

$$
\pi_j(u)=a_j+B_j\operatorname{clip}_{\rho_j}(v_j),
$$


$$
d_j(u)^2=\|u-a_j\|^2-\|v_j\|^2+(\|v_j\|-\rho_j)_+^2.
$$

Để giảm cancellation, reference kernel tính phần trực giao trực tiếp. MVP dùng full-search; ANN không được dùng thay full-search khi chưa có lower bounds loại trừ mọi patch bỏ qua.

### 8.2. Quy tắc nhận sửa

~~~text
REPAIR_FRAME(h, artifact):
    nếu outside declared scope/cap hoặc artifact inactive: return h
    u = Q.T @ (h - mean)
    tính distances của toàn bộ patch; min theo lớp
    A = min class distance
    nếu A <= q_off: return h
    feasible = {p: d_p <= eta_used + nu}
    nếu feasible không singleton: return h
    nếu numerical confidence hoặc deadline không đạt: return h
    candidate = exact nearest-patch projection của lớp duy nhất
    nếu norm(candidate - u) > declared correction budget: return h
    return h + Q @ (candidate - u)
~~~

Correction budget default $$\eta_{\rm used}+\nu$$; nếu candidate vượt budget, reject, không clip lại rồi viện chứng chỉ cũ. Numerical/latency guards có thể chỉ làm giảm tỷ lệ nhận.

### 8.3. Numerical status

- Float64 reference offline; float32 online ban đầu.
- Test orthogonality và projection errors trên declared shapes/range.
- Với interval distances, nhận lớp $$p$$ khi $$u_p\le\tau$$, mọi lớp khác $$l_q>\tau$$; mở gate khi lower score $$>q_{\rm off}$$.
- Bounds của distance chưa chặn lỗi vector projection. Nếu certificate số học chưa có, report empirical agreement; thêm measured tolerance là heuristic guard, không certificate.
- Margin $$g>2\delta$$ bảo toàn nearest-class dưới perturbation có cận $$\delta$$, không xác nhận input nằm trong noise budget.

Zero-intervention trên clean support ở số học thực cần thêm margin cho score error: dùng $$\eta_{\rm cert}+\epsilon_A$$ trong envelope threshold nếu đã chặn $$|\tilde A-A|\le\epsilon_A$$, hoặc chỉ mở khi certified lower bound vượt envelope. Calibration vẫn dùng đúng implemented score đã khóa; runtime luôn kiểm tra score này vượt $$q_{\rm cal}$$.

### 8.4. Identity, baseline và trạng thái

Khi gate đóng, trả original features. Không casting/re-normalization hoặc reconstruct $$h$$ bằng basis. Keep native output tree, lengths/masks and state.

Bảo toàn transcript khi không có intervention trong episode cần cùng initial state, deterministic decode/tie-breaking và scheduling không làm drop frame. Một repair trước đó có thể làm cache/history khác baseline; tắt gate ở frame sau không tự khôi phục trạng thái. Reset phải thuộc baseline protocol; không thêm reset chỉ để lấy certificate rồi gọi là bảo toàn native baseline.

### 8.5. Không gọi gate là detector bệnh hoàn hảo

Residual chỉ đo sự phù hợp với dictionary. Clean tail có thể vượt ngưỡng; corruption có thể ở ngay trên dictionary. Singleton có thể sai ngoài miền noise assumptions. Fallback trả baseline, không bảo đảm baseline đúng.

Nếu input nằm ngay trên patch lớp sai thì score bằng 0 và gate trả identity: đây là missed correction, không phải correction củng cố lỗi. Accepted wrong routing chỉ xảy ra khi input vượt q_off, có đúng một lớp feasible nhưng lớp đó sai và vượt các guard còn lại. Báo hai trường hợp riêng; không dùng singleton tự chứng minh noise budget. Positive toy A.18 cho thấy baseline sai có thể được sửa đúng, nhưng chưa cho biết tần suất trên speech.

### 8.6. Decoder contract

Projection đúng và routing đúng nhãn hình học chưa bảo đảm WER. Với greedy CTC dùng Appendix A.13; với deterministic decoder khác dùng comparison-trace theorem A.14. Cần include all relevant branch/pruning/EOS decisions, continuous state dependencies và same initial state. Gradient/teacher forcing trên clean samples là diagnostic, không thay free-running certificate.

Không đặt một “decoder certificate checker” giả ở runtime cần clean counterpart $$X$$ hoặc transcript thật. Nếu không có runtime-accessible certificate, empirical WER được đánh giá và claim vẫn conditional.

Certificate transcript trên cả ba decoder không là gate của nghiên cứu empirical. Với CTC có thể tính một số affine-head margins/bounds thuận lợi hơn, nhưng vẫn phải xác lập clean counterpart, coverage và miền sai số. Paper ưu tiên kết quả trực tiếp giải thích thiết kế; standard lemmas và comparison-trace details được giữ trong appendix, không tính như novelty độc lập.

<a id="s9"></a>
## 9. Yêu cầu phần mềm, tài nguyên và realtime

### 9.1. Các interface cần triển khai

~~~text
Binding.extract(native_output) -> MemoryPacket
MemoryPacket:
    features, valid_lengths/mask, timebase, stream_id, memory_version
Binding.replace(native_output, repaired_features) -> native_output
ArtifactBuilder.fit(clean_manifest, binding, shared_policy) -> Artifact
RepairCore(packet, artifact) -> packet
Evaluator(native_pipeline, locked_artifact, eval_manifest) -> Result
~~~

State tensors thuộc model, core không sở hữu. Native decoder loop giữ nguyên; encoder wrapper trả features đã sửa vào pipeline gốc. Không yêu cầu viết lại toàn decoder cho từng model nếu native invocation cho phép chặn đúng output.

### 9.2. Cấu trúc repository mục tiêu

~~~text
phonorepair/
  core/             # geometry, projection, gate, lift
  binding/          # tensor-tree/layout contracts and runtime insertion
  artifacts/        # builders, schemas, validators
  evaluation/       # text normalization, metrics, CI, timing
  tests/            # meaningful kernel/state/certificate tests
configs/
  shared_policy.yaml
  bindings/         # declarations only
manifests/
  source_splits.jsonl
  locked_run.json
outputs/
  artifacts/
  evaluations/
~~~

Đây là cấu trúc phải implement, chưa tồn tại trong project. Python reference; dùng compiled tensor kernels/C++ sau profiling. Không phụ thuộc control flow theo tên model. Library/backend differences có thể có runtime bindings; phải báo manual LOC và giờ tích hợp.

### 9.3. Độ phức tạp

| Bước | Time | Space chính |
| --- | --- | --- |
| Read/lift $$n$$ frames | $$O(nd_br_b)$$ | Basis $$O(d_br_b)$$, xử lý theo block |
| Farthest-first | $$O(\sum_p n_pM_pr_b)$$ | $$O(nr_b+M r_b+n)$$ |
| Local SVD $$n_j\times r_b$$ | $$O(n_jr_b\min(n_j,r_b))$$ mỗi patch | $$O(n_jr_b)$$ |
| Fit max-residual audit | $$O(nMr_b(s+1))$$ | Có thể streaming blocks |
| Pair separation lower bound | $$O(M^2r_b)$$ | $$O(1)$$ phụ khi duyệt cặp |
| Repair chunk $$B$$ | $$O(B[d_br_b+Mr_b(s+1)])$$ | $$O(d_br_b+Mr_b(s+1)+B(d_b+r_b))$$ |
| Text edit metric | $$O(L_{\rm ref}L_{\rm hyp})$$ | $$O(\min(L_{\rm ref},L_{\rm hyp}))$$ với rolling rows |
| Candidate search | Tổng cost của $$G$$ candidates | Có thể đánh giá tuần tự |

Backbone, alignment, native decode và certificate solvers cộng riêng. Full-rank identity basis có fast path để tránh dense $$d_b^2$$ storage/compute.

### 9.4. Resource targets và deadline

Artifact size, working memory và measured repair overhead là kiểm tra bắt buộc cho main panel. Các bất đẳng thức deadline và 250-ms SLA dưới đây chỉ áp dụng realtime extension đã đăng ký; không dùng chúng để loại W1/C1 offline khỏi đánh giá recognition.

Hardware reference đề xuất: workstation 32 GiB RAM, GPU 24 GiB; CPU-only run được report riêng. Ghi exact CPU/GPU, driver, runtime, clocks/power mode, threads, batch=1. Mục tiêu chính trên một hardware đã khóa; không so timing từ các máy khác nhau thành lợi ích thuật toán.

Với chunk arrival interval $$\Delta=Bh$$, total compute upper bound $$C_{\rm tot}$$, lookahead $$\ell$$:

$$
C_{\rm tot}\le\Delta,\qquad
\ell+\Delta+C_{\rm tot}\le250\ {\rm ms}.
$$

Appendix A.16 cho proof scheduling lý tưởng. Mean/p99/max observed chỉ là soft-realtime evidence, chưa WCET.

Deadline guard reserve thời gian cho suffix/decoder/identity; không chờ repair hết deadline rồi mới bypass. Output word finalization và acoustic chunk availability là hai latency khác nhau.

### 9.5. Streaming scope

Whisper default sử dụng sliding 30-s windows và autoregressive decoding [R23]. Main Whisper test là native buffered/offline; không gắn nhãn hard causal realtime. Một buffered low-latency configuration có thể là secondary setting nhưng phải tạo lại clean artifacts và báo latency thực, không đổi decoder/masks âm thầm.

Zipformer streaming và cache-aware FastConformer là hai deployment modes cho realtime study [R18], [R24]. Conformer CTC offline vẫn là target integration/accuracy. Checkpoint/context mới luôn tạo artifact mới.

<a id="s10"></a>
## 10. Dữ liệu, checkpoint và quyền truy cập

### 10.1. Clean source split

Các split sau là protocol đề xuất, không phải official split mới của corpus [R19]:

| Vai trò | Data | Quy tắc |
| --- | --- | --- |
| Basis-fit | LibriSpeech train-clean-100 | 20% speakers đầu theo hash seed 17 |
| Dictionary-fit | train-clean-100 còn lại | 80% speakers, không trùng basis-fit |
| Clean-selection | train-clean-360 | 100 speakers đầu theo hash; dùng tất cả utterances hợp lệ |
| Calibration | train-clean-360 | 100 speakers tiếp theo; mỗi speaker chọn 10 episodes theo hash |
| Clean stress-validation | train-clean-360: 100 speakers tiếp theo | Chỉ trước lock cho bugs/resource; không disease |
| Clean speaker holdout | train-clean-360 sau 300 speakers đầu | Không mở trước lock; kiểm tra block-risk E3 và synthetic evaluation E2/E10, không dùng selection |
| Final clean test | test-clean và test-other | Không fit/select/calibrate |
| Alignment audit/phone diagnostics | TIMIT official TRAIN/TEST | Protocol E1; không trộn TEST vào fit |

Hash ordering: SHA256(seed || corpus || speaker_id), rồi utterance ID. Resolve actual IDs, freeze manifests và báo số speakers/utterances sau exclusions. Nếu cross-split speaker overlap phát hiện, chuyển speaker theo thứ tự ưu tiên fit→selection→calibration, rồi chọn bù từ chưa dùng; ghi log trước feature extraction.

Calibration lấy episodes ≤30 s, không cắt giữa phone bằng transcript runtime. Nếu speaker thiếu 10 episodes hợp lệ, skip khỏi block-calibration và chọn speaker tiếp theo theo hash. Không dùng một người nhiều lần để đủ $$n=100$$.

Synthetic development dùng tối đa 100 utterances ≤30 s từ clean-selection, speaker-balanced theo hash; evaluation dùng tối đa 100 utterances từ 100 speakers đầu đủ điều kiện của clean speaker holdout, mỗi speaker một utterance theo hash. Freeze IDs trước extraction; không đưa các speakers đó vào development hoặc calibration. E3 có thể đánh giá cùng holdout speakers nhưng các phân tích không được gọi là các cohort độc lập. Thiếu số lượng thì báo actual counts, không lấy bù từ calibration/test.

Nguồn pretrained model/aligner có thể đã thấy corpus source hoặc disease; kiểm tra provenance khi có. “Không học bệnh” ở đây là quyền truy cập của PhonoRepair, không là chứng minh pretraining chưa từng thấy bất kỳ tiếng nói bệnh nào.

### 10.2. Disease test

| Corpus | Vai trò | Quyền dùng |
| --- | --- | --- |
| UASpeech [R25], [R33] | Primary word-level dysarthric recognition | Final blind evaluation |
| TORGO [R26], [R32] | Primary sentence-level và secondary words | Final blind evaluation |
| Healthy controls của hai corpus | Secondary domain-shift protection test | Không fit/calibrate trong setting chính |

Không dùng disease corpus để chọn layer, basis, radius, decoder prompt, threshold, pronunciation variants hoặc model nào “tốt nhất”. Không LOSO fine-tuning; toàn bộ speakers bệnh là held-out đối với PhonoRepair.

UASpeech đặt điều kiện sử dụng nghiên cứu và hạn chế thương mại/redistribution; TORGO công bố academic non-profit use [R25], [R26]. Kiểm tra agreement của bản dataset được cấp trước download. Không đóng gói raw corpus vào artifact được phát hành.

### 10.3. Corruption test

LibriSpeech test-clean kết hợp MUSAN noise [R27] và RIRs [R28], chỉ sau lock. Đây là acoustic corruption benchmark, không được gọi là mô phỏng bệnh và không dùng để fit model/gate. Recipe E6 là protocol mới của nghiên cứu; không tuyên bố official benchmark score.

### 10.4. Checkpoint matrix

| ID | Candidate checkpoint | Native decoding cố định | Vai trò |
| --- | --- | --- | --- |
| W1 | OpenAI Whisper base.en [R23] | English transcription, temperature 0, greedy, prompt rỗng, không carry previous-window text trong episode độc lập | Main accuracy/integration; buffered |
| Z1 | Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17 [R29] | Official native greedy transducer recipe và state contract | Main accuracy + streaming |
| C1 | nvidia/stt_en_conformer_ctc_large [R30] | Native greedy CTC, no external LM | Main Conformer accuracy/integration |
| C2 | nvidia/stt_en_fastconformer_hybrid_large_streaming_multi [R31] | Native streaming RNNT; optional CTC is separate preregistered run | Realtime extension và decoder portability |

Model cards/repos đã được kiểm tra ngày 13-10-2026. Exact revisions, files, hashes, runtime versions và native limits phải được resolve và ghi vào locked_run.json trước chạy test; không bịa commit SHA trong proposal. Nếu checkpoint unavailable, substitution phải được đăng ký trước mở target data, chọn theo architecture/context/resource policy, không theo disease WER.

Realtime đồng đều trên cả ba backbone không thuộc tiêu chí recognition chính. Nếu một mode đăng ký realtime không đạt SLA, báo thất bại extension đó, không thay đổi WER claim. C2 không thay thế âm thầm kết quả C1; W1 buffered không được dùng làm bằng chứng causal realtime.

Default decoding ở trên là benchmark baseline của nghiên cứu; không mặc định bằng mọi default của API. Baseline và repair dùng đúng cùng options, normalizer, model weights và context.

W1 và C1 không cần cùng parameter count với Z1: comparisons chính là paired within-checkpoint. Không suy rằng một model có WER tuyệt đối thấp hơn chứng minh adapter tốt hơn.


<a id="s11"></a>
## 11. Protocol đánh giá và thống kê chung

### 11.1. Baseline cố định và text normalization

Mỗi run phải so native baseline và repair trên cùng audio, checkpoint, decode options, precision và reset policy. Baseline không tự đổi prompt, LM, beam hoặc word lexicon khi bật repair. Không lấy từ đầu tiên trong hypothesis UASpeech để làm score đẹp hơn.

Normalizer v1 áp dụng đối xứng:
1. Unicode NFKC và lowercase.
2. Thay punctuation bằng space, giữ apostrophe nội bộ từ.
3. Collapse whitespace.
4. Với English, token là đúng một digit được map bằng bảng cố định: 0→zero, 1→one, 2→two, 3→three, 4→four, 5→five, 6→six, 7→seven, 8→eight, 9→nine. Multi-digit tokens giữ nguyên trong primary normalizer; không tự viết pronunciation khác theo hypothesis/corpus. Lưu version/hash.
5. Ghi raw transcript và normalized transcript. Các trường hợp normalization ambiguous được báo, không chỉnh theo model hypothesis.

Primary WER:

$$
\mathrm{WER}=\frac{S+D+I}{N_{\rm ref}}.
$$

Corpus WER cộng numerator/denominator toàn corpus, không trung bình utterance WER. Report thêm speaker-macro WER, CER, exact-match accuracy, $$S,D,I$$ riêng, và $$\Delta\mathrm{WER}$$ theo điểm phần trăm.

$$
\mathrm{WERR}_{\rm rel} =
\frac{\mathrm{WER}_{\rm base}-\mathrm{WER}_{\rm repair}}
{\mathrm{WER}_{\rm base}},
$$

chỉ khi baseline WER >0.

Một utterance đúng nguyên chuỗi khác với WER 0 trung bình. Báo baseline đúng→repair sai (harm), baseline sai→repair đúng (correction), cả hai đúng và cả hai sai. Report sửa latent nhưng transcript không đổi. WER có thể >1 do insertion.

### 11.2. PER và probes

PER là edit distance giữa phone sequences sau mapping chung, chia số phone reference. Không gọi frame accuracy hoặc transcript→G2P edit là acoustic PER.

Main ASR evaluation luôn dùng native decoder. Auxiliary phone probe chỉ được fit từ clean data và khóa trước target test; nó đo phone evidence của latent, không thay native ASR. Nếu không có phone reference đủ tin cậy, không báo PER như ground truth.

E1/E2 dùng frame routing accuracy với boundaries sạch để audit offline. Boundaries thật không được đưa vào runtime repair.

### 11.3. Paired CI và multiplicity

- Bootstrap cluster theo speaker; chọn session/utterances bên trong speaker khi cần hierarchical bootstrap. 10,000 replicates, seed 20261013.
- Mỗi replicate giữ baseline/repair cùng audio và recompute corpus numerator/denominator; không bootstrap từng frame độc lập.
- Primary disease family có 6 comparisons W1/Z1/C1 × UASpeech/TORGO. Report unadjusted 95% paired CI và simultaneous Bonferroni intervals mức $$1-0.05/6\approx99.17\%$$; success primary dùng simultaneous intervals.
- Clean family gồm 3 models; mọi quyết định đạt clean non-inferiority ở §4.2 dùng cận trên một phía Bonferroni $$1-0.05/3\approx98.33\%$$. Cận một phía 95% được báo thêm để mô tả, không thay cận quyết định khi thuận lợi.
- Secondary analyses/ablations dùng 95% CI và Benjamini–Hochberg hoặc label exploratory theo plan đã đăng ký. Không đổi primary family sau khi thấy kết quả.
- TORGO ít speakers: CI có thể bất ổn. Report leave-one-speaker-out sensitivity của paired effect, không coi nhiều utterances là nhiều speakers độc lập.

Cluster bootstrap là công cụ thực nghiệm xấp xỉ, không thay conformal theorem hoặc finite-population support certificate.

**Estimand và decision rule.** Primary effect là paired difference của corpus WER trên manifest cố định; speaker-macro effect là secondary. Bootstrap recompute ratio tổng edits/tổng reference words trong từng replicate, giữ paired outputs và trọng số dữ liệu tương ứng. Family sáu cặp được giữ ngay cả khi một backbone dừng feasibility hoặc unsupported; cell đó không có positive claim và không được dùng để giảm mức hiệu chỉnh của các cell còn lại.

Một claim intersection–union chỉ nói “cả sáu đều cải thiện” không tự bắt buộc Bonferroni nếu mọi test thành phần hợp lệ; nhưng nghiên cứu này còn báo simultaneous intervals và partial claims. Vì vậy giữ Bonferroni cho toàn bộ cell-level claims và các mức §4.3, không đổi rule sau test. Các mức này là tổ hợp của cell claims đã hiệu chỉnh, không phải test mới trên một tập winners được chọn.

**Power/sensitivity trước target.** Khóa một phân tích mô phỏng cho số speakers khả dụng, effect sizes và heterogeneity theo nhiều kịch bản, nêu nguồn giả định từ literature hoặc development sạch. Mô phỏng paired edit counts/reference lengths rồi chạy đúng estimator/CI trên, báo power, CI width và minimum detectable effect theo từng kịch bản; không gọi là power đã biết của dysarthria. Không dùng disease outcomes để chọn artifact, threshold hoặc primary family. TORGO ít clusters: report leave-one-speaker-out effects và kiểm tra độ ổn định đuôi bootstrap; nếu 10,000 replicates chưa ổn định thì tăng theo tolerance/seed rule khóa trước test. Mixed-effects analysis chỉ là sensitivity phụ được preregister, với model count/denominator phù hợp; không thay primary estimator vì nó cho kết quả đẹp hơn.

Point WERR ≥5% cùng CI ΔWER dưới 0 không đồng nghĩa lower bound WERR ≥5%. Khi baseline WER bằng 0 thì WERR không xác định; báo absolute effect và không đánh dấu cell đạt relative-reduction target. WER >100% do insertions vẫn hợp lệ; không clip score.

### 11.4. Không có bằng chứng khác với không suy giảm

Các trạng thái:
- **Tốt:** đạt target effect và CI phù hợp.
- **Xấu:** harm vượt budget với CI rõ, thất bại engineering gate, hoặc không có usefulness.
- **Inconclusive:** CI quá rộng để kết luận; tăng nguồn đánh giá độc lập hoặc thu hẹp claim.
- **Unsupported:** assumption/capability không đáp ứng; không gán accuracy bằng 0 để che lỗi tích hợp.

Không có significant regression chưa chứng minh non-inferiority. Gate không mở chưa chứng minh recovery. Một model tốt không chứng minh mọi model tốt.

### 11.5. Artifact lock và quyền test

Trước target run, lưu immutable bundle: artifact hash, checkpoint/runtime/config hashes, split/normalizer, E0–E10 recipe, seed, hardware và analysis code. Eval loader không được truyền target transcript/alignment vào encoder wrapper.

Disease-test và acoustic-corruption-test không được gọi bởi ArtifactBuilder. Một lần mở kết quả đích kết thúc trạng thái blind của phiên bản đó. Phân tích nguyên nhân được phép; thay đổi tiếp theo cần preregister phiên bản mới và confirmation data chưa dùng.

Bundle phải ghi vai trò từng synthetic dataset, recipe/seed, lịch sử policy development và danh sách nguồn từng được xem. Final calibration không phải vòng thử candidates: sau khi khóa một candidate mới mở calibration. Nếu gate rỗng sau bước đó, báo failure của phiên bản; không hạ beta, đổi envelope hoặc thử lại grid trên chính calibration. Một phiên bản mới cần development log và calibration units độc lập chưa dùng. Synthetic holdout và final clean test chỉ có vai trò đánh giá, không tái chọn candidate.

<a id="s12"></a>
## 12. Thực nghiệm E0–E10

Mỗi E dưới đây nêu benchmark, setup, metric, cách đánh giá và bước tiếp theo. E0/E1/E2-development → candidate lock → independent calibration → E2-holdout/E3 là các gate trước blind target. E4/E5/E6/E10-target chạy sau lock. Gate holdout chỉ quyết định status/tiếp tục, không cung cấp dữ liệu để chọn lại artifact. E7 shared-core audit, E8 state correctness và E9 overhead cơ bản là bắt buộc; realtime/onboarding extension được đánh dấu riêng. Các failure actions không cho phép tối ưu trên disease-test rồi gọi lại là blind success.

### E0. Kernel correctness và identity integration

**Mục đích:** xác nhận kernel và đường tích hợp hoạt động đúng trước khi diễn giải E1–E5. Đây là gate kỹ thuật, không phải bằng chứng clean non-inferiority hoặc WER improvement.

**Data:** bộ ca hình học ở Appendix A và một batch ngẫu nhiên có seed cố định; tối đa 100 utterances speaker-balanced từ clean stress-validation để kiểm tra native invocation. W1/Z1/C1 dùng đúng checkpoint, decoder mode và entrypoint của main run. C2 hoặc export khác chỉ kiểm tra nếu đã đăng ký extension tương ứng.

**Setup bắt buộc:**

1. **Kernel độc lập:** so projection/distance với reference float64 độc lập, gồm điểm trong/ngoài radius, $$s=0$$, radius 0, clipping, ties, gần ngưỡng và scale khác nhau; invalid shapes phải bị từ chối rõ ràng. Kiểm tra read/lift, phần trực giao được giữ và correction budget. Chạy bộ ca biên cùng một batch ngẫu nhiên tái lập; lưu số ca, seed và độ phủ nhánh. Quadratic-program reference chỉ cần cho các ca đại diện; không đặt 10,000 solves × 5 seeds làm điều kiện thành công.
2. **Gate đóng:** từ cùng initial state và native decoding options, so pipeline gốc, adapter gắn nhưng trả original features, và adapter được tháo. Đối chiếu tensor decoder thực sự nhận, valid lengths/masks, output tree, transcript và state ở các boundary có state. Không casting, tái tạo latent hoặc mutate in-place vùng features alias với encoder state. So transcript đơn thuần không đủ vì hai memory khác nhau có thể cho cùng transcript.
3. **Đường sửa hoạt động:** dùng toy artifact với quyết định nhận đã biết để xác nhận RepairCore tạo correction kỳ vọng. Sau đó ép một thay đổi hữu hạn, xác định trước trên valid encoder frame của clean utterance và kiểm tra tensor tại đúng memory consumer của native decoder thay đổi đúng vị trí/giá trị; lengths, masks, padding và unrelated state giữ đúng contract. Thay đổi ép này là probe tích hợp, không phải kết quả ASR hoặc candidate dùng để fit. Không yêu cầu transcript phải đổi. Kiểm tra callback/wrapper được gọi trên chính native entrypoint, kể cả đường thay features sau export nếu export thuộc run.
4. **State và mode:** với mode có state, so state ngay từ stream start và sau từng chunk khi gate luôn đóng; kiểm tra cache được tạo từ memory version phù hợp khi đường sửa mở. Mode offline không bị buộc giả lập persistent stream. E8 tiếp tục đánh giá carryover sau một correction thật; E0 chỉ chứng minh đường identity và ownership cơ bản.

**Metrics:** sai số projection/distance float64/float32, constraint violation, số ca/nhánh đã kiểm tra, số lần consumer nhận đúng tensor, mismatch features/lengths/masks/output tree/transcript/state, và alias/cache-version violations. Báo status riêng cho từng checkpoint/mode.

**Đạt gate:** kernel nằm trong tolerance đã khai báo trên miền test (targets ban đầu: float64 normalized error ≤$$10^{-10}$$; float32 error ≤$$10^{-5}$$ lần source RMS scale), không vi phạm constraint vượt tolerance; gate đóng không gây mismatch trong deterministic backend; probe đường sửa tới đúng consumer; không có lỗi mask, alias hoặc cache ownership. Với backend không deterministic, đo biến thiên khi lặp baseline cùng cấu hình, khóa backend nếu khả thi và chỉ claim mức tương đương thực sự kiểm chứng được. Tolerance đo hữu hạn không là universal rounding certificate.

**Không đạt:** sai khác khi gate đóng, đường correction không tới decoder, hook bị bỏ qua ở invocation đã đăng ký, mutation/cached memory sai version hoặc kernel lệch reference. Sửa binding/kernel rồi chạy lại E0 trước candidate lock, final calibration và disease test; nếu một mode không thể qua gate, báo unsupported cho mode đó thay vì dùng WER để bỏ qua lỗi.

### E1. Clean labels, time mapping và phone evidence

**Benchmark:** TIMIT official TRAIN/TEST [R22]; 200 utterances từ clean source để audit alignment. TIMIT không dùng làm calibration bệnh.

**Setup:**
- Dùng TIMIT hand annotations để so offline aligner; chọn 200 TRAIN utterances speaker-balanced theo hash, không chọn câu “dễ” bằng model.
- Phone mapping được khóa; report unmapped labels/closures.
- Check nominal frame-time mapping bằng impulses/length propagation và phone boundaries. Timestamp mapping không được suy từ attention weights rồi gọi là ground truth.
- Phone probe diagnostic: fit một linear CTC phone probe trên TIMIT TRAIN 80% speakers, chọn regularization trên 20% còn lại; cùng grids/optimizer policy cho mọi backbone. Evaluate TEST, remove SA sentences nếu muốn so protocol đó và ghi rõ trong manifest.
- Probe không được dùng online, không update ASR; không đưa TIMIT TEST vào ArtifactBuilder.

**Metrics:** boundary median/p95 absolute deviation, phone label agreement trên interior frames, frame routing accuracy, macro phone accuracy, probe PER trên TEST.

Tách ba bảng: (a) MFA so manual annotation; (b) nominal frame-time mapping; (c) latent phone geometry/probe. Label agreement ≥95% bên dưới thuộc (a), không là yêu cầu hidden-state phone classification đạt 95%. Alignment/timebase đúng chưa xác nhận geometry hữu ích. Peaky/blank là đặc tính output của decoder tương ứng; không gán trực tiếp nhãn blank cho mọi encoder state, và Z1 phải được phân tích theo transducer contract.

**Tốt:** proposed audit targets: median boundary deviation ≤20 ms, p95 ≤60 ms; label agreement ≥95% trên các frame cách manual boundary ≥20 ms; frame-time offset consistent với declared frontend. Đây là targets audit, không published MFA performance. Phone evidence baseline và repaired được so paired trong E10.

**Xấu:** một model có systematic offset, nhiều label không map, frame labels sai ngay trên interior hoặc probe gần majority predictor.

**Bước tiếp:** sửa timebase/subsampling contract; audit pronunciation lexicon và stress handling bằng clean data; tăng manual clean annotation; thử full-rank basis theo common grid. Nếu labels vẫn không đáng tin, không cấp routing certificate dựa trên chúng; native WER có thể tiếp tục như empirical exploratory setting với nhãn status rõ.

Report representation diagnostics riêng cho interior/transition và các phone classes, không bỏ transition khỏi WER. Đổi layer hoặc dùng phone posterior là biến thể mới cần boundary, uncertainty và proof/claim phù hợp; không là fallback tự động của policy hiện tại.

### E2. Dictionary feasibility, coverage và khoảng cách tới optimum

**Benchmark:** dictionary-fit và clean-selection LibriSpeech; TIMIT TEST chỉ là external diagnostic sau lock nếu không dùng cho selection; synthetic finite supports cho exact certificates.

**Setup:**
- Chạy shared candidate grid $$M,s,r$$, seeds anchor deterministic; optional repeated seeds là secondary robustness.
- Evaluate fit residual và unseen-source residual separately; không gọi test max residual là population $$\eta$$.
- Compute class distances, local gap, global lower separation, singleton feasibility và gate envelope.
- Trước lock, báo source diagnostics cho toàn grid với provisional threshold; sau final calibration, tính w của §7.7 cho đúng một candidate đã khóa. Nếu w=0, dừng usefulness và ghi cell failure; không tìm candidate thay thế bằng calibration.
- Với synthetic finite support đã liệt kê đầy đủ, $$r_{\rm cov}=0$$. Với speech support chưa có net bound, ghi support coverage “conditional”.
- Small finite anchor-only comparator: lấy 30 points/class và M nhỏ, solve exact discrete covering bằng enumeration/MILP với solver gap báo rõ. Chỉ so anchor objective cùng family. Affine optimum chỉ có lower bound 0 hoặc relaxation thực sự hợp lệ; best multistart là upper bound.

**Metrics:** $$\widehat E$$, p50/p95/p99/max residual, anchor radius, class-gap distribution, $$\underline\Gamma$$, $$\nu_{\rm global/local}$$, theoretical/working $$\eta$$, $$U_{\rm gap}$$, fit→selection growth, wrong routing, coverage status, storage và selection cost.

**Synthetic mechanism pilot, không mô phỏng bệnh:** dùng manifests §10.1; development seed 20261017, holdout seed 20261018. Với mỗi valid frame có source annotation hợp lệ, xác định nearest patch của annotated class để tạo perturbation offline. Chọn các hướng (i) vector Gaussian được chiếu lên tangent của patch, (ii) lên phần trực giao, (iii) về nearest point của nearest competing class. Chuẩn hóa hướng khác 0; nếu tangent dimension 0, không có competing class hoặc hướng suy biến thì skip hướng đó và báo counts, không thay bằng hướng thuận lợi.

Với mỗi hướng, chạy riêng các mức $$\|\xi_t\|=\kappa\nu$$, $$\kappa\in\{0,0.25,0.5,1,2\}$$; chuẩn hóa theo ν của artifact, không fit từ disease severity. Tạo chuỗi perturbed giữ cùng shape/lengths, lift qua Q, giữ phần trực giao với Q và padding nguyên. Decoder phải chạy lại native free-running từ cùng initial state/cache hợp lệ cho no-repair và repaired inputs. Annotation/patch thật chỉ ở bộ tạo perturbation và evaluator, không đi vào RepairCore. Đây là diagnostic hình học có dùng nhãn sạch offline, không phải một pipeline inference có oracle.

Report acceptance, accepted wrong-class rate so annotation và coverage annotation, correct→wrong/wrong→correct transcript, WER/S/D/I và latent displacement cho từng hướng/mức, không chỉ trung bình có lợi. Mức κ>1 cố ý ngoài threat budget; mức κ≤1 vẫn cần coverage/label assumptions để áp theorem. Development có thể thúc đẩy đăng ký policy mới trước lock theo §7.1, nhưng không thay xếp hạng candidate ngầm. Holdout chạy sau lock với candidate duy nhất và không dùng chỉnh phương pháp.

**Reject accounting:** ghi first exit theo đúng thứ tự REPAIR_FRAME (inactive/scope, below q_off, feasible empty, feasible multiple, numerical/deadline, correction budget) và acceptance. Báo thêm từng điều kiện độc lập trên cùng offline full-search scores để phân biệt gate nào che gate nào; không diễn giải first-exit counts như nguyên nhân độc lập.

**Tốt về feasibility:** candidate trong budget có positive declared radius lớn hơn numerical uncertainty, observed correct singleton trên source, w>0 sau calibration và có acceptance trong synthetic holdout. Corrected/harmed cases phải được báo dù bằng 0; chỉ có acceptance chưa là task usefulness. Muốn claim usefulness synthetic phải có corrected cases và báo toàn bộ harms. Nếu holdout không có acceptance/corrected case, ghi no observed usefulness và dừng mở rộng target trong phiên bản này; không suy từ một mẫu hữu hạn rằng mọi input đều bất khả sửa. Reconstruction đẹp hoặc positive toy chưa chứng minh WER bệnh.

**Xấu:** global separation 0, local singleton rất hiếm, incorrect routing ở clean interior, $$\eta$$ lớn khiến $$q_{\rm off}\ge\eta+\nu$$, hoặc budget tăng không cải thiện residual/gap.

**Bước tiếp:** trước lock, kiểm tra labels/time mapping; tăng M hoặc r trong grid chung; thêm clean speaker/context coverage; cải thiện bounded local radii hoặc minimax initialization. Nếu global gap xấu nhưng local regions tốt, giữ local selective design. Nếu mọi candidate không hữu ích, báo feasibility failure; không thêm disease learning để cứu cùng claim.

### E3. Bảo toàn clean baseline và false activation

**Benchmark:** LibriSpeech test-clean (primary WER), test-other (secondary healthy acoustic-domain shift), clean speaker holdout của train-clean-360 để kiểm tra speaker blocks; healthy controls TORGO/UASpeech chỉ là secondary sau lock [R19], [R25], [R26].

**Setup:**
- Native baseline và locked repair trên cùng utterances; independent episode reset theo cùng protocol.
- Với holdout speakers, lấy 10 episodes/speaker như calibration block. Không đưa true labels vào gate.
- Test-clean/test-other không ghép thành một exchangeable population nếu chưa có lý do; report riêng.
- Thử ba đường đã preregistered: attached identity, locked default $$q_{\rm off}$$, và calibration-only gate ablation. Không chọn winner sau test.
- Report zero activation và zero harm separately: intervention có thể không đổi transcript.

**Metrics:** corpus/speaker WER, $$\Delta$$WER, any-intervention per block, transcript-change per block, correct→wrong harm, repaired→same transcript, CI theo speakers. Report số calibration/test units thật.

**Tốt:** upper clean $$\Delta$$WER bound một phía Bonferroni 98.33% ≤+0.1 pp theo §11.3; block activation phù hợp β dưới assumptions; identity exact. Với zero events trên $$n$$ independent test blocks, upper 95% rate là $$1-0.05^{1/n}$$, không tự bằng 0. Đây là interval riêng cho intervention rate, không thay cận WER quyết định. Ít nhất 299 zero-event blocks mới có upper bound khoảng 1%; 99.9% target cần nhiều hơn, Appendix A.17.

**Xấu:** identity mismatch; source clean harm >budget; empirical tail-rate cao rõ; gate chỉ an toàn vì $$q=+\infty$$. Healthy-control target shift có thể fail exchangeability, phải report failure thay vì viện theorem như vẫn đúng.

**Bước tiếp:** identity mismatch→E0/cache; source false activation→collect thêm independent clean units, audit source coverage, recalibrate phiên bản mới; harm khi accepted→audit patch geometry/decoder response. Không chỉnh threshold từ healthy controls của disease corpus trong source-only main setting; nếu làm target-matched healthy calibration, báo setting riêng và test mới.

### E4. UASpeech: word-level dysarthric ASR

**Benchmark:** bản UASpeech được cấp quyền [R25], toàn bộ dysarthric speakers available trong release, các word blocks có reference lexical hợp lệ. Không fine-tune/LOSO.

**Setup:**
- Native decoder open vocabulary của W1/Z1/C1; không cung cấp word list target cho decoder trong primary setting.
- Primary channel M5 nếu release có microphone-labelled files đó. Nếu format/release không có M5, dừng và preregister channel binding trước decode; không chọn mic theo WER. Secondary chạy tất cả available channels, cluster repeated words theo speaker/session.
- Dùng mọi blocks hợp lệ theo manifest, report block IDs và recognized item count. Không mô tả all-block zero-shot protocol này là B2-only supervised protocol.
- Reference là intended prompt theo corpus. Không loại utterance vì speaker phát âm khó hoặc baseline sai.
- Predeclared exclusions chỉ missing/corrupt audio, missing reference và technical-invalid metadata. Report từng exclusion.
- Decode toàn hypothesis; giữ insertion. Severity/intelligibility metadata chỉ cho subgroup reporting sau lock.

**Metrics:** word-level corpus WER, exact-word accuracy, speaker-macro WER, S/D/I, relative reduction, corrected/harmed cases, acceptance rate theo severity/channel, paired CI.

**Tốt:** ≥5% relative WER reduction và simultaneous CI của $$\Delta$$WER <0; có corrected cases; clean E3 đạt. Exact-word accuracy tăng mà insertion WER tăng không đủ đạt primary criterion.

**Xấu:** WER tăng, severe speakers bị harm có hệ thống, singleton acceptance nhiều nhưng corrections ít, hoặc chỉ improvement sau oracle closed vocabulary.

**Bước tiếp:** sau run, phân tích residual/gap/accepted wrong-label patterns như exploratory; kiểm tra length/normalization/channel bugs. Candidate version mới chỉ thay shared core từ hypothesis có cơ sở và clean evidence; confirmation trên dữ liệu chưa dùng. Nếu decoder đã mất phone evidence, report scope limit thay vì tăng correction strength từ disease severity.

### E5. TORGO: sentence-level dysarthric ASR và phone diagnostics

**Benchmark:** TORGO [R26]. Primary: dysarthric recordings có orthographic reference và ≥2 words; head-mounted microphone path theo metadata. Single-word recordings là secondary riêng.

**Setup:**
- Chọn một acoustic channel/headMic theo contract, không double-count same utterance across microphones.
- Prompts xxx, missing transcript, non-word tasks và image prompts không có transcription đáng tin được loại theo rules đã đăng ký; không “suy transcript” từ tên ảnh.
- Spontaneous speech chỉ vào subset khi có actual orthographic transcript đủ tin cậy. Prompt text không mặc định là lời được nói nguyên văn.
- Các recording có phone annotation phù hợp microphone tạo PER subset; không giả định mọi file TORGO có PHN.
- W1/Z1/C1 cùng frozen artifacts source-only, native decoder. Ground-truth boundaries/PHN chỉ evaluator dùng sau decode.
- Report words và sentences riêng, không lấy isolated-word win đại diện dictation.

**Metrics:** primary sentence corpus WER, speaker-macro WER, CER, exact-sentence accuracy; secondary word WER; probe PER trên annotated subset, annotation coverage; S/D/I, harms/corrections, CI và leave-one-speaker-out sensitivity.

**Tốt:** sentence WER criterion như E4; insertion/deletion không bị che bởi PER improvement; effect không phụ thuộc một speaker duy nhất. PER tốt nhưng WER xấu là decoder/sequence failure.

**Xấu:** word subset tốt nhưng sentences xấu; repeated-token/deletion tăng; annotation coverage quá ít để suy luận PER; CI quá rộng thì inconclusive.

**Bước tiếp:** audit sequence/cache, timing assumption và native decoder influence. Clean-only temporal-state dictionary có thể là version mới, với budget và proof mới; không thêm oracle alignment vào runtime. Nếu transcript references không đủ, thu human annotations chỉ để evaluation, evaluator tách khỏi model fitting.

### E6. Biến dạng âm học chưa dùng để fit

**Benchmark tự xây:** LibriSpeech test-clean + MUSAN noise và SLR28 RIRs [R19], [R27], [R28].

**Setup:**
- Lấy tối đa 1,000 test-clean utterances bằng speaker-balanced hash selection; danh sách khóa trước decode.
- Additive noise: chỉ MUSAN noise category, SNR $$\{20,10,5,0\}$$ dB; noise file/offset theo hash(seed,utterance,SNR). Tile nếu noise ngắn; lưu IDs/offsets.
- Mix $$y=x+\alpha n$$, $$\alpha=\sqrt{\sum x^2/(\sum n^2\,10^{\mathrm{SNR}/10})}$$. Nếu clipping, scale toàn mixture bởi cùng constant và ghi factor; không clip cứng waveform.
- RIR: chọn 100 real RIR files theo hash từ release; assign theo utterance hash. Với RIR multichannel, lấy channel index 0; chuẩn hóa $$\ell_2$$ với floor $$10^{-8}$$, bỏ leading delay tới sample có absolute peak, rồi full convolution và giữ tail. Lưu shift/tail length; không chọn RIR bởi score.
- Reverberation+noise 10 dB là secondary fixed condition.
- Time stretch rates $$\{0.8,1.2\}$$ pitch-preserving là stress secondary: dùng librosa 0.11.0, effects.time_stretch(y, rate=f, n_fft=2048, hop_length=512), lưu environment/hash [R35]; rate 0.8 làm dài duration theo $$1/0.8$$, không nhầm rate với duration multiplier. Không giả định nó thỏa latent theorem.
- Chỉ evaluation sau lock; không chọn SNR/transform bằng kết quả repair.

**Metrics:** WER/CER theo condition, paired difference, clean-to-corrupted degradation, acceptance/correction/harm, noise amplitude, latency/RTF. Có paired waveform cho diagnosis encoder displacement khi frame correspondence hợp lệ; resampling/stretch làm shape thay đổi thì không tính Frobenius “true noise” bằng forced padding.

**Tốt:** consistent improvement ở nhiều conditions mà E3 vẫn đạt; report từng cell, không chọn cell tốt nhất.

**Xấu:** chỉ thắng một SNR, thấp SNR tạo confident wrong routing, thời gian biến dạng làm repair xóa token.

**Bước tiếp:** phân tích miền recoverable và cutoff; thêm clean recording diversity hoặc revise basis theo same source policy cho version mới. Không fit trên MUSAN/RIR test rồi tiếp tục gọi corruption unseen. Timing failure thúc đẩy nghiên cứu temporal repair riêng, không biến thành claim phiên bản hiện tại.

### E7. Portability và mức dễ onboarding

**Benchmark:** W1/Z1/C1; C2 và Whisper small.en là secondary checkpoints khóa trước test.

Main audit chỉ bắt buộc chứng minh shared core/policy trên ba checkpoint. Repeat-onboarding stopwatch, developer thứ hai và secondary checkpoints là extension; không trì hoãn main accuracy study để đạt chỉ tiêu số phút.

**Setup:**
- Main: audit bindings và core trên W1/Z1/C1. Extension: một developer tích hợp runtime đầu; developer khác dùng docs và source clean manifests để onboarding repeat checkpoint.
- Ẩn model display name khỏi RepairCore/ArtifactBuilder; đổi sang opaque ID, kết quả policy không được đổi.
- Nếu C2 hỗ trợ CTC/RNNT, thay native decoding config trong một run riêng nhưng không sửa core.
- Binding declarations được audit: không threshold/phone rule theo brand.
- Auto builders dùng cùng grid/selection/splits; d/r tự suy từ valid features. Failure ghi unsupported, không thêm manual exception.
- Ghi từng thay đổi, manual LOC, engineer active time và machine onboarding time separately. Existing library internals không bị tính là code PhonoRepair, nhưng mọi wrapper mới phải tính.

**Metrics:** core code delta, binding LOC, manual threshold count, engineer minutes, compute hours, artifact size, success rate, WER/clean harm trên từng backbone.

**Tốt main:** 3/3 main targets integrated; cùng core/policy, 0 brand branches và 0 manual thresholds. Điều này xác nhận mức tích hợp của §4.3, không đủ cho claim WER general. **Extension:** repeat same-contract checkpoint 0 manual code và ≤15 min setup; first interface ≤100 manual binding LOC excluding tests/docs là targets chưa đo.

**Xấu:** config trở thành interpreter chứa repair logic riêng, rename model đổi result, hoặc một backbone chỉ chạy được khi đổi dictionary algorithm bằng tay.

**Bước tiếp:** chuẩn hóa MemoryPacket/output-tree binding; wrapper invocation thay brand adapter; reject ambiguous contract. Nếu runtime mới thật sự cần thêm binding code, report limitation và adapter cost. Không đổi tên specialized repair thành registry để che specialization.

### E8. Stateful carryover: đoạn lỗi rồi đoạn sạch

**Benchmark:** 100 ordered source-clean episode pairs và 100 paired E6 corruption→clean pairs, với stream boundaries cố định; raw disease sequences chỉ dùng exploratory sau E4/E5.

State/cache correctness trên clean pairs là bắt buộc với mọi mode có state/history. Corruption→clean pairs chỉ chạy sau lock cùng E6; raw disease carryover không được dùng để tuning. Mode không có persistent state phải ghi rõ not applicable, vẫn kiểm tra same-state identity trên episode.

**Setup:**
- Compare (a) baseline không repair cả stream, (b) repair ở đoạn trước nhưng force identity ở đoạn sạch sau, (c) identity toàn stream, (d) cùng reset boundary chuẩn giữa hai đoạn.
- Nếu model native không hỗ trợ persistent stream, ghi buffered-window/history setting riêng; không giả lập cache contract chưa có.
- Compare clean suffix transcript và state trajectories. Same initial state chỉ được assert ở stream start/reset đã có trong baseline.
- Kiểm tra cache version đúng; encoder-state không nhận repair feedback.

**Metrics:** suffix transcript divergence, state divergence, duration of influence, identity recovery after legitimate reset, cache invalidations/recompute cost.

**Tốt:** identity toàn stream bằng baseline; reset setting đúng baseline contract khôi phục equivalence; carryover nếu có được đo và scope claim ghi rõ.

**Xấu:** cache memory stale, carryover xuất hiện dù identity từ đầu, hoặc đổi reset để đạt proof nhưng so với baseline khác protocol.

**Bước tiếp:** sửa version/invalidation và state ownership; chuyển claim bảo vệ sạch về whole episode bắt đầu bằng same state. Một clean suffix sau dirty history không được tự gán 99%/99.9% baseline-preservation theorem.

### E9. Latency, throughput, deadline và memory

**Benchmark:** clean stress-validation cho profiling trước lock; final long stream từ heldout clean/E6 evaluation. Z1/C2 native streaming; W1/C1 offline/buffered riêng.

Main panel bắt buộc đo artifact/scratch memory và repair/native end-to-end timing trên clean validation rồi xác nhận sau lock. Concurrency=4, long-stream deadline SLA, cold-start campaign và C2 thuộc extension nếu đăng ký; chi phí cơ bản không được bỏ vì paper không claim realtime.

**Setup:**
- Batch=1, concurrency=1 primary; concurrency=4 secondary, lock CPU/GPU/threads/power/precision.
- Warm-up 100 chunks; đo ≥10,000 chunks bằng real-time paced replay, synchronize device timing đúng boundaries.
- Cold-start 20 repeats; 30-minute stream memory/cache growth.
- Timing encoder, repair, decoder, I/O và end-to-end; không dùng riêng GPU kernel time làm latency sản phẩm.
- Native chunk/lookahead giữ nguyên cho baseline và repair. Word finalization và first partial transcript report riêng.
- Deadline guard có reserved identity/suffix budget; collect bypass count và drop count.
- Artifact builder CPU/GPU hours, energy nếu có meter, peak host RAM cũng được báo.

**Metrics:** repair/end-to-end p50/p95/p99/max observed, RTF, deadline miss%, dropped frames, peak artifact/scratch/state memory, cache slope, onboarding cost.

**Tốt main:** đạt repair overhead/memory budgets §4.2, không unintended frame drops, state/cache theo native contract. **Tốt extension:** eligible mode đạt realtime SLA đã đăng ký. W1 có thể đạt useful buffered RTF nhưng không đạt 250-ms streaming; kết quả này không phủ định riêng paired WER improvement.

**Xấu:** p99 tốt nhưng rare backlog/drop cao, guard chỉ quyết định sau deadline, timing hides overlaps/native buffering, hoặc artifact fit vượt resource budget.

**Bước tiếp:** vectorize/batch distances, fuse matmul/clip, reduce candidates qua common budget policy, compile graph và đo lại. ANN chỉ sau exclusion bounds; đổi context/chunk/rank cần artifact mới và clean revalidation. Hard realtime claim chỉ khi có scheduling/WCET evidence hợp lệ.

### E10. Ablation, cơ chế tác vụ và các phản ví dụ

**Benchmark:** source-clean selection trước lock; locked runs trên E3–E6 sau lock. Toy/adversarial mathematical cases Appendix A.18 không dùng train.

Synthetic matched-sequence diagnostic dùng recipe và development/holdout split của E2; không mở thêm vòng tìm noise direction, amplitude hoặc candidate dựa trên holdout. Core ablations ưu tiên identity, anchor-only/centroid, no-label, no-clean-gate và calibration-only; các ablation khác phải được đánh dấu secondary/extension trong lock bundle.

**Ablations phải đăng ký trước:**

| Ablation | Điều kiểm tra |
| --- | --- |
| Identity | Hiệu quả tổng và overhead |
| Anchor-only $$s=0$$, same M | Giá trị local affine directions |
| Nearest centroid clean gate | Giá trị nhiều patch/shape |
| Unlabelled projection | Giá trị routing label; report harm |
| No clean gate | Vai trò protection |
| Calibration-only vs max(calibration,envelope) | Trade-off clean tail và usefulness |
| Global-only vs local singleton | Giá trị sửa vùng cục bộ |
| Full-rank vs reduced basis | Thông tin mất do read space |
| Fixed M/r/s candidates | Budget–accuracy trade-off; không chọn disease winner |
| Projection without cache-correct binding | Negative control chỉ trong test harness |
| Entropy ln4 + sigmoid gate lịch sử | Optional diagnostic; không certified/deployed |

**Setup:** cùng checkpoint/clean splits. Nếu ablation thay score hoặc sửa feature distribution, nó cần calibration độc lập theo chính score đó; không reuse q của khác score. Khi thay encoder context phải refit toàn artifact. Artifact selection cuối vẫn không dùng disease ablation results.

**Metrics:** paired WER, probe PER, harms/corrections, acceptance, geometry, numerical/runtime costs; decoder output changes vs latent reconstruction changes. Với clean matched synthetic latent perturbations, plot task correctness đối chiếu $$\|\hat H-X\|$$; với disease không có true clean counterpart thì chỉ surrogate.

**Tốt:** default thắng identity và có đóng góp cơ chế rõ; local routing giúp usefulness mà không vượt clean harm budget; reconstruction và task accuracy được báo riêng.

**Xấu:** default không hơn anchor/centroid, gate làm tất cả identity, improvement chỉ do external decoder choice, hoặc entropy gate tạo confidence không phản ánh correctness.

**Bước tiếp:** giảm claim novelty/complexity nếu simpler baseline đủ tốt; chọn scope mà artifact dùng được. Các nghiên cứu nonlinear/temporal/decoder-directed repair thuộc version mới, cần clean-only protocol và proof tương ứng.

### Kết quả xấu không được xử lý bằng cách “chọn lại trên test”

Một bug tái lập độc lập từ E0 có thể sửa rồi rerun kỹ thuật. Nếu đã xem disease scores, run sửa là version mới; report cả bản lỗi và bản sửa khi thích hợp. Một thay đổi thuật toán vì failure mode đích là exploration, không được gọi confirmatory trên chính dữ liệu đó.

<a id="s13"></a>
## 13. Baselines, ablations và bảng báo cáo

### 13.1. Main baselines

1. Native frozen ASR identity.
2. Attached identity, đo integration overhead.
3. Same-budget anchor-only.
4. Nearest-centroid với protection policy sạch.
5. Exact affine projection không label routing.
6. Exact affine projection không clean gate.
7. Frozen native ASR với time-stretch cố định, không PhonoRepair: hai arms rate 0.8 và 1.2, pitch-preserving bằng librosa 0.11.0 với n_fft=2048, hop_length=512 như E6 [R35]. Rate 1.0 là native identity; không chọn arm thắng theo corpus/speaker/reference. Hai arms là secondary comparisons được preregister và báo đầy đủ trên clean/disease, có overhead riêng; không thêm chúng vào primary family sáu cặp của default repair.

Time-stretch baseline là preprocessing đối chứng, khác E6 dùng stretch tạo stress inputs cho repair. Giữ native decoder options và text normalization; xử lý audio dài sau stretch bằng chính native segmentation/window policy đã khóa, không cắt mất phần cuối hoặc chỉ lấy từ đầu hypothesis. Không fit rate vào bệnh hoặc gọi một rate cố định là normalization tối ưu. Nếu nghiên cứu adaptive tempo sau này, phải đăng ký estimator/source-only policy và evaluation mới.

SUTA/DSUTA [R06], [R07] chỉ là optional **separate target-adaptation setting**; chúng dùng input đích để cập nhật nên vi phạm quyền truy cập main PhonoRepair. Nếu chạy, báo optimizer/iterations/time/target access riêng; không gộp như cùng protocol. Supervised disease fine-tuning không cần chạy để thực hiện nghiên cứu này.

| Phương pháp/nhóm | Fit/development | Cập nhật từ target lúc sử dụng | Vai trò so sánh |
| --- | --- | --- | --- |
| Native, attached identity | Pretrained checkpoint cố định | Không | Main reference và integration |
| PhonoRepair, anchor/centroid/projection ablations | Clean source; artifact khóa | Không | Đối chứng cơ chế cùng quyền dữ liệu |
| Fixed time-stretch | Hai rates cố định, không chọn trên target | Không | Secondary preprocessing control |
| SUTA/DSUTA | Recipe TTA tương ứng; báo rõ source access nếu sửa recipe | Có, từ target không nhãn | Bảng riêng, khác setting và compute |
| Supervised dysarthric ASR [R01], [R37] | Có disease supervision/augmentation theo paper | Tùy paper | Literature context; không là upper bound |
| Enhancement/codebook/retrieval [R38]–[R40] | Theo objective/data của từng paper | Tùy phương pháp | Related mechanisms, không mặc định baseline cùng task |

Literature results phải kèm corpus release, split/speaker protocol, microphone, vocabulary/LM, decoder và quyền dữ liệu; thiếu thông tin thì ghi không đối chiếu trực tiếp. Không gọi một nguồn là SOTA hiện hành hoặc tính chênh lệch trực tiếp với PhonoRepair khi protocol không khớp. Nếu simpler baseline ngang/better, hạ claim về lợi ích của geometry thay vì chọn bỏ arm đó.

### 13.2. Bảng bắt buộc

~~~text
Table A — Data/model manifest
model revision | decoder mode | source counts | calibration units | exclusions

Table B — Geometry/coverage
M | r | s | fit max | selection max | eta provenance | gap | nu | w | status

Table C — Recognition
model | corpus/subset | N speakers | N words | baseline WER | repair WER
      | absolute delta | relative reduction | adjusted CI | S/D/I

Table D — Protection/usefulness
block activation | transcript change | harm | correction
                | first-exit reasons | independent gate failures
                | q_cal | q_off | w | finite threshold? | synthetic split/condition

Table E — Deployment
native context | p99 full/repair | RTF | memory | main/extension status
               | extension: misses | engineer/compute onboarding
~~~

Xuất utterance-level JSONL để tái tính metric. Không phát hành audio bệnh hoặc latent có thể bị hạn chế bởi agreement. Aggregate plots được export bằng standard plotting tools; axes/units rõ.

<a id="s14"></a>
## 14. Quy tắc xử lý kết quả xấu và tránh leakage

| Quan sát | Diagnosis đầu tiên | Hành động hợp lệ tiếp theo |
| --- | --- | --- |
| Gate off nhưng baseline khác | Binding/state/precision/scheduling | Sửa E0 trước khoa học |
| Q giảm phone evidence | Representation thiếu thông tin | Full-rank/common grid, clean labels tốt hơn |
| Residual nhỏ, WER xấu | Geometry không khớp decoder task | Decoder sensitivity audit; revise clean-only candidate objective |
| Global gap 0 | Support giao/transition | Local singleton và report rejected scope |
| Local routing confident sai | Coverage/label/noise budget assumptions sai | Audit scope; không dùng singleton tự chứng minh noise bound |
| q infinite hoặc q_off≥tau | Calibration resolution/envelope quá bảo thủ | Stop usefulness của candidate khóa; phiên bản mới cần development log và calibration units chưa dùng, không retry grid trên calibration cũ |
| Severe subgroup bị harm | Target nằm ngoài miền hoặc ambiguity | Thu hẹp claim; next-version confirmation chưa dùng |
| PER tốt nhưng WER xấu | Timing, blank/repetition, decoder | Sequence-aware research mới; giữ native decoder baseline |
| Latency xấu | Search/layout copies/native buffering | Shared optimization, artifact mới nếu đổi geometry/context |
| Chỉ 1 model tốt | Usefulness xuyên backbone chưa được xác nhận | Report model-specific result và mọi cell còn lại; phân biệt với integration portability; không thêm brand branch |
| Simplest baseline ngang/better | Contribution không đủ | Thu gọn phương pháp hoặc chứng minh ích lợi khác có đo |
| CI rộng | Thiếu independent speakers | Inconclusive; tăng cohort evaluation độc lập |

Stop gates:
- Không tích hợp được identity/state contract → không chạy target accuracy.
- Không artifact nào hữu ích trong budget → report feasibility failure.
- Không đạt clean non-inferiority → không deploy dù disease average tốt.
- Học bệnh để cứu kết quả → setting mới, không còn main claim.
- Không có decoder certificate → có thể chạy empirical study, nhưng không tuyên bố guaranteed WER.

Pilot thất bại không phải giấy phép đi tìm checkpoint/mode thắng trên bệnh. Trước target có thể sửa shared policy dựa trên development sạch đã khai báo, áp dụng lại cùng policy cho cả ba và ghi version. Sau khi xem calibration/holdout phải dùng dữ liệu độc lập mới cho những vai trò đã bị ảnh hưởng; sau khi xem disease results phải có confirmation target chưa dùng. Không thay denominator panel hoặc tiêu chí general của §4.3.

<a id="s15"></a>
## 15. Lộ trình và deliverables

| Phase | Việc phải làm | Exit gate |
| --- | --- | --- |
| P0 | Resolve rights, pin W1/Z1/C1, source roles, synthetic recipe và statistical rules | Builder không đọc target/calibration/holdout trong development |
| P1 | Một RepairCore; có thể làm C1 trước rồi nối sớm W1/Z1 | E0 trên cả ba; E7 shared-core audit, không chọn backbone theo bệnh |
| P2 | Tách alignment/timebase/geometry; E2 source và synthetic development theo policy | Có candidate khả thi hoặc báo failure; không mở disease |
| P3 | Khóa một candidate/checkpoint rồi calibration độc lập | Tính w, reject accounting; gate rỗng → failure, không tái chọn |
| P4 | E2 synthetic holdout, E3 final clean, E8 state và E9 overhead cơ bản | Báo pass/inconclusive/harm/no usefulness; holdout không dùng sửa candidate |
| P5 | Blind disease E4/E5, core ablations E10; E6 nếu đã đăng ký | Báo toàn panel, gồm cells dừng/unsupported; không cherry-pick |
| P6 | Extensions đã đăng ký: C2, repeat onboarding, realtime SLA, stress/carryover mở rộng | Báo riêng; không thay main checkpoints hoặc cứu primary failure |
| P7 | Analysis và paper | Claims theo ba mức §4.3; release configs/results và failure report |

Unit tests bắt buộc tập trung vào projection/clipping/s=0, tie-breaking, intervals, masks/padding, episode limits, identity/state, alternate streaming invocation, cache versions và deadline. Không viết test chỉ lặp lại implementation làm reference.

Deliverables: code reference; frozen artifacts; contracts; manifests; logs; evaluator; metric scripts; figures; proof assumptions checklist; failure report. Không giả định đã có training infrastructure trong workspace hiện tại.

Paper tập trung vào shared-core clean-only repair, usefulness/harm và cơ chế thành bại qua ba decoder. Giữ kiểm tra kỹ thuật thiết yếu dù không trình bày hết trong thân bài; chuyển nội dung sang appendix không được coi là đã giảm công thực hiện. Chưa có evidence để cam kết acceptance tại hội nghị.

<a id="s16"></a>
## 16. Claim được phép và giới hạn

**Có thể nói hiện tại:** có đặc tả một clean-only geometric repair framework với core chung; có chứng minh có điều kiện và protocol có thể triển khai; artifact onboarding dự kiến tự động sau semantic binding.

**Chỉ được nói sau evidence:** checkpoint/mode cụ thể có dictionary hữu ích; paired WER improvement trên corpus/split cụ thể; clean risk đạt target; deployment đạt latency trên hardware đã đo.

Claims general phải dùng đúng mức §4.3: tích hợp chung; hiệu quả trên nhiều backbone với các cặp được nêu tên; hoặc hiệu quả nhất quán toàn panel. Báo mọi harms/inconclusive/unsupported cùng với positives. Shared policy được phép chọn tham số khác theo clean data của mỗi checkpoint; không cho phép brand-specific repair hoặc manual thresholds để cứu một model.

**Không được suy ra:** 64 atoms phủ mọi manifold ngữ âm; PCA gần optimum minimax; entropy ln4 là hallucination detector; mọi clean input được nhận diện từ calibration hữu hạn; mọi backbone auto-bind từ weights; mọi distortion recoverable; phone routing đúng luôn cho WER đúng; offline Whisper mặc nhiên đạt 250-ms causal streaming.

Phần architecture mới trong tài liệu này hợp nhất các trao đổi sau Proof_1.tex v5. Appendix A là proof source tự chứa của proposal; Proof_4.tex chỉ cung cấp phần Voronoi/entropy được audit, không là justification cho gate sigmoid lịch sử.

<a id="app-a"></a>
## Appendix A. Chứng minh toán học

Các mệnh đề được viết cho số học chính xác trừ khi có nói khác. Không có định lý nào dưới đây tự xác nhận các giả định trên SSL/encoder output thực tế. Những phần bắt nguồn từ công cụ chuẩn được dẫn nguồn; proof được viết lại để tài liệu tự chứa.

### A.1. Bất khả phân biệt và giới hạn tổng quát

**Mệnh đề.** Nếu hai input với $$Y_1\ne Y_2$$ có thể tạo cùng toàn bộ quan sát $$o$$ mà bộ sửa truy cập, không thuật toán chỉ dùng $$o$$ và cùng source artifact luôn đúng cả hai.

**Chứng minh.** Thuật toán tất định trả cùng đầu ra cho cùng $$o$$, nên sai ít nhất một trường hợp. Với randomized algorithm, xác suất xuất $$Y_1,Y_2$$ là $$p_1,p_2$$, $$p_1+p_2\le1$$. Với hai trường hợp đồng xác suất, accuracy $$\le(p_1+p_2)/2\le1/2$$. Abstain giảm wrong decisions nhưng không khôi phục đầy đủ cả hai. $$\square$$

Encoder hằng trên hai lớp tạo đúng vấn đề này tại latent boundary. Chứng minh không nói speech thực luôn nhập nhằng; nó bác bỏ bảo đảm “mọi backbone/mọi biến dạng” không có điều kiện. Các impossibility results domain adaptation có vấn đề giả định nguồn–đích liên quan [R08].

### A.2. Read/lift và giữ thành phần trực giao

Cho $$Q\in\mathbb R^{d\times r}$$, $$Q^\top Q=I$$, $$u=Q^\top(h-\mu)$$. Với $$\hat h=h+Q(\hat u-u)$$:

$$
Q^\top(\hat h-\mu)
=Q^\top(h-\mu)+Q^\top Q(\hat u-u)=\hat u.
$$

Vì $$(I-QQ^\top)Q=0$$:

$$
(I-QQ^\top)(\hat h-\mu)=(I-QQ^\top)(h-\mu).
$$

Ngoài ra:

$$
\|\hat h-h\|^2
=(\hat u-u)^\top Q^\top Q(\hat u-u)
=\|\hat u-u\|^2.
$$

$$\square$$

Đây là bảo đảm tọa độ và biên độ, không là certificate semantics. Nếu $$r=d$$, $$Q$$ là orthogonal change of coordinates. Với gate identity, triển khai phải trả $$h$$ nguyên vẹn để tránh sai số không cần thiết.

### A.3. Exact projection lên patch và union

Cho $$S_j=\{a_j+B_jw:\|w\|\le\rho_j\}$$, $$B_j^\top B_j=I$$. Viết:

$$
u-a_j=B_jv_j+e_j,\qquad v_j=B_j^\top(u-a_j),\quad B_j^\top e_j=0.
$$

Với $$\|w\|\le\rho_j$$:

$$
\|u-a_j-B_jw\|^2=\|e_j\|^2+\|v_j-w\|^2.
$$

Minimizer của phần thứ hai là projection quả cầu:

$$
\operatorname{clip}_{\rho}(v)=
\begin{cases}
v\min(1,\rho/\|v\|),&v\ne0,\\
0,&v=0.
\end{cases}
$$

Suy ra công thức Section 8.1. Hợp hữu hạn compact patches đạt min; lấy nearest patch trên tất cả patches cho union projection. Ties theo smallest fixed ID tạo một phép chọn xác định. $$s=0$$ hoặc $$\rho=0$$ trả anchor. $$\square$$

Mixture của các patch projections có thể ra ngoài union. LISTA vài bước/ANN không có exclusion bounds chưa cho nghiệm này.

### A.4. Coverage từ mẫu và xác suất lấy mẫu

**Giả định.** $$K_p$$ compact không rỗng, sample $$X_p\subset K_p$$ nhãn đúng. Representation $$Q,\mu$$ và metric đã cố định trước khi dùng probabilistic sample argument. $$X_p$$ là $$r_{{\rm cov},p}$$-net của $$K_p$$.

Distance tới một tập không rỗng là 1-Lipschitz:

$$
|\operatorname{dist}(x,S)-\operatorname{dist}(y,S)|\le\|x-y\|.
$$

Với mỗi $$x\in K_p$$, có $$y\in X_p$$ cách không quá $$r_{{\rm cov},p}$$:

$$
\operatorname{dist}(x,S_p)\le r_{{\rm cov},p}+\widehat E_p.
$$

Vì $$X_p\subset K_p$$:

$$
\boxed{
\widehat E_p\le E_p\le\widehat E_p+r_{{\rm cov},p}.
}
$$

Lập luận đúng đồng thời cho mọi dictionary của cùng representation, kể cả dictionary được fit từ sample. $$\square$$

Nếu anchors thuộc sample, đặt $$R_p=\max_{x\in X_p}\min_{\ell_j=p}\|x-a_j\|$$. Anchor thuộc patch cho $$\widehat E_p\le R_p$$. Mỗi point trong patch cách một clean anchor không quá $$\rho_j$$, nên:

$$
d_H(K_p,S_p)
\le
\max\{\widehat E_p+r_{{\rm cov},p},\max_{\ell_j=p}\rho_j\}.
$$

**Probabilistic đủ điều kiện.** Có một fixed $$r_p/2$$-net của $$K_p$$ không quá $$N_p$$ centers; mọi ball radius $$r_p/2$$ tại center có mass ≥$$b_p>0$$. Với $$n_p$$ i.i.d. samples mỗi lớp, probability một ball rỗng:

$$
(1-b_p)^{n_p}\le e^{-n_pb_p}.
$$

Gọi $$\mathcal E_{\mathrm{cov}}$$ là biến cố các mẫu không đồng thời tạo thành các $$r_p$$-net. Union bound cho:

$$
P(\mathcal E_{\mathrm{cov}})
\le
\delta_{\rm cov}:=\min(1,\sum_pN_pe^{-n_pb_p}).
$$

Nếu mọi ball có sample, hai khoảng cách $$r_p/2$$ cộng thành $$r_p$$. $$\square$$

Compact chỉ bảo đảm finite net, không cung cấp $$N_p,b_p,r_p$$ hữu ích. Frames cùng speaker/recording không thỏa i.i.d. tự động. Chọn representation/net parameters từ cùng sample cần bound adaptive/simultaneous riêng hoặc sample độc lập sau khóa representation.

### A.5. Farthest-first và giới hạn PCA

Farthest-first trên finite $$X$$, $$m$$ anchors, cho covering radius $$R_{\rm FF}$$. Chọn $$m$$ anchors và point xa nhất cuối cùng: mọi cặp trong $$m+1$$ points cách nhau ≥$$R_{\rm FF}$$. Bất kỳ $$m$$ centers phủ chúng với radius $$R$$ phải có hai points chung center, nên:

$$
R_{\rm FF}\le2R.
$$

Lấy infimum:

$$
\boxed{R_{\rm FF}\le2R^\star_{\rm anchor}.}
$$

Đây là packing argument của farthest-first [R05]. Nó áp dụng anchor covering objective, không thay $$R^\star_{\rm anchor}$$ bằng affine optimum. $$\square$$

PCA quanh fixed anchor $$a$$ tối thiểu:

$$
\sum_i\|(I-BB^\top)(x_i-a)\|^2
$$

với $$B^\top B=I_s$$, vì objective là $$n\operatorname{tr}G-n\operatorname{tr}(B^\top GB)$$, $$G=n^{-1}\sum_i(x_i-a)(x_i-a)^\top$$, và leading eigenvectors tối đa hóa trace.

**Phản ví dụ minimax.** Một lớp có support $$\{0,\pm e_1,\pm A e_2\}$$, $$A>1$$. Fit gồm $$N$$ points cân bằng ở $$\pm e_1$$, một ở mỗi $$\pm A e_2$$, một ở 0; $$N>2A^2$$. Fixed anchor 0, một 1D patch radius A. PCA chọn ngang, max-residual A; patch dọc có max-residual 1. Với $$r_{\rm cov}=0$$, ratio tới optimum ≥A và unbounded. $$\square$$

Vì vậy explained variance/mean squared loss chưa bảo đảm gần optimum max-residual.

### A.6. Khoảng cách tới dictionary lý tưởng

Cố định cùng representation, metric, labels và feasible family $$\mathfrak D_B$$ gồm budgets/anchor domain/radius/constraints đã khai báo. Đầu ra $$\widehat{\mathcal D}$$ thuộc family. Đặt:

$$
E=\max_pE_p,\quad\widehat E=\max_p\widehat E_p,\quad
E_B^\star=\inf_{\mathcal D\in\mathfrak D_B}E(\mathcal D),\quad
\widehat E_B^\star=\inf_{\mathcal D\in\mathfrak D_B}\widehat E(\mathcal D).
$$


$$
\Delta_{\rm opt} =
\widehat E(\widehat{\mathcal D})-\widehat E_B^\star,\qquad
r_{\rm cov}=\max_p r_{{\rm cov},p}.
$$

Coverage cho mọi dictionary:

$$
E(\mathcal D)\le\widehat E(\mathcal D)+r_{\rm cov},\quad
\widehat E(\mathcal D)\le E(\mathcal D).
$$

Suy ra $$\widehat E_B^\star\le E_B^\star$$ và:

$$
\boxed{
0\le E(\widehat{\mathcal D})-E_B^\star
\le r_{\rm cov}+\Delta_{\rm opt}.
}
$$

Nếu $$L\le\widehat E_B^\star$$ là lower bound hợp lệ:

$$
E(\widehat{\mathcal D})-E_B^\star
\le r_{\rm cov}+\widehat E(\widehat{\mathcal D})-L.
$$

$$\square$$

Lower bound 0 hợp lệ nhưng có thể lỏng. Best multistart là feasible upper bound, không phải L. Family phụ thuộc sample được so theo chính family ấy; không âm thầm đổi thành oracle có anchors tự do. Đổi Q/metric làm thay comparator.

### A.7. Margin/Voronoi dùng được từ Proof_4

Với centroid $$\mu_p\ne\mu_q$$, $$u$$ thuộc closed Voronoi cell $$p$$:

$$
b_{pq}(u)=
\frac{\|u-\mu_q\|^2-\|u-\mu_p\|^2}
{2\|\mu_q-\mu_p\|}\ge0.
$$

Đây là distance có dấu tới pairwise bisector, hướng vào phía p. Nếu $$\|u'-u\|\le\delta$$, $$b_{pq}(u')\ge b_{pq}(u)-\delta$$. Vì thế $$b_{pq}(u)\ge\delta$$ giữ thuộc closed halfspace; dấu $$>$$ loại tie.

Viết margin $$m_{pq}=\|u-\mu_q\|-\|u-\mu_p\|$$ và $$R_{pq}=(\|u-\mu_p\|+\|u-\mu_q\|)/2$$:

$$
b_{pq}=\frac{R_{pq}}{\|\mu_q-\mu_p\|}m_{pq}.
$$

Tam giác cho $$R_{pq}\ge\|\mu_q-\mu_p\|/2$$, nên $$m_{pq}>2\delta$$ bảo đảm strict safety. Exact bisector expression chỉ áp dụng centroid, không affine patches. $$\square$$

**Generalization tới class unions.** $$d_p(u)=\operatorname{dist}(u,S_p)$$ là 1-Lipschitz. Với $$p$$ gần nhất duy nhất:

$$
g(u)=\min_{q\ne p}[d_q(u)-d_p(u)].
$$

Do difference là 2-Lipschitz:

$$
g(u)>2\delta\Rightarrow d_q(u')>d_p(u')\quad\forall q\ne p.
$$

Đây là bảo toàn observed nearest-class, không chứng minh observed class là thật. $$\square$$

### A.8. Routing đúng cục bộ và radius global

Giả sử $$x\in K_p$$, $$E_p\le\eta$$, $$u=x+\xi$$, $$\|\xi\|\le\nu$$. Khi đó:

$$
d_p(u)\le\|\xi\|+\operatorname{dist}(x,S_p)\le\nu+\eta.
$$

Vì vậy true p thuộc $$\mathcal P(u)=\{q:d_q(u)\le\eta+\nu\}$$. Nếu singleton, $$\mathcal P(u)=\{p\}$$. Equivalent implementable condition:

$$
d_{(1)}(u)\le\eta+\nu<d_{(2)}(u).
$$

$$\square$$

Nếu $$\Gamma=\min_{p\ne q}\operatorname{dist}(S_p,S_q)>2(\eta+\nu)$$, không có hai lớp trong ball quanh u: hai nearest points khi đó cách nhau ≤$$2(\eta+\nu)$$, contradiction. Global nearest class đúng cho mọi point trong threat domain.

Vì mọi point trên patch cách anchor ≤radius:

$$
\Gamma\ge\underline\Gamma =
\max\{0,\min_{\ell_j\ne\ell_k}(\|a_j-a_k\|-\rho_j-\rho_k)\}.
$$

Nếu $$\underline\Gamma>2\eta$$, với fixed $$0<\lambda<1$$:

$$
\nu=\lambda(\underline\Gamma/2-\eta)
$$

cho $$2(\eta+\nu)=\lambda\underline\Gamma+2(1-\lambda)\eta<\underline\Gamma$$. Proposal cố định $$\lambda=1/2$$. $$\square$$

Local singleton không cần global separation. Noise budget vẫn là giả định; singleton/residual không ước lượng noise thật hoặc severity.

### A.9. Nhãn patch, coarse error và local denoising

Với head phụ affine $$g_c(v)=w_c^\top v+b_c$$, patch j label p:

$$
\kappa_j=\min_{q\ne p}
[g_p(a_j)-g_q(a_j)-\rho_j\|B_j^\top(w_p-w_q)\|]>0.
$$

Mọi $$v=a_j+B_jw$$ trong patch có:

$$
g_p(v)-g_q(v)
\ge g_p(a_j)-g_q(a_j)-\rho_j\|B_j^\top(w_p-w_q)\|>0.
$$

Vậy head phụ trả p trên patch. Head phụ đúng không là native-decoder certificate. $$\square$$

Chiếu đúng lớp cho:

$$
\|\hat u-x\|
\le\|\hat u-u\|+\|\xi\|
=d_p(u)+\|\xi\|\le\eta+2\nu.
$$

Nếu clean x thuộc mô hình đúng patch:

$$
x=a+Bw+e,\quad B^\top e=0,\quad \|w\|\le\rho,\quad\|e\|\le\varepsilon_{\rm loc},
$$

thì $$B^\top(u-a)=w+B^\top\xi$$. Projection quả cầu là nonexpansive:

$$
\|\operatorname{clip}_\rho(w+B^\top\xi)-w\|\le\|B^\top\xi\|.
$$

Hai components trực giao nên:

$$
\boxed{
\|\hat u-x\|^2\le\varepsilon_{\rm loc}^2+\|B^\top\xi\|^2.
}
$$

$$\square$$

Nonexpansiveness có thể chứng minh trực tiếp: với projections $$p=\Pi_C(a)$$, $$q=\Pi_C(b)$$ trên ball lồi, điều kiện tối ưu cho $$\langle a-p,q-p\rangle\le0$$ và $$\langle b-q,p-q\rangle\le0$$. Cộng và sắp xếp cho $$\|p-q\|^2\le\langle p-q,a-b\rangle\le\|p-q\|\|a-b\|$$, suy ra $$\|p-q\|\le\|a-b\|$$.

Đúng lớp chưa bảo đảm đúng local patch. Một sufficient patch stability condition là $$d_k(x)-d_j(x)>2\nu$$ với mọi competing patch k. Local denoising không bảo đảm improvement khi approximation residual lớn hơn thành phần noise bị loại.

### A.10. Clean envelope và zero intervention

Giả sử $$A(x)=\min_p d_p(x)\le\eta$$ với mọi declared clean point. Gate chỉ mở nếu:

$$
A(u)>\max(q_{\rm cal},\eta).
$$

Với u=x clean, điều kiện không bao giờ đúng. Vậy R(x)=x trên toàn support. Với same initial state và execution contract, toàn episode bằng baseline bằng quy nạp. $$\square$$

Nếu coverage event có probability ≥$$1-\delta_{\rm cov}$$, zero-intervention conclusion đúng đồng đều trên event ấy. Nếu $$\eta$$ chỉ empirical maximum, kết luận population không theo.

**Số học:** nếu implemented score $$\tilde A$$ có certified error $$|\tilde A-A|\le\epsilon_A$$, dùng $$q_{\rm off}\ge\eta+\epsilon_A$$ để zero-intervention conclusion giữ. Hoặc chỉ mở khi certified lower bound của A vượt η, đồng thời implemented calibration score vượt q_cal. Chưa có error bound thì zero-intervention claim chỉ cho ideal arithmetic.

### A.11. Calibration hữu hạn mẫu và bảo vệ baseline

Khóa pipeline và score; n calibration scores $$Z_i$$ và score mới $$Z_{n+1}$$ exchangeable. Với $$\beta\in(0,1)$$:

$$
k=\lceil(n+1)(1-\beta)\rceil,\quad
q=Z_{(k)}\ (k\le n),\quad q=+\infty\ (k>n).
$$

Random tie-breaking cho rank uniform trong n+1 slots; strict exceedance $$Z_{n+1}>q$$ nằm trong event rank>k. Do đó:

$$
P(Z_{n+1}>q)\le\frac{n+1-k}{n+1}\le\beta.
$$

Ties làm bound conservative. $$\square$$

Lấy $$Z_i=\max_{k\le K_{\rm ep},t}A(u^0_{i,k,t})$$. Prefix score không phụ thuộc repairs downstream. Có bất kỳ intervention trong block kéo theo $$Z_{\rm new}>q_{\rm cal}$$; mọi gate bổ sung chỉ thu hẹp event.

$$
P_{\rm clean}(\mathcal I_{\rm block})\le\beta.
$$

Nếu khi không repair, tensor/state/decoder contract bằng baseline:

$$
P_{\rm clean}(Y_{\rm repair}\ne Y_{\rm base}\text{ trong block})\le\beta.
$$

Exchangeability cần giữa **đơn vị calibration được chọn**, không là độc lập giữa frames. Group theo speaker giúp nhận diện phụ thuộc, nhưng không tự chứng minh các source/new-speaker blocks exchangeable. Bound marginal, không tự conditional sau observing q hoặc subgroup.

Finite threshold cần:

$$
n\ge\lceil1/\beta\rceil-1.
$$

Với J blocks có cùng marginal guarantee, union bound chỉ cho min(1,Jβ), không bảo vệ stream vô hạn.

Nếu repair WER≤$$W_{\max}$$ trên nonempty-reference episodes:

$$
\mathbb E[(\mathrm{WER}_{\rm repair}-\mathrm{WER}_{\rm base})_+]
\le W_{\max}\beta.
$$

Proof: loss increase zero nếu không intervention, ≤Wmax nếu có; lấy kỳ vọng. Output cap $$N_{\rm out}^{\max}$$ và reference length≥$$N_{\rm ref}^{\min}\ge1$$ cho conservative $$W_{\max}=1+N_{\rm out}^{\max}/N_{\rm ref}^{\min}$$. $$\square$$

### A.12. Sai số chuỗi sau read/lift

Gọi clean raw latent $$h_t^\star$$, corrupted $$h_t$$, $$\xi_t=h_t-h_t^\star$$. Sau sửa:

$$
Q^\top(\hat h_t-h_t^\star)=\hat u_t-u_t^\star,
$$


$$
(I-QQ^\top)(\hat h_t-h_t^\star)
=(I-QQ^\top)\xi_t.
$$

Do orthogonality:

$$
\|\hat H-H^\star\|_F^2 =
\sum_t\|\hat u_t-u_t^\star\|^2
+\sum_t\|(I-QQ^\top)\xi_t\|^2.
$$

Cho accepted set $$\mathcal I$$ đúng local patch, A.9 cho:

$$
E_{\rm seq}^2\le
\sum_{t\in\mathcal I}
[\varepsilon_{{\rm loc},t}^2+\|B_{j_t}^\top Q^\top\xi_t\|^2]
+
\sum_{t\notin\mathcal I}\|Q^\top\xi_t\|^2
+
\sum_t\|(I-QQ^\top)\xi_t\|^2.
$$

Nếu chỉ biết đúng lớp, thay accepted term bằng $$(\eta+2\nu)^2$$. Numeric projection error norm total≤$$\epsilon_{\rm seq,num}$$ cho $$E_{\rm actual}\le E_{\rm ideal}+\epsilon_{\rm seq,num}$$, không bỏ cross terms bằng cách cộng squared bounds sai.

Không có bound phần trực giao thì không suy ra E_seq hữu hạn hữu ích. Phần đó được phép arbitrary chỉ nếu native task thật sự invariant với nó trên miền xét.

### A.13. Greedy CTC và finite lexicon

**Greedy CTC.** Với clean $$H^\star$$ và actual $$\hat H$$ cùng shape, fixed suffix logits $$G_{t,c}$$. Unique clean winner $$\pi_t^\star$$, margins:

$$
m_{t,c}=G_{t,\pi_t^\star}(H^\star)-G_{t,c}(H^\star)>0.
$$

Nếu hiệu logits $$L_{t,c}$$-Lipschitz trên miền chứa hai sequences và:

$$
L_{t,c}E_{\rm seq}<m_{t,c}\quad\forall t,c\ne\pi_t^\star,
$$

thì margin tại repaired sequence ≥$$m_{t,c}-L_{t,c}E_{\rm seq}>0$$; winners giữ nguyên. Fixed CTC collapse trả same transcript [R13]. Nếu clean transcript đúng, repaired WER=0; nếu clean sai, chỉ được same transcript/WER như clean. $$\square$$

**Finite lexicon.** Với transcript set $$\mathcal W$$ chứa truth $$w^\star$$, pronunciation map $$\phi$$ đơn ánh, minimum edit distance $$d_{\min}>0$$. Nếu predicted phone sequence $$\hat a$$ có edit error≤k và $$2k<d_{\min}$$, thì với $$w\ne w^\star$$:

$$
d_{\rm edit}(\hat a,\phi(w))
\ge d_{\min}-k>k
\ge d_{\rm edit}(\hat a,\phi(w^\star)).
$$

Nearest-edit decoder trả truth duy nhất. $$\square$$

k bao gồm lỗi segmentation/insertion/deletion. Homophones phá injectivity. Lexicon theorem không được áp cho native Whisper hoặc free dictation nếu không có đúng lexicon decoder.

### A.14. Hợp đồng quyết định chung cho decoder xác định

Xét deterministic decoder program trên một bounded horizon, fixed initial state/config. Clean execution có finite decision trace; mọi comparison cần để chọn branch có strict margin $$m_i>0$$.

Định nghĩa $$f_i(H)$$ là score difference cho comparison i khi chương trình replay cùng preceding discrete choices như clean trace; continuous states/caches vẫn được tính từ H qua các preceding operations. Giả sử bounds sound:

$$
|f_i(\hat H)-f_i(H^\star)|
\le L_i^\parallel E_\parallel+L_i^\perp E_\perp<m_i.
$$

Comparison đầu giữ sign; bằng induction, preceding branch history của comparison kế cũng đúng như replay. Vì tất cả comparisons giữ sign, toàn trace, termination và output giữ nguyên. $$\square$$

Với beam search phải include branch/pruning/reordering/termination relevant, không chỉ final token argmax. Teacher forcing một history đúng không tự chứng minh free-running history khi chưa có all-step induction. Random sampling không thuộc theorem nếu chưa cố định và mô hình hóa randomness phù hợp.

Khuôn này không cho free, tight, automatically computable bounds cho mọi model. Whisper/RNNT cần state/cache domain và actual decoder graph; đây là research requirement, không deploy module đã hoàn thành.

### A.15. Điều khiển head affine: trường hợp giới hạn riêng

Nếu boundary nằm ngay trước affine native head $$g_k(h)=w_k^\top h+b_k$$, target c đã được xác định đúng độc lập, và clean anchor $$v_c$$ có:

$$
\kappa_c=\min_{k\ne c}[g_c(v_c)-g_k(v_c)]>0,\quad\mu_c=\kappa_c/2,
$$

đặt $$d_k(h)=g_c(h)-g_k(h)$$. Với $$d_k(h)<\mu_c$$:

$$
\alpha_k=\frac{\mu_c-d_k(h)}{d_k(v_c)-d_k(h)},
$$

còn $$\alpha_k=0$$ nếu $$d_k(h)\ge\mu_c$$. Chọn $$\alpha=\max_k\alpha_k$$, $$\hat h=(1-\alpha)h+\alpha v_c$$. Head affine cho:

$$
g_c(\hat h)-g_k(\hat h)
=(1-\alpha)d_k(h)+\alpha d_k(v_c)\ge\mu_c>0.
$$

Vậy native head trả c. $$\square$$

Đây là proof khả năng task-directed latent repair, không là core chung encoder-output đã chọn trong proposal. Không cần giảm distance tới clean x. Target correctness vẫn là nút thắt. Sửa ở final head về token có thể tương đương override token bằng router; nếu nghiên cứu mode này cần baseline override.

### A.16. Streaming scheduling và deadline

Một worker tuần tự; chunks đến mỗi Δ; service worst-case $$C_{\rm tot}\le\Delta$$. Chunk 1 hoàn thành trước arrival chunk 2; induction cho mọi chunk không có backlog. Earliest sample trong chunk chờ tối đa Δ để đủ chunk, thêm lookahead ℓ và compute C_tot, nên latency≤ℓ+Δ+C_tot.

Cận không gồm word-finalization future context ngoài ℓ đã khai báo, không chứng minh một offline backbone tự thành streaming, và không suy ra WCET từ p99. Guard phải reserve completion budget trước repair bắt đầu. $$\square$$

### A.17. Zero observed harms và sample-size resolution

Với n independent Bernoulli harm indicators có probability h, probability 0 events là $$(1-h)^n$$. Upper one-sided confidence $$1-\delta$$:

$$
h_U=1-\delta^{1/n}.
$$

Với $$\delta=0.05$$, khoảng 299 zero-event independent units cần để $$h_U\le0.01$$; khoảng 2995 để $$h_U\le0.001$$. $$\square$$

Đây là test-confidence bound trên harm probability, khác số calibration units và khác expected WER magnitude. Nếu observations clustered phải dùng independent blocks hoặc analysis dependence hợp lệ.

### A.18. Counterexamples và positive toy bắt buộc

**Low entropy không sparse.**

$$
P=(0.99,0.0025,0.0025,0.0025,0.0025)
$$

có H≈0.06986<ln4 nhưng support size 5.

**Entropy theorem đúng.** Với k positive probabilities, Jensen:

$$
H(P)=\sum_iP_i\ln(1/P_i)
\le\ln\!\left(\sum_iP_i/P_i\right)=\ln k.
$$

Vậy sup trên ≤s support là ln s, attained uniform; H>ln s implies support>s. Đây là support result, không “phase transition” của recognition.

**Convex hull theorem có điều kiện.** Nếu toàn centroid set độc lập affine, representation $$z=\sum_iP_i\mu_i$$, $$\sum P_i=1$$, là duy nhất. Một representation với ≤s positive weights sẽ có cùng P, contradiction khi support>s. C≤d+1 chỉ là điều kiện cần, không certificate. Sau rank reduction C>r+1 làm affine independence bất khả.

**Convex sparsity khác linear sparsity.** Các centroids $$e_1,e_2,e_3,e_4,2e_1$$ độc lập affine; uniform 5 weights có H=ln5>ln4 nhưng mean $$0.6e_1+0.2e_2+0.2e_3+0.2e_4$$ thuộc span 4 atoms. Không áp convex theorem cho linear UoS.

**Outside nhưng arbitrarily close.** Với centroids $$(0,0),(1,0),(2,\epsilon)$$, $$\epsilon>0$$, uniform mean $$(1,\epsilon/3)$$ ngoài mọi 2-point hull nhưng cách single centroid $$(1,0)$$ chỉ ε/3→0. Strict exclusion không cho useful distance bound đồng đều.

**Baseline false activation.** Dictionary points±1; K_plus chứa 1 và 1.08; η=0.1, q_cal=0 có thể xảy ra khi calibration không gặp rare clean variant. Clean u=1.08 có score .08 nên gate cũ mở, gate max(q_cal,η) đóng. Nếu η chưa support-certified thì chỉ minh họa, không proof thực tế.

**Outside-budget wrong routing.** Patches $$S_\pm=\{(v,\pm1):|v|\le0.1\}$$, clean x=(0,1), corrupted u=(0,-1.2). Score tới S_minus=.2; singleton có thể nhận sai lớp khi τ=.5. Actual noise 2.2 ngoài certified budget .4. Không dùng singleton tự chứng minh noise budget.

**Positive clean-only repair.** Cùng $$K_\pm=S_\pm$$, anchors $$(0,\pm1)$$, tangent $$e_1$$, radius .1; native illustrative head $$g_+=4u_1+u_2,\ g_-=-g_+$$ đúng trên clean patches với margin≥1.2. Conservative η=.1, $$\underline\Gamma=1.8$$ cho ν=.4, τ=.5. Clean score 0, q_off=.1. Sau lock xét x=(0,1), noise=(-.3,0); score=.2, singleton plus. Baseline g_plus=-.2<.2, projection=(-.1,1) cho g_plus=.6>-.6. Recovery xảy ra mà không fit disease. Đây là existence toy, không speech benchmark.

**Task đúng dù latent xa hơn.** Native scores $$g_A(h)=h-.9,\ g_B(h)=0$$; K_A={1,2}, K_B={-1}. Truth x=1, observed h=.8, routing A đúng với ν=.5. Head raw chọn B. A.15 dùng anchor2, μ=.55 cho α=13/24, repaired h=1.45, head chọn A. Distance tới x tăng .2→.45. Vì vậy reconstruction decrease không là điều kiện cần của task recovery.

### A.19. Tail mass nếu nghiên cứu mixtures sau này

Đặt $$\delta_s(P)=1-\sum_{\mathrm{TopS}}P_i$$. Với mọi s-sparse Q, mass outside supp(Q)≥δ_s; ℓ1 difference ở outside và inside mỗi phần≥δ_s. Top-s renormalization đạt equality:

$$
\min_{Q\in\Delta,\ |\operatorname{supp}Q|\le s}\|P-Q\|_1=2\delta_s(P).
$$

Nếu centroid diameter≤D, $$z(P)=(1-\delta)z_{\rm top}+\delta z_{\rm out}$$:

$$
\|z(P)-z_{\rm top}\|\le D\delta.
$$

$$\square$$

Điều này định lượng approximation của mixture, không xác nhận nhãn hoặc native-decoder margin. Không thêm mixture/entropy vào core vì chỉ có result này.

### A.20. Checklist assumptions cho từng artifact

| Bảo đảm | Đối tượng phải xác lập |
| --- | --- |
| Projection | Orthonormal B, compact radii, full-search, numerical bounds |
| Population coverage | Correct labels, fixed representation, net/mass/dependence assumptions |
| Near ideal dictionary | Same feasible family; valid lower bound hoặc Δ_opt |
| Correct accepted class | Coverage η và actual projected noise≤ν |
| Zero intervention clean | Whole-support envelope và arithmetic score error |
| Statistical clean protection | Frozen score, exchangeable calibration units, correct scope |
| Same baseline transcript | Same initial state/execution when identity |
| Recovered transcript | Native decoder margins/domain/state bounds, clean correctness nếu WER0 |
| Arbitrary discarded-direction noise | Exact task invariance, không chỉ local Jacobian nhỏ |
| Realtime | Native context, bounded state/service and scheduler contract |

Không có assumption nào được xác lập chỉ bằng UMAP đẹp, finite residual thấp hoặc “backbone pretrained trên data lớn”.

<a id="app-b"></a>
## Appendix B. Cấu hình và checklist triển khai tái lập

### B.1. Shared policy v7.1

~~~yaml
policy_version: phonorepair-v7.1
seeds:
  source_split: 17
  bootstrap: 20261013
  kernel_cases: [0, 1, 2, 3, 4]
source_only: true
online_updates: false
boundary: acoustic_encoder_output
scope:
  primary_models: [W1, Z1, C1]
  primary_disease_pairs: 6
  general_claim_rule: section_4_3
  realtime_sla_required_for_main: false
  onboarding_minutes_required_for_main: false
  transcript_certificate_required_for_main: false
synthetic_pilot:
  development_source: clean_selection
  evaluation_source: clean_speaker_holdout
  max_utterances_per_split: 100
  development_seed: 20261017
  evaluation_seed: 20261018
  directions: [patch_tangent, patch_normal, competing_class]
  noise_to_nu_ratios: [0, 0.25, 0.5, 1, 2]
  candidate_ranking_uses_synthetic: false
  final_calibration_reselection: false
statistics:
  family_alpha: 0.05
  disease_ci: two_sided_bonferroni_6
  clean_upper_bound: one_sided_bonferroni_3
  relative_werr_point_target: 0.05
preprocessing_controls:
  time_stretch_rates: [0.8, 1.2]
  choose_target_winner: false
basis:
  kinds: [pca, fisher_qr, identity]
  ranks: [32, 64, 128, 256, full]
dictionary:
  patch_counts: [64, 128, 256]
  patch_dimensions: [0, 2, 4]
  allocation: equal_per_class_then_label_order
  max_fit_frames_per_class: 5000
  neighborhood_size: 128
  anchor_init: smallest_sample_id
  anchor_selection: farthest_first
  radius: max_neighbor_projected_norm
routing:
  full_search: true
  global_lambda: 0.5
  local_gap_quantile: 0.10
  local_gap_multiplier: 0.25
gate:
  beta: 0.01
  calibration_unit: speaker_block
  speakers: 100
  episodes_per_block: 10
  max_episode_seconds: 30
  final_threshold: max_calibration_and_envelope
  fusion: hard_identity_or_projection
resources:
  artifact_mib: 16
  repair_working_memory_mib: 64
  repair_p99_ms: 5
realtime_extension:
  eligible_streaming_p99_ms: 250
  end_to_end_rtf: 0.8
  deadline_miss_percent: 0.1
~~~

Ranks clip vào feature dimension; de-duplicate candidates; invalid ranks/absent labels không được fake. Các budget áp dụng shared policy, không ghi theo brand trong YAML.

### B.2. Command sequence mục tiêu

Các lệnh sau là CLI phải implement, không phải executable sẵn có trong workspace:

~~~text
phonorepair prepare-source --policy shared_policy.yaml
phonorepair bind --model model_package --contract contract.yaml
phonorepair verify-identity --contract contract.yaml
phonorepair build --source source_manifest --policy shared_policy.yaml
phonorepair select --clean-selection selection_manifest
phonorepair calibrate --speaker-blocks calibration_manifest
phonorepair lock --artifact selected_artifact --protocol evaluation_protocol
phonorepair evaluate --locked-run locked_run.json --suite clean
phonorepair evaluate --locked-run locked_run.json --suite disease
phonorepair evaluate --locked-run locked_run.json --suite corruption
phonorepair report --results result_directory
~~~

Artifact builder process không được mount target corpus directory. Eval process có artifact read-only; target annotations chỉ ở evaluator, không ở inference callback.

### B.3. Per-utterance output schema

~~~json
{
  "run_hash": "computed",
  "artifact_hash": "computed",
  "model_id": "opaque",
  "corpus": "evaluation_only",
  "speaker_id": "evaluation_only",
  "session_id": "evaluation_only",
  "utterance_id": "manifest_id",
  "reference_raw": "evaluator_only",
  "hypothesis_base_raw": "native_output",
  "hypothesis_repair_raw": "native_output",
  "reference_normalized": "same_normalizer",
  "hypothesis_base_normalized": "same_normalizer",
  "hypothesis_repair_normalized": "same_normalizer",
  "edit_counts_base": {"S": 0, "D": 0, "I": 0, "N": 0},
  "edit_counts_repair": {"S": 0, "D": 0, "I": 0, "N": 0},
  "valid_frames": 0,
  "accepted_frames": 0,
  "identity_reasons": {},
  "latency_ms": {},
  "certificate_status": "empirical_or_conditional_or_certified"
}
~~~

Values là schema placeholders, không kết quả. Không lưu diagnosis/speaker ID vào repair feature input.

### B.4. Checklist trước target test

- [ ] Dataset rights và checkpoint/runtime versions được lưu.
- [ ] Source split IDs, mapping và timebase freeze.
- [ ] E0 identity, kernel và cache tests pass.
- [ ] E1 alignment, timebase và latent phone geometry có bảng riêng.
- [ ] Candidate policy và tie-breaking deterministic.
- [ ] Envelope provenance rõ; chưa certified thì không label certified.
- [ ] Calibration độc lập, units/dependence và β resolution rõ.
- [ ] Final artifact không all-identity được báo; không giấu no-op.
- [ ] w và reject accounting của candidate khóa được báo; không tái chọn bằng final calibration.
- [ ] Synthetic development/holdout có manifests, seeds và vai trò riêng; không gọi là bệnh mô phỏng.
- [ ] Mức general, confidence levels và full/partial claim rules khóa trước target.
- [ ] Fixed time-stretch controls báo cả hai arms, không chọn winner theo bệnh.
- [ ] E7 shared-core, E8 state và E9 overhead được tách khỏi optional extensions.
- [ ] E3 và primary/secondary analysis families đã preregister.
- [ ] Model/decoder context giống baseline; no external target prompts/LM.
- [ ] Runtime error/deadline/exclusion rules freeze.
- [ ] Target transcripts không đi vào repair.
- [ ] Source-only builder và target evaluator tách quyền đọc.
- [ ] Commit/artifact/protocol hashes lưu trước target mount.

### B.5. Inventory, mapping và phone scoring

Router inventory v1 là:

~~~text
AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH
K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH SIL
~~~

Đây là 39 speech labels cộng SIL. Với MFA ARPA labels, uppercase rồi bỏ stress suffix 0/1/2; acoustic-model silence symbols được map sang SIL bằng declared lexicon metadata, không đoán mọi unknown token là silence.

Mapping TIMIT dùng cho diagnostic v1:

| Input labels | Output |
| --- | --- |
| Base labels có trong inventory trên, không stress | Uppercase tương ứng |
| ax, ax-h | AH |
| axr | ER |
| ix | IH |
| ux | UW |
| hv | HH |
| el, em, en, eng | L, M, N, NG tương ứng |
| nx | N |
| dx | T |
| h#, pau, epi, bcl, dcl, gcl, kcl, pcl, tcl | SIL |
| q | UNSUPPORTED; report count riêng |
| Bất kỳ label khác | UNSUPPORTED; audit lỗi mapping |

Không gọi mapping này là standard TIMIT 61→39 score: nó giữ AO và ZH để khớp ARPABET router. Nếu báo standard TIMIT score, phải thêm một mapping/protocol riêng được preregister, dùng cùng cho baseline và repair.

Auxiliary probe có thể dự đoán original TIMIT phone labels, rồi evaluator map theo bảng trên. Native ASR không đọc probe. Diagnostic PER v1 bỏ SIL khỏi reference/hypothesis sau mapping; Q/UNSUPPORTED được loại đối xứng và báo original-phone coverage. Không merge hai token phone liên tiếp chỉ vì chúng cùng label trong final phone sequence; CTC collapse chỉ áp lên raw per-frame path của probe. Tách kết quả subset có high-quality phone references khỏi full-corpus WER.

Mapping phải là machine-readable JSON và có unit tests bao phủ tất cả labels thực gặp. Chưa có label mapping hoặc timebase hợp lệ là lỗi E1, không phải lý do để cho decoder dùng ground-truth boundaries.


<a id="refs"></a>
## References

**Nguồn R01–R35 được đối chiếu trong bản 13-10-2026; R36–R40 bổ sung ngày 17-10-2026.** Các citation [Rxx] trong thân bài là link trực tiếp tới primary paper, tài liệu của tác giả/maintainer hoặc trang corpus chính thức. Những recipe, mục tiêu threshold và tiêu chí thực nghiệm do proposal đề xuất không phải published results của nguồn được dẫn. Repository/docs mutable phải pin revision khi triển khai. Danh sách là related work có chọn lọc, không phải xác nhận exhaustive novelty hoặc bảng SOTA hiện hành.

### Papers và nền tảng

- **[R01]** Violeta, D. P. L., Huang, W.-C., & Toda, T. (2022). *Investigating Self-supervised Pretraining Frameworks for Pathological Speech Recognition*. Interspeech 2022. [ISCA](https://www.isca-archive.org/interspeech_2022/violeta22_interspeech.html).
- **[R02]** Johansson, F. D., Sontag, D., & Ranganath, R. (2019). *Support and Invertibility in Domain-Invariant Representations*. AISTATS, PMLR 89:527–536. [Paper page](https://proceedings.mlr.press/v89/johansson19a.html).
- **[R03]** Muller, B., Ortiz Barrañón, A. A., & Roberts, L. (2026). *Training-Free Cross-Lingual Dysarthria Severity Assessment via Phonological Subspace Analysis in Self-Supervised Speech Representations*. arXiv:2604.10123, preprint. [arXiv](https://arxiv.org/abs/2604.10123).
- **[R04]** Wang et al. (2011). *Localized K-Flats*. Proceedings of AAAI. [Publisher page](https://ojs.aaai.org/index.php/AAAI/article/view/7912).
- **[R05]** Gonzalez, T. F. (1985). *Clustering to minimize the maximum intercluster distance*. Theoretical Computer Science 38:293–306. DOI:10.1016/0304-3975(85)90224-5. [Publisher page](https://www.sciencedirect.com/science/article/pii/0304397585902245). Appendix A.5 supplies the anchor-radius packing proof explicitly.
- **[R06]** Lin, G.-T., Li, S.-W., & Lee, H.-y. (2022). *Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition*. Interspeech, 2198–2202. DOI:10.21437/Interspeech.2022-600. [ISCA](https://www.isca-archive.org/interspeech_2022/lin22b_interspeech.html).
- **[R07]** Lin, G.-T., Huang, W.-P., & Lee, H.-y. (2024). *Continual Test-time Adaptation for End-to-end Speech Recognition on Noisy Speech*. EMNLP 2024; arXiv:2406.11064. [Paper](https://arxiv.org/abs/2406.11064).
- **[R08]** Ben-David, S., Lu, T., Luu, T., & Pál, D. (2010). *Impossibility Theorems for Domain Adaptation*. AISTATS, PMLR 9:129–136. [Paper page](https://proceedings.mlr.press/v9/david10a.html).
- **[R09]** Angelopoulos, A. N., & Bates, S. (2021; revised 2022). *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. arXiv:2107.07511. [Paper](https://arxiv.org/abs/2107.07511).
- **[R10]** Radford, A., et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision*. arXiv:2212.04356. [Paper](https://arxiv.org/abs/2212.04356).
- **[R11]** Yao, Z., et al. (2024). *Zipformer: A faster and better encoder for automatic speech recognition*. ICLR 2024; arXiv:2310.11230. [Paper](https://arxiv.org/abs/2310.11230).
- **[R12]** Gulati, A., et al. (2020). *Conformer: Convolution-augmented Transformer for Speech Recognition*. arXiv:2005.08100. [Paper](https://arxiv.org/abs/2005.08100).
- **[R13]** Graves, A., Fernández, S., Gomez, F., & Schmidhuber, J. (2006). *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks*. ICML 2006. [Author-hosted paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf).

### Implementation và model interfaces

- **[R14]** OpenAI. *Whisper model implementation*, encoder/decoder và KV-cache interfaces. [Source](https://github.com/openai/whisper/blob/main/whisper/model.py).
- **[R15]** k2-fsa / Icefall maintainers. *Zipformer ASR model implementation*. [Source](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/model.py).
- **[R16]** NVIDIA. *NeMo ASR models*, Conformer-CTC/transducer và model configuration. [Official documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html).
- **[R17]** PyTorch maintainers. *torch.nn.Module*, module invocation và forward hooks. [Official documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html).
- **[R18]** k2-fsa / Icefall maintainers. *Zipformer streaming ONNX export*. [Source](https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/export-onnx-streaming.py).
- **[R23]** OpenAI. *Whisper README*, checkpoint names và default transcription workflow. [Official repository](https://github.com/openai/whisper/blob/main/README.md).
- **[R24]** NVIDIA. *Cache-aware Streaming Conformer*. [Official documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#cache-aware-streaming-conformer).
- **[R29]** Zengwei Yao / Icefall. *icefall-asr-librispeech-streaming-zipformer-2023-05-17*. [Checkpoint repository](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17).
- **[R30]** NVIDIA. *stt_en_conformer_ctc_large*. [Model card](https://huggingface.co/nvidia/stt_en_conformer_ctc_large).
- **[R31]** NVIDIA. *stt_en_fastconformer_hybrid_large_streaming_multi*. [Model card](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi).

### Dữ liệu và alignment

- **[R19]** Panayotov, V., Chen, G., Povey, D., & Khudanpur, S. (2015). *LibriSpeech: An ASR corpus based on public domain audio books*. Corpus SLR12 và official splits. [OpenSLR](https://www.openslr.org/12).
- **[R20]** Montreal Forced Aligner maintainers. *Align with an acoustic model*. [Official workflow documentation](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html).
- **[R21]** Montreal Forced Aligner maintainers. *English (US) ARPA acoustic model v3.0.0*. [Model documentation](https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20(US)%20ARPA%20acoustic%20model%20v3_0_0.html).
- **[R22]** Garofolo, J. S., et al. (1993). *TIMIT Acoustic-Phonetic Continuous Speech Corpus*, LDC93S1. [Official catalog](https://catalog.ldc.upenn.edu/LDC93S1).
- **[R25]** University of Illinois Statistical Speech Technology Group. *UASpeech corpus, availability và license conditions*. [Official corpus page](https://speechtechnology.web.illinois.edu/uaspeech/). Cite acquisition paper [R33] for publications.
- **[R26]** University of Toronto. *TORGO database, structure và conditions*. [Official corpus page](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html). Cite acquisition paper [R32] for publications.
- **[R27]** Snyder, D., Chen, G., & Povey, D. (2015). *MUSAN: A Music, Speech, and Noise Corpus*. SLR17; arXiv:1510.08484. [Official corpus page](https://www.openslr.org/17).
- **[R28]** OpenSLR maintainers. *Room Impulse Response and Noise Database*, SLR28. [Official corpus page](https://www.openslr.org/28).
- **[R32]** Rudzicz, F., Namasivayam, A. K., & Wolff, T. (2012; online 2011). *The TORGO database of acoustic and articulatory speech from speakers with dysarthria*. Language Resources and Evaluation 46:523–541. DOI:10.1007/s10579-011-9145-0. [Publisher page](https://link.springer.com/article/10.1007/s10579-011-9145-0).
- **[R33]** Kim, H., et al. (2008). *Dysarthric speech database for universal access research*. Interspeech, 1741–1744. DOI:10.21437/Interspeech.2008-480. [ISCA](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.html).
- **[R34]** McAuliffe, M., Socolof, M., Mihuc, S., Wagner, M., & Sonderegger, M. (2017). *Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi*. Interspeech, 498–502. DOI:10.21437/Interspeech.2017-1386. [ISCA](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html).
- **[R35]** librosa maintainers. *librosa.effects.time_stretch*, version 0.11.0. [Official API documentation](https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html).

### Nguồn bổ sung sau review

- **[R36]** Prananta et al. (2022). *The Effectiveness of Time Stretching for Enhancing Dysarthric Speech for Improved Dysarthric Speech Recognition*. Interspeech 2022. [ISCA](https://www.isca-archive.org/interspeech_2022/prananta22_interspeech.html).
- **[R37]** Leung, W.-Z., Cross, M., Ragni, A., & Goetze, S. (2024). *Training Data Augmentation for Dysarthric Automatic Speech Recognition by Text-to-Dysarthric-Speech Synthesis*. Interspeech 2024. [ISCA](https://www.isca-archive.org/interspeech_2024/leung24_interspeech.html).
- **[R38]** Plantinga, P., Bagchi, D., & Fosler-Lussier, E. (2021). *Perceptual Loss with Recognition Model for Single-Channel Enhancement and Robust ASR*. arXiv:2112.06068. [Author manuscript](https://arxiv.org/abs/2112.06068).
- **[R39]** Baevski, A., Zhou, H., Mohamed, A., & Auli, M. (2020). *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations*. NeurIPS 2020. [Proceedings](https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf).
- **[R40]** *Voice Conversion With Just Nearest Neighbors* (2023). arXiv:2305.18975. [Author manuscript](https://arxiv.org/abs/2305.18975).

### Truy vết phiên bản

Proof_1.tex v5 là nền tảng coverage/GAP/projection/routing/calibration/decoder hữu hạn. Proof_4.tex được audit: giữ kết quả Voronoi/entropy có điều kiện, không giữ các suy luận không được chứng minh trong gate lịch sử. PhonoRepair_theory_v3.md và technical_rp.md v6 là tài liệu trước. Revision 7.1 giữ core/boundary/gate của v7, bổ sung gate-feasibility diagnostic, synthetic pilot roles, thống nhất thống kê, baseline preprocessing và ba mức claim general. Chưa triển khai quantile envelope, early-layer repair hoặc phone posterior.

[R01]: https://www.isca-archive.org/interspeech_2022/violeta22_interspeech.html
[R02]: https://proceedings.mlr.press/v89/johansson19a.html
[R03]: https://arxiv.org/abs/2604.10123
[R04]: https://ojs.aaai.org/index.php/AAAI/article/view/7912
[R05]: https://www.sciencedirect.com/science/article/pii/0304397585902245
[R06]: https://www.isca-archive.org/interspeech_2022/lin22b_interspeech.html
[R07]: https://arxiv.org/abs/2406.11064
[R08]: https://proceedings.mlr.press/v9/david10a.html
[R09]: https://arxiv.org/abs/2107.07511
[R10]: https://arxiv.org/abs/2212.04356
[R11]: https://arxiv.org/abs/2310.11230
[R12]: https://arxiv.org/abs/2005.08100
[R13]: https://www.cs.toronto.edu/~graves/icml_2006.pdf
[R14]: https://github.com/openai/whisper/blob/main/whisper/model.py
[R15]: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/model.py
[R16]: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html
[R17]: https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html
[R18]: https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/export-onnx-streaming.py
[R19]: https://www.openslr.org/12
[R20]: https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html
[R21]: https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20(US)%20ARPA%20acoustic%20model%20v3_0_0.html
[R22]: https://catalog.ldc.upenn.edu/LDC93S1
[R23]: https://github.com/openai/whisper/blob/main/README.md
[R24]: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html#cache-aware-streaming-conformer
[R25]: https://speechtechnology.web.illinois.edu/uaspeech/
[R26]: https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html
[R27]: https://www.openslr.org/17
[R28]: https://www.openslr.org/28
[R29]: https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17
[R30]: https://huggingface.co/nvidia/stt_en_conformer_ctc_large
[R31]: https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi
[R32]: https://link.springer.com/article/10.1007/s10579-011-9145-0
[R33]: https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.html
[R34]: https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html
[R35]: https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html
[R36]: https://www.isca-archive.org/interspeech_2022/prananta22_interspeech.html
[R37]: https://www.isca-archive.org/interspeech_2024/leung24_interspeech.html
[R38]: https://arxiv.org/abs/2112.06068
[R39]: https://proceedings.neurips.cc/paper/2020/file/92d1e1eb1cd6f9fba3227870bb6d7f07-Paper.pdf
[R40]: https://arxiv.org/abs/2305.18975



