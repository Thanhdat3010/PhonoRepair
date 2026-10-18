# PhonoRepair — Related works, dữ liệu và phép đo cho E0–E3

**Hoàn thiện: 18-10-2026. Nguồn được tra cứu ngày 17-10-2026.** Đối chiếu proposal nội bộ v7.1 trong PhonoRepair_Proposal_v7.md.

Đây là tài liệu chuẩn bị thực nghiệm, chưa phải kết quả chạy. Các ngưỡng đạt, seed và split nội bộ là protocol của proposal. Chi tiết bổ sung được đánh dấu **đề xuất triển khai**, cần ghi vào manifest trước chạy. File JSON đi kèm là danh mục nguồn, không phải dữ liệu thực nghiệm đã thu thập.

## 1. Tổng quan

E0–E3 sử dụng **LibriSpeech, TIMIT được cấp quyền, ba checkpoint đã chọn và dữ liệu hình học tự sinh**. Chưa có sẵn: nhãn phone thủ công cho audit LibriSpeech, latent theo đúng checkpoint/context, bộ perturbation E2 và kết quả PhonoRepair.

| Thực nghiệm | Paper/nguồn trực tiếp | Data | Đối chứng | Cần tự tạo |
| --- | --- | --- | --- | --- |
| E0: kernel và integration | Paper W1/Z1/C1; code native; CVXPY | Toy có seed; tối đa 100 câu clean stress-validation | Float64 reference; native; attached identity; detached adapter; forced correction probe | Geometry cases, tensor/state traces, lỗi số học/mismatch |
| E1: labels/timebase/phone evidence | MFA; TIMIT; CTC | TIMIT TRAIN/TEST có PHN; clean source được manual audit | Manual phones; majority predictor; linear CTC probe | Alignment CSV/JSON; phone mapping; manual source annotations; probe predictions |
| E2: geometry/usefulness | Gonzalez; Localized K-Flats; PCA/Fisher; lý thuyết proposal | LibriSpeech fit/selection/holdout; finite supports tự sinh | Anchor-only; centroid; unlabelled projection; exact finite anchor covering | Dictionary, frame scores, synthetic sequences, acceptance/rejection |
| E3: clean protection | LibriSpeech; conformal calibration; WER tools; bootstrap ASR | test-clean; test-other; clean speaker holdout; healthy controls tùy chọn | Native; attached identity; locked default; calibration-only | Paired hypotheses, edits, speaker/block events, confidence intervals |

Phân biệt: **paper** mô tả phương pháp; **model/data card** xác định artifact; **code/docs** xác định cách chạy; **công thức/protocol riêng** xác định phép đo của nghiên cứu. Paper gần chủ đề không đồng nghĩa toàn bộ protocol PhonoRepair đã được paper đó kiểm chứng.

## 2. Backbone: paper, checkpoint và code

| ID | Paper trực tiếp | Artifact/code | Chế độ E0–E3 |
| --- | --- | --- | --- |
| W1 | Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision*, ICML 2023. [Paper](https://proceedings.mlr.press/v202/radford23a.html) | [OpenAI Whisper](https://github.com/openai/whisper); [HF base.en](https://huggingface.co/openai/whisper-base.en) | Frozen base.en; English, temperature 0, greedy, prompt rỗng, không carry previous-window text; buffered/offline |
| Z1 | Yao et al., *Zipformer: A faster and better encoder for automatic speech recognition*, ICLR 2024. [Paper](https://arxiv.org/abs/2310.11230) | [Checkpoint Zengwei](https://huggingface.co/Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17); [recipe tác giả](https://github.com/k2-fsa/icefall/pull/1058); [icefall](https://github.com/k2-fsa/icefall) | Native greedy transducer; khóa chunk/context/cache contract trước extraction |
| C1 | Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition*, Interspeech 2020. [Paper](https://arxiv.org/abs/2005.08100) | [NVIDIA Conformer-CTC Large](https://huggingface.co/nvidia/stt_en_conformer_ctc_large); [NeMo models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html) | Native greedy CTC, không external LM |

Nguồn decoder/loss:

- Graves et al. (2006), *Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks* — [CTC paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf), cho C1 và phone probe.
- Graves (2012), *Sequence Transduction with Recurrent Neural Networks* — [transducer paper](https://arxiv.org/abs/1211.3711), nền tảng decoder Z1; cache contract phải đọc code checkpoint.
- [PyTorch Module hooks](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html) và code phiên bản chạy là nguồn triển khai E0. Paper kiến trúc không chứng minh wrapper đã can thiệp đúng consumer.

### Revision metadata đã xác minh

Đây là SHA repository Hugging Face ngày tra cứu, **không phải SHA256 weights và chưa phải locked_run**:

| Repo | Revision |
| --- | --- |
| openai/whisper-base.en | 911407f4214e0e1d82085af863093ec0b66f9cd6 |
| Zengwei/icefall-asr-librispeech-streaming-zipformer-2023-05-17 | 37cb5606808f3d5e55a3fc73554bdf757d82465a |
| nvidia/stt_en_conformer_ctc_large | 5f18b90411b9fdfc9b52b8caf636f204491afb01 |

W1 main dùng loader native OpenAI: lưu URL/hash weights thực sự tải; HF conversion không tự chứng minh numerical pipeline tương đương. C1 repo có stt_en_conformer_ctc_large.nemo. Z1 có exp/pretrained.pt, exp/epoch-30.pt, scripts và tokenizer; không đồng nhất hai files hoặc trộn native/JIT artifacts.

Z1 card trỏ tới PR recipe; có nhiều cấu hình chunk/context và averaging. Phải khóa file, hash, code commit, averaging provenance, tokenizer và entrypoint; không chọn theo published test WER trong repo. Paper Conformer mô tả kiến trúc, không phải paper huấn luyện riêng của checkpoint NeMo này. Pretraining của backbone/aligner có dùng LibriSpeech phải khai báo riêng với split clean-only của PhonoRepair.

## 3. Dữ liệu và nguồn tải

### 3.1. LibriSpeech — E0/E2/E3, source audit E1

Paper: Panayotov, Chen, Povey & Khudanpur (2015), *LibriSpeech: an ASR corpus based on public domain audio books*, ICASSP — [PDF tác giả](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf).

Nguồn tải/JSON:

- [OpenSLR 12 — archive gốc](https://www.openslr.org/12).
- [HF openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr).
- [JSON metadata repository](https://huggingface.co/api/datasets/openslr/librispeech_asr).
- [JSON config/split](https://datasets-server.huggingface.co/splits?dataset=openslr%2Flibrispeech_asr).
- [JSON danh sách Parquet](https://datasets-server.huggingface.co/parquet?dataset=openslr%2Flibrispeech_asr).
- [JSON một trang train.clean.100](https://datasets-server.huggingface.co/rows?dataset=openslr%2Flibrispeech_asr&config=all&split=train.clean.100&offset=0&length=100).
- [Dataset Viewer API](https://huggingface.co/docs/dataset-viewer/en/parquet).

Đã xác minh API trả HTTP 200. Revision dataset: **71cacbfb7e2354c4226d01e70d77d5fca3d04ba1**. Schema gồm file, audio 16 kHz, text, speaker_id, chapter_id, id; định dạng Parquet; license CC BY 4.0 theo [dataset card](https://huggingface.co/datasets/openslr/librispeech_asr).

Chọn thống nhất config **all**:

| Corpus | Split HF | Utterances metadata | Vai trò |
| --- | --- | ---: | --- |
| train-clean-100 | train.clean.100 | 28,539 | Basis-fit/dictionary-fit theo speaker |
| train-clean-360 | train.clean.360 | 104,014 | Selection/calibration/stress-validation/holdout |
| dev-clean | validation.clean | 2,703 | Không tự thêm vào protocol |
| dev-other | validation.other | 2,864 | Không tự thêm vào protocol |
| test-clean | test.clean | 2,620 | E3 primary |
| test-other | test.other | 2,939 | E3 secondary, báo riêng |

Config clean dùng tên train.100, train.360, validation, test; không trộn tên split giữa configs.

Ví dụ truy cập sau khi khóa môi trường tương thích:

~~~python
from datasets import load_dataset

source = load_dataset(
    "openslr/librispeech_asr", "all",
    split="train.clean.100",
    revision="71cacbfb7e2354c4226d01e70d77d5fca3d04ba1",
    streaming=True,
)
~~~

Đoạn này là hướng dẫn, chưa chạy tải audio. Chỉ mở final test sau lock.

CSV/JSON manifest cần metadata và path tới audio thực. Trường file ở một số hàng là cache path của người tạo dataset, không dùng trực tiếp trên máy mình. URL audio trong API rows có chữ ký hết hạn; không dùng làm định danh bền vững. JSON parquet chứa nhiều configs có nội dung trùng; chỉ tải config/split cần dùng. API rows có phân trang; một trang không phải toàn corpus.

### 3.2. Split nội bộ: phải tự sinh manifest

Theo §10.1 proposal:

1. train.clean.100: 20% speakers đầu theo hash seed 17 cho basis-fit; còn lại dictionary-fit.
2. train.clean.360: 100 speakers selection, 100 calibration, 100 stress-validation; còn lại speaker holdout.
3. Calibration: mỗi speaker 10 episodes hợp lệ, mỗi episode ≤30 giây. Speaker thiếu điều kiện thì skip, chọn bù và cập nhật roles để vẫn không giao nhau.
4. E0: tối đa 100 utterances speaker-balanced từ stress-validation.
5. E2 development: tối đa 100 câu từ selection; evaluation: tối đa 100 câu từ 100 holdout speakers đủ điều kiện đầu tiên, một câu/người. Perturbation seeds: 20261017/20261018.
6. E3: test.clean/test.other và holdout speaker blocks. E2-holdout và E3 dùng chung speakers không tạo ra hai cohort độc lập.

**Đề xuất triển khai cần khóa:** hash UTF-8 với delimiter rõ, ví dụ 17|librispeech|speaker_id; floor khi lấy 20%; quy tắc chọn utterances và chọn bù cụ thể. Nếu đã có manifest thì giữ convention đã khóa. Audit speaker overlap/exclusions trước extraction. Không có CSV chính thức của các split nội bộ này.

### 3.3. TIMIT — phone ground truth E1

Citation dataset: Garofolo et al., *TIMIT Acoustic-Phonetic Continuous Speech Corpus*, LDC93S1 — [LDC catalog](https://catalog.ldc.upenn.edu/LDC93S1).

- [HF timit-asr/timit_asr](https://huggingface.co/datasets/timit-asr/timit_asr).
- [JSON metadata](https://huggingface.co/api/datasets/timit-asr/timit_asr).
- [Loader đã kiểm tra](https://huggingface.co/datasets/timit-asr/timit_asr/blob/a92396bd1b08354e525a44cd966b019df2139d6c/timit_asr.py).

**HF repo chỉ có loader/README, không chứa corpus audio/PHN**, tại revision a92396bd1b08354e525a44cd966b019df2139d6c. Loader yêu cầu bản TIMIT từ LDC. License script hoặc trạng thái ungated không cấp quyền corpus. Chưa xác minh nguồn CSV/JSON TIMIT đầy đủ được cấp phép tải tự do.

Từ bản được cấp quyền:

1. Giữ official TRAIN/TEST, speaker/utterance IDs.
2. Đọc .PHN: start_sample, end_sample, phone; .WRD cho word boundaries; .TXT cho transcript.
3. Chuyển thời gian theo sample rate; với 16,000 Hz:

$$
t_{\rm start}=s_{\rm start}/16000,\qquad t_{\rm end}=s_{\rm end}/16000.
$$

4. Giữ sample indices, khoảng nửa mở [start,end), nhãn raw và mapped. Kiểm tra order/duration. WAV có thể là NIST/SPHERE, cần decoder đúng định dạng.
5. Xuất phones.csv/jsonl cùng provenance. Corpus và annotation gốc chỉ phân phối theo quyền được cấp.

Không thay PHN bằng ASR pseudo-label rồi gọi manual ground truth. Script HF kiểu cũ có thể không tương thích thư viện datasets mới; parser local từ corpus được cấp quyền là đường triển khai rõ ràng.

### 3.4. MFA — tạo alignment CSV/JSON

Paper: McAuliffe et al. (2017), *Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi*, Interspeech — [paper](https://www.isca-archive.org/interspeech_2017/mcauliffe17_interspeech.html).

Artifacts:

- [English (US) ARPA acoustic model v3.0.0](https://mfa-models.readthedocs.io/en/latest/acoustic/English/English%20%28US%29%20ARPA%20acoustic%20model%20v3_0_0.html).
- [Acoustic model release](https://github.com/MontrealCorpusTools/mfa-models/releases/tag/acoustic-english_us_arpa-v3.0.0).
- [Dictionary v3.0.0](https://mfa-models.readthedocs.io/en/latest/dictionary/English/English%20%28US%29%20ARPA%20dictionary%20v3_0_0.html).
- [CLI/output_format](https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/workflows/alignment.html): có JSON, CSV, long/short TextGrid.

Ví dụ cú pháp MFA 3.x tương ứng docs:

~~~text
mfa align SOURCE_CORPUS DICTIONARY_PATH ACOUSTIC_MODEL_PATH OUTPUT_DIRECTORY --output_format json
~~~

Pin version, model/dictionary hashes, beam, normalization, speaker adaptation và OOV policy. Dùng file release cụ thể; lệnh tải theo tên không khóa vĩnh viễn version. Model card ghi training trên LibriSpeech. MFA output là nhãn tự động offline, không phải manual truth.

**Điểm chưa rõ trong E1:** benchmark nói 200 câu clean source, setup nói 200 TIMIT TRAIN. Đề xuất tách hai cohort: TIMIT có PHN sẵn; LibriSpeech phải được annotator gán/kiểm tra phone boundaries thủ công. Nếu chỉ làm TIMIT, chưa hoàn tất source-domain manual audit. Chọn câu theo hash trước khi nghe; annotator không xem repair routing/WER; lưu guideline và adjudication. Không có file HF LibriSpeech chứa sẵn manual phones đã xác minh.

### 3.5. Toy/synthetic E0/E2

Không có dataset tải sẵn đúng protocol PhonoRepair:

| Bộ data | Cách tạo | Lưu |
| --- | --- | --- |
| E0 geometry | QR Gaussian tạo orthonormal basis; anchors/radii/scales có seed; thêm ca biên §4 | JSON cases + NPZ arrays, expected output/reference |
| E2 finite supports | Liệt kê universe hữu hạn và labels; allowable centers/budget rõ | Tất cả points, metric, solver bounds |
| E2 speech perturbations | Trích latent sạch đúng checkpoint; tạo tangent/normal/competitor directions §6 | Manifest source, artifact hash, seed, direction, kappa, arrays, skip reasons |

Latent/dictionary phụ thuộc checkpoint/layer/context; không dùng chung arrays giữa backbones.

### 3.6. Healthy controls tùy chọn E3

| Corpus | Paper | Data source | Chuyển thành CSV/JSON |
| --- | --- | --- | --- |
| TORGO | Rudzicz, Namasivayam & Wolff, *The TORGO database of acoustic and articulatory speech from speakers with dysarthria*. [Paper](https://link.springer.com/article/10.1007/s10579-011-9145-0) | [Corpus/download](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html) | Chọn controls theo metadata; một channel khóa trước; join audio với reference; tách words/sentences |
| UASpeech | Kim et al. (2008), *Dysarthric speech database for universal access research*. [Paper](https://www.isca-archive.org/interspeech_2008/kim08c_interspeech.html) | [Trang cấp quyền](https://speechtechnology.web.illinois.edu/uaspeech/) | Sau cấp quyền: controls + word prompts + blocks + microphone metadata |

Chưa xác minh CSV/JSON công khai chính thức đầy đủ cho hai nguồn. Giữ điều kiện sử dụng của corpus. Chỉ secondary sau lock, không calibrate main gate. Thiếu quyền/metadata → not run; không chọn mirror không rõ nguồn.

## 4. E0 — kernel và integration

### 4.1. Reference và data

Với patch:

$$
S_j=\{a_j+B_jw:\|w\|_2\le\rho_j\},\quad B_j^\top B_j=I,\quad
v=u-a_j,\quad z=B_j^\top v.
$$

$$
c=\begin{cases}z,&\|z\|_2\le\rho_j,\\
\rho_jz/\|z\|_2,&\|z\|_2>\rho_j,\end{cases}
\qquad \pi_j(u)=a_j+B_jc.
$$

$$
d_j(u)^2=\|(I-B_jB_j^\top)v\|_2^2+
\bigl(\max(0,\|z\|_2-\rho_j)\bigr)^2.
$$

Patch dimension/radius bằng 0 thì projection về anchor. Đây là công thức của proposal; không có paper benchmark riêng cho kernel PhonoRepair.

Reference độc lập giải:

$$
\min_w \tfrac12\|u-a_j-B_jw\|_2^2
\quad\text{subject to}\quad \|w\|_2\le\rho_j.
$$

Nguồn tool: Diamond & Boyd (2016), *CVXPY: A Python-Embedded Modeling Language for Convex Optimization* — [paper](https://www.jmlr.org/papers/v17/15-408.html). Dùng solver hỗ trợ norm-ball constraint; ghi solver/status/tolerance. Đây là convex QCQP/conic formulation, không phải QP chỉ có ràng buộc tuyến tính để mặc định dùng OSQP. Chỉ cần representative cases và một công thức float64 độc lập; solver không hội tụ không là oracle.

**Đề xuất triển khai:** 256 random cases, seed 20261017, read dimensions 2/8/32/128, patch dimensions hợp lệ 0/2/4, scales 0.001/1/1000; bổ sung ca biên cố định. Đây là ngân sách test đề xuất, không sample size bảo đảm thống kê.

Ca bắt buộc: trong patch, ngoài radius, ngoài affine span, zero dimension/radius, ties, sát ngưỡng, invalid shapes, NaN/Inf fail-closed. Không âm thầm sửa invalid basis để che lỗi. Audio E0 lấy stress-validation, so native/attached identity/detached cùng initial state; forced correction trên valid frame để kiểm tra memory consumer.

### 4.2. Metric vận hành

| Metric | Công thức/cách đo |
| --- | --- |
| Projection error | $$e_\pi=\|\pi_{\rm impl}(u)-\pi_{\rm ref}(u)\|_2$$ |
| Distance error | $$e_d=\lvert d_{\rm impl}(u)-d_{\rm ref}(u)\rvert$$ |
| Span violation | $$v_{\rm span}=\|(I-BB^\top)(\pi_{\rm impl}-a)\|_2$$ |
| Radius violation | $$v_{\rm rad}=\max(0,\|B^\top(\pi_{\rm impl}-a)\|_2-\rho)$$ |
| Read/lift error | $$e_{\rm lift}=\|Q^\top(\hat h-\mu)-\hat u\|_2$$ với $$\hat h=h+Q(\hat u-u)$$ |
| Orthogonal preservation | $$e_\perp=\|(I-QQ^\top)(\hat h-h)\|_2$$ |
| Budget violation | $$v_{\rm budget}=\max(0,\|\hat h-h\|_2-c_{\max})$$, dùng đúng norm/budget của core |
| Mismatch | Đếm riêng tensor values/shapes, lengths, masks, output tree, transcript, state |
| Consumer correctness | Số invocation nhận đúng tensor / số invocation được probe; kèm indices/values kỳ vọng |
| Ownership violation | Số mutation vùng alias, stale cache version, state corruption |
| Branch coverage | Nhánh đã chạy / nhánh đăng ký, kèm danh sách nhánh |

Ký hiệu c_max ở đây là budget đã khóa, không thêm tham số mới. Nếu core dùng budget theo episode thay vì frame, tính trên đúng đơn vị đó.

**Đề xuất chuẩn hóa tolerance:** scale cố định từ basis-fit, ví dụ RMS của vector norms; toy dùng scale sinh đã biết. Normalized error = raw error / max(scale, floor khóa trước). Báo raw lẫn normalized max theo dtype/scale. Target proposal: float64 normalized ≤1e-10; float32 raw ≤1e-5 lần source RMS. Đây là kiểm chứng hữu hạn, không universal rounding certificate.

Identity exact ở deterministic backend. Với nondeterminism, đo biến thiên native lặp lại trước; không nới tolerance theo kết quả repair để đạt gate. Transcript giống nhau không đủ chứng minh memory/state đúng.

## 5. E1 — labels, timebase và phone evidence

### 5.1. Phương thức thực hiện

MFA paper hỗ trợ chọn aligner; TIMIT cung cấp manual intervals; CTC paper hỗ trợ objective probe. **Median 20 ms, p95 60 ms và interior agreement 95% là audit targets của proposal**, không phải published MFA performance.

1. Lấy 200 TIMIT TRAIN utterances speaker-balanced theo hash; LibriSpeech manual audit là cohort riêng §3.4.
2. Khóa mapping B.5, stress stripping, unsupported/closure handling. Mapping giữ AO/ZH, map dx về T, q unsupported; **khác TIMIT 61→39 thông dụng**. Không so trực tiếp PER với leaderboard khác mapping.
3. Timebase: frontend stride/padding/length propagation và impulse probes. Attention weights hoặc thời điểm CTC token phát ra không là ground truth thời gian của encoder frame. Tách nominal frame coordinates khỏi receptive-field span, nhất là encoder có context rộng.
4. **Đề xuất boundary correspondence:** map inventory, align hai chuỗi phones bằng edit distance với tie-breaking cố định; chỉ so boundary giữa hai cặp phones liên tiếp được match đúng trong cả hai chuỗi. Báo matched/unmatched counts. Không chọn boundary gần nhất theo thời gian để làm nhỏ lỗi; không gộp phone lặp chỉ để tăng agreement.
5. Phone probe: encoder frozen; linear head + CTC; TIMIT TRAIN chia 80/20 theo speaker để fit/chọn regularization, TEST chỉ evaluate. Chốt SA inclusion/exclusion và full TEST/core TEST trước chạy. Full TEST không được gọi là standard core-test score.

Với head:

$$
P_t=\operatorname{softmax}(Wh_t+b),\qquad
\mathcal L_{\rm CTC}=-\log\sum_{\pi:\mathcal B(\pi)=y}\prod_tP_t(\pi_t).
$$

Đây là **linear CTC probe tùy chỉnh**; chưa tìm thấy paper định nghĩa nguyên bộ W1/Z1/C1 + mapping của proposal. Cùng optimizer/grid/early-stopping policy cho ba models; lưu seeds, epochs, regularization, output-length failures. Probe không đi vào inference và không update backbone.

Để biến mô tả thành recipe chạy được, cần khóa trước test: feature normalization nếu có, optimizer, learning-rate grid, batch policy, số epochs tối đa, stopping metric trên TRAIN-dev và deterministic tie-break khi chọn hyperparameters. Không lấy TIMIT TEST để chọn những giá trị này. Majority baseline chọn class phổ biến nhất trên TRAIN-fit.

### 5.2. Metric E1

Đặt V là frames có nhãn manual hợp lệ, J là boundary pairs ghép được:

$$
e_j^{\rm ms}=1000|\hat t_j-t_j|,\quad
\operatorname{BoundaryMedian}=Q_{0.5}(e),\quad
\operatorname{BoundaryP95}=Q_{0.95}(e).
$$

Khóa quantile convention; đề xuất nearest rank. Báo signed offsets, matched-boundary coverage và unmatched counts. Quantiles trên matched subset không đại diện mọi boundaries nếu coverage thấp.

$$
V_{\rm int}=\{t\in V:\min_{b\in\text{manual boundaries}}|t_{\rm frame}-b|\ge0.020\ {\rm s}\},
$$

$$
\operatorname{LabelAgreement}_{\rm int}
=\frac{\sum_{t\in V_{\rm int}}\mathbf1[\ell_t^{\rm MFA}=\ell_t^{\rm manual}]}{|V_{\rm int}|}.
$$

Đây là alignment agreement, khác routing accuracy:

$$
\operatorname{RouteAcc}
=\frac{\sum_{t\in V}\mathbf1[\arg\min_p d_p(u_t)=\ell_t]}{|V|}.
$$

$$
\operatorname{MacroPhoneAcc}
=\frac1{|P_V|}\sum_{p\in P_V}
\frac{\sum_{t\in V:\ell_t=p}\mathbf1[\hat p_t=p]}{|\{t\in V:\ell_t=p\}|}.
$$

P_V chỉ gồm classes có reference observations; báo classes thiếu, SIL riêng, interior/transition riêng. Routing trên LibriSpeech MFA labels là agreement với pseudo-labels, không được ghi manual accuracy.

$$
\operatorname{PER}
=\frac{\sum_i(S_i^{\rm phone}+D_i^{\rm phone}+I_i^{\rm phone})}
{\sum_iN_i^{\rm phone}}.
$$

PER dùng edit distance trên phone sequence. Collapse raw CTC path theo quy tắc CTC, rồi áp scoring mapping; bỏ SIL/unsupported đối xứng theo proposal, báo coverage và số tokens bị loại. Không merge thêm phone lặp sau mapping. Mẫu số 0 → NA. PER/WER có thể vượt 100% vì insertions.

Với nominal timebase, báo length mismatch count, max absolute timestamp-coordinate error so với frontend contract và offset theo input length/chunk position:

$$
N_{\rm length\ mismatch}=\sum_i\mathbf1[T_i^{\rm observed}\ne T_i^{\rm contract}],
\qquad
e_{\rm time,max}=\max_{i,t}1000\lvert \tau_{it}^{\rm implementation}-\tau_{it}^{\rm contract}\rvert.
$$

Thời gian tau tính bằng giây nên e_time,max có đơn vị ms. Nếu time coordinate quy ước là center của một receptive field, ghi center rule cụ thể. Đây là kiểm tra kỹ thuật tùy chỉnh, không có published metric riêng cho binding PhonoRepair.

Tách ba bảng: MFA–manual, timebase, geometry/probe. Mục tiêu agreement 95% ở bảng thứ nhất không yêu cầu latent phone classification phải đạt 95%.

## 6. E2 — geometry, competitors và synthetic mechanism

### 6.1. Related works trực tiếp

| Thành phần | Nguồn | Liên quan và giới hạn |
| --- | --- | --- |
| Farthest-first / covering | Gonzalez (1985), *Clustering to minimize the maximum intercluster distance*. [Paper](https://www.cs.columbia.edu/~verma/classes/uml/ref/clustering_minimize_intercluster_distance_gonzalez.pdf) | Chọn anchors; không chứng minh optimum affine dictionary |
| Local affine approximation | Wang et al. (2011), *Localized K-Flats*. [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/7912) | Related method trực tiếp; PhonoRepair không tự là tái lập LKF |
| PCA | Pearson (1901), *On lines and planes of closest fit to systems of points in space*. [DOI](https://doi.org/10.1080/14786440109462720) | Nền tảng least-squares subspace; trang publisher bị hạn chế truy cập trong lượt tra cứu |
| Fisher span | Fisher (1936), *The use of multiple measurements in taxonomic problems*. [Paper](https://doi.org/10.1111/j.1469-1809.1936.tb02137.x) | Discriminant directions; regularization/QR/grid là lựa chọn triển khai riêng |
| Rank calibration | Angelopoulos & Bates, *A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification*. [Paper](https://arxiv.org/abs/2107.07511) | Nền tảng calibration; speaker-block repair gate là construction của proposal |

PCA: mean/covariance trên basis-fit, eigenvectors lớn nhất. Fisher: between/within-class scatter, regularized generalized eigenproblem và QR để basis Euclid trực chuẩn. Nếu requested rank vượt numerical Fisher rank, báo actual rank hoặc khóa cách bổ sung directions trước chạy. Identity: Q là ma trận đơn vị.

Dictionary theo §7.4: farthest-first per class, tối đa 128 neighbors, SVD quanh anchor, bounded radius, shared M/s/r grid. Khóa initialization/tie-break; không đổi anchor thành local mean mà không khai báo. Các bước này cần tự implement; không có checkpoint PhonoRepair công khai đã xác minh để tải.

### 6.2. Đối chứng và cách thực hiện

Đây là **baselines/ablations do nghiên cứu định nghĩa**. Paper nền không cung cấp nguyên bộ clean gate/native bindings này.

| Arm | Cách làm | Kiểm soát |
| --- | --- | --- |
| Native / attached identity | Trả nguyên memory; attached identity không read/lift để tái tạo | Cùng weights/options/initial state |
| Anchor-only | Same M, s=0; patch là điểm anchor | Giữ source roles, Q và budget policy |
| Nearest centroid + clean gate | Mỗi class một centroid từ dictionary-fit; distance Euclid, projection về centroid khi nhận | Ít atoms hơn M; không gọi same-storage comparison |
| Unlabelled projection | Nearest patch toàn dictionary; bỏ singleton theo class, giữ clean gate/guards | Tie-break cố định; báo routing/harm; mất guarantee theo labels |
| No clean gate | Bỏ điều kiện A>q_off, giữ singleton và guards khác | Không bỏ correction budget/numerical guards |
| Calibration-only | q_off=q_cal thay vì max(q_cal,eta_used) | Nếu score đổi thì calibrate chính score đó |
| Exact finite anchor covering | Cùng allowable centers, labels, metric, budget; enumeration/MILP | Chỉ gọi exact khi chứng minh optimum |

Mỗi arm thay geometry phải fit/calibrate theo arm đã đăng ký; không reuse threshold từ một score khác. Các calibration scores có thể đánh giá trên cùng cohort dành trước cho ablations nhưng không được dùng chọn winner; báo rõ sự phụ thuộc giữa arms.

Exact finite covering: tại radius thử R, D_ij là khoảng cách từ point i tới candidate center j:

$$
z_j\in\{0,1\},\qquad \sum_jz_j\le M,\qquad
\sum_{j:D_{ij}\le R}z_j\ge1\quad\forall i.
$$

Thêm đúng class/budget constraints. Tìm R nhỏ nhất trong tập khoảng cách hữu hạn. Solver chưa chứng minh optimum thì báo incumbent và valid bound/gap. Không lấy anchor optimum làm affine optimum hoặc affine lower bound nếu chưa chứng minh quan hệ phù hợp. Multistart best là upper bound; 0 là lower bound hợp lệ nhưng lỏng cho loss không âm.

### 6.3. Data generator E2

Với valid clean frame có annotation, chọn nearest patch j thuộc annotated class. Gaussian g dùng seed khóa:

$$
v_{\rm tan}=B_jB_j^\top g,\qquad
v_{\rm normal}=(I-B_jB_j^\top)g.
$$

Hướng competitor đi từ u tới projection lên nearest patch của class khác. Với mỗi hướng khác 0:

$$
\xi=\kappa\nu\frac{v}{\|v\|_2},\quad
\kappa\in\{0,0.25,0.5,1,2\},\qquad h'=h+Q\xi.
$$

Giữ shapes/lengths/masks/padding và phần trực giao Q. Tangent rank 0, thiếu competitor hoặc vector norm 0 → skip có reason/count; không thay bằng hướng thuận lợi. Nu=0 → perturbation degenerate, không claim positive robustness.

Chạy hai nhánh native free-running từ cùng initial state/cache: perturbed-no-repair và perturbed-repair. Annotation chỉ ở generator/evaluator. Báo từng direction/kappa. Kappa>1 cố ý ngoài budget; kappa≤1 vẫn cần coverage/label assumptions. Recipe này tự thiết kế, không phải dataset dysarthria simulation được xác nhận bởi paper.

Development có thể dẫn tới policy version mới trước lock; không rerank candidates ngầm theo synthetic WER. Holdout chỉ đánh giá candidate đã khóa.

### 6.4. Geometry metrics

Với true/source annotation p_i:

$$
d_p(u)=\min_{j:\ell_j=p}d_j(u),\quad A(u)=\min_p d_p(u),\quad
e_i=d_{p_i}(u_i),\quad
\widehat E_p=\max_{i:p_i=p}e_i,\quad \widehat E=\max_p\widehat E_p.
$$

**Correct-class residual khác gate score min toàn classes.** Dùng min toàn classes thay residual có thể che wrong routing.

| Metric | Định nghĩa | Diễn giải |
| --- | --- | --- |
| Residual quantiles/max | Q_0.50, Q_0.95, Q_0.99, max của e_i | Fit/selection/holdout và classes riêng |
| Anchor cover radius | $$R_p=\max_{i:p_i=p}\min_{j:\ell_j=p}\|u_i-a_j\|_2$$ | Khác patch radius rho_j; báo cả hai |
| Class gap | $$g_i=d_{(2)}(u_i)-d_{(1)}(u_i)$$ | Hai classes khác nhau, không phải hai patches cùng class |
| Separation lower bound | $$\underline\Gamma=\max(0,\min_{\ell_j\ne\ell_k}(\|a_j-a_k\|_2-\rho_j-\rho_k))$$ | Bằng 0 không chứng minh supports chắc chắn overlap |
| Working eta | Max correct-class residual trên dictionary-fit và geometry audit selection | Empirical, không population certificate |
| Certified eta | $$\eta_{\rm cert}=\max_p(r_{{\rm cov},p}+\widehat E_p)$$ | Chỉ certified khi net bound/assumptions hợp lệ |
| Local radius | $$\nu_{\rm local}=0.25Q_{0.10}(g_i)$$ trên source frames nearest class đúng, gap>0 | Tập rỗng → 0; empirical policy |
| Global radius | $$\nu_{\rm global}=\tfrac12(\underline\Gamma/2-\eta_{\rm cert})$$ khi certified và dương | Không đủ điều kiện → unavailable |
| Gate width | $$w=\max(0,\eta+\nu-\max(q_{\rm cal},\eta))$$ | w=0 không có miền score nhận sửa; w>0 chưa đủ acceptance |
| Fit→selection growth | $$G_{\rm abs}=\widehat E_{\rm selection}-\widehat E_{\rm fit}$$ | Giữ artifact/metric; giá trị có thể âm |
| Relative growth tùy chọn | $$G_{\rm rel}=\widehat E_{\rm selection}/\widehat E_{\rm fit}$$ | Fit=0 → NA |

Coverage status phải tách ít nhất: finite-complete; population-certified với bằng chứng/assumptions; empirical/conditional; unsupported labels. Báo số classes và tỷ lệ frames được bao phủ, không chỉ một nhãn “certified”.

Theo Appendix A.6:

$$
\Delta_{\rm opt}=\widehat E(\widehat{\mathcal D})-\widehat E_B^\star,\qquad
U_{\rm gap}=r_{\rm cov}+\widehat E(\widehat{\mathcal D})-L,\quad
L\le\widehat E_B^\star.
$$

Đây là population suboptimality bound theo assumptions proposal. Universe hữu hạn liệt kê đầy đủ → r_cov=0. Speech chưa có net bound → **population U_gap = NA/conditional**; empirical gap bound chỉ là Ehat−L. Test max residual không phải r_cov. Best multistart không phải L.

### 6.5. Activation, routing và usefulness

Đặt V là valid frames, V_L là subset có annotation hợp lệ, a_t là accepted:

$$
\operatorname{AcceptRate}=\frac{\sum_{t\in V}a_t}{|V|},\qquad
\operatorname{WrongAmongAccepted}=
\frac{\sum_{t\in V_L}a_t\mathbf1[\hat p_t\ne\ell_t]}{\sum_{t\in V_L}a_t}.
$$

Báo annotation coverage trên mọi frames và riêng accepted frames. Accepted denominator=0 → NA, không phải 0% wrong.

Với normalized transcript y0, y1 và reference y*:

$$
H_i=\mathbf1[y_i^0=y_i^*,\,y_i^1\ne y_i^*],\qquad
C_i=\mathbf1[y_i^0\ne y_i^*,\,y_i^1=y_i^*].
$$

Báo counts H/C và rates trên mọi utterances; conditional rates dùng baseline-correct/baseline-wrong denominator riêng. Cần WER/S/D/I để phản ánh sửa sai một phần. Latent displacement: norm repaired−perturbed per frame, mean/p95/max và sequence Frobenius norm. Reconstruction tới true clean chỉ xác định ở synthetic có counterpart.

First-exit counts: inactive/scope → below threshold → feasible empty → feasible multiple → numerical/deadline → budget → accepted. Tổng bằng số frames xét. Thêm từng điều kiện độc lập từ offline full-search, vì first-exit không đo nguyên nhân độc lập.

Storage = tổng bytes serialize; MiB = bytes/2^20. Selection cost = wall-clock toàn grid, gồm failures, peak RAM/VRAM; ghi hardware/threads/precision/warmup/cache policy và synchronize GPU khi đo. Đây là custom operational metrics, không có paper riêng để gán.

## 7. E3 — clean WER, activation và confidence intervals

### 7.1. Nguồn trực tiếp cho metric

- [NIST SCTK](https://github.com/usnistgov/SCTK): scoring toolkit cho ASR.
- [JiWER](https://jitsi.github.io/jiwer/): word/character edit metrics; pin version/normalizer.
- Bisani & Ney (2004), *Bootstrap estimates for confidence intervals in ASR performance evaluation* — [RWTH record](https://publications.rwth-aachen.de/record/112688/). Nguồn bootstrap ASR; paired speaker clusters và Bonferroni dưới đây là protocol cụ thể của nghiên cứu.
- Clopper & Pearson (1934), *The use of confidence or fiducial limits illustrated in the case of the binomial* — [DOI](https://doi.org/10.1093/biomet/26.4.404). Publisher toàn văn không truy cập được trong lượt tra cứu; zero-event formula cũng suy trực tiếp từ binomial probability.

### 7.2. WER và delta

Đếm edits từng utterance sau cùng normalizer/tokenizer, cộng counts rồi chia:

$$
\operatorname{WER}_{\rm corpus}=
\frac{\sum_i(S_i+D_i+I_i)}{\sum_iN_i},
$$

$$
\operatorname{WER}_{\rm spk}=
\frac1{|\mathcal S|}\sum_{s\in\mathcal S}
\frac{\sum_{i\in s}(S_i+D_i+I_i)}{\sum_{i\in s}N_i}.
$$

$$
\Delta\operatorname{WER}_{\rm pp}
=100(\operatorname{WER}_{\rm repair}-\operatorname{WER}_{\rm native}).
$$

CER tùy chọn: cùng edit-rate formula trên characters, khóa xử lý spaces. Corpus WER khác mean utterance WER; 0.1 pp khác 0.1% relative. Lưu S/D/I/N từng utterance. Tách test-clean và test-other.

### 7.3. Calibration và block events

Một block là một speaker với 10 episodes theo cùng selection rule calibration/holdout:

$$
M_b=\max_{i\in b}\max_{t\in i}A(u_{it}),\qquad
k=\left\lceil(n_{\rm cal}+1)(1-\beta)\right\rceil,
$$

$$
q_{\rm cal}=\begin{cases}
M_{(k)},&k\le n_{\rm cal},\\
+\infty,&k>n_{\rm cal}.
\end{cases}
$$

n_cal=100, beta=0.01 → k=100, threshold là max block score. Gate dùng strict A>q_off. Rank argument yêu cầu exchangeability của blocks với score/artifact đã khóa; không coi frames là independent units. Không tự chuyển guarantee sang test-other hoặc healthy target controls khác phân phối.

Phân biệt:

$$
Z_b^{\rm accept}=\mathbf1[\exists(i,t)\in b:a_{it}=1],\qquad
Z_b^{\rm edit}=\mathbf1[\exists(i,t)\in b:\hat h_{it}\ne h_{it}],
$$

$$
Z_b^{\rm text}=\mathbf1[\exists i\in b:y_i^1\ne y_i^0].
$$

Mỗi event rate = tổng indicators / số blocks. Khóa numerical equality cho latent-change; báo accept riêng để thấy accepted nhưng correction zero. Repaired-same-transcript rate: số utterances có đổi latent nhưng y1=y0 / số utterances đổi latent; denominator 0 → NA. Harm/correction theo H/C §6.5.

Với 0 events trên n independent blocks có cùng event probability:

$$
p_{\rm upper,95\%}=1-0.05^{1/n}.
$$

Suy từ (1−p)^n=0.05. Cần ít nhất 299 zero-event blocks để upper ≤1%. Không nhân 100 speakers × 10 episodes thành 1,000 independent blocks. n=0 không tính bound; assumptions không phù hợp thì chỉ descriptive rate.

### 7.4. Paired speaker bootstrap

**Protocol của proposal, không gán toàn bộ cho một paper:**

1. Từng corpus riêng: resample S speaker IDs có hoàn lại, giữ toàn bộ paired native/repair utterances và multiplicity.
2. Mỗi replicate tính lại corpus WER từ summed edits/reference words, rồi delta pp.
3. B=10,000, seed 20261013; descriptive two-sided 95% CI bằng quantiles 0.025/0.975.
4. Main clean family ba backbones: upper một phía tại **1−0.05/3 = 0.983333…**; không làm tròn alpha trước tính.
5. Đạt non-inferiority nếu upper ≤+0.1 pp. Bootstrap là approximate inference, không exact finite-sample theorem với ít speakers.

Không bootstrap frames. Nếu thêm primary arms/corpora phải khai báo lại multiplicity family; secondary ablations không mặc nhiên được hưởng family chỉ ba tests. Zero-event activation bound không thay CI delta WER.

## 8. Competitors khác và vai trò

| Related work | Vai trò E0–E3 | Cách dùng |
| --- | --- | --- |
| Lin et al. (2022), *Listen, Adapt, Better WER: Source-free Single-utterance Test-time Adaptation for Automatic Speech Recognition*. [SUTA](https://www.isca-archive.org/interspeech_2022/lin22b_interspeech.html) | Optional setting riêng; không cần để pass E0–E3 | Theo recipe paper; report target updates/optimizer/iterations/reset/compute; không chuyển objective CTC sang RNNT/Whisper rồi gọi đúng SUTA |
| *Continual Test-time Adaptation for End-to-end Speech Recognition on Noisy Speech* (2024). [DSUTA](https://arxiv.org/abs/2406.11064) | Optional target-adaptation comparator | Khóa stream order/reset/continual state; khác quyền dùng target của frozen artifact |
| Prananta et al. (2022), *The Effectiveness of Time Stretching for Enhancing Dysarthric Speech for Improved Dysarthric Speech Recognition*. [Paper](https://www.isca-archive.org/interspeech_2022/prananta22_interspeech.html) | Secondary preprocessing arm; E3 đo clean harm nếu đăng ký | Proposal: rates 0.8/1.2, librosa 0.11.0, n_fft=2048, hop_length=512; không claim tái lập toàn paper |

[librosa time_stretch](https://librosa.org/doc/0.11.0/generated/librosa.effects.time_stretch.html) là nguồn implementation. Giữ pitch, native segmentation cho audio dài hơn, không cắt phần cuối; báo cả hai rates, không chọn theo reference. Data của arm tự tạo từ audio clean tương ứng, không cần corpus mới.

PhonoRepair core comparisons cần ưu tiên native, identity, anchor-only, centroid và gate ablations. TTA dùng target updates nên báo riêng, không trình bày như cùng data-access protocol.

## 9. Schema đầu ra để tái lập

Các file sau **cần sinh khi triển khai**, chưa được tạo như experimental results:

| File | Trường tối thiểu |
| --- | --- |
| audio_manifest.csv/jsonl | corpus, revision, official_split, role, speaker_id, utterance_id, audio_path, audio_sha256, sample_rate, duration_s, text, exclusion_reason |
| phones.jsonl | utterance_id, start/end samples hoặc seconds, raw_phone, mapped_phone, annotation_source, mapping_version, annotator/adjudication nếu manual |
| locked_run.json | model_id, repo_revision, weights_sha256, code_commit, tokenizer_hash, mode, chunk/context, decoder_options, normalizer, source_manifest_hash, artifact_hash, calibration_manifest_hash |
| e0_cases.json + arrays.npz | seed, case_id, dimensions, dtype, scale, inputs, expected/reference outputs, tolerance, pass/fail |
| e1_metrics.csv | model, corpus/subset, mapping, boundary coverage/median/p95, interior coverage/agreement, routing/macro accuracy, probe PER |
| frame_metrics.parquet | run/utterance/frame, annotation provenance, evaluator label, routed class, correct-class residual, nearest/second distances, score, threshold, acceptance, first_exit, displacement |
| synthetic_manifest.jsonl | source utterance, artifact/generator version, seed, direction, kappa, nu, skip reason, array checksum |
| utterance_results.csv | run, arm, speaker, utterance, reference, hypothesis, S/D/I/N, accepted/changed frames, harm, correction |
| block_results.csv | speaker/block ID, episode IDs, accepted_any, latent_changed_any, transcript_changed_any, harms/corrections |
| geometry_summary.json | candidate/split, Ehat/quantiles, Gamma_lower, eta/nu/provenance, w, L/provenance, r_cov/status, U_gap/status, storage/compute |

CSV/JSON giữ metadata/path; arrays lớn dùng NPZ/Parquet. Trường chưa biết dùng null/NA kèm reason, không điền 0. Không đóng gói raw corpus ngoài quyền sử dụng.

## 10. Các điểm cần chốt trước chạy

1. **TIMIT access:** cần corpus được cấp quyền; HF loader không cung cấp audio/PHN.
2. **Manual audit E1:** tách 200 TIMIT TRAIN khỏi clean-source audit; cohort LibriSpeech cần annotation.
3. **Native artifact:** khóa weights/code/tokenizer/mode; Z1 cần chunk/context/averaging provenance.
4. **Phone protocol:** mapping B.5 là tùy chỉnh; chốt SA, full/core TEST, boundary matching và unsupported coverage.
5. **Metric conventions:** norm/scale E0, quantiles, denominators/NA, acceptance/latent/text events, paired bootstrap.
6. **Geometry claims:** eta_cert/U_gap population chỉ báo khi assumptions hợp lệ; speech có thể chỉ empirical/conditional.
7. **Lock:** E2-holdout/E3 không dùng chọn lại candidate/checkpoint/gate. Failure là kết quả của phiên bản.

**Mức sẵn sàng:** nguồn công khai đủ để chuẩn bị E0, trích clean features và thiết kế E2/E3. E1 đầy đủ phụ thuộc TIMIT và manual source annotations. Việc có đủ nguồn không chứng minh geometry khả thi hoặc WER sẽ cải thiện.
