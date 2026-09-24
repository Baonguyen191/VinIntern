# Statistical Baseline — BDG2 (Rat_office_Colby)

Xây dựng **mô hình statistical baseline** dự đoán mức tiêu thụ điện bình thường cho tòa nhà văn phòng, sử dụng dữ liệu từ **Building Data Genome Project 2**.

Đánh giá theo chuẩn **ASHRAE Guideline 14**: CV(RMSE) ≤ 25%, |NMBE| ≤ 10%.

---

## Baseline Là Gì?

Mô hình baseline học các yếu tố ảnh hưởng đến tiêu thụ điện bình thường:

- **Ngày trong tuần** → weekday có tải cao hơn weekend
- **Chu kỳ mùa** → hệ số mùa theo tháng
- **Xu hướng dài hạn** → tải tăng/giảm dần theo năm

Pipeline statistical dùng **seasonal decomposition + adaptive rolling median**, không phụ thuộc thời tiết — phù hợp cho tòa nhà có tải ổn định.

```
                    ┌─────────────────┐
Dữ liệu thô ──→   │  Làm sạch       │
                    │  Trích đặc trưng│──→  Statistical Model ──→  Predicted (dự đoán)
                    │  Huấn luyện     │                               │
                                                                      │
                                          Actual (thực tế) ───────────┤
                                                                      ▼
                                                            Residual = Actual − Predicted
                                                                      │
                                                                      ▼
                                                        (Downstream: Anomaly Detection)
```

---

## Kết Quả

Đánh giá trên 3 biến thể dữ liệu (Train 2016, Test 2017):

| Variant | Model | CV(RMSE) | NMBE | R² | ASHRAE |
|---------|-------|----------|------|-----|--------|
| raw | dow_median | 17.67% | -1.27% | -0.074 | PASS |
| raw | overall_median | 17.86% | -1.72% | -0.096 | PASS |
| bdg2_cleaned | dow_median | 17.67% | -1.27% | -0.074 | PASS |
| bdg2_cleaned | overall_median | 17.86% | -1.72% | -0.096 | PASS |
| **our_cleaned** | **dow_median** | **7.29%** | **+0.92%** | **0.218** | **PASS** |
| **our_cleaned** | **overall_median** | **7.75%** | **+0.38%** | **0.115** | **PASS** |

**Nhận xét**: Làm sạch dữ liệu (our_cleaned) giảm CV(RMSE) từ ~18% xuống ~7% — cải thiện hơn 2 lần. Model `dow_median` tốt hơn nhờ phân biệt weekday/weekend.

### Model Weights

Pipeline lưu trọng số mô hình vào `fold_N_<model>_weights.json`:

```json
{
  "model": "dow_median",
  "overall_median": 50763.12,
  "seasonal_index": {
    "1": 1.018, "2": 1.014, "3": 0.943, "4": 0.947,
    "5": 0.981, "6": 1.052, "7": 1.070, "8": 1.091,
    "9": 1.059, "10": 0.893, "11": 0.859, "12": 0.940
  },
  "dow_medians": {
    "0": 50983.33, "1": 51430.51, "2": 52028.69,
    "3": 51686.34, "4": 51854.32, "5": 47277.88, "6": 45941.18
  },
  "trend_slope": 180.08,
  "window": 28,
  "train_period": "2016-01-02 to 2017-01-02"
}
```

| Trường | Ý nghĩa |
|--------|---------|
| `overall_median` | Median tiêu thụ toàn bộ train (kWh/ngày) — mức nền |
| `seasonal_index` | Hệ số mùa theo tháng. >1 = tháng tiêu thụ cao, <1 = thấp |
| `dow_medians` | Median tiêu thụ theo ngày trong tuần (0=Thứ 2, 6=Chủ nhật) |
| `trend_slope` | Xu hướng tuyến tính (kWh/ngày). Dương = tải tăng dần |
| `window` | Cửa sổ rolling đã dùng (ngày) |

---

## Hướng Dẫn Sử Dụng

### Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu: Python ≥ 3.10, pandas, numpy, scikit-learn, matplotlib, lightgbm, scipy, holidays.

### Chạy baseline

```bash
python main.py --pipeline statistical --building Rat_office_Colby --window 28
```

### Tham số

| Tham số | Mặc định | Ý nghĩa |
|---------|---------|---------|
| `--building` | Rat_office_Colby | Mã tòa nhà BDG2 |
| `--output` | tự động | Thư mục xuất kết quả |
| `--eta` | 3.0 | Hệ số làm sạch outlier. Tăng → ít loại bỏ hơn |
| `--window` | 28 | Cửa sổ rolling (ngày) |
| `--k` | 3.0 | Hệ số ngưỡng anomaly (k·σ) |

---

## Output

Sau khi chạy pipeline, thư mục output chứa:

```
results/bdg2/statistical/Rat_office_Colby/
├── comparison.csv                 # So sánh metrics giữa các variant
├── <variant>/
│   ├── dow_profiles.png           # Biểu đồ tải theo ngày trong tuần
│   └── fold_1/
│       ├── fold_1_metrics.json
│       ├── fold_1_<model>_weights.json
│       ├── fold_1_*_actual_vs_pred.png
│       ├── fold_1_*_baseline.png
│       ├── fold_1_*_residuals.png
│       ├── fold_1_*_scatter.png
│       └── fold_1_*_cusum.png
```

### Giải thích metrics

| Metric | Ý nghĩa | Chuẩn ASHRAE |
|--------|---------|-------------|
| **CV(RMSE)** | Sai số dự đoán trung bình (%). Càng thấp = baseline càng chính xác | ≤ 25% |
| **NMBE** | Thiên lệch hệ thống (%). Gần 0 = không thiên lệch | \|NMBE\| ≤ 10% |
| **R²** | Phần biến thiên được giải thích. Gần 1 = tốt | — |

### Biểu đồ

| File | Nội dung |
|------|---------|
| `*_actual_vs_pred.png` | Chuỗi thời gian thực tế vs dự đoán |
| `*_baseline.png` | Đường baseline theo thời gian |
| `*_residuals.png` | Phân bố residual |
| `*_scatter.png` | Scatter actual vs predicted |
| `*_cusum.png` | Biểu đồ CUSUM — phát hiện xu hướng lệch dần |
| `dow_profiles.png` | Tải trung bình theo ngày trong tuần |

---

## Sprint 1 — Bộ Ca Bất Thường Có Nhãn + Bộ Phát Hiện Ban Đầu

Dùng bộ dữ liệu chung `data_normalized` (BDG2 năm 2017, lưới 1 giờ) và định dạng output O1/O2 trong `problem.md` mục 5.

### Chạy

```bash
# bo du lieu chung dat o data_normalized/ (hoac data/normalized/)
python main.py --pipeline anomaly-cases --seed 42 --k 3.0
```

Tham số: `--data-dir` (mặc định `data_normalized`, nếu không có thì `data/normalized`), `--output` (mặc định `results/anomaly_cases`), `--seed` (sinh ca tái lập được), `--k` (độ rộng dải baseline).

### Nguyên tắc: chỉ tạo ca trên dữ liệu thật và sạch

Chỉ tạo ca trên key có số đo thật trong bộ dữ liệu chung:

| Key | Nguồn | Có ca |
|---|---|---|
| `power_active_kw` | M1, đo thật | Có |
| `cooling_kw` | M2, đo thật | Có |
| CO2, đồng hồ nước (M3), điện chiller | Không có trong bộ dữ liệu | **Không** — không giả lập kênh để tạo ca |
| `energy_active_kwh_total` | Counter mô phỏng (`cumsum`) | Không |

Không chọn đồng hồ rác. Trên toàn bộ dữ liệu, 75/806 đồng hồ M1 và 48/349 đồng hồ M2 bị loại vì coverage < 50%, > 50% giá trị 0 hoặc > 30% giá trị lặp. Tiêu chí chọn đồng hồ tạo ca chặt hơn:

- **M1 (6 đồng hồ điện văn phòng):** ≥ 98% mẫu, < 2% mẫu lặp giá trị khác 0, tải đêm > 1 kW, tỷ lệ tải ngày/đêm ≥ 1.5. `Rat_office_Colby` không đạt (tải ngày/đêm quá phẳng).
- **M2 (4 đồng hồ nước lạnh văn phòng):** ≥ 98% mẫu, < 2% mẫu lặp, < 20% giá trị 0, p95 ≥ 50 kW, tỷ lệ tải ngày/đêm mùa hè (05–08) ≥ 1.5. Nhà máy lạnh chạy phẳng 24/7 không có chế độ "ngoài giờ" nên không dùng được cho `off_hours_run`.

01/01–14/03 giữ sạch để baseline trượt 8 tuần khởi động. Mỗi đồng hồ có một chuỗi ô 21 ngày, mỗi ô 1 ca, các ca không chồng nhau. `stuck_sensor` chỉ đặt vào đoạn giá trị khác 0 (giá trị 0 đóng băng không phân biệt được với đồng hồ tắt).

| Loại ca | Nhóm | Key | Cách tạo | Thời lượng |
|---|---|---|---|---|
| `spike` | A | cả hai | + 1.0–2.0 × p95 của đồng hồ | 1–2 h |
| `off_hours_run` | A | cả hai | Từ 20:00 giữ tải ở 70–100% median giờ làm việc, nhiễu 3%; bỏ ca nếu mức tăng trung bình < 30% chênh lệch ngày–đêm | 8–10 h |
| `baseload_rise` | A | `power_active_kw` | + 30–60% tải nền đêm, liên tục 24/7 | 3–7 ngày |
| `data_loss` | DQ | cả hai | Xoá mẫu | 6–48 h |
| `stuck_sensor` | DQ | cả hai | Đóng băng giá trị đầu cửa sổ | 8–24 h |

Thứ tự ca M1: `spike, off_hours_run, baseload_rise, data_loss, stuck_sensor, spike, off_hours_run, baseload_rise`. Thứ tự ca M2: `spike, data_loss, off_hours_run, stuck_sensor, spike, off_hours_run` (ca cần chiller chạy đặt vào cuối xuân/hè).

**Đánh dấu ca giả lập:** mọi ca bơm vào có `is_synthetic=True`, `source=injected`. Lỗi dữ liệu **đã có sẵn** trong chuỗi thật (mất mẫu ≥ 3 h, đơ số ≥ 4 h) được gắn `source=natural` để không bị tính là báo động giả. Chuỗi gốc luôn được giữ ở cột `<key>_clean` cạnh chuỗi đã bơm lỗi.

**Chưa có ca cho:** rò rỉ nước (M3), CO2 (M4), suy giảm hiệu suất chiller (nhóm B). Các nhóm này cần dữ liệu đo thật (đồng hồ nước, cảm biến CO2, điện từng chiller) trước khi tạo ca.

### Bộ phát hiện ban đầu

| Detector | Nhóm | Phương pháp | Tham chiếu |
|---|---|---|---|
| `profile_band` | A | Dải median ± k·IQR/1.349 theo ô giờ × loại ngày (ngày lễ dùng profile cuối tuần). Vượt dải < 3 h với z ≥ 5 → `spike`; vượt ≥ 3 h → `off_hours_run` / `high_consumption` theo tỷ lệ giờ vận hành (07–19h ngày làm việc) | Trượt 8 tuần, nhân quả |
| `cusum_night_residual` | A | CUSUM (`src/core/anomaly_detector.py`) trên phần dư trung bình 01–04h mỗi ngày → `baseload_rise` (chỉ chạy trên `power_active_kw`) | Trượt 8 tuần |
| `dq_gap` / `dq_flatline` | DQ | Mất mẫu ≥ 3 h / giá trị khác 0 lặp lại ≥ 4 h | — |

Mỗi sự kiện có `start`, `end`, `detected_at` (thời điểm đầu tiên luật đủ dữ liệu để báo; dùng để tính độ trễ), baseline, độ lệch, `excess_kwh` — theo O2.

### Kết quả (seed 42, k = 3)

Sự kiện khớp nhãn nếu cùng đồng hồ + key và chồng lấn thời gian (cho phép trễ 24 h). Precision tính **theo sự kiện**.

| Key | Loại ca | Số ca | Event recall | Đúng loại | Trễ trung vị |
|---|---|---|---|---|---|
| `power_active_kw` | spike | 12 | 1.00 | 0.92 | 0 h |
| `power_active_kw` | off_hours_run | 12 | 1.00 | 1.00 | 3 h |
| `power_active_kw` | baseload_rise | 12 | 1.00 | 0.92 | 2 h |
| `power_active_kw` | data_loss | 6 | 1.00 | 1.00 | 2 h |
| `power_active_kw` | stuck_sensor | 6 | 1.00 | 1.00 | 3 h |
| `cooling_kw` | spike | 8 | 1.00 | 0.75 | 0 h |
| `cooling_kw` | off_hours_run | 8 | 1.00 | 1.00 | 4.5 h |
| `cooling_kw` | data_loss | 4 | 1.00 | 1.00 | 2 h |
| `cooling_kw` | stuck_sensor | 4 | 1.00 | 1.00 | 3 h |

| Key | Event precision | Báo động giả / thiết bị / ngày |
|---|---|---|
| `power_active_kw` | **0.346** | **0.097** |
| `cooling_kw` | 0.112 | 0.157 |

Kênh điện đạt tiêu chí nhóm A đề xuất trong `problem.md` mục 10 (precision ≥ 0.20 tại recall ≥ 0.35, ngân sách 0.2 báo động/thiết bị/ngày). Kênh nước lạnh nằm trong ngân sách báo động nhưng precision chưa đạt 0.20.

**Kiểm tra độ lệch baseline:** `eval_cases.csv` so sánh lượng đã bơm (`value − clean`) với lượng baseline đo được (`value − expected`). Tỷ số `recovery` có trung vị 1.01, 27/52 ca biên độ nằm trong 0.9–1.1.

**Giới hạn:**
- Biên độ ca đang lớn nên recall cao. Chưa thử ca biên độ nhỏ.
- Chuỗi thật có thể chứa bất thường thật chưa được gán nhãn, nên precision ở đây là **cận dưới**.
- `cooling_kw` thay đổi mạnh theo mùa và thời tiết. Dải trượt 8 tuần theo sau mùa chậm, nên `off_hours_run` / `high_consumption` báo giả nhiều (159 + 40 sự kiện, 15 đúng). Cần baseline có biến nhiệt độ cho kênh này.
- `cusum_night_residual` dùng mean/σ của cả năm (hàm CUSUM có sẵn), chưa hoàn toàn nhân quả.

### Output (`results/anomaly_cases/`)

| File | Nội dung |
|## Sprint 2 — Bộ Phát Hiện Bất Thường: Luật vs Isolation Forest vs LightGBM

Chạy trên cùng bộ ca có nhãn của Sprint 1 (`build_case_set`, cùng seed). Bộ phát hiện chỉ cần dữ liệu đo chuẩn hoá. Không phụ thuộc mô hình dự báo nào.

```bash
python main.py --pipeline anomaly-detect --seed 42
```

### Luồng xử lý

```
dữ liệu đo ──► lớp DQ (mất mẫu ≥ 3 h, đơ số ≥ 4 h) ──────────────────────────┐
          └──► nhánh phát hiện (1 trong 4 bộ bên dưới) ──► tách lỗi cảm biến ──► gộp cảnh báo liên tiếp ──► danh sách cảnh báo O2
```

| Bộ | Cách làm | Tham chiếu |
|---|---|---|
| `rules` | Luật đã thống nhất: `profile_band` + `cusum_night_residual`, **k = 3 giữ nguyên** | Trượt 8 tuần, nhân quả |
| `lightgbm` | Nhánh so sai lệch với baseline: LightGBM quantile (q10/q50/q90) theo giờ, thứ, loại ngày, nhiệt độ; **không dùng lag** (lag sẽ kéo bất thường vào dự báo). Dải O1 = q50 ± k·σ, với σ = (q90 − q10)/2.563. Dùng lại luật chạy/phân loại và CUSUM đêm trên phần dư | Cross-fit theo tháng |
| `iforest` | Isolation Forest trên số đo thô: giá trị, thay đổi 1 h, trung bình 24 h, giờ, ngày làm việc, nhiệt độ | Cross-fit theo tháng |
| `iforest_ctx` | Isolation Forest trên độ lệch so với median cùng khung giờ (hiện tại, trung bình 24 h) + thay đổi 1 h | Cross-fit theo tháng |

- **Cross-fit theo tháng:** mô hình chấm tháng m được huấn luyện trên 11 tháng còn lại của 2017. Cách này thay cho "có 1 năm lịch sử", vì bộ dữ liệu chỉ có 1 năm. Cách này **không nhân quả hoàn toàn**, khác với luật trượt 8 tuần.
- **Isolation Forest:** chỉ giữ chiều tăng, giống các bộ khác (bộ ca không có loại "quá thấp"). Chuỗi ngắn < 3 h chỉ thành `spike` nếu vượt ngưỡng chặt hơn 10 lần. Chuỗi ≥ 24 h trên điện được gắn `baseload_rise`.
- **Tách lỗi cảm biến:** cảnh báo nhóm A nằm ≥ 50% trong cửa sổ DQ của cùng đồng hồ bị bỏ. Lỗi này chỉ báo một lần, dưới dạng `data_loss` / `stuck_sensor`. Mô hình IF/LightGBM không học trên giờ DQ.
- **Gộp cảnh báo:** gộp cảnh báo cùng đồng hồ, key, bộ phát hiện và loại nếu cách nhau ≤ 2 h. Giữ `detected_at` sớm nhất. Trung bình tính theo thời lượng, lấy đỉnh lớn nhất, cộng dồn `excess_kwh`. Ghi `n_merged` vào bằng chứng.
- **Bằng chứng mỗi cảnh báo (O2):** `actual_mean`, `expected_mean`, `deviation_pct`, `peak_z` (luật/LightGBM) hoặc `score` (IF), `excess_kwh`, `duration_h`, `n_merged`, `reference_window`.

### Kết quả (seed 42)

Ngưỡng mô hình được chọn trên dải quét. Tiêu chí: recall nhóm A cao nhất mà báo động giả ≤ 0.2/thiết bị/ngày trên mọi key; hoà thì chọn precision cao hơn. Luật giữ k = 3.

| Bộ | Ngưỡng | Recall (A) | Đúng loại | Precision | Precision điện | Precision nước lạnh | BĐ giả/TB/ngày điện | BĐ giả/TB/ngày nước lạnh |
|---|---|---|---|---|---|---|---|---|
| `rules` | k = 3 | **1.00** | **0.94** | 0.234 | 0.333 | 0.111 | 0.095 | 0.153 |
| `lightgbm` | k = 3.5 | 0.92 | 0.93 | 0.233 | **0.384** | 0.080 | 0.082 | 0.181 |
| `iforest` | q = 0.95 | 0.60 | 0.56 | 0.164 | 0.235 | 0.086 | 0.092 | 0.152 |
| `iforest_ctx` | q = 0.95 | 0.65 | 0.61 | 0.198 | 0.296 | 0.101 | 0.069 | 0.134 |

Recall theo loại ca tại điểm vận hành:

| Key / loại | rules | lightgbm | iforest | iforest_ctx |
|---|---|---|---|---|
| điện / spike | 1.00 | 1.00 | 0.83 | 1.00 |
| điện / off_hours_run | 1.00 | 1.00 | 0.17 | 0.08 |
| điện / baseload_rise | 1.00 | 1.00 | 0.83 | 0.83 |
| nước lạnh / spike | 1.00 | 0.88 | 0.75 | 1.00 |
| nước lạnh / off_hours_run | 1.00 | 0.63 | 0.38 | 0.38 |
| DQ (data_loss, stuck_sensor) | 1.00 | 1.00 | 1.00 | 1.00 |

**Nhận xét:**
- **Luật đã thống nhất vẫn tốt nhất về recall và đúng loại.** LightGBM đạt precision cao hơn trên điện (0.384 so với 0.333, báo động giả thấp hơn một chút) nhưng bỏ sót 4 ca nước lạnh (3 `off_hours_run`, 1 `spike`).
- **Isolation Forest không thay được luật.** Nó bỏ sót gần hết `off_hours_run`: tải ngày kéo sang đêm có giá trị bình thường trong năm. Một lệch vừa phải trên 1 đặc trưng cũng bị loãng giữa nhiều đặc trưng khác. Đặc trưng ngữ cảnh giúp `spike` và precision, nhưng không cứu được `off_hours_run`. IF phù hợp làm lớp phụ bắt `spike` với precision cao ở ngưỡng chặt (q = 0.998: precision 0.75, báo động giả 0.002–0.006/thiết bị/ngày).
- **LightGBM trên nước lạnh:** baseline theo nhiệt độ chưa cải thiện precision (0.080 so với 0.111 của luật). Giả thuyết chưa kiểm chứng: báo động giả đến từ biến động vận hành của nhà máy lạnh (lịch bật/tắt, chế độ chạy) mà giờ + nhiệt độ không giải thích được.
- Lớp DQ phát hiện 100% ca lỗi cảm biến ở mọi bộ. Bước tách lỗi cảm biến bỏ 3 (luật) / 2 (LightGBM) cảnh báo nhóm A nằm trên cửa sổ DQ; IF không có cảnh báo nào bị bỏ vì giờ DQ đã bị loại trước khi chấm điểm.

### Giao diện

Bộ phát hiện theo luật (k = 3) được gắn vào demo chung `dashboard_v1.py`, thay phần mock `is_anomaly` / `anomaly_reason` (`src/anomaly/rules.py`). Luật chạy trên chuỗi thật cả năm của tòa nhà đang chọn (baseline trượt 8 tuần cần lịch sử), rồi gắn cờ cho các giờ trong tập test.

```bash
python scripts/build_baseline_forecast.py   # tao data_normalized/forecast_test_results.parquet
streamlit run dashboard_v1.py
```

### Output (`results/anomaly_detection/`)

| File | Nội dung |
|---|---|
| `alerts_<bộ>.csv`, `alerts_<bộ>_O2.json` | Danh sách cảnh báo cuối (sau tách lỗi cảm biến + gộp) kèm bằng chứng và trạng thái khớp nhãn |
| `comparison_summary.csv` | Chỉ số tại điểm vận hành của từng bộ |
| `comparison_sweep.csv` | Toàn bộ dải quét ngưỡng |
| `comparison_by_type.csv` | Recall / đúng loại theo key × loại ca × bộ |
| `eval_cases_<bộ>.csv` | Kết quả từng ca |
| `baseline_lgbm_O1.csv` | Baseline LightGBM dạng O1; không commit (~13 MB), chạy lại để tạo |
| `tradeoff_recall_vs_false_alarms.png`, `recall_by_type_by_suite.png` | Đường recall – báo động giả; recall theo loại |

---|---|
| `labels.csv` | Nhãn: `case_id, entity_id, module, key, group, anomaly_type, start, end, magnitude, is_synthetic, source, note` |
| `series_M1.csv`, `series_M2.csv` | Chuỗi đã bơm lỗi + cột `_clean` + `case_id` từng giờ |
| `baseline_O1.csv` | Baseline dạng O1 (`expected/lower/upper`); không commit (~12 MB), chạy lại để tạo |
| `events.csv`, `events_O2.json` | Sự kiện phát hiện (O2) + trạng thái khớp nhãn |
| `eval_cases.csv`, `eval_by_type.csv`, `eval_overall.json` | Đánh giá theo ca / key × loại / tổng |
| `plots/<case_id>_<type>.png`, `recall_by_type.png` | Hình từng ca: chuỗi gốc, chuỗi có lỗi, dải baseline, sự kiện phát hiện |

---|---|
| `labels.csv` | Nhãn: `case_id, entity_id, module, key, group, anomaly_type, start, end, magnitude, is_synthetic, source, note` |
| `series_M1.csv`, `series_M2.csv` | Chuỗi đã bơm lỗi + cột `_clean` + `case_id` từng giờ |
| `baseline_O1.csv` | Baseline dạng O1 (`expected/lower/upper`); không commit (~12 MB), chạy lại để tạo |
| `events.csv`, `events_O2.json` | Sự kiện phát hiện (O2) + trạng thái khớp nhãn |
| `eval_cases.csv`, `eval_by_type.csv`, `eval_overall.json` | Đánh giá theo ca / loại / tổng |
| `plots/<case_id>_<type>.png`, `recall_by_type.png` | Hình từng ca: chuỗi gốc, chuỗi có lỗi, dải baseline, sự kiện phát hiện |

---

## Cấu Trúc Dự Án

```
├── main.py                      # Điểm vào chính
├── requirements.txt
│
├── src/
│   ├── io/                      # Đọc & ghép dữ liệu
│   ├── core/                    # Metrics, cross-validation, làm sạch
│   ├── features/                # Trích xuất đặc trưng
│   ├── models/                  # Mô hình baseline
│   ├── pipelines/               # Pipeline chính
│   ├── anomaly/                 # Sinh ca có nhãn, baseline giờ, detector luật/IF/LightGBM, hậu xử lý, đánh giá
│   └── visualization/           # Vẽ biểu đồ
│
├── scripts/                     # Script EDA & đánh giá bổ sung
│
└── results/
    ├── anomaly_cases/           # Kết quả Sprint 1
    ├── anomaly_detection/       # Kết quả Sprint 2 (so sánh bộ phát hiện)
    └── bdg2/
        └── statistical/         # Kết quả Pipeline Statistical
```

---

## Tiêu Chuẩn Đánh Giá

Theo **ASHRAE Guideline 14-2014**:

| Chỉ số | Ngưỡng | Ý nghĩa |
|--------|--------|---------|
| CV(RMSE) | ≤ 25% (dữ liệu ngày) | Độ chính xác dự đoán |
| \|NMBE\| | ≤ 10% | Độ thiên lệch hệ thống |

Phương pháp: **Expanding-window cross-validation** — train trên dữ liệu tích lũy, test trên năm tiếp theo.
