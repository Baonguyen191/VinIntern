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
# copy bo du lieu chung vao data/normalized/ (thu muc data/ khong duoc commit)
python main.py --pipeline anomaly-cases --seed 42 --k 3.0
```

Tham số: `--data-dir` (mặc định `data/normalized`), `--output` (mặc định `results/anomaly_cases`), `--seed` (sinh ca tái lập được), `--k` (độ rộng dải baseline).

### Bộ ca

- **6 đồng hồ điện văn phòng (M1)** chọn ngẫu nhiên theo seed trong các đồng hồ có ≥ 98% mẫu, tỷ lệ tải ngày/đêm ≥ 1.5 và < 2% mẫu lặp giá trị. `Rat_office_Colby` không đạt tiêu chí này (tải ngày/đêm quá phẳng).
- **2 đồng hồ nước lạnh văn phòng (M2)** có chiller chạy cả trong cửa sổ tham chiếu (04/2017–05/2017) lẫn mùa hè.
- 01/01–14/03 giữ sạch để baseline trượt 8 tuần khởi động. Mỗi đồng hồ M1 có 8 ô 21 ngày, mỗi ô 1 ca, các ca không chồng nhau.

| Loại ca | Nhóm | Key | Cách tạo | Thời lượng |
|---|---|---|---|---|
| `spike` | A | `power_active_kw` | + 1.0–2.0 × p95 của đồng hồ | 1–2 h |
| `off_hours_run` | A | `power_active_kw` | Từ 20:00 giữ tải ở 70–100% median giờ làm việc, nhiễu 3%; bỏ ca nếu mức tăng trung bình < 30% chênh lệch ngày–đêm | 8–10 h |
| `leak` | A | `power_active_kw` | + 30–60% tải nền đêm, liên tục 24/7 | 3–7 ngày |
| `co2_sustained_high` | A | `co2_ppm` | + 250–450 ppm | 2–3 ngày |
| `data_loss` | DQ | `power_active_kw` | Xoá mẫu | 6–48 h |
| `stuck_sensor` | DQ | `power_active_kw` | Đóng băng giá trị đầu cửa sổ | 8–24 h |
| `efficiency_degradation` | B | `kw_per_kw_cooling` | kW/kW lạnh tăng tuyến tính 12–20% trong 7 ngày rồi giữ 21 ngày | 28 ngày |

**Đánh dấu ca giả lập:** mọi ca bơm vào có `is_synthetic=True`, `source=injected`. Lỗi dữ liệu **đã có sẵn** trong chuỗi thật (mất mẫu, đơ số) được gắn `source=natural` để không bị tính là báo động giả. Chuỗi gốc luôn được giữ ở cột `<key>_clean` cạnh chuỗi đã bơm lỗi.

**Kênh giả lập — cần lưu ý khi đọc kết quả:**
- `co2_ppm` là kênh **hoàn toàn giả lập** (420 ppm + proxy số người lấy từ profile điện). Bộ dữ liệu chung không có CO2 (M4 ngoài phạm vi MVP).
- `leak` được bơm vào chuỗi **điện** để thay thế cho rò rỉ nước M3, vì bộ dữ liệu chung chưa có đồng hồ nước. Dạng lỗi giống nhau (tải nền đêm tăng liên tục).
- Điện chiller (`chiller_power_kw`) được **tổng hợp** từ `cooling_kw` thật qua đường cong EIR-FPLR trong `equipment_params.csv` (máy mẫu gần công suất nhất, Q định mức = 1.1 × p99 tải lạnh, nhiễu 2%). Cấu hình từng máy ở `virtual_chillers.json`.

### Bộ phát hiện ban đầu

| Detector | Nhóm | Phương pháp | Tham chiếu |
|---|---|---|---|
| `profile_band` | A | Dải median ± k·IQR/1.349 theo ô giờ × loại ngày (ngày lễ dùng profile cuối tuần). Vượt dải < 3 h với z ≥ 5 → `spike`; vượt ≥ 3 h → `off_hours_run` / `high_consumption` theo tỷ lệ giờ vận hành (07–19h ngày làm việc), hoặc `co2_sustained_high` | Trượt 8 tuần, nhân quả |
| `cusum_night_residual` | A | CUSUM (`src/core/anomaly_detector.py`) trên phần dư trung bình 01–04h mỗi ngày → `leak` | Trượt 8 tuần |
| `fixed_reference_regression` | B | Hồi quy bậc 2 của kW/kW lạnh theo PLR, fit trên cửa sổ **cố định** 01/04–27/05 (giả định bảo trì 01/04). Báo khi median 3 ngày của độ lệch ≥ 8% | Cố định |
| `dq_gap` / `dq_flatline` | DQ | Mất mẫu ≥ 3 h / giá trị khác 0 lặp lại ≥ 4 h | — |

Mỗi sự kiện có `start`, `end`, `detected_at` (thời điểm đầu tiên luật đủ dữ liệu để báo; dùng để tính độ trễ), baseline, độ lệch, `excess_kwh` — theo O2.

### Kết quả (seed 42, k = 3)

Sự kiện khớp nhãn nếu cùng đồng hồ + key và chồng lấn thời gian (cho phép trễ 24 h). Precision tính **theo sự kiện**.

| Loại ca | Số ca | Event recall | Đúng loại | Trễ trung vị |
|---|---|---|---|---|
| spike | 12 | 1.00 | 0.92 | 0 h |
| off_hours_run | 12 | 0.92 | 0.92 | 2 h |
| leak | 6 | 1.00 | 1.00 | 3.5 h |
| co2_sustained_high | 6 | 1.00 | 1.00 | 2 h |
| data_loss | 6 | 1.00 | 1.00 | 2 h |
| stuck_sensor | 6 | 1.00 | 1.00 | 3 h |
| efficiency_degradation | 2 | 1.00 | 1.00 | 155 h |

| Key | Event precision | Báo động giả / thiết bị / ngày |
|---|---|---|
| `power_active_kw` | **0.263** | **0.104** |
| `co2_ppm` (giả lập) | 0.050 | 0.078 |
| `kw_per_kw_cooling` | 1.000 | 0 |

Kênh điện đạt tiêu chí nhóm A đề xuất trong `problem.md` mục 10 (precision ≥ 0.20 tại recall ≥ 0.35, ngân sách 0.2 báo động/thiết bị/ngày). Suy giảm hiệu suất được phát hiện trong khoảng 6–7 ngày kể từ lúc bắt đầu (đạt mục tiêu ≤ 7 ngày kể từ khi suy giảm ổn định).

**Kiểm tra độ lệch baseline:** `eval_cases.csv` so sánh lượng đã bơm (`value − clean`) với lượng baseline đo được (`value − expected`). Tỷ số `recovery` có trung vị 0.99, 27/38 ca nằm trong 0.9–1.1, tức độ lệch báo ra phản ánh đúng độ lớn ca.

**Giới hạn:**
- Biên độ ca đang lớn nên recall cao. Chưa thử ca biên độ nhỏ.
- Chuỗi thật có thể chứa bất thường thật chưa được gán nhãn, nên precision ở đây là **cận dưới**.
- Ca bị bỏ sót (`SYN-M1-05-02`, Rat_office_Loyd): tải lịch sử dao động mạnh nên dải baseline ban đêm quá rộng.
- CO2 báo động giả nhiều vì kênh này lấy proxy từ điện thật, nên dao động thật của điện lan sang CO2.
- Kết quả nhóm B chỉ kiểm tra được đường ống xử lý: điện chiller là số tổng hợp từ chính đường cong dùng để tham chiếu.
- `cusum_night_residual` dùng mean/σ của cả năm (hàm CUSUM có sẵn), chưa hoàn toàn nhân quả.

### Output (`results/anomaly_cases/`)

| File | Nội dung |
|---|---|
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
│   ├── anomaly/                 # Sinh ca có nhãn, baseline giờ, detector, đánh giá (Sprint 1)
│   └── visualization/           # Vẽ biểu đồ
│
├── scripts/                     # Script EDA & đánh giá bổ sung
│
└── results/
    ├── anomaly_cases/           # Kết quả Sprint 1
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
