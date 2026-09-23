# Baseline Tiêu Thụ Điện Năng Tòa Nhà Thông Minh

Xây dựng **mô hình baseline** dự đoán mức tiêu thụ điện bình thường cho tòa nhà thương mại/công nghiệp. Baseline là nền tảng để phát hiện bất thường: khi tiêu thụ thực tế lệch đáng kể so với dự đoán, đó là tín hiệu cần điều tra.

Đánh giá theo chuẩn **ASHRAE Guideline 14**: CV(RMSE) ≤ 25%, |NMBE| ≤ 10%.

---

## Baseline Là Gì?

Mô hình baseline học các yếu tố ảnh hưởng đến tiêu thụ điện bình thường:

- **Nhiệt độ** → Heating/Cooling Degree-Day (HDD/CDD)
- **Ngày trong tuần** → weekday có tải cao hơn weekend
- **Ngày lễ** → tải giảm sâu (đặc biệt Tết Âm Lịch)
- **Chu kỳ mùa** → mùa hè tiêu thụ nhiều hơn mùa đông
- **Xu hướng dài hạn** → tải tăng/giảm dần theo năm

Sau khi huấn luyện, baseline xuất ra **giá trị dự đoán** cho mỗi ngày. Hiệu số giữa thực tế và dự đoán gọi là **residual** — đây là đầu vào chính để các module phát hiện bất thường phía sau sử dụng.

```
                    ┌─────────────────┐
Dữ liệu thô ──→   │  Làm sạch       │
                    │  Trích đặc trưng│──→  Baseline Model ──→  Predicted (dự đoán)
Dữ liệu thời tiết ─┘  Huấn luyện     │                          │
                                                                  │
                                          Actual (thực tế) ───────┤
                                                                  ▼
                                                        Residual = Actual − Predicted
                                                                  │
                                                                  ▼
                                                    (Downstream: Anomaly Detection)
```

---

## Kết Quả Pipeline Statistical (BDG2 — Rat_office_Colby)

Pipeline statistical dùng **seasonal decomposition + adaptive rolling median**, không phụ thuộc thời tiết — phù hợp cho tòa nhà có tải ổn định.

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

## Hướng Dẫn Sử Dụng Baseline

### Cài đặt

```bash
pip install -r requirements.txt
```

Yêu cầu: Python ≥ 3.10, pandas, numpy, scikit-learn, matplotlib, lightgbm, scipy, holidays.

### Chạy baseline

```bash
# Baseline Degree-Day cho user EWELD (khuyến nghị)
python main.py --pipeline degree-day --user U317

# Baseline Statistical cho building BDG2
python main.py --pipeline statistical --building Rat_office_Colby --window 28
```

### Chọn pipeline nào?

| Tình huống | Pipeline | Lý do |
|-----------|----------|-------|
| Tòa nhà có **tiêu thụ phụ thuộc nhiệt độ** rõ ràng (có điều hòa/sưởi) | `degree-day` | Dùng HDD/CDD nắm bắt quan hệ nhiệt độ–tải |
| Tòa nhà **ổn định**, ít phụ thuộc thời tiết (văn phòng, IT) | `statistical` | Dùng seasonal decomposition + rolling median, không cần weather |

### Tham số quan trọng

| Tham số | Mặc định | Ý nghĩa |
|---------|---------|---------|
| `--user` | U317 | Mã user EWELD (U1–U386) |
| `--building` | Rat_office_Colby | Mã tòa nhà BDG2 |
| `--output` | tự động | Thư mục xuất kết quả |
| `--eta` | 3.0 | Hệ số làm sạch outlier. Tăng → ít loại bỏ hơn |
| `--balance-step` | 0.5 | Bước tìm balance point (°F). Giảm → chính xác hơn nhưng chậm hơn |
| `--window` | 28 | Cửa sổ rolling cho statistical baseline (ngày) |
| `--k` | 3.0 | Hệ số ngưỡng anomaly (k·σ) |

### Chọn user EWELD phù hợp

Không phải user nào cũng cho baseline tốt. Tiêu chí lọc:

| Tier | Tiêu chí | Số user | Khuyến nghị |
|------|----------|---------|-------------|
| **Tier 1** | ≥ 2 năm, < 50% zero | 298 | Dùng được |
| Tier 2 | 1–2 năm, < 50% zero | 22 | Cẩn thận |
| Tier 3 | < 1 năm HOẶC ≥ 50% zero | 66 | Không dùng |

---

## Output: Đọc Kết Quả Baseline

Sau khi chạy pipeline, thư mục output chứa:

```
results/eweld/degree_day/
├── summary_table.csv          # Metrics tất cả model × fold
├── ashrae_pass_fail.csv       # Đạt/không đạt ASHRAE từng fold
├── outlier_report.csv         # Danh sách ngày bị loại khi làm sạch
├── anomaly_report.csv         # Residual + cờ anomaly cho mỗi ngày
├── scatter_temp_kwh.png       # Biểu đồ nhiệt độ vs tiêu thụ
├── model_comparison.png       # So sánh hiệu năng 4 model
├── metric_trend.png           # Xu hướng metrics qua các fold
└── fold_N/
    ├── fold_N_metrics.json
    ├── fold_N_<model>_weights.json   # Trọng số mô hình (statistical pipeline)
    ├── fold_N_*_actual_vs_pred.png
    ├── fold_N_*_residuals.png
    └── fold_N_*_cusum.png
```


### Giải thích metrics

| Metric | Ý nghĩa | Chuẩn ASHRAE |
|--------|---------|-------------|
| **CV(RMSE)** | Sai số dự đoán trung bình (%). Càng thấp = baseline càng chính xác | ≤ 25% |
| **NMBE** | Thiên lệch hệ thống (%). Gần 0 = không thiên lệch. Âm = baseline dự đoán cao hơn thực tế | \|NMBE\| ≤ 10% |
| **R²** | Phần biến thiên được giải thích. Gần 1 = tốt | — |

---

### Biểu đồ

| File | Nội dung |
|------|---------|
| `*_actual_vs_pred.png` | Chuỗi thời gian thực tế vs dự đoán — kiểm tra trực quan baseline có khớp không |
| `*_residuals.png` | Phân bố residual — kiểm tra residual có phân bố chuẩn quanh 0 không |
| `*_cusum.png` | Biểu đồ CUSUM — phát hiện xu hướng lệch dần |
| `scatter_temp_kwh.png` | Nhiệt độ vs tiêu thụ — kiểm tra quan hệ degree-day |
| `model_comparison.png` | So sánh CV(RMSE) giữa 4 model |

---

## Cấu Trúc Dự Án

```
├── main.py                      # Điểm vào chính
├── requirements.txt
│
├── data/
│   ├── eweld/                   # Bộ dữ liệu EWELD (386 user, Đài Loan)
│   └── bdg2/                    # Bộ dữ liệu BDG2 (1.636 tòa nhà, đa quốc gia)
│
├── src/
│   ├── io/                      # Đọc & ghép dữ liệu
│   ├── core/                    # Metrics, cross-validation, làm sạch
│   ├── features/                # Trích xuất đặc trưng (degree-day, TOWT)
│   ├── models/                  # Mô hình baseline (Ridge, OLS, LightGBM, 5P, Statistical)
│   ├── pipelines/               # 4 pipeline chính
│   └── visualization/           # Vẽ biểu đồ
│
├── scripts/                     # Script EDA & đánh giá bổ sung
│
├── results/
│   ├── eweld/
│   │   ├── towt/                # Kết quả Pipeline TOWT
│   │   └── degree_day/          # Kết quả Pipeline Degree-Day
│   └── bdg2/
│       ├── degree_day/          # Kết quả Pipeline BDG2 Degree-Day
│       └── statistical/         # Kết quả Pipeline Statistical
│
└── docs/                        # Tài liệu, báo cáo làm sạch, biểu đồ EDA
```

Tham khảo `CLAUDE.md` để biết chi tiết kỹ thuật về từng module.

---

## Dữ Liệu

### EWELD

| File | Cột | Mô tả |
|------|-----|-------|
| `data/eweld/Electricity Consumption/**/U*.csv` | `Time, Value` | Tiêu thụ điện 15 phút (kWh) |
| `data/eweld/Weather Data/W*.csv` | `Time, Temperature(F), Humidity(%), ...` | Thời tiết 15 phút |
| `data/eweld/Extreme Weather/EW_CT*/*_interval.csv` | `Start Time, End Time, Weather` | Khoảng thời tiết cực đoan |
| `data/eweld/User Location/U_CT*.csv` | `User No.` | Ánh xạ user → cụm thời tiết |

386 user thuộc 17 ngành NACE, chia thành 3 cụm thời tiết (CT1: 12 user, CT2: 371 user, CT3: 2 user).

### BDG2

Dữ liệu điện theo giờ + thời tiết từ Building Data Genome Project 2. Chi tiết tại `data/bdg2/README.md`.

---

## User MVP & Ứng Viên Mở Rộng

### MVP hiện tại: U317 (Bất động sản)

| Thuộc tính | Giá trị |
|-----------|---------|
| Thời gian | 5.8 năm (2016-09 → 2022-08) |
| Bản ghi | 205.439 |
| Tỷ lệ zero | 0% |
| Tiêu thụ TB | 19.4 kWh/15min |
| Cụm thời tiết | CT2 (cụm lớn nhất) |

### Ứng viên tiếp theo

| User | Ngành | Thời gian | Tiêu thụ TB | Đặc điểm |
|------|-------|-----------|------------|----------|
| **U290** | Tài chính | 4.6 năm | 44.6 kWh | Mẫu văn phòng |
| **U352** | Bất động sản | 4.6 năm | 80.8 kWh | Đỉnh lúc 19h, cuối tuần > ngày thường |
| **U169** | Sản xuất | 4.4 năm | 48.7 kWh | Mẫu công nghiệp ban ngày |
| **U380** | Giáo dục | 4.4 năm | 4.9 kWh | Phân tách weekday/weekend mạnh nhất |

---

## Tiêu Chuẩn Đánh Giá

Theo **ASHRAE Guideline 14-2014**:

| Chỉ số | Ngưỡng | Ý nghĩa |
|--------|--------|---------|
| CV(RMSE) | ≤ 25% (dữ liệu ngày) | Độ chính xác dự đoán |
| \|NMBE\| | ≤ 10% | Độ thiên lệch hệ thống |

Phương pháp: **Expanding-window cross-validation** — train trên dữ liệu tích lũy, test trên năm tiếp theo.
