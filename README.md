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
│   └── visualization/           # Vẽ biểu đồ
│
├── scripts/                     # Script EDA & đánh giá bổ sung
│
└── results/
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
