# VinIntern

Pipeline Năng lượng & M&E: *Baseline → Bất thường → Dự báo → Tối ưu MILP*.

## Cấu trúc

```
src/            code dùng chung + dashboard (paths.py giữ mọi đường dẫn)
scripts/        script chạy pipeline từ đầu đến cuối
data/normalized dữ liệu đầu vào đã chuẩn hóa (parquet không commit)
data/teammates  file kết quả thật từ Bạn 1 & Bạn 2 (xem docs/swap_checklist.md)
outputs/        mọi file sinh ra bởi pipeline (CSV report, parquet dự báo)
tests/fixtures  dữ liệu mock / mẫu kiểm thử
docs/           đặc tả & hướng dẫn
```

## Chạy

```powershell
pip install -r requirements.txt
python scripts/train_ridge.py        # -> outputs/regression_vs_baseline.csv, forecast_test_results.parquet
python src/forecast.py               # -> outputs/forecast_final_report.csv
python src/optimizer.py              # -> outputs/recommendation_ranking.csv
python -m streamlit run src/dashboard.py
```
