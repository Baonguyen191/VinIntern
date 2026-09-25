# Quy Chuẩn Schema Dữ Liệu Demo Khung Phân Hệ (Demo Schema Spec)

Tài liệu này định nghĩa **Schema dữ liệu thống nhất (Unified Demo Schema)** được thống nhất giữa cả 3 thành viên phụ trách các phần:
1. **Thành viên 1**: Baseline kỳ vọng (`baseline_expected`)
2. **Thành viên 2**: Bắt sự kiện bất thường (`is_anomaly`, `anomaly_reason`)
3. **Thành viên 3 (Tôi)**: Dự báo 24h & Khoảng tin cậy (`forecast_next_24h`, `forecast_lower`, `forecast_upper`)

---

## 1. Cấu Trúc Các Trường (Field Definitions)

Mỗi điểm bản ghi (record) trên hệ thống Dashboard/Demo bao gồm 10 trường dữ liệu chính:

| Tên trường | Kiểu dữ liệu | Người phụ trách | Mô tả | Ví dụ |
|---|---|---|---|---|
| `timestamp` | `datetime64[ns]` / ISO8601 | Dùng chung | Mốc thời gian theo chu kỳ giờ (YYYY-MM-DD HH:MM:SS) | `2017-10-15 14:00:00` |
| `building_id` | `string` | Dùng chung | Mã định danh tòa nhà / entity | `Panther_office_space_Bear` |
| `metric` | `string` | Dùng chung | Tên chỉ số đo (`power_active_kw` hoặc `cooling_kw`) | `power_active_kw` |
| `value_actual` | `float` | Dùng chung | Giá trị thực tế đo được từ đồng hồ / cảm biến | `412.50` |
| `baseline_expected` | `float` | **Thành viên 1 (Baseline)** | Giá trị baseline tiêu chuẩn kỳ vọng tại thời điểm đó | `398.20` *(Mock ở V1: Rolling 7-day mean)* |
| `is_anomaly` | `boolean` | **Thành viên 2 (Anomaly)** | Cờ đánh dấu có bất thường hay không (`True`/`False`) | `False` *(Mock ở V1)* |
| `anomaly_reason` | `string` / `null` | **Thành viên 2 (Anomaly)** | Lý do / Loại bất thường nếu `is_anomaly = True` | `"Spike tải đột biến (+25%)"` *(Mock ở V1)* |
| `forecast_next_24h` | `float` | **Thành viên 3 (Tôi - Dự báo)** | Giá trị dự báo phụ tải cho điểm giờ đích (P50/Mean) | `405.10` *(THẬT từ Hour-of-Week Model)* |
| `forecast_lower` | `float` | **Thành viên 3 (Tôi - Dự báo)** | Giới hạn dưới khoảng tin cậy 80% (P10) | `365.00` *(THẬT từ Model residual std)* |
| `forecast_upper` | `float` | **Thành viên 3 (Tôi - Dự báo)** | Giới hạn trên khoảng tin cậy 80% (P90) | `445.20` *(THẬT từ Model residual std)* |

---

## 2. Quy Định Phân Định Dữ Liệu THẬT vs MOCK Tại Sprint 1

Do 2 thành viên phụ trách Baseline và Anomaly đang hoàn thiện mô hình ở Sprint 1, dashboard (Sprint 1: `dashboard_v1.py`, nay là `src/dashboard.py` qua `src/teammate_adapter.py`) sẽ áp dụng cơ chế:

- **Dữ liệu THẬT (Real Data)**:
  - `timestamp`, `building_id`, `metric`, `value_actual`: Đọc trực tiếp từ `telemetry_M1.parquet` và `telemetry_M2.parquet`.
  - `forecast_next_24h`, `forecast_lower`, `forecast_upper`: Tính toán trực tiếp từ mô hình baseline dự báo (Hour-of-Week Profile / Seasonal Naive) huấn luyện trên 80% tập dữ liệu lịch sử.
- **Dữ liệu MOCK (Placeholder Data - Sẽ được thay thế ở Sprint 2)**:
  - `baseline_expected`: Giả lập bằng giá trị trung bình trượt 7 ngày (Rolling 7-day mean).
  - `is_anomaly`: Giả lập cờ ngẫu nhiên/quy tắc lệch > 2.5 std so với rolling mean.
  - `anomaly_reason`: Giả lập văn bản lý do bất thường.

---

## 3. Mẫu Dữ Liệu (JSON Example)

```json
{
  "timestamp": "2017-10-15T14:00:00+07:00",
  "building_id": "Panther_office_space_Bear",
  "metric": "power_active_kw",
  "value_actual": 412.5,
  "baseline_expected": 398.2,
  "is_anomaly": false,
  "anomaly_reason": null,
  "forecast_next_24h": 405.1,
  "forecast_lower": 365.0,
  "forecast_upper": 445.2
}
```
