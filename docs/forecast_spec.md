# Quy Cách Dự Báo Năng Lượng & Phụ Tải Cấp Lạnh (Forecast Specification)

Tài liệu này xác định các quy chuẩn, chỉ số, khoảng thời gian dự báo và nguyên tắc chống rò rỉ dữ liệu (Data Leakage Prevention) phục vụ bài toán **Dự báo Phụ tải Điện & Cấp lạnh** thuộc dự án *Năng lượng & M&E Thông minh*.

---

## 1. Chỉ Số Dự Báo (Target Metrics)

Dự báo được thực hiện độc lập cho 2 phân hệ telemetry chính:

| Phân hệ | File Dữ Liệu | Chỉ Số Dự Báo | Đơn Vị | Mô Tả |
|---|---|---|---|---|
| **M1 — Điện năng** | `data_normalized/telemetry_M1.parquet` | `power_active_kw` | kW | Công suất điện chủ động của toàn tòa nhà / phụ tải tổng |
| **M2 — Chiller** | `data_normalized/telemetry_M2.parquet` | `cooling_kw` | kW | Phụ tải lạnh cần cung cấp cho tòa nhà |

---

## 2. Khoảng Dự Báo & Chu Kỳ Cập Nhật (Horizon & Frequency)

1. **Thời điểm phát hành (Issue Time)**:
   - Dự báo được phát hành định kỳ vào lúc **00:00 hằng ngày** cho **24 giờ tiếp theo** ($t + 1$ đến $t + 24$).
   - Cho phép **cập nhật rolling mỗi giờ** (Hourly rolling update) khi nhận thêm điểm đo thực tế mới nhất.
2. **Độ phân giải thời gian (Step / Resolution)**:
   - Bước thời gian: **1 giờ** (1-hour resolution).
   - *Ghi chú về giới hạn dữ liệu*: Dữ liệu gốc trong `telemetry_M1.parquet` và `telemetry_M2.parquet` (nguồn Kaggle BDG2) được ghi nhận theo tần suất 1 giờ. Bài toán thực tế DMP hỗ trợ luồng 15 phút, tuy nhiên ở cấp độ dữ liệu hiện tại, mô hình giữ nguyên độ phân giải 1 giờ để đảm bảo tính nguyên bản, không dùng nội suy nhân tạo cho tập dự báo baseline.

---

## 3. Quy Tắc Chống Rò Rỉ Dữ Liệu & Tập Đặc Trưng Hợp Lệ

### 3.1 Quy tắc thời điểm thông tin ($t_0 = 00:00$)
Tại thời điểm phát hành dự báo $t_0$, mô hình **CHỈ ĐƯỢC PHÉP** sử dụng các điểm dữ liệu đã xảy ra tính đến $t \le t_0$. Tuyệt đối không sử dụng bất kỳ thông tin thực tế nào của khoảng thời gian $[t_0 + 1\text{h}, t_0 + 24\text{h}]$.

### 3.2 Tập đặc trưng hợp lệ (Valid Feature Set)

| Nhóm đặc trưng | Tên đặc trưng | Công thức / Mô tả | Tính hợp lệ tại $t_0$ |
|---|---|---|---|
| **Lags (Độ trễ)** | `lag_24h` | $y(t - 24\text{h})$ | Hợp lệ (đã đo 24h trước) |
| | `lag_48h` | $y(t - 48\text{h})$ | Hợp lệ (đã đo 48h trước) |
| | `lag_168h` | $y(t - 168\text{h})$ | Hợp lệ (đã đo 1 tuần trước) |
| **Profile lịch sử** | `hour_of_week_mean` | Trung bình lịch sử của đúng (thứ, giờ) đó trên tập học | Hợp lệ (tính từ tập Train) |
| **Đặc trưng lịch** | `hour_of_day` | Slot giờ trong ngày ($0 \dots 23$) | Hợp lệ (biết trước) |
| | `day_of_week` | Ngày trong tuần ($0 = \text{Thứ 2} \dots 6 = \text{Chủ nhật}$) | Hợp lệ (biết trước) |
| | `day_type` | Loại ngày (`ngay_lam_viec`, `cuoi_tuan`, `le`) | Hợp lệ (biết trước) |
| **Thời tiết** | `outdoor_temp_target` | Nhiệt độ ngoài trời tại giờ đích $t$ | **Giả định weather forecast hoàn hảo** |

> **[!IMPORTANT] Giới Hạn Cần Nêu Trong Báo Cáo:**
> Việc sử dụng cột `temperature` tại giờ đích $t$ dựa trên giả định hệ thống có dịch vụ **Dự báo thời tiết hoàn hảo (Perfect Weather Forecast)**. Trong triển khai thực tế, nhiệt độ này phải lấy từ API dự báo thời tiết (ví dụ OpenWeatherMap/AccuWeather), do đó sai số dự báo thời tiết có thể ảnh hưởng một phần tới độ chính xác của dự báo phụ tải.
