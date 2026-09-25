# Hướng Dẫn & Checklist Thay Thế Dữ Liệu Mock Bằng Dữ Liệu Thật (Swap Checklist)

Tài liệu này hướng dẫn chi tiết quy trình tiếp nhận kết quả từ **Bạn 1 (Baseline)** và **Bạn 2 (Bắt sự kiện bất thường - Anomaly Detection)** khi họ hoàn thành, để tích hợp vào hệ thống mà không làm gián đoạn hay phải sửa đổi logic cốt lõi của Dashboard và Bộ giải tối ưu.

---

## 1. Danh Sách 2 File Cần Tiếp Nhận

Cả hai file phải được đặt tại **thư mục gốc của repository** (`d:\VIN_AITC\VinIntern\`) và tuân thủ đúng quy chuẩn [demo_schema.md](file:///d:/VIN_AITC/VinIntern/docs/demo_schema.md):

### File 1: `baseline_output.csv` (Từ Bạn 1 — Phụ trách Baseline kỳ vọng)
* **Vị trí file**: `d:\VIN_AITC\VinIntern\baseline_output.csv`
* **Cấu trúc các cột bắt buộc**:
  | Tên cột | Kiểu dữ liệu | Mô tả | Ví dụ |
  |---|---|---|---|
  | `timestamp` | datetime / ISO8601 | Mốc thời gian từng giờ | `2017-10-20 00:00:00` |
  | `building_id` | string | Mã định danh tòa nhà | `Bull_education_Luke` |
  | `metric` | string | Chỉ số đo | `power_active_kw` hoặc `cooling_kw` |
  | `baseline_expected` | float | Giá trị công suất baseline kỳ vọng (kW) | `812.45` |

### File 2: `anomaly_output.csv` (Từ Bạn 2 — Phụ trách Phát hiện Bất thường)
* **Vị trí file**: `d:\VIN_AITC\VinIntern\anomaly_output.csv`
* **Cấu trúc các cột bắt buộc**:
  | Tên cột | Kiểu dữ liệu | Mô tả | Ví dụ |
  |---|---|---|---|
  | `timestamp` | datetime / ISO8601 | Mốc thời gian từng giờ | `2017-10-20 00:00:00` |
  | `building_id` | string | Mã định danh tòa nhà | `Bull_education_Luke` |
  | `metric` | string | Chỉ số đo | `power_active_kw` hoặc `cooling_kw` |
  | `is_anomaly` | boolean | Cờ bất thường (`True` / `False`) | `True` |
  | `anomaly_reason` | string / null | Diễn giải lý do bất thường | `Tải tăng đột biến +35% ngoài giờ` |

---

## 2. Vị Trí & Dòng Code Chính Xác Cần Đổi Khi Thay File

Mở file adapter [teammate_outputs_adapter.py](file:///d:/VIN_AITC/VinIntern/teammate_outputs_adapter.py):

* **Tại Dòng 10**:
  ```python
  # TRƯỚC KHI THAY (Đang dùng dữ liệu mock kỷ luật):
  USE_MOCK = True
  
  # SAU KHI THAY (Chuyển sang dùng dữ liệu thật của 2 bạn):
  USE_MOCK = False
  ```

* **Tại Dòng 13 & 14** (Kiểm tra lại đường dẫn nếu 2 bạn đặt tên file khác):
  ```python
  REAL_BASELINE_PATH = os.path.join(BASE_DIR, 'baseline_output.csv')
  REAL_ANOMALY_PATH = os.path.join(BASE_DIR, 'anomaly_output.csv')
  ```

> [!NOTE]
> Khi chuyển `USE_MOCK = False`, toàn bộ hàm `get_baseline_expected()` và `get_anomaly_flags()` sẽ tự động chuyển sang đọc file thật của 2 bạn. Bạn **KHÔNG CẦN CHỈNH SỬA BẤT KỲ DÒNG CODE NÀO** trong [forecast_final.py](file:///d:/VIN_AITC/VinIntern/forecast_final.py), [opt_solver.py](file:///d:/VIN_AITC/VinIntern/opt_solver.py) hay [dashboard_final.py](file:///d:/VIN_AITC/VinIntern/dashboard_final.py).

---

## 3. Quy Trình Kiểm Thử Hồi Quy (Regression Testing) Với `test_fixtures/`

Thư mục [test_fixtures/](file:///d:/VIN_AITC/VinIntern/test_fixtures/) đã lưu trữ đầy đủ dữ liệu kiểm thử chuẩn của cả 3 sprint:
* `test_fixtures/mock_baseline.csv`
* `test_fixtures/mock_anomaly.csv`
* `test_fixtures/sample_test_telemetry.parquet`

Sau khi nhận file thật từ 2 bạn và đổi `USE_MOCK = False`, chạy lệnh kiểm tra tính toàn vẹn:

```powershell
# 1. Chạy test adapter độc lập
python teammate_outputs_adapter.py

# 2. Khởi chạy Dashboard để kiểm tra giao diện demo
python -m streamlit run dashboard_final.py
```

Nếu xuất hiện lỗi thiếu cột hoặc sai định dạng thời gian, chuyển lại `USE_MOCK = True` để hệ thống demo tiếp tục hoạt động bình thường trong lúc chờ 2 bạn hiệu chỉnh định dạng file.
