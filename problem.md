# Bài toán A — Năng lượng & M&E trên DMP

**Baseline → Phát hiện bất thường → Dự báo → Gợi ý tối ưu**

Phạm vi: 3 domain **Điện · Chiller · Nước**. Xây dựng baseline cho cả 3; **tập trung dự báo và tối ưu cho Chiller**.
Nguồn key: *DMP telemetry keys v1.13 (132 telemetry core + 13 NORIS + 12 PHATTAI + 16 attribute)*.
Dữ liệu cho MVP: **Building Data Genome 2** (mục 8) trong khi chờ luồng DMP thật.

---

## 1. Phát biểu bài toán

Hệ M&E của tòa nhà đang vận hành theo lịch cố định và logic ngưỡng đơn giản. Hệ thống **không biết mức vận hành "bình thường" của từng thiết bị là bao nhiêu**, nên:

1. Lãng phí âm thầm (thiết bị chạy ngoài giờ, chạy non tải, hiệu suất suy giảm dần) không ai nhìn thấy.
2. Sự cố (rò rỉ nước, chiller mất hiệu suất, bơm suy giảm) chỉ lộ ra khi đã gây hậu quả.
3. Không tính trước được phụ tải nên không khai thác được chênh lệch giá điện giữa các khung giờ.

**Bài toán:** xây dựng một lớp phân tích dùng chung trên DMP, nhận telemetry sẵn có, và với mỗi thiết bị / khu vực:

- học **baseline** — mức vận hành kỳ vọng theo ô giờ × loại ngày, có điều kiện theo thời tiết và lượng người;
- **phát hiện bất thường** theo ba nhóm: lượng tiêu thụ, hiệu suất, mẫu hình thời gian (mục 3.2);
- **dự báo** tải lạnh và phụ tải 1–24h tới;
- đưa ra **gợi ý vận hành** kèm ước tính kWh và tiền điện tiết kiệm được.

Người vận hành là người ra quyết định cuối cùng. Hệ thống **không ghi lệnh xuống thiết bị**.

---

## 2. Phạm vi

### 2.1 Trong phạm vi

| Hạng mục | Mô tả |
|---|---|
| Framework dùng chung | Pipeline 5 bước (mục 3), chạy được trên lưới 15 phút lẫn lưới giờ, cấu hình theo module |
| M1 — Điện năng | Anchor dữ liệu. Baseline tiêu thụ, phát hiện lãng phí, dự báo phụ tải tổng |
| M2 — Chiller | Vertical chủ đạo. Dự báo tải lạnh, mô hình hiệu suất từng máy, gợi ý điểm vận hành |
| M3 — Nước / bơm | Baseline tiêu thụ nước, phát hiện rò rỉ và suy giảm bơm |
| Đầu ra | Sự kiện bất thường, chuỗi dự báo, gợi ý vận hành, báo cáo tiết kiệm (mục 5) |

### 2.2 Ngoài phạm vi

- **Không closed-loop**: không ghi các key `*_cmd`, `*_sp` xuống thiết bị. Các key này chỉ được **đọc** để so lệnh với phản hồi.
- **M4 — Không khí / CO2 / AQI**: để lại cho giai đoạn sau. Khung phân tích giữ nguyên nên mở rộng được mà không phải làm lại.
- Không thay thế logic an toàn cục bộ của PLC/BMS (phao bồn, interlock bơm, bảo vệ chiller).
- Không lắp thêm cảm biến trong phạm vi thực tập; chỉ ghi nhận thành đề xuất nếu thiếu dữ liệu.

---

## 3. Kiến trúc pipeline dùng chung

```
Telemetry DMP ──┐
Attribute DMP ──┼─► [1] Chuẩn hóa ─► [2] Baseline ─► [3] Bất thường ─► [4] Dự báo ─► [5] Gợi ý
Dữ liệu ngoài ──┘         │               │                 │                │             │
                          ▼               ▼                 ▼                ▼             ▼
                  Chuỗi sạch trên    Mức kỳ vọng +     Sự kiện có         Chuỗi dự báo  Khuyến nghị +
                  lưới đều + cờ      dải tin cậy       bằng chứng         + khoảng tin  ước tính tiết
                  chất lượng                                              cậy           kiệm
```

### 3.1 Năm bước

| Bước | Việc làm | Ghi chú |
|---|---|---|
| 1. Chuẩn hóa | Resample về lưới đều (15 phút với DMP, 1 giờ với BDG2); chuyển counter `*_total` thành lượng theo kỳ, bỏ Δ âm khi reset; đồng nhất đơn vị và tên key giữa DMP chuẩn và NORIS; gắn cờ chất lượng | Phân biệt "mất dữ liệu" với "giá trị bằng 0" — đây là nguồn báo động giả lớn nhất |
| 2. Baseline — 3 lớp | **(a) Luật vật lý**: ràng buộc không cần lịch sử. **(b) Lịch sử chính thiết bị**: profile theo ô giờ × loại ngày, điều kiện theo nhiệt độ ngoài trời và lượng người. **(c) So ngang tập thiết bị**: so với thiết bị cùng model | Lớp (b) là lớp chính; lớp (c) cần ≥ 2 thiết bị cùng model |
| 3. Bất thường | Ba nhóm ở mục 3.2 | Mỗi sự kiện kèm key, khoảng thời gian, baseline, độ lệch |
| 4. Dự báo | Tải lạnh (chính), phụ tải điện, nhu cầu nước | Horizon 24h, bước theo lưới dữ liệu |
| 5. Gợi ý | Tối ưu có ràng buộc: an toàn → tiện nghi → chi phí điện | Chỉ đề xuất, người vận hành xác nhận |

### 3.2 Ba nhóm bất thường

| Nhóm | Câu hỏi | Biến đích | Cửa sổ tham chiếu |
|---|---|---|---|
| **A. Lượng tiêu thụ** | Mức này có cao hơn mức đáng lẽ phải dùng không? | `kwh_15m`, `m3_15m`, điện năng chiller | **Trượt** — refit định kỳ trên 8 tuần liền trước |
| **B. Hiệu suất** | Cùng lượng đầu ra, có tốn hơn trước không? | `kw_per_kw_cooling`, `pump_head_proxy`, `chw_delta_t`, `kwh_per_m3` | **Cố định** — 4–8 tuần kể từ lần bảo trì gần nhất |
| **C. Mẫu hình thời gian** | Ngày hôm nay có hình dạng giống những ngày cùng loại không? | Vector profile ngày; giờ bật/tắt, thời lượng chạy, giờ đạt đỉnh, số lần đóng cắt | Tập ngày sạch cùng `day_type` trong 8–12 tuần |

Khác biệt cửa sổ tham chiếu giữa nhóm A và B là **bắt buộc**: nếu nhóm B cũng dùng cửa sổ trượt thì mô hình sẽ hấp thụ chính sự suy giảm cần phát hiện và không bao giờ báo. Hệ quả: **nhật ký bảo trì trở thành dữ liệu bắt buộc**, và nhóm B không được dùng đặc trưng tự hồi quy (lag) của chính biến đích.

Kèm theo cả ba nhóm là **lớp kiểm tra chất lượng dữ liệu** (mất mẫu, counter đơ, counter nhảy lùi, giá trị ngoài dải vật lý, lệch đồng hồ thời gian, bậc thang sau khi đổi firmware). Lớp này không tốn thêm key nào và phải chạy cùng lúc với nhóm A — mọi báo động đều vô nghĩa nếu chuỗi số đầu vào sai. Đầu ra của lớp này gửi cho đội vận hành hệ thống, không phải quản lý tòa nhà.

### 3.3 Phương pháp theo loại lỗi

Kết quả benchmark nội bộ (19 phương pháp × 12 loại lỗi trên dữ liệu mô phỏng theo key v1.13; hiệu chỉnh mọi ngưỡng về cùng ngân sách 1 báo động/thiết bị/ngày). Không mô hình nào thắng mọi loại lỗi:

| Loại lỗi | Nhóm | Phương pháp tốt nhất | Point recall | Trễ |
|---|---|---|---|---|
| Spike | A | Profile band giờ × loại ngày | 1.000 | 0 phút |
| Chạy ngoài giờ | A | Profile band giờ × loại ngày | 1.000 | 0 phút |
| Đơ số / flatline | A | GBM quantile band (không lag) | 0.836 | 45 phút |
| Rò rỉ nước | A | CUSUM trên phần dư lưu lượng | 0.995 | 15 phút |
| Tải nền tăng dần | A | CUSUM trên phần dư | 0.906 | ~35 giờ |
| Suy giảm COP chiller | B | Baseline hồi quy không lag trên `kw_per_kw_cooling` | 0.996 | 0 phút |
| Suy giảm cột áp bơm | B | CUSUM trên phần dư `pump_head_proxy` | 0.970 | 0 phút |
| ΔT nước lạnh thấp | B | Ngưỡng tĩnh cũng đủ (dải vật lý hẹp) | 1.000 | 0 phút |
| Hình dạng ngày lạ | C | PCA tái tạo profile ngày | *(xem dưới)* | theo ngày |

Kiểm định trên 200 công tơ thật có nhãn chuyên gia cho thấy thứ hạng đổi một phần: baseline hồi quy **có lag** đứng đầu (event recall 0.635) vì 53% sự cố thật chỉ dài 1–2 giờ, PCA tái tạo profile ngày xếp hạng 2 (0.558), profile band 0.481. Kết luận triển khai: **chạy song song hai lớp** — baseline hồi quy (cả biến thể có và không lag) cho lỗi ngắn, control chart trên phần dư cho lỗi trôi chậm — và **không dùng** matrix profile / kNN discord (0.266, trễ trung vị 3.8 giờ).

---

## 4. Input

### 4.1 Telemetry — tổng quan

| Module | Đích (lập baseline) | Giải thích / điểm vận hành | Trạng thái & đọc lệnh | Chất lượng |
|---|---|---|---|---|
| Điện | `energy_active_kwh_total`, `power_active_kw` | *(biến ngoại sinh, mục 4.3)* | `power_factor`, `voltage_l1/l2/l3_v`, `current_l1/l2/l3_a` | `reachable`, `fault`, `mode` |
| Chiller | `cooling_kw`, `active_power_kw` (từng máy), `active_energy_kwh` | `chiller_load_pct`, `chilled_water_supply/return_temp_c`, `runtime_min` | `chiller_state`, `chiller_state_cmd`, `compressor_status`, `op_mode` | `reachable`, `fault`, `mode` |
| Nước | `water_volume_m3_total`, `water_flow_lpm`, `power_active_kw` (bơm) | `pump_speed_pct`, `pressure_supply_bar`, `pressure_return_bar` | `pump_state`, `pump_state_cmd` | `reachable`, `fault`, `mode` |

Chi tiết từng key ở mục 6; ma trận đầy đủ ở mục 7.

### 4.2 Attribute DMP (metadata)

| Key | Dùng để |
|---|---|
| `device_type` | Định tuyến thiết bị vào module |
| `device_model` | Gom nhóm cho baseline lớp (c) — so ngang cùng model |
| `device_label` | Hiển thị trong cảnh báo / gợi ý |
| `vendor_device_id`, `serial_number` | Đối soát thiết bị giữa DMP và NORIS |
| `firmware_version` | Loại trừ bất thường giả do đổi firmware |

### 4.3 Biến ngoại sinh và dữ liệu ngoài telemetry

| Dữ liệu | Dạng | Bắt buộc | Nguồn |
|---|---|---|---|
| `temperature`, `humidity` ngoài trời | Time-series cùng lưới | ✔ | PHATTAI hoặc API thời tiết |
| Δ`record_count` — lượt người ra vào | Time-series cùng lưới | ✔ | ZKTeco. **Không dùng `user_count`** (gap G4) |
| Lịch tòa nhà: `day_type` 4 mức (workday / saturday / off / **holiday**), giờ vận hành → `is_op`, ô thời gian `tod_slot` | Calendar | ✔ | Ban quản lý tòa nhà |
| **Nhật ký bảo trì** (thay lọc, vệ sinh dàn, sửa bơm, nạp gas) | Log | ✔ | Phiếu bảo trì — mốc reset cửa sổ tham chiếu của nhóm B **và** nhãn đánh giá |
| Biểu giá điện theo khung giờ | Bảng cấu hình | ✔ | Hợp đồng điện của tòa nhà |
| Topology: thiết bị → đồng hồ điện → khu vực / tầng | Bảng quan hệ | ✔ | SMCP / hồ sơ M&E |
| Thông số định mức: công suất lạnh chiller, công suất và đường cong bơm, dung tích bồn | Bảng tĩnh | ✔ (M2, M3) | Hồ sơ thiết bị |
| Diện tích sàn và số người thiết kế theo khu vực | Bảng tĩnh | ✔ (M1) | Hồ sơ tòa nhà — để chuẩn hoá kWh/m², kWh/người |
| Dự báo thời tiết 24h | API | Tùy chọn | Thiếu thì dùng lịch sử PHATTAI |

---

## 5. Output

### 5.1 O1 — Baseline

```json
{
  "entity_id": "PM-B1-MAIN",
  "key": "power_active_kw",
  "ts": "2026-09-16T14:00:00+07:00",
  "expected": 412.5,
  "lower": 380.1,
  "upper": 446.0,
  "baseline_layer": "history",
  "window": {"type": "rolling", "fit_weeks": 8, "refit_on": "2026-09-01"},
  "context": {"day_type": "workday", "tod_slot": 56, "outdoor_temp_c": 33.1}
}
```

### 5.2 O2 — Sự kiện bất thường

```json
{
  "event_id": "A-20260916-0042",
  "group": "B_efficiency",
  "module": "M2_chiller",
  "entity_id": "CH-02",
  "type": "cop_degradation",
  "severity": "info",
  "rule_or_model": "baseline_regression+cusum",
  "start": "2026-09-09T08:00:00+07:00",
  "end": null,
  "evidence": {
    "kw_per_kw_cooling": 0.213,
    "reference": 0.187,
    "deviation_pct": 13.9,
    "reference_window": "2026-06-20..2026-08-15 (sau bảo trì 2026-06-18)",
    "load_pct_range": [45, 75]
  },
  "estimated_waste_kwh_per_month": 2140,
  "suggested_action": "Kiểm tra dàn ngưng và lượng gas CH-02"
}
```

Mức `severity`: `critical` (an toàn, báo ngay, không cần baseline) · `warning` (lệch baseline có ý nghĩa) · `info` (xu hướng suy giảm chậm).
Quy tắc gán cảnh báo gồm ba tham số bắt buộc: **ngân sách báo động/thiết bị/ngày**, **thời lượng tối thiểu**, và **cách gộp các ô liền kề thành một sự kiện**.

### 5.3 O3 — Dự báo

```json
{
  "entity_id": "CHILLER-PLANT",
  "key": "cooling_kw",
  "issued_at": "2026-09-16T00:00:00+07:00",
  "horizon": "24h",
  "step": "15min",
  "points": [{"ts": "...", "p50": 1850, "p10": 1700, "p90": 2010}]
}
```

### 5.4 O4 — Gợi ý vận hành

```json
{
  "rec_id": "R-20260916-007",
  "module": "M2_chiller",
  "valid_from": "2026-09-16T09:00:00+07:00",
  "valid_to": "2026-09-16T11:30:00+07:00",
  "action": "Chạy CH-01 + CH-03 thay vì CH-01 + CH-02",
  "reason": "CH-02 có kW/kW lạnh cao hơn 14% so với cửa sổ tham chiếu ở cùng mức tải",
  "constraints_checked": ["chw_supply_temp <= 7.5C", "min_runtime", "comfort"],
  "est_saving_kwh": 96,
  "est_saving_vnd": 250000,
  "status": "pending_operator"
}
```

### 5.5 O5 — Báo cáo định kỳ (ngày / tuần / tháng)

- Tiêu thụ thực tế vs baseline theo hệ, theo khu vực.
- Top thiết bị lãng phí (nhóm A) và top thiết bị suy giảm hiệu suất (nhóm B).
- Danh sách ngày có mẫu hình bất thường (nhóm C), kèm hình so sánh profile ngày đó với dải bình thường cùng loại ngày.
- Số gợi ý đã đưa ra, số được chấp nhận, tiết kiệm ước tính.
- Chất lượng dữ liệu: tỷ lệ mẫu hợp lệ theo thiết bị.

---

## 6. Chi tiết theo module

Quy ước cột **Vai trò**: `Đích` = biến cần baseline / dự báo · `Giải thích` = biến ngữ cảnh hoặc điểm vận hành đưa vào mô hình · `Trạng thái` = dùng cho luật và để lọc dữ liệu · `Đọc lệnh` = key `_cmd`/`_sp`, chỉ đọc · `Chất lượng` = cờ dữ liệu.

### 6.1 M1 — Điện năng (anchor dữ liệu)

**Câu hỏi cần trả lời**
- Tòa nhà / từng tủ / từng khu đang tiêu thụ nhiều hơn mức kỳ vọng bao nhiêu, vào lúc nào?
- Phụ tải 24h tới là bao nhiêu, bao nhiêu rơi vào giờ cao điểm?
- Có dấu hiệu bất thường về chất lượng điện (lệch pha, PF thấp) không?

| Key | Đơn vị | Vai trò | Dùng cho |
|---|---|---|---|
| `energy_active_kwh_total` | kWh | Đích | Counter → `kwh_15m`; baseline tiêu thụ, báo cáo tiền điện |
| `power_active_kw` | kW | Đích | Baseline & dự báo phụ tải, phát hiện spike / chạy ngoài giờ |
| `power_factor` | – | Trạng thái | **Luật riêng**: PF thấp kéo dài → nguy cơ phạt công suất phản kháng. Không phải biến giải thích cho baseline kWh |
| `voltage_l1_v`, `voltage_l2_v`, `voltage_l3_v` | V | Trạng thái | Luật lệch áp giữa các pha |
| `current_l1_a`, `current_l2_a`, `current_l3_a` | A | Trạng thái | Luật mất cân bằng dòng pha |
| `voltage_v`, `current_a` | V, A | Trạng thái | Dự phòng khi đồng hồ không đẩy theo pha; kiểm chéo `kW ≈ √3·U·I·PF` |
| `frequency_hz` | Hz | Trạng thái | Tần số ngoài dải = đang chạy máy phát → **loại** khoảng đó khỏi baseline |
| `energy_reactive_kvarh_total` | kvarh | Đích | Theo dõi công suất phản kháng, ước lượng tiền phạt |
| `lighting_knx_state`, `state_on_off` | – | Giải thích | Giải thích bước nhảy tải |
| `active_energy_kwh`, `active_power_kw` (NORIS) | kWh, kW | Đích | Nguồn thay thế khi đồng hồ chưa đẩy real-time vào DMP |

**Đại lượng dẫn xuất**
- `kwh_15m = Δ energy_active_kwh_total` (bỏ Δ âm do reset counter).
- `phase_voltage_unbalance_pct = max|V_Lx − V_avg| / V_avg × 100`; tương tự cho dòng.
- `peak_share_pct` = kWh giờ cao điểm / tổng kWh ngày · `after_hours_kwh` = kWh ngoài giờ vận hành.
- Chỉ số chuẩn hoá: kWh/m²/ngày, kWh/người/ngày, tỷ lệ tải nền = kW đáy đêm / kW đỉnh ngày.

**Output cụ thể**
- Baseline `power_active_kw` và `kwh_15m` theo đồng hồ / khu vực.
- Bất thường nhóm A: chạy ngoài giờ, tải nền tăng dần, spike, đơ số. Nhóm C: lệch lịch bật/tắt, sai loại ngày, dịch đỉnh sang giờ cao điểm. Luật riêng: PF thấp, mất cân bằng pha.
- Dự báo phụ tải 24h kèm phân rã theo khung giá.

### 6.2 M2 — Chiller (vertical chủ đạo — trọng tâm dự báo và tối ưu)

**Câu hỏi cần trả lời**
- Tải lạnh 24h tới là bao nhiêu?
- Mỗi chiller đang chạy với hiệu suất bao nhiêu so với chính nó sau lần bảo trì gần nhất, và so với máy cùng loại?
- Với tải dự báo, nên chạy máy nào, bao nhiêu máy, khởi động lúc mấy giờ, nhiệt độ nước lạnh cấp bao nhiêu?

| Key (tên NORIS) | Đơn vị | Vai trò | Dùng cho |
|---|---|---|---|
| `cooling_kw` (`cooling_load_kw`) | kW | Đích | Tải lạnh — biến dự báo chính, tử số của hiệu suất |
| `active_power_kw` | kW | Đích | Điện tiêu thụ chiller — mẫu số của hiệu suất. **Bắt buộc có đồng thời `cooling_kw`, và phải theo từng máy** |
| `active_energy_kwh` | kWh | Đích | Quy sự kiện suy giảm ra kWh và tiền |
| `chiller_load_pct` | % | Giải thích | Trục PLR — phải so hiệu suất ở cùng mức tải |
| `chilled_water_supply_temp_c` (`chw_supply_temp`) | °C | Giải thích | Ràng buộc tiện nghi; vế đầu của `chw_delta_t` |
| `chilled_water_return_temp_c` (`chw_return_temp`) | °C | Giải thích | ΔT thấp kéo dài = low ΔT syndrome |
| `chiller_state` | – | Trạng thái | Lọc khoảng máy dừng — tính hiệu suất lúc máy dừng cho số vô nghĩa |
| `chiller_state_cmd` | – | Đọc lệnh | Luật lệnh ≠ phản hồi |
| `compressor_status`, `op_mode` | – | Trạng thái | Phân biệt đang nén thật; loại khoảng chạy test / bảo trì |
| `runtime_min` / `run_hours` | min / h | Giải thích | Min runtime, cân bằng giờ chạy giữa các máy |
| `evaporator_pressure` | bar | Giải thích | Thiếu gas, dàn bám bẩn — bằng chứng bổ sung |
| `chilled_water_supply/return_pressure_bar` | bar | Giải thích | Suy giảm lưu lượng, nghẽn lọc Y |
| `cooling_tower_fan_state` | – | Trạng thái | Tính `plant_kw` (chiller + bơm + tháp) |
| `temperature`, `humidity`, Δ`record_count` | °C, %, – | Giải thích | Biến chính cho dự báo tải lạnh |

**Đại lượng dẫn xuất**
- `chw_delta_t = return − supply` (chỉ tính khi tải plant trên ngưỡng).
- `kw_per_kw_cooling = active_power_kw / cooling_kw`; `COP = 1 / kw_per_kw_cooling`. Chỉ tính khi máy chạy, tải > ~5% định mức, đã ổn định ≥ 15 phút.
- `plant_kw` = chiller + bơm + quạt tháp (khi đủ dữ liệu).
- Kiểm chéo: `Q ≈ 4.186 × (water_flow_lpm / 60) × chw_delta_t` nếu có lưu lượng trên vòng CHW.

**Output cụ thể**
- Dự báo `cooling_kw` 24h, P10/P50/P90.
- Đường cong hiệu suất `kw_per_kw_cooling` theo `chiller_load_pct` và nhiệt độ ngoài trời, cho **từng máy**, gắn với cửa sổ tham chiếu sau lần bảo trì gần nhất.
- Bất thường nhóm B: suy giảm hiệu suất so với cửa sổ tham chiếu, ΔT thấp kéo dài, chạy non tải kéo dài.
- Gợi ý: tổ hợp máy theo từng khung giờ, thời điểm khởi động buổi sáng, setpoint nước lạnh cấp trong dải cho phép, kèm kWh / tiền tiết kiệm ước tính.

### 6.3 M3 — Nước / bơm

**Câu hỏi cần trả lời**
- Có rò rỉ không (tiêu thụ ban đêm cao bất thường)?
- Bơm nào đang suy giảm (cùng tốc độ nhưng cột áp giảm)?
- Suất tiêu thụ điện trên mỗi m³ nước có xấu đi không?

| Key | Đơn vị | Vai trò | Dùng cho |
|---|---|---|---|
| `water_volume_m3_total` | m³ | Đích | Counter → `m3_15m`; baseline tiêu thụ nước |
| `water_flow_lpm` | L/min | Đích | `night_min_flow` → rò rỉ; cần bên cạnh Δ thể tích vì Δ không phân biệt "bơm chạy không ra nước" với "bơm không chạy" |
| `power_active_kw`, `energy_active_kwh_total` (của bơm) | kW, kWh | Đích | `kwh_per_m3`; ước tính kWh lãng phí trong mỗi sự kiện |
| `pump_speed_pct` | % | Giải thích | Chuẩn hoá cột áp theo tốc độ — tách "bơm yếu" khỏi "bơm chạy chậm" |
| `pressure_supply_bar`, `pressure_return_bar` | bar | Giải thích | `pump_head_proxy` = hiệu hai giá trị |
| `water_supply_temp_c`, `water_return_temp_c` | °C | Giải thích | Bơm nóng khi chạy khô / vòng kín bị đóng van |
| `pump_state` | – | Trạng thái | Chỉ tính cột áp và suất tiêu thụ khi bơm chạy; đếm số lần đóng cắt |
| `pump_state_cmd`, `pump_speed_pct_sp` | – | Đọc lệnh | Luật lệnh không đáp ứng, biến tần không theo lệnh |
| `tank_level_low/medium/high` | bool | Trạng thái | Ràng buộc an toàn cho gợi ý lịch bơm; `tank_fill_time` |
| `supply_valve_open/close_limit`, `supply_valve_state_cmd` | bool, – | Trạng thái / Đọc lệnh | Luật van kẹt |
| `return_valve_*`, `bypass_valve_state_cmd` | bool, – | Trạng thái / Đọc lệnh | Van vòng hồi; giải thích thay đổi áp khi mở bypass |

**Luật an toàn (lớp a — không cần baseline)**

Nhóm luật này cho kết quả chắc chắn nhất trong benchmark (chạy khô: event recall 1.00, precision 1.00, trễ 0 phút, 0 báo động giả) nhưng **chỉ chạy được khi có các key trạng thái từ DMP thật** — không làm được trên dữ liệu MVP.

| Luật | Điều kiện (cần hiệu chỉnh) | Mức |
|---|---|---|
| Chạy khô | `pump_state` = ON và (`tank_level_low` báo cạn **hoặc** `water_flow_lpm` ≈ 0) kéo dài > N phút | critical |
| Tràn bồn | `tank_level_high` báo đầy và bơm lên bồn vẫn ON | critical |
| Van kẹt | `*_valve_state_cmd` = OPEN nhưng `*_valve_open_limit` không lên sau T giây | warning |
| Lệnh không đáp ứng | `pump_state_cmd` ≠ `pump_state` sau T giây | warning |
| Limit switch mâu thuẫn | `*_open_limit` và `*_close_limit` cùng active | warning |
| Bơm đóng cắt liên tục | Số lần `pump_state` đổi trạng thái / giờ > ngưỡng | warning |

> Phải xác nhận với hiện trường ngữ nghĩa `tank_level_*` (tiếp điểm thường đóng hay thường mở) trước khi bật luật — đảo dấu một tiếp điểm là đảo ngược toàn bộ cảnh báo.

**Đại lượng dẫn xuất**
- `m3_15m = Δ water_volume_m3_total` · `night_min_flow` = lưu lượng tối thiểu 01–04h.
- `pump_head_proxy = pressure_supply_bar − pressure_return_bar`, chuẩn hoá theo `pump_speed_pct`.
- `kwh_per_m3` = điện bơm / m³ bơm · `tank_fill_time` = thời gian từ `tank_level_low` đến `tank_level_high`.

**Output cụ thể**
- Baseline tiêu thụ nước; bất thường nhóm A (rò rỉ ban đêm, đơ số); nhóm B (suy giảm cột áp, `kwh_per_m3` xấu đi).
- Dự báo nhu cầu nước theo giờ.
- Gợi ý lịch bơm lên bồn dồn vào giờ thấp điểm, **ràng buộc cứng**: mực bồn không bao giờ xuống `tank_level_low` trong dự báo, tôn trọng min on/off time.

---

## 7. Ma trận key × module

✔ = key chính · ○ = bổ trợ / tùy chọn

| Nhóm | Key | M1 Điện | M2 Chiller | M3 Nước |
|---|---|:-:|:-:|:-:|
| Power meter | `power_active_kw`, `energy_active_kwh_total` | ✔ | ○ | ✔ |
| Power meter | `power_factor`, `energy_reactive_kvarh_total`, `frequency_hz` | ✔ | | |
| Power meter | `voltage_v`, `voltage_l1_v`, `voltage_l2_v`, `voltage_l3_v` | ✔ | | |
| Power meter | `current_a`, `current_l1_a`, `current_l2_a`, `current_l3_a` | ✔ | | |
| Chiller | 9 key nhóm chiller | | ✔ | |
| NORIS | `active_power_kw`, `active_energy_kwh`, `cooling_load_kw`, `chw_supply/return_temp`, `compressor_status`, `op_mode`, `run_hours`, `evaporator_pressure` | ○ | ✔ | |
| NORIS | `voltage_l1..l3`, `water_volume_m3` | | ○ | ○ |
| Air side | `cooling_tower_fan_state` | | ✔ | |
| Nước | 20 key nhóm nước | | ○ | ✔ |
| PHATTAI | `temperature`, `humidity` | ✔ | ✔ | ○ |
| ZKTeco | `record_count` | ✔ | ✔ | ○ |
| Dùng chung | `fault`, `mode` | ✔ | ✔ | ✔ |
| Dùng chung | `state_on_off`, `lighting_knx_state` | ○ | | ○ |
| ICMP | `reachable` | ✔ | ✔ | ✔ |

### 7.1 Tổng hợp số key

| Nguồn | Tổng trong v1.13 | Dùng trong bài toán A (3 domain) |
|---|---:|---:|
| Power meter | 13 | 13 |
| HVAC · chiller | 9 | 9 |
| Nước / bơm / van / bồn | 20 | 20 |
| §2.1 Siemens NORIS | 13 | 13 |
| HVAC · air side | 16 | 1 (`cooling_tower_fan_state`) |
| §2.2 PHATTAI | 12 | 2 (`temperature`, `humidity`) |
| Dùng chung | 6 | 4 (`fault`, `mode`, `state_on_off`, `lighting_knx_state`) |
| Access control · ZKTeco | 11 | 1 (`record_count`) |
| ICMP ping | 3 | 1 (`reachable`) |
| Attribute | 16 | 6 |

**Bộ tối thiểu để khởi động (P0): 30 key** — dùng chung 6 (`temperature`, `humidity`, `record_count`, `fault`, `mode`, `reachable`) + M1 2 + M3 12 + M2 10.

### 7.2 Không dùng

- Host / gateway compute (20), cellular/WLAN (9), GPS (6), Camera Hik ISAPI (8), Báo cháy Notifier (9), Thang máy Schindler (2) → thuộc bài toán B.
- Air side còn lại (15) và PHATTAI còn lại (10) → thuộc M4, giai đoạn sau.
- Tất cả key `*_cmd`, `*_sp` **không bao giờ được ghi**; chỉ đọc.

---

## 8. Dữ liệu cho MVP

**Building Data Genome 2 (BDG2)** — https://github.com/buds-lab/building-data-genome-project-2

- Quy mô: **3.053 đồng hồ đo từ 1.636 tòa nhà** phi dân dụng tại 19 khu vực, hai năm đầy đủ **2016–2017**, tần suất **theo giờ**, kèm dữ liệu thời tiết và metadata tòa nhà.
- Loại đồng hồ: điện, nước lạnh (chilled water), hơi, nước nóng, gas, nước sinh hoạt, tưới tiêu, điện mặt trời → phủ được cả 3 domain ở **mức tiêu thụ**.

| Làm được trên BDG2 | Không làm được |
|---|---|
| Nhóm A — baseline tiêu thụ, chạy ngoài giờ, tải nền tăng dần, rò rỉ, đơ số | Các lỗi ngắn trong ô 15 phút (dữ liệu chỉ theo giờ) |
| Nhóm C — mẫu hình ngày, lệch lịch, sai loại ngày, dịch đỉnh | Nhóm B — hiệu suất: BDG2 không có công suất điện từng chiller, không có áp suất / tốc độ bơm |
| Lớp chất lượng dữ liệu | Luật an toàn M3: không có biến trạng thái |
| So ngang tập thiết bị (nhiều tòa nhà cùng loại) | Gợi ý tối ưu có ràng buộc thiết bị |

Vì vậy MVP trên BDG2 chốt được **khung xử lý, thư viện phương pháp và bộ tham số vận hành**; phần hiệu suất và luật trạng thái phải đợi dữ liệu DMP thật.

---

## 9. Gap dữ liệu & câu hỏi cần chốt

| # | Vấn đề | Ảnh hưởng | Việc cần làm | Chặn |
|---|---|---|---|---|
| G1 | Điện / chiller chưa có luồng real-time trong DMP, phải qua NORIS | M1, M2 | Chốt: NORIS có đủ tần suất (≤ 15 phút) và độ trễ chấp nhận được không | **Chặn cả bài toán A — làm đầu tiên** |
| G2 | Hai bộ tên key cho cùng đại lượng (DMP chuẩn vs NORIS) | M1, M2 | Lập bảng mapping + kiểm tra đơn vị, dấu, counter | Chặn bước chuẩn hóa |
| G3 | Không có công suất điện riêng từng chiller trong nhóm chiller chuẩn | M2 | Xác nhận `active_power_kw` NORIS là theo từng máy hay cả plant | **Chặn toàn bộ nhóm bất thường hiệu suất của M2** |
| G4 | `user_count` là số người đã đăng ký, không phải số người có mặt | M1, M2 | Dùng Δ`record_count`; xác nhận counter có quay vòng khi chạm `record_capacity` không | Không chặn (baseline vẫn chạy, kém chính xác hơn) |
| G6 | `tank_level_*` chỉ 3 mức rời rạc, chưa rõ ngữ nghĩa tiếp điểm | M3 | Xác nhận thường đóng / thường mở; chấp nhận ước lượng rời rạc | Chặn luật an toàn M3 |
| G7 | Không có lưu lượng nước lạnh riêng cho vòng chiller | M2 | Xác nhận có đồng hồ nào nằm trên vòng CHW không | Không chặn (có `cooling_kw`) |
| G8 | Thiếu nhãn sự cố thật để đánh giá | Tất cả | Thu nhật ký bảo trì / sự cố 6–12 tháng; dựng tập kịch bản tái hiện cho M3 | Chặn đánh giá precision |
| G9 | Độ dài lịch sử dữ liệu | Tất cả | Kiểm tra mỗi key có tối thiểu bao nhiêu tháng; cần ≥ 8 tuần liên tục cho lớp baseline | Không chặn GĐ1 (lớp luật chạy được ngay) |
| G10 | Nhật ký bảo trì chưa có dạng dữ liệu | M2, M3 | Số hoá mốc bảo trì từng thiết bị | **Chặn nhóm bất thường hiệu suất** (không có mốc reset cửa sổ tham chiếu) |

---

## 10. Tiêu chí nghiệm thu (đề xuất, cần thống nhất với mentor)

| Hạng mục | Chỉ số | Mục tiêu đề xuất |
|---|---|---|
| Chất lượng dữ liệu | Tỷ lệ mẫu hợp lệ sau chuẩn hóa | Báo cáo được cho 100% thiết bị trong phạm vi |
| Bất thường nhóm A | **Event precision** ở ngân sách 0.2 báo động/thiết bị/ngày, thời lượng tối thiểu 1 giờ | ≥ 0.20 tại event recall ≥ 0.35 |
| Bất thường nhóm B | Phát hiện được mức suy giảm ≥ 10% so với cửa sổ tham chiếu | Trong vòng 7 ngày kể từ khi suy giảm ổn định |
| Bất thường nhóm C | Số ngày lạ báo ra | ≤ 3 ngày/thiết bị/tháng, mỗi ngày kèm hình so sánh profile |
| Luật an toàn M3 *(khi có dữ liệu DMP)* | Bỏ sót trên tập kịch bản tái hiện | = 0 |
| Dự báo tải lạnh 24h | MAPE (giờ vận hành) | ≤ 15% |
| Dự báo phụ tải điện 24h | MAPE (giờ vận hành) | ≤ 10–15% |
| Gợi ý | Tiết kiệm ước tính (kWh, VND) trên backtest, ghi rõ giả định | Có số liệu cho M2 |
| Vận hành | Mỗi cảnh báo / gợi ý có bằng chứng (key, thời gian, baseline, cửa sổ tham chiếu) | 100% |

**Lưu ý về mức kỳ vọng.** Tiêu chí "precision ≥ 80%" theo mẫu không đạt được với dữ liệu tòa nhà thật: đo trên 200 công tơ có nhãn chuyên gia, ở ngân sách 1 báo động/ngày event precision chỉ 0.089–0.153; hạ xuống 0.2 báo động/ngày thì lên 0.198 (hợp OR, thời lượng ≥ 1 giờ) và 0.291 (hợp AND). Muốn precision cao hơn phải hy sinh recall: siết thời lượng tối thiểu lên 3 giờ cho precision 0.728 nhưng recall chỉ còn 0.014. Vì vậy precision phải tính **theo sự kiện**, không theo mẫu, và mức kỳ vọng nên chốt từ đầu chứ không đợi lúc nghiệm thu.

---

## 11. Lộ trình

| GĐ | Nội dung | Module | Đầu ra chính |
|---|---|---|---|
| GĐ0 (tuần 1–2) | Chốt G1–G3, G10; kiểm kê key **thực có dữ liệu** (số mẫu/ngày, ngày khuyết trong 30 ngày gần nhất); lập mapping NORIS; thu calendar, biểu giá, nhật ký bảo trì | Tất cả | Báo cáo kiểm kê dữ liệu, bảng mapping |
| GĐ1 (tuần 3–6) | MVP trên BDG2: chuẩn hóa + lớp chất lượng dữ liệu + baseline 3 lớp + bất thường nhóm A và C; chốt bộ tham số vận hành (ngân sách, thời lượng, cách gộp sự kiện) | M1, M3 | O1, O2, dashboard baseline |
| GĐ2 (tuần 7–10) | Chuyển sang dữ liệu DMP thật: bất thường nhóm B (hiệu suất) + luật an toàn M3; dự báo tải lạnh và phụ tải điện | M2 (chính), M1, M3 | O3, đường cong hiệu suất từng máy |
| GĐ3 (tuần 11–14) | Gợi ý tối ưu theo giá điện có ràng buộc; backtest tiết kiệm; báo cáo | M2 (chính), M3 | O4, O5, báo cáo tiết kiệm ước tính |

**Phân công theo module:** M1 Điện — Đức · M2 Chiller — Đôn · M3 Nước — Bảo. Framework (chuẩn hóa, baseline, định dạng output, lớp chất lượng dữ liệu) làm chung, thống nhất interface từ GĐ0. M4 Không khí/CO2 giữ nguyên khung, triển khai sau khi 3 domain chính chạy ổn.
