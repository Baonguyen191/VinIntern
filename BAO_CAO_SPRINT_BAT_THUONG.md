# Báo cáo 2 sprint: Bộ ca bất thường và Bộ phát hiện bất thường

Bảo Nguyễn · nhánh `anomaly` · 25/09/2026

## Dữ liệu và phạm vi

- Dữ liệu điện và chilled water BDG2 của 20 tòa nhà Office. Năm 2016 dùng để học, năm 2017 dùng để phát hiện và đánh giá.
- Không làm ca rò rỉ nước và CO₂, vì dữ liệu hiện có không có chuỗi nước hay CO₂.

## Sprint 1 – Bộ ca bất thường

- Chèn 116 ca có nhãn vào dữ liệu năm 2017. Mọi ca có cột `is_synthetic`.
  - Nhóm A: 40 ca tăng vọt, 20 ca chạy ngoài giờ.
  - Nhóm B: 16 ca giảm hiệu suất.
  - Chất lượng dữ liệu: 20 ca mất dữ liệu, 20 ca cảm biến đứng yên.
- Chèn 9 ca vào CSV baseline mẫu (dạng ngày) để thử nhánh so sai lệch.
- Bộ phát hiện ban đầu là lớp kiểm tra chất lượng dữ liệu: mất mẫu, giá trị đứng yên, ngoài dải đo, counter nhảy lùi.

## Sprint 2 – Bộ phát hiện bất thường

- Các detector đã làm:
  - Luật profile band: baseline là median 4 tuần gần nhất cùng giờ và loại ngày.
  - Isolation Forest.
  - LightGBM.
  - Gộp điểm LightGBM với IF.
  - Detector hiệu suất: điện so với tải lạnh.
  - Nhánh so sai lệch: đọc mọi CSV theo cấu trúc baseline mẫu.
- Luật tự tính baseline, nên không phụ thuộc tiến độ mô hình của module khác.
- Gộp các giờ bất thường liền nhau thành một sự kiện, rồi phân loại **tại điểm** (dài tối đa 2 giờ) hoặc **trên khoảng**.
- Lỗi cảm biến được tách riêng. Cảnh báo vận hành nào trùng khoảng lỗi cảm biến thì bị bỏ.
- Mỗi cảnh báo xuất theo format O2, có bằng chứng: giá trị, baseline, % lệch, cửa sổ tham chiếu.

## Kết quả (trên nhãn giả lập)

| Hạng mục | Precision / Recall |
|---|---|
| Cảnh báo cuối, ngân sách 0,2/thiết bị/ngày (mục tiêu ≥ 0,20 / ≥ 0,35) | **0,205 / 0,934** |
| Ngân sách 0,02: Luật / IF / LightGBM | 0,272 / 0,597 · 0,219 / 0,468 · 0,265 / 0,581 |
| Ngân sách 0,02: Gộp max LightGBM + IF | **0,286 / 0,645** |

- Cảnh báo cuối đạt mục tiêu của spec. Ca giảm hiệu suất được phát hiện sau khoảng 1 ngày; spec cho phép tối đa 7 ngày.
- Không nên đưa điểm IF cùng giờ làm feature cho LightGBM. Làm vậy baseline bám theo bất thường và che bớt nó: kết quả giảm còn 0,221 / 0,500.

## Demo

`streamlit run demo_anomaly.py`

So sánh tải thực tế với baseline và hiển thị bất thường tại điểm và trên khoảng. Người dùng chọn được cách phát hiện, ngưỡng, ngân sách và khoảng ngày.

## Việc tiếp theo

1. Thay cách chọn cảnh báo top-K theo cả năm (dùng cả dữ liệu tương lai) bằng trần cảnh báo theo ngày và ngưỡng hiệu chỉnh riêng từng tòa nhà.
2. Chốt ngân sách vận hành với đội vận hành. Đề xuất 0,02–0,05 cảnh báo/thiết bị/ngày.
3. Chốt lịch ngày lễ: theo nước sở tại hay lịch Việt Nam.
4. Làm ca rò rỉ nước và CO₂ khi có dữ liệu DMP thật.
