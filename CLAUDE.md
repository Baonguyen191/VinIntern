# CLAUDE.md — VinIntern

Pipeline dự báo & tối ưu năng lượng (Python, pandas, scipy MILP, Streamlit). Chi tiết chạy: xem `README.md`.

## Cấu trúc thư mục — đặt file ĐÚNG chỗ

| Thư mục | Chứa gì | Commit? |
|---|---|---|
| `src/` | Module dùng chung + `dashboard.py`. Import phẳng (`from forecast import ...`) | Có |
| `src/paths.py` | **Nguồn duy nhất** cho mọi đường dẫn file | Có |
| `scripts/` | Script chạy pipeline (entry point), không chứa logic dùng lại | Có |
| `data/normalized/` | Dữ liệu đầu vào đã chuẩn hóa — **chỉ đọc** | CSV có, parquet không |
| `data/teammates/` | File kết quả thật từ Bạn 1/Bạn 2 | Theo thỏa thuận |
| `outputs/` | **Mọi** file do code sinh ra (report CSV, parquet dự báo) | CSV có, parquet không |
| `tests/fixtures/` | Dữ liệu mock / mẫu kiểm thử | Có |
| `docs/` | Đặc tả, hướng dẫn, checklist (`.md`) | Có |

Thư mục gốc chỉ chứa: `README.md`, `CLAUDE.md`, `requirements.txt`, `.gitignore`. Không để `.py`, `.csv`, `.md` khác ở gốc.

## Quy tắc đặt tên

- **KHÔNG** dùng hậu tố phiên bản/trạng thái trong tên file: `_v1`, `_v2`, `_final`, `_new`, `_old`, `_backup`, `_copy`, `_sprintN`, `_preliminary` cho code. Phiên bản là việc của **git** (commit, branch, tag), không phải tên file.
- Muốn làm bản mới của một file → **sửa trực tiếp file đó** trên branch mới. Muốn giữ bản cũ để so sánh → dùng `git show <commit>:<path>` hoặc tag (vd. `git tag sprint2-demo`), không copy file.
- Tên file theo **chức năng**, `snake_case`, danh từ ngắn: `forecast.py`, `optimizer.py`, `dashboard.py`. Script trong `scripts/` dùng động từ: `train_ridge.py`, `build_dataset.py`.
- Output đặt tên theo **nội dung**, không theo sprint/ngày: `regression_vs_baseline.csv`, `recommendation_ranking.csv`.
- Không thay thế một module bằng cách tạo module song song. Nếu hành vi cũ vẫn cần giữ → thêm tham số/hàm trong cùng module.

## Quy tắc code

- Mọi đường dẫn khai báo trong `src/paths.py`; module khác `from paths import ...`. Không `os.path.join(BASE_DIR, ...)` rải rác.
- Script trong `scripts/` import `src/` bằng `sys.path.insert(0, <root>/src)` như trong `scripts/train_ridge.py`.
- Code chỉ **ghi** vào `outputs/`; không ghi vào `data/normalized/` hay `tests/fixtures/`.
- Hai script không được ghi cùng một file output trừ khi script sau cố ý thay thế script trước — ghi rõ trong docstring.
- Chạy dashboard: `python -m streamlit run src/dashboard.py` (từ thư mục gốc).

## Trước khi thêm file mới, tự hỏi

1. Đã có file nào làm việc này chưa? → sửa file đó.
2. File này thuộc thư mục nào trong bảng trên?
3. Tên có chứa `_v`, `_final`, `_new`, số sprint không? → đổi tên.
4. Nếu thay thế file cũ → `git mv` / `git rm` file cũ trong **cùng commit**.
