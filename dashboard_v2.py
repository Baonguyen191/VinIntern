import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Setup page layout
st.set_page_config(
    page_title="Năng lượng & M&E Thông minh — Khung Demo Sprint 2",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, 'regression_vs_baseline.csv')
FORECAST_PATH = os.path.join(BASE_DIR, 'data_normalized', 'forecast_test_results.parquet')
SCENARIOS_PATH = os.path.join(BASE_DIR, 'optimization_scenarios_preliminary.csv')

@st.cache_data
def load_regression_report():
    if os.path.exists(REPORT_PATH):
        return pd.read_csv(REPORT_PATH)
    return pd.DataFrame()

@st.cache_data
def load_forecast_data():
    if os.path.exists(FORECAST_PATH):
        df = pd.read_parquet(FORECAST_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    return pd.DataFrame()

@st.cache_data
def load_scenarios_data():
    if os.path.exists(SCENARIOS_PATH):
        return pd.read_csv(SCENARIOS_PATH, comment='#')
    return pd.DataFrame()

# ---------------------------------------------------------
# HEADER & VERIFICATION STATUS (SPRINT 2)
# ---------------------------------------------------------
st.title("⚡ Khung Demo Tích Hợp Sprint 2 — Pipeline Năng Lượng & M&E")
st.markdown("""
**Hệ thống Dự báo & Gợi ý Tối ưu** | Pipeline 4 bước: *Baseline → Bất thường → Dự báo → Gợi ý tối ưu*
""")

# Bảng phân định RÕ RÀNG dữ liệu THẬT vs MOCK theo yêu cầu Sprint 2
st.info("""
📌 **PHÂN ĐỊNH TRẠNG THÁI DỮ LIỆU TRÊN DASHBOARD (SPRINT 2):**
- 🟢 **DỮ LIỆU THẬT (Mô hình Hồi quy Ridge Regression - Việc 1 & 2)**:
  - `value_actual`: Dữ liệu đo đạc thực tế từ telemetry cảm biến / công tơ.
  - `forecast_next_24h`: Dự báo **THẬT từ mô hình hồi quy Ridge Regression** (tập đặc trưng chuẩn `forecast_spec.md`: lag 24h/48h/168h, mean cùng giờ tuần trước, lịch, nhiệt độ ngoài trời). Thay thế hoàn toàn mô hình naive ở Sprint 1.
  - `forecast_lower`, `forecast_upper`: Khoảng tin cậy 80% (P10 - P90) tính từ độ lệch chuẩn phần dư tập học của mô hình hồi quy.
- 🟡 **DỮ LIỆU MOCK / PLACEHOLDER (VẪN CHƯA CÓ KẾT QUẢ THẬT TỪ BẠN 1 & BẠN 2 - TIẾP TỤC ĐỂ PLACEHOLDER)**:
  - `baseline_expected`: Giả lập bằng *Trung bình trượt 7 ngày (Rolling 7-day mean)* (Chờ Bạn 1).
  - `is_anomaly`, `anomaly_reason`: Giả lập quy tắc lệch > 2.0 std so với baseline kỳ vọng (Chờ Bạn 2).
""")

# Load Data
df_report = load_regression_report()
df_fc = load_forecast_data()
df_scenarios = load_scenarios_data()

if df_fc.empty:
    st.error("Không tìm thấy file `forecast_test_results.parquet`. Vui lòng chạy `python scripts/run_sprint2_pipeline.py` trước.")
    st.stop()

# ---------------------------------------------------------
# SIDEBAR FILTERS
# ---------------------------------------------------------
st.sidebar.header("🔍 Bộ Lọc Dữ Liệu")

# Select Metric
metric_options = sorted(df_fc['metric'].unique())
selected_metric = st.sidebar.selectbox("Chọn Chỉ Số (Metric):", metric_options, index=0)

# Select Building
filtered_by_metric = df_fc[df_fc['metric'] == selected_metric]
building_options = sorted(filtered_by_metric['building_id'].unique())
selected_building = st.sidebar.selectbox("Chọn Tòa Nhà (Building ID):", building_options, index=0)

# Filter dataset for selected building & metric
building_df = filtered_by_metric[filtered_by_metric['building_id'] == selected_building].sort_values('timestamp').copy()

# Date range slider
min_ts = building_df['timestamp'].min()
max_ts = building_df['timestamp'].max()

st.sidebar.subheader("📅 Khoảng Thời Gian (Tập Test)")
date_range = st.sidebar.date_input(
    "Chọn khoảng ngày:",
    value=(min_ts.date(), min_ts.date() + pd.Timedelta(days=14)),
    min_value=min_ts.date(),
    max_value=max_ts.date()
)

if len(date_range) == 2:
    start_date, end_date = date_range
    mask = (building_df['timestamp'].dt.date >= start_date) & (building_df['timestamp'].dt.date <= end_date)
    building_df = building_df[mask]

# ---------------------------------------------------------
# MOCK PLACEHOLDER GENERATION (FOR TEAMMATE 1 & 2)
# ---------------------------------------------------------
# [PLACEHOLDER TEAMMATE 1 - BASELINE]
# Mock logic: Rolling 7-day mean
building_df['baseline_expected'] = building_df['value_actual'].rolling(window=168, min_periods=24).mean()

# [PLACEHOLDER TEAMMATE 2 - ANOMALY]
# Mock logic: Lệch > 2.0 std so với rolling mean
diff = np.abs(building_df['value_actual'] - building_df['baseline_expected'])
threshold = building_df['value_actual'].std() * 2.0
building_df['is_anomaly'] = diff > threshold
building_df['anomaly_reason'] = np.where(
    building_df['is_anomaly'],
    "MOCK (Bạn 2): Phụ tải lệch quá 2.0 std so với baseline kỳ vọng",
    None
)

# ---------------------------------------------------------
# TOP METRICS CARD: RIDGE VS SEASONAL-NAIVE EVALUATION
# ---------------------------------------------------------
st.subheader(f"📊 Đánh Giá Hiệu Năng Mô Hình: Hồi Quy Ridge vs Seasonal-Naive — {selected_building}")

if not df_report.empty:
    b_report = df_report[(df_report['building_id'] == selected_building) & (df_report['metric'] == selected_metric)]
    sn_row = b_report[b_report['model'] == 'seasonal_naive_168h']
    ridge_row = b_report[b_report['model'] == 'ridge_regression']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    sn_mae = sn_row['MAE'].values[0] if not sn_row.empty else "N/A"
    sn_cvrmse = sn_row['CV(RMSE)'].values[0] if not sn_row.empty else "N/A"
    ridge_mae = ridge_row['MAE'].values[0] if not ridge_row.empty else "N/A"
    ridge_cvrmse = ridge_row['CV(RMSE)'].values[0] if not ridge_row.empty else "N/A"
    delta_pct = ridge_row['delta_cvrmse_vs_seasonal_naive_pct'].values[0] if not ridge_row.empty else "N/A"
    
    unit_str = "kW"
    with col1:
        st.metric("Seasonal-Naive MAE", f"{sn_mae} {unit_str}" if sn_mae != "N/A" else "N/A")
    with col2:
        st.metric("Seasonal-Naive CV(RMSE)", f"{sn_cvrmse}%" if sn_cvrmse != "N/A" else "N/A")
    with col3:
        st.metric("Ridge Regression MAE", f"{ridge_mae} {unit_str}" if ridge_mae != "N/A" else "N/A", 
                  delta=f"{round(ridge_mae - sn_mae, 2)} {unit_str}" if (ridge_mae != "N/A" and sn_mae != "N/A") else None,
                  delta_color="inverse")
    with col4:
        st.metric("Ridge Regression CV(RMSE)", f"{ridge_cvrmse}%" if ridge_cvrmse != "N/A" else "N/A",
                  delta=f"{delta_pct}%" if delta_pct != "N/A" else None,
                  delta_color="inverse")
    with col5:
        if delta_pct != "N/A":
            if delta_pct < 0:
                st.success(f"✅ ĐẠT TIÊU CHÍ\n(CV(RMSE) giảm {abs(delta_pct)}%)")
            else:
                st.warning(f"⚠️ Chưa vượt Naive (+{delta_pct}%)")
        else:
            st.info("N/A")

# Expander for portfolio overview
with st.expander("🌐 Xem Tóm Tắt Tỷ Lệ Thắng Toàn Portfolio (1.140 Tòa Nhà)"):
    st.markdown("""
    - **Tổng số tòa nhà kiểm thử**: 1.140 tòa nhà (799 tòa nhà Điện năng M1, 341 hệ thống Chiller M2).
    - **Tỷ lệ mô hình Hồi quy Ridge chiến thắng Seasonal-naive (t-168h)**: **91,14%** (1.039 / 1.140 tòa nhà).
      - **Điện năng (`power_active_kw`)**: Thắng **89,6%** (716/799). CV(RMSE) trung bình giảm từ 34.08% xuống 26.94%.
      - **Chiller (`cooling_kw`)**: Thắng **94,7%** (323/341). CV(RMSE) trung bình giảm từ 248.83% xuống 163.43%.
    - *Ghi chú tính hợp lệ*: Các trường hợp Ridge chưa vượt được Seasonal-naive (8,86%) chủ yếu do phụ tải có tính chu kỳ tuần cực kỳ cố định và chuỗi lịch sử học bị gián đoạn cảm biến, đây là thông tin thực tế được ghi nhận đầy đủ trong báo cáo.
    """)

# ---------------------------------------------------------
# MAIN TIME-SERIES CHART
# ---------------------------------------------------------
st.subheader(f"📈 Biểu Đồ Chuỗi Thời Gian & Dự Báo Hồi Quy (Ridge) 24h — {selected_building}")

fig, ax = plt.subplots(figsize=(15, 5))

# Actual values (Real)
ax.plot(building_df['timestamp'], building_df['value_actual'], label='Thực tế (Real Telemetry Actual)', color='#1f77b4', linewidth=1.5)

# Baseline expected (Mock Teammate 1)
ax.plot(building_df['timestamp'], building_df['baseline_expected'], label='Baseline kỳ vọng (Mock Bạn 1)', color='#ff7f0e', linestyle='--', alpha=0.8)

# Forecast next 24h (Real Ridge Regression Forecast)
ax.plot(building_df['timestamp'], building_df['forecast_next_24h'], label='Dự báo Hồi quy Ridge (THẬT - Me)', color='#2ca02c', linestyle='-', linewidth=2)

# Upper & Lower bounds band (Real 80% Confidence Interval)
ax.fill_between(
    building_df['timestamp'],
    building_df['forecast_lower'],
    building_df['forecast_upper'],
    color='#2ca02c',
    alpha=0.18,
    label='Khoảng tin cậy 80% (P10 - P90) từ phần dư Ridge'
)

# Anomaly markers (Mock Teammate 2)
anomalies = building_df[building_df['is_anomaly']]
if not anomalies.empty:
    ax.scatter(anomalies['timestamp'], anomalies['value_actual'], color='red', s=45, zorder=5, label='Bất thường (Mock Bạn 2)')

ax.set_xlabel("Thời gian (Timestamp)")
ax.set_ylabel(f"Giá trị ({selected_metric})")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')

st.pyplot(fig)

# ---------------------------------------------------------
# TABLE 1: UNIFIED DEMO SCHEMA TABLE
# ---------------------------------------------------------
st.subheader("📋 Bảng Dữ Liệu Khung Theo Schema Thống Nhất (`demo_schema.md`)")

display_cols = [
    'timestamp', 'building_id', 'metric', 
    'value_actual', 'baseline_expected', 
    'is_anomaly', 'anomaly_reason', 
    'forecast_next_24h', 'forecast_lower', 'forecast_upper'
]

st.dataframe(
    building_df[display_cols].head(100),
    use_container_width=True
)
st.caption("Hiển thị 100 dòng đầu tiên theo khung bộ lọc đã chọn tuân thủ 100% 10 cột theo quy chuẩn `demo_schema.md`.")

# ---------------------------------------------------------
# TABLE 2: PRELIMINARY OPTIMIZATION SCENARIOS (VIỆC 3 & 4)
# ---------------------------------------------------------
st.markdown("---")
st.subheader(f"💡 Bảng Phụ: Ước Tính Kịch Bản Tối Ưu Sơ Bộ (Preliminary Optimization) — {selected_building}")

st.warning("""
⚠️ **GHI CHÚ QUAN TRỌNG (DISCLAIMER):**
Kịch bản sơ bộ sử dụng luật đơn giản (if-then rule-based), **CHƯA phải kết quả từ bộ giải tối ưu toán học (MILP)**.
Không dùng số liệu này để khẳng định mức tiết kiệm chắc chắn. Sprint 3 sẽ tinh chỉnh với bộ giải tối ưu chi tiết và thông số thiết bị chính thức.
""")

if not df_scenarios.empty:
    bld_scenarios = df_scenarios[(df_scenarios['building_id'] == selected_building) & (df_scenarios['metric'] == selected_metric)]
    
    if not bld_scenarios.empty:
        # KPI summary cards
        c1, c2, c3 = st.columns(3)
        row_ht = bld_scenarios[bld_scenarios['scenario'] == 'hien_trang']
        row_shift = bld_scenarios[bld_scenarios['scenario'] == 'dich_tai_thu_cong']
        row_shave = bld_scenarios[bld_scenarios['scenario'] == 'giam_dinh_don_gian']
        
        with c1:
            st.markdown("#### (a) Hiện Trạng")
            if not row_ht.empty:
                st.write(f"**Điện năng**: {row_ht['total_kwh'].values[0]:,.1f} kWh")
                st.write(f"**Chi phí**: {row_ht['cost_vnd'].values[0]:,.0f} VND")
                st.write("**Tiết kiệm**: 0 VND (0.0%)")
        with c2:
            st.markdown("#### (b) Dịch Tải Thủ Công")
            if not row_shift.empty:
                st.write(f"**Điện năng**: {row_shift['total_kwh'].values[0]:,.1f} kWh")
                st.write(f"**Chi phí**: {row_shift['cost_vnd'].values[0]:,.0f} VND")
                st.success(f"**Tiết kiệm**: {row_shift['savings_vnd'].values[0]:,.0f} VND (**{row_shift['savings_pct'].values[0]}%**)")
        with c3:
            st.markdown("#### (c) Giảm Đỉnh Đơn Giản")
            if not row_shave.empty:
                st.write(f"**Điện năng**: {row_shave['total_kwh'].values[0]:,.1f} kWh")
                st.write(f"**Chi phí**: {row_shave['cost_vnd'].values[0]:,.0f} VND")
                st.success(f"**Tiết kiệm**: {row_shave['savings_vnd'].values[0]:,.0f} VND (**{row_shave['savings_pct'].values[0]}%**)")

        st.markdown("**Chi tiết bảng kịch bản cho tòa nhà:**")
        scenario_display = bld_scenarios[['scenario', 'total_kwh', 'cost_vnd', 'savings_vnd', 'savings_pct', 'assumptions']].copy()
        scenario_display.columns = ['Kịch Bản', 'Tổng kWh Tiêu Thụ', 'Chi Phí (VND)', 'Tiết Kiệm (VND)', 'Tỷ Lệ Tiết Kiệm (%)', 'Giả Định & Luật Áp Dụng']
        st.dataframe(scenario_display, use_container_width=True)
    else:
        st.info(f"Chưa có kịch bản tối ưu chi tiết cho tòa nhà `{selected_building}`.")
        
    # Also show Portfolio total in an expander
    with st.expander("🏢 Xem Kịch Bản Ước Tính Toàn Portfolio (PORTFOLIO_TOTAL)"):
        port_df = df_scenarios[(df_scenarios['building_id'] == 'PORTFOLIO_TOTAL') & (df_scenarios['metric'] == selected_metric)].copy()
        if not port_df.empty:
            port_disp = port_df[['scenario', 'total_kwh', 'cost_vnd', 'savings_vnd', 'savings_pct', 'assumptions']].copy()
            port_disp.columns = ['Kịch Bản', 'Tổng kWh Tiêu Thụ', 'Chi Phí (VND)', 'Tiết Kiệm (VND)', 'Tỷ Lệ Tiết Kiệm (%)', 'Giả Định & Luật Áp Dụng']
            st.dataframe(port_disp, use_container_width=True)
            
st.markdown("---")
st.caption("Sprint 2 Pipeline & Demo hoàn tất — Các đầu ra được lưu tại `regression_vs_baseline.csv`, `optimization_scenarios_preliminary.csv`, `dashboard_v2.py`.")
