import os
import pandas as pd
import numpy as np
import streamlit as st

# Setup page layout
st.set_page_config(
    page_title="Năng lượng & M&E Thông minh — Khung Demo Sprint 1",
    page_icon="⚡",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORT_PATH = os.path.join(BASE_DIR, 'baseline_forecast_report.csv')
FORECAST_PATH = os.path.join(BASE_DIR, 'data_normalized', 'forecast_test_results.parquet')

@st.cache_data
def load_baseline_report():
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

# ---------------------------------------------------------
# HEADER & VERIFICATION STATUS
# ---------------------------------------------------------
st.title("⚡ Khung Demo Tích Hợp Sprint 1 — Pipeline Năng Lượng & M&E")
st.markdown("""
**Hệ thống Dự báo & Gợi ý Tối ưu** | Pipeline 4 bước: *Baseline → Bất thường → Dự báo → Gợi ý tối ưu*
""")

# Bảng phân định RÕ RÀNG dữ liệu THẬT vs MOCK
st.info("""
📌 **PHÂN ĐỊNH TRẠNG THÁI DỮ LIỆU TRÊN DASHBOARD (SPRINT 1):**
- 🟢 **DỮ LIỆU THẬT (Dự báo của tôi)**: `value_actual`, `forecast_next_24h`, `forecast_lower`, `forecast_upper` (được tính từ mô hình Hour-of-Week Profile / Seasonal Naive trên 20% tập test cuối chuỗi).
- 🟡 **DỮ LIỆU MOCK / PLACEHOLDER (Chờ Bạn 1 & Bạn 2 ở Sprint 2)**:
  - `baseline_expected`: Giả lập bằng *Trung bình trượt 7 ngày (Rolling 7-day mean)*.
  - `is_anomaly`, `anomaly_reason`: Giả lập quy tắc lệch > 2.5 std so với rolling mean.
""")

# Load Data
df_report = load_baseline_report()
df_fc = load_forecast_data()

if df_fc.empty:
    st.error("Không tìm thấy file `forecast_test_results.parquet`. Vui lòng chạy `python scripts/build_baseline_forecast.py` trước.")
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
    value=(min_ts.date(), max_ts.date()),
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
# Sprint 1 mock logic: Rolling 7-day mean
building_df['baseline_expected'] = building_df['value_actual'].rolling(window=168, min_periods=24).mean()

# [PLACEHOLDER TEAMMATE 2 - ANOMALY]
# Sprint 1 mock logic: Lệch > 2.5 std so với rolling mean
diff = np.abs(building_df['value_actual'] - building_df['baseline_expected'])
threshold = building_df['value_actual'].std() * 2.0
building_df['is_anomaly'] = diff > threshold
building_df['anomaly_reason'] = np.where(
    building_df['is_anomaly'],
    "MOCK: Phụ tải lệch quá 2.0 std so với baseline kỳ vọng",
    None
)

# ---------------------------------------------------------
# TOP METRICS CARD (REAL BASELINE REPORT EVALUATION)
# ---------------------------------------------------------
st.subheader(f"📊 Đánh Giá Hiệu Năng Mô Hình Naive — {selected_building} ({selected_metric})")

if not df_report.empty:
    b_report = df_report[(df_report['building_id'] == selected_building) & (df_report['metric'] == selected_metric)]
    col1, col2, col3, col4 = st.columns(4)
    
    sn_row = b_report[b_report['model'] == 'seasonal_naive_168h']
    hw_row = b_report[b_report['model'] == 'hour_of_week_profile']
    
    with col1:
        sn_mae = sn_row['MAE'].values[0] if not sn_row.empty else "N/A"
        st.metric("Seasonal Naive (168h) MAE", f"{sn_mae} kW" if sn_mae != "N/A" else "N/A")
    with col2:
        sn_mape = sn_row['MAPE'].values[0] if not sn_row.empty else "N/A"
        st.metric("Seasonal Naive (168h) MAPE", f"{sn_mape}%" if sn_mape != "N/A" else "N/A")
    with col3:
        hw_mae = hw_row['MAE'].values[0] if not hw_row.empty else "N/A"
        st.metric("Hour-of-Week Profile MAE", f"{hw_mae} kW" if hw_mae != "N/A" else "N/A")
    with col4:
        hw_mape = hw_row['MAPE'].values[0] if not hw_row.empty else "N/A"
        st.metric("Hour-of-Week Profile MAPE", f"{hw_mape}%" if hw_mape != "N/A" else "N/A")

# ---------------------------------------------------------
# MAIN TIME-SERIES CHART
# ---------------------------------------------------------
st.subheader(f"📈 Biểu Đồ Chuỗi Thời Gian & Dự Báo 24h — {selected_building}")

# Use Streamlit line chart / native matplotlib for clean display
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(14, 5))

# Actual values (Real)
ax.plot(building_df['timestamp'], building_df['value_actual'], label='Thực tế (Real Actual)', color='#1f77b4', linewidth=1.5)

# Baseline expected (Mock Teammate 1)
ax.plot(building_df['timestamp'], building_df['baseline_expected'], label='Baseline kỳ vọng (Mock Bạn 1)', color='#ff7f0e', linestyle='--', alpha=0.8)

# Forecast next 24h (Real Forecast)
ax.plot(building_df['timestamp'], building_df['forecast_next_24h'], label='Dự báo (Real Forecast - Me)', color='#2ca02c', linestyle=':', linewidth=2)

# Upper & Lower bounds band (Real Confidence Interval)
ax.fill_between(
    building_df['timestamp'],
    building_df['forecast_lower'],
    building_df['forecast_upper'],
    color='#2ca02c',
    alpha=0.15,
    label='Khoảng tin cậy 80% (P10 - P90)'
)

# Anomaly markers (Mock Teammate 2)
anomalies = building_df[building_df['is_anomaly']]
if not anomalies.empty:
    ax.scatter(anomalies['timestamp'], anomalies['value_actual'], color='red', s=40, zorder=5, label='Bất thường (Mock Bạn 2)')

ax.set_xlabel("Thời gian (Timestamp)")
ax.set_ylabel(f"Giá trị ({selected_metric})")
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend(loc='upper right')

st.pyplot(fig)

# ---------------------------------------------------------
# UNIFIED DEMO SCHEMA TABLE
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
    width='stretch'
)

st.caption("Hiển thị 100 dòng đầu tiên theo khung bộ lọc đã chọn.")
